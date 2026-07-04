#!/usr/bin/env python3
import argparse
import csv
import os
import struct
from collections import defaultdict
from pathlib import Path


PACK_MAGIC = b"GGMLMOEPACKv1\0\0\0"
PACK_HEADER = struct.Struct("<16sIIQQ")
PACK_ENTRY = struct.Struct("<128siIQQ")
ALIGNMENT = 4096


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def load_trace_first_use(path: Path) -> dict[str, list[int]]:
    order: dict[str, list[int]] = defaultdict(list)
    seen: dict[str, set[int]] = defaultdict(set)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("batch_seq") == "batch_seq":
                continue
            tensor = row["tensor"]
            expert_idx = int(row["expert_idx"])
            if expert_idx not in seen[tensor]:
                seen[tensor].add(expert_idx)
                order[tensor].append(expert_idx)
    return order


def load_pack(path: Path) -> tuple[int, list[dict]]:
    entries = []
    with path.open("rb") as f:
        header = f.read(PACK_HEADER.size)
        if len(header) != PACK_HEADER.size:
            raise RuntimeError(f"{path}: short pack header")
        magic, version, header_size, n_entries, data_start = PACK_HEADER.unpack(header)
        if magic != PACK_MAGIC or version != 1 or header_size < PACK_HEADER.size or data_start < header_size:
            raise RuntimeError(f"{path}: invalid pack header")
        for entry_idx in range(n_entries):
            raw = f.read(PACK_ENTRY.size)
            if len(raw) != PACK_ENTRY.size:
                raise RuntimeError(f"{path}: short pack index")
            name_raw, expert_idx, _reserved, offset, nbytes = PACK_ENTRY.unpack(raw)
            name = name_raw.split(b"\0", 1)[0].decode("utf-8")
            if len(name.encode("utf-8")) >= 128:
                raise RuntimeError(f"{path}: tensor name too long: {name}")
            entries.append({
                "entry_idx": entry_idx,
                "name": name,
                "expert_idx": expert_idx,
                "source_offset": offset,
                "nbytes": nbytes,
            })
    return data_start, entries


def build_physical_order(entries: list[dict], first_use: dict[str, list[int]], mode: str) -> list[dict]:
    by_tensor: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        by_tensor[entry["name"]].append(entry)

    ordered = []
    for tensor in sorted(by_tensor):
        tensor_entries = sorted(by_tensor[tensor], key=lambda e: e["expert_idx"])
        by_expert = {e["expert_idx"]: e for e in tensor_entries}
        if mode == "current":
            tensor_order = [e["expert_idx"] for e in tensor_entries]
        elif mode == "first-use":
            used = [e for e in first_use.get(tensor, []) if e in by_expert]
            used_set = set(used)
            tensor_order = used + [e["expert_idx"] for e in tensor_entries if e["expert_idx"] not in used_set]
        else:
            raise RuntimeError(f"unsupported mode: {mode}")
        ordered.extend(by_expert[expert_idx] for expert_idx in tensor_order)
    return ordered


def copy_exact(src, dst, nbytes: int, chunk_size: int) -> None:
    remaining = nbytes
    while remaining:
        chunk = min(chunk_size, remaining)
        data = src.read(chunk)
        if len(data) != chunk:
            raise RuntimeError("short read while copying expert-pack data")
        dst.write(data)
        remaining -= chunk


def write_reordered_pack(src_path: Path, out_path: Path, entries: list[dict], physical_order: list[dict], chunk_size: int) -> None:
    index_bytes = len(entries) * PACK_ENTRY.size
    data_start = align_up(PACK_HEADER.size + index_bytes)
    offset = data_start
    by_entry_idx = {}
    for entry in physical_order:
        offset = align_up(offset)
        copied = dict(entry)
        copied["pack_offset"] = offset
        by_entry_idx[entry["entry_idx"]] = copied
        offset += entry["nbytes"]
    total_size = offset

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_path.with_suffix(out_path.suffix + ".tmp")
    copied_bytes = 0
    next_report = 16 * 1024 * 1024 * 1024
    with src_path.open("rb") as src, tmp_out.open("wb") as out:
        out.write(PACK_HEADER.pack(PACK_MAGIC, 1, PACK_HEADER.size, len(entries), data_start))
        for original in entries:
            item = by_entry_idx[original["entry_idx"]]
            name_bytes = item["name"].encode("utf-8")
            out.write(PACK_ENTRY.pack(
                name_bytes + b"\0" * (128 - len(name_bytes)),
                item["expert_idx"],
                0,
                item["pack_offset"],
                item["nbytes"],
            ))
        pad = data_start - out.tell()
        if pad < 0:
            raise RuntimeError("internal error: negative index padding")
        out.write(b"\0" * pad)

        for item in physical_order:
            pos = out.tell()
            planned = by_entry_idx[item["entry_idx"]]
            if pos > planned["pack_offset"]:
                raise RuntimeError("internal error: overlapping output offsets")
            out.write(b"\0" * (planned["pack_offset"] - pos))
            src.seek(item["source_offset"])
            copy_exact(src, out, item["nbytes"], chunk_size)
            copied_bytes += item["nbytes"]
            if copied_bytes >= next_report:
                print(f"copied={copied_bytes} bytes", flush=True)
                next_report += 16 * 1024 * 1024 * 1024

    os.replace(tmp_out, out_path)
    print(f"wrote {out_path}")
    print(f"entries={len(entries)} data_start={data_start} size={total_size} copied_bytes={copied_bytes}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Reorder a GGMLMOEPACKv1 file by route first-use without changing tensor bytes.")
    parser.add_argument("--pack", required=True, type=Path, help="Input GGMLMOEPACKv1 file.")
    parser.add_argument("--trace", required=True, type=Path, help="io-read-trace.csv.")
    parser.add_argument("--out", required=True, type=Path, help="Output reordered pack.")
    parser.add_argument("--mode", choices=["first-use", "current"], default="first-use")
    parser.add_argument("--chunk-size", type=int, default=64 * 1024 * 1024)
    args = parser.parse_args()

    if args.pack.resolve() == args.out.resolve():
        raise RuntimeError("--out must differ from --pack")
    if args.chunk_size <= 0:
        raise RuntimeError("--chunk-size must be positive")

    first_use = load_trace_first_use(args.trace)
    _data_start, entries = load_pack(args.pack)
    physical_order = build_physical_order(entries, first_use, args.mode)
    if len(physical_order) != len(entries):
        raise RuntimeError("internal error: physical order length mismatch")
    print(f"pack={args.pack}")
    print(f"trace={args.trace}")
    print(f"out={args.out}")
    print(f"mode={args.mode}")
    print(f"entries={len(entries)} tensors={len({e['name'] for e in entries})}")
    write_reordered_pack(args.pack, args.out, entries, physical_order, args.chunk_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
