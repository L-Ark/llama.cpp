#!/usr/bin/env python3
import argparse
import csv
import itertools
import os
import struct
from collections import Counter, defaultdict
from pathlib import Path


PACK_MAGIC = b"GGMLMOEPACKv1\0\0\0"
PACK_HEADER = struct.Struct("<16sIIQQ")
PACK_ENTRY = struct.Struct("<128siIQQ")
ALIGNMENT = 4096


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def load_trace_stats(paths: list[Path], max_jobs: int | None) -> tuple[
        dict[str, list[int]],
        dict[str, set[int]],
        dict[str, Counter[int]],
        dict[str, Counter[tuple[int, int]]],
        int,
        int]:
    first_use: dict[str, list[int]] = defaultdict(list)
    seen: dict[str, set[int]] = defaultdict(set)
    freq: dict[str, Counter[int]] = defaultdict(Counter)
    by_batch_tensor: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    rows = 0
    skipped_jobs = 0
    for path in paths:
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("expert_idx") in (None, "", "expert_idx"):
                    continue
                if max_jobs is not None and row.get("jobs") not in (None, "", "jobs"):
                    try:
                        if int(row["jobs"]) > max_jobs:
                            skipped_jobs += 1
                            continue
                    except ValueError:
                        skipped_jobs += 1
                        continue
                tensor = row["tensor"]
                expert_idx = int(row["expert_idx"])
                rows += 1
                if expert_idx not in seen[tensor]:
                    seen[tensor].add(expert_idx)
                    first_use[tensor].append(expert_idx)
                freq[tensor][expert_idx] += 1
                batch_seq = row.get("batch_seq", "")
                by_batch_tensor[(str(path), batch_seq, tensor)].add(expert_idx)

    edges: dict[str, Counter[tuple[int, int]]] = defaultdict(Counter)
    for (_path, _batch_seq, tensor), experts in by_batch_tensor.items():
        for a, b in itertools.combinations(sorted(experts), 2):
            edges[tensor][(a, b)] += 1
    return first_use, seen, freq, edges, rows, skipped_jobs


def greedy_pair_order(experts: list[int], freq: Counter[int], edges: Counter[tuple[int, int]]) -> list[int]:
    remaining = set(experts)
    order = []
    while remaining:
        if not order:
            cur = max(remaining, key=lambda e: (freq[e], -e))
        else:
            last = order[-1]

            def score(e: int) -> tuple[int, int, int]:
                key = (last, e) if last < e else (e, last)
                return edges[key], freq[e], -e

            cur = max(remaining, key=score)
        order.append(cur)
        remaining.remove(cur)
    return order


def build_trace_order(
        mode: str,
        first_use: dict[str, list[int]],
        seen: dict[str, set[int]],
        freq: dict[str, Counter[int]],
        edges: dict[str, Counter[tuple[int, int]]]) -> dict[str, list[int]]:
    order: dict[str, list[int]] = {}
    for tensor in sorted(seen):
        experts = sorted(seen[tensor])
        if mode == "first-use":
            order[tensor] = first_use[tensor]
        elif mode == "frequency":
            order[tensor] = sorted(experts, key=lambda e: (-freq[tensor][e], e))
        elif mode == "greedy-pair":
            order[tensor] = greedy_pair_order(experts, freq[tensor], edges[tensor])
        else:
            raise RuntimeError(f"unsupported mode: {mode}")
    return order


def load_pack(path: Path, source_order: int) -> list[dict]:
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
                "pack": path,
                "source_order": source_order,
                "entry_idx": entry_idx,
                "name": name,
                "expert_idx": expert_idx,
                "source_offset": offset,
                "nbytes": nbytes,
            })
    return entries


def select_entries(packs: list[Path], trace_order: dict[str, list[int]]) -> list[dict]:
    by_key: dict[tuple[str, int], dict] = {}
    for source_order, pack in enumerate(packs):
        for entry in load_pack(pack, source_order):
            key = (entry["name"], entry["expert_idx"])
            # Match runtime duplicate replacement semantics: later packs win.
            if key not in by_key or entry["source_order"] >= by_key[key]["source_order"]:
                by_key[key] = entry

    selected = []
    missing = 0
    for tensor in sorted(trace_order):
        used = []
        for expert_idx in trace_order[tensor]:
            entry = by_key.get((tensor, expert_idx))
            if entry is None:
                missing += 1
                continue
            used.append(entry)
        selected.extend(used)
    if missing:
        print(f"missing_trace_keys={missing}")
    return selected


def copy_exact(src, dst, nbytes: int, chunk_size: int) -> None:
    remaining = nbytes
    while remaining:
        chunk = min(chunk_size, remaining)
        data = src.read(chunk)
        if len(data) != chunk:
            raise RuntimeError("short read while copying expert-pack data")
        dst.write(data)
        remaining -= chunk


def write_pack(entries: list[dict], out_path: Path, chunk_size: int) -> None:
    index_bytes = len(entries) * PACK_ENTRY.size
    data_start = align_up(PACK_HEADER.size + index_bytes)
    offset = data_start
    planned = []
    for entry in entries:
        offset = align_up(offset)
        item = dict(entry)
        item["pack_offset"] = offset
        planned.append(item)
        offset += entry["nbytes"]
    total_size = offset

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_path.with_suffix(out_path.suffix + ".tmp")
    copied_bytes = 0
    next_report = 8 * 1024 * 1024 * 1024
    open_packs = {}
    try:
        with tmp_out.open("wb") as out:
            out.write(PACK_HEADER.pack(PACK_MAGIC, 1, PACK_HEADER.size, len(planned), data_start))
            for item in planned:
                name_bytes = item["name"].encode("utf-8")
                out.write(PACK_ENTRY.pack(
                    name_bytes + b"\0" * (128 - len(name_bytes)),
                    item["expert_idx"],
                    0,
                    item["pack_offset"],
                    item["nbytes"],
                ))
            out.write(b"\0" * (data_start - out.tell()))

            for item in planned:
                pos = out.tell()
                if pos > item["pack_offset"]:
                    raise RuntimeError("internal error: overlapping output offsets")
                out.write(b"\0" * (item["pack_offset"] - pos))
                pack = item["pack"]
                if pack not in open_packs:
                    open_packs[pack] = pack.open("rb")
                src = open_packs[pack]
                src.seek(item["source_offset"])
                copy_exact(src, out, item["nbytes"], chunk_size)
                copied_bytes += item["nbytes"]
                if copied_bytes >= next_report:
                    print(f"copied={copied_bytes} bytes", flush=True)
                    next_report += 8 * 1024 * 1024 * 1024
    finally:
        for f in open_packs.values():
            f.close()

    os.replace(tmp_out, out_path)
    print(f"wrote {out_path}")
    print(f"entries={len(planned)} data_start={data_start} size={total_size} copied_bytes={copied_bytes}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a trace-only overlay from one or more GGMLMOEPACKv1 files.")
    parser.add_argument("--trace", action="append", required=True, type=Path,
                        help="io-read-trace.csv; repeat for dev-prompt layout training.")
    parser.add_argument("--pack", action="append", required=True, type=Path, help="Input pack; repeat in runtime source order.")
    parser.add_argument("--out", required=True, type=Path, help="Output trace-only overlay pack.")
    parser.add_argument("--mode", choices=["first-use", "frequency", "greedy-pair"], default="first-use",
                        help="Physical order for selected entries.")
    parser.add_argument("--max-jobs", type=int, default=None,
                        help="Ignore trace rows whose jobs column is larger than this value.")
    parser.add_argument("--chunk-size", type=int, default=64 * 1024 * 1024)
    args = parser.parse_args()

    if args.chunk_size <= 0:
        raise RuntimeError("--chunk-size must be positive")
    out_resolved = args.out.resolve()
    if any(pack.resolve() == out_resolved for pack in args.pack):
        raise RuntimeError("--out must differ from all --pack inputs")
    if args.max_jobs is not None and args.max_jobs <= 0:
        raise RuntimeError("--max-jobs must be positive when set")

    first_use, seen, freq, edges, rows, skipped_jobs = load_trace_stats(args.trace, args.max_jobs)
    trace_order = build_trace_order(args.mode, first_use, seen, freq, edges)
    entries = select_entries(args.pack, trace_order)
    total_bytes = sum(entry["nbytes"] for entry in entries)
    print("traces=" + ",".join(str(trace) for trace in args.trace))
    print("packs=" + ",".join(str(pack) for pack in args.pack))
    print(f"out={args.out}")
    print(f"mode={args.mode} max_jobs={args.max_jobs if args.max_jobs is not None else 'none'}")
    print(f"trace_rows={rows} skipped_jobs={skipped_jobs} trace_tensors={len(trace_order)} "
          f"entries={len(entries)} bytes={total_bytes}")
    write_pack(entries, args.out, args.chunk_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
