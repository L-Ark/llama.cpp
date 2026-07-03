#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
import struct
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/root/lfz/ik_llama/gguf-py")
from gguf import GGUFReader

MAGIC = b"GGMLMOEPACKv1\0\0\0"
HEADER_STRUCT = struct.Struct("<16sIIQQ")
ENTRY_STRUCT = struct.Struct("<128siIQQ")
COPY_CHUNK = 16 * 1024 * 1024


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def copy_slice(src, off: int, nbytes: int, out) -> None:
    src.seek(off)
    left = nbytes
    while left:
        chunk = src.read(min(COPY_CHUNK, left))
        if not chunk:
            raise RuntimeError(f"short read at offset {off}")
        out.write(chunk)
        left -= len(chunk)


def read_trace_order(paths: list[Path], *, miss_only: bool) -> tuple[list[tuple[str, int]], Counter[tuple[str, int]]]:
    seen: set[tuple[str, int]] = set()
    order: list[tuple[str, int]] = []
    counts: Counter[tuple[str, int]] = Counter()

    for path in paths:
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                if miss_only and int(row.get("cache_hit", "0") or "0"):
                    continue
                tensor = row["tensor"]
                expert = int(row["expert"])
                key = (tensor, expert)
                counts[key] += 1
                if key not in seen:
                    seen.add(key)
                    order.append(key)
    return order, counts


def write_profile(path: Path, keys: list[tuple[str, int]], counts: Counter[tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for tensor, expert in keys:
            f.write(f"{tensor}\t{expert}\t{counts[(tensor, expert)]}\n")


def write_pack(model: Path, outp: Path, keys: list[tuple[str, int]], align: int) -> None:
    reader = GGUFReader(str(model), "r")
    tensors = {t.name: t for t in reader.tensors}
    slices: list[tuple[str, int, int, int]] = []

    for name, expert in keys:
        t = tensors[name]
        shape = [int(x) for x in t.shape.tolist()]
        n_expert = shape[2]
        expert_bytes = int(t.n_bytes) // n_expert
        off = int(t.data_offset) + expert * expert_bytes
        slices.append((name, expert, off, expert_bytes))

    total = sum(s[3] for s in slices)
    print(f"unique pairs: {len(keys)}", file=sys.stderr)
    print(f"payload bytes={total} gib={total / (1024 ** 3):.3f}", file=sys.stderr)

    index_size = ENTRY_STRUCT.size * len(slices)
    data_start = align_up(HEADER_STRUCT.size + index_size, align)
    outp.parent.mkdir(parents=True, exist_ok=True)
    tmp = outp.with_suffix(outp.suffix + ".tmp")
    entries: list[tuple[str, int, int, int]] = []

    with model.open("rb") as src, tmp.open("wb") as out:
        out.write(b"\0" * data_start)
        for i, (name, expert, off, nbytes) in enumerate(slices, 1):
            pos = out.tell()
            pad = align_up(pos, align) - pos
            if pad:
                out.write(b"\0" * pad)
            dst_off = out.tell()
            copy_slice(src, off, nbytes, out)
            tail = align_up(out.tell(), align) - out.tell()
            if tail:
                out.write(b"\0" * tail)
            entries.append((name, expert, dst_off, nbytes))
            if i % 256 == 0:
                print(f"packed {i}/{len(slices)}", file=sys.stderr)

        out.seek(0)
        out.write(HEADER_STRUCT.pack(MAGIC, 1, HEADER_STRUCT.size, len(entries), data_start))
        for name, expert, dst_off, nbytes in entries:
            nb = name.encode("utf-8")
            if len(nb) >= 128:
                raise ValueError(f"tensor name too long for pack entry: {name}")
            out.write(ENTRY_STRUCT.pack(nb + b"\0" * (128 - len(nb)), expert, 0, dst_off, nbytes))

    shutil.move(str(tmp), str(outp))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf")
    ap.add_argument("--trace", action="append", type=Path, required=True)
    ap.add_argument("--profile-out", type=Path, required=True)
    ap.add_argument("--pack-out", type=Path)
    ap.add_argument("--include-cache-hits", action="store_true", help="Include all trace rows. Default packs cold cache misses only.")
    ap.add_argument("--align", type=int, default=4096)
    args = ap.parse_args()

    keys, counts = read_trace_order(args.trace, miss_only=not args.include_cache_hits)
    write_profile(args.profile_out, keys, counts)
    print(args.profile_out)
    if args.pack_out:
        write_pack(Path(args.model), args.pack_out, keys, args.align)
        print(args.pack_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
