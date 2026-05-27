#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import shutil
import struct
from pathlib import Path


MAGIC = b"GGMLMOEPACKv1\0\0\0"
HEADER_STRUCT = struct.Struct("<16sIIQQ")
ENTRY_STRUCT = struct.Struct("<128siIQQ")
COPY_CHUNK = 16 * 1024 * 1024


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def read_source_index(path: Path) -> dict[tuple[str, int], tuple[int, int]]:
    with path.open("rb") as f:
        header = f.read(HEADER_STRUCT.size)
        if len(header) != HEADER_STRUCT.size:
            raise SystemExit(f"source expert pack is truncated: {path}")
        magic, version, _header_size, n_entries, _data_start = HEADER_STRUCT.unpack(header)
        if magic != MAGIC or version != 1:
            raise SystemExit(f"not a v1 MoE expert pack: {path}")

        index: dict[tuple[str, int], tuple[int, int]] = {}
        for _ in range(n_entries):
            raw = f.read(ENTRY_STRUCT.size)
            if len(raw) != ENTRY_STRUCT.size:
                raise SystemExit(f"source expert pack index is truncated: {path}")
            name_raw, expert_idx, _reserved, offset, nbytes = ENTRY_STRUCT.unpack(raw)
            name = name_raw.split(b"\0", 1)[0].decode("utf-8", errors="replace")
            index[(name, int(expert_idx))] = (int(offset), int(nbytes))
        return index


def read_trace_routes(path: Path) -> list[tuple[str, int]]:
    allowed_ops = {"cpu_up_route", "cpu_gate_route", "cpu_down_route"}
    routes: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("op") not in allowed_ops:
                continue
            tensor = row.get("tensor") or ""
            if not tensor:
                continue
            try:
                expert = int(row.get("expert_idx") or "-1")
            except ValueError:
                continue
            if expert < 0:
                continue
            key = (tensor, expert)
            if key in seen:
                continue
            seen.add(key)
            routes.append(key)
    if not routes:
        raise SystemExit(f"trace contains no routed prompt expert rows: {path}")
    return routes


def tensor_sort_key(name: str) -> tuple[int, int, str]:
    match = re.search(r"(?:^|\.)blk\.(\d+)\.", name)
    layer = int(match.group(1)) if match else 1_000_000
    if ".ffn_up_exps." in name:
        kind = 0
    elif ".ffn_gate_exps." in name:
        kind = 1
    elif ".ffn_down_exps." in name:
        kind = 2
    else:
        kind = 9
    return layer, kind, name


def order_routes(routes: list[tuple[str, int]], order: str) -> list[tuple[str, int]]:
    if order == "trace":
        return routes
    if order == "compute":
        return sorted(routes, key=lambda item: (*tensor_sort_key(item[0]), item[1]))
    raise SystemExit(f"unknown route order: {order}")


def copy_slice(src, dst, offset: int, nbytes: int) -> None:
    src.seek(offset)
    remaining = nbytes
    while remaining > 0:
        chunk = src.read(min(COPY_CHUNK, remaining))
        if not chunk:
            raise RuntimeError(f"short read from source expert pack at {offset}")
        dst.write(chunk)
        remaining -= len(chunk)


def write_ttft_pack(
    source_pack: Path,
    output: Path,
    routes: list[tuple[str, int]],
    source_index: dict[tuple[str, int], tuple[int, int]],
    alignment: int,
) -> tuple[int, int, int]:
    entries: list[tuple[str, int, int, int]] = []
    missing = 0
    for tensor, expert in routes:
        item = source_index.get((tensor, expert))
        if item is None:
            missing += 1
            continue
        offset, nbytes = item
        if len(tensor.encode("utf-8")) > 127:
            raise SystemExit(f"tensor name too long for pack index: {tensor}")
        entries.append((tensor, expert, offset, nbytes))
    if not entries:
        raise SystemExit("no traced routes were present in the source expert pack")

    index_size = ENTRY_STRUCT.size * len(entries)
    data_start = align_up(HEADER_STRUCT.size + index_size, alignment)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    payload = 0
    written_index: list[tuple[str, int, int, int]] = []
    with source_pack.open("rb") as src, tmp.open("wb") as out:
        out.write(b"\0" * data_start)
        for tensor, expert, src_offset, nbytes in entries:
            pos = out.tell()
            pad = align_up(pos, alignment) - pos
            if pad:
                out.write(b"\0" * pad)
            dst_offset = out.tell()
            copy_slice(src, out, src_offset, nbytes)
            payload += nbytes
            written_index.append((tensor, expert, dst_offset, nbytes))
            tail = align_up(out.tell(), alignment) - out.tell()
            if tail:
                out.write(b"\0" * tail)

        out.seek(0)
        out.write(HEADER_STRUCT.pack(MAGIC, 1, HEADER_STRUCT.size, len(written_index), data_start))
        for tensor, expert, offset, nbytes in written_index:
            name = tensor.encode("utf-8")
            name_buf = name + b"\0" * (128 - len(name))
            out.write(ENTRY_STRUCT.pack(name_buf, expert, 0, offset, nbytes))
    shutil.move(str(tmp), str(output))
    return len(written_index), missing, payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a route-ordered TTFT expert pack from a TTFT trace and full expert pack.")
    parser.add_argument("--trace", type=Path, required=True, help="TTFT trace CSV with cpu_*_route rows.")
    parser.add_argument("--expert-pack", type=Path, required=True, help="Full source .expert-pack.")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output TTFT expert pack.")
    parser.add_argument("--max-expert-rows", type=int, default=512, help="Maximum unique routed expert rows to include; <=0 means all.")
    parser.add_argument("--align", type=int, default=4096, help="Output slice alignment.")
    parser.add_argument("--order", choices=("trace", "compute"), default="trace", help="Output route order.")
    args = parser.parse_args()

    if args.align <= 0 or args.align & (args.align - 1):
        raise SystemExit("--align must be a positive power of two")
    if not args.trace.is_file():
        raise SystemExit(f"missing trace: {args.trace}")
    if not args.expert_pack.is_file():
        raise SystemExit(f"missing expert pack: {args.expert_pack}")

    routes = order_routes(read_trace_routes(args.trace), args.order)
    if args.max_expert_rows > 0:
        routes = routes[:args.max_expert_rows]
    source_index = read_source_index(args.expert_pack)
    count, missing, payload = write_ttft_pack(args.expert_pack, args.output, routes, source_index, args.align)
    print(
        f"wrote {args.output} entries={count} missing={missing} "
        f"payload_gib={payload / float(1024 ** 3):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
