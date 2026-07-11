#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import csv
import os
from pathlib import Path


PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
GIB = 1024 ** 3
MIB = 1024 ** 2

PROT_NONE = 0
MAP_SHARED = 0x01
MAP_FAILED = ctypes.c_void_p(-1).value

libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_long,
]
libc.mincore.restype = ctypes.c_int
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
libc.munmap.restype = ctypes.c_int
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]


def cached_bytes(path: Path, chunk_bytes: int) -> tuple[int, int]:
    size = path.stat().st_size
    if size == 0:
        return 0, 0
    total_cached_pages = 0
    total_pages = (size + PAGE_SIZE - 1) // PAGE_SIZE
    fd = os.open(path, os.O_RDONLY)
    try:
        offset = 0
        while offset < size:
            length = min(chunk_bytes, size - offset)
            map_len = ((length + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE
            addr = libc.mmap(None, map_len, PROT_NONE, MAP_SHARED, fd, offset)
            if addr == MAP_FAILED:
                err = ctypes.get_errno()
                raise OSError(err, os.strerror(err), str(path))
            try:
                npages = map_len // PAGE_SIZE
                vec = (ctypes.c_ubyte * npages)()
                ret = libc.mincore(ctypes.c_void_p(addr), map_len, ctypes.byref(vec))
                if ret != 0:
                    err = ctypes.get_errno()
                    raise OSError(err, os.strerror(err), str(path))
                valid_pages = (length + PAGE_SIZE - 1) // PAGE_SIZE
                total_cached_pages += sum(1 for i in range(valid_pages) if vec[i] & 1)
            finally:
                libc.munmap(ctypes.c_void_p(addr), map_len)
            offset += length
    finally:
        os.close(fd)
    return min(total_cached_pages * PAGE_SIZE, size), total_pages * PAGE_SIZE


def label_for(path: Path) -> str:
    name = path.name
    if name.endswith(".gguf"):
        return "gguf_shard"
    if name.endswith(".expert-pack"):
        return "expert_pack"
    if name.endswith(".tsv"):
        return "alias_tsv"
    return "other"


def discover(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path.is_dir():
            candidates = sorted(path.glob("*.gguf")) + sorted(path.glob("*.expert-pack")) + sorted(path.glob("*.tsv"))
        else:
            candidates = [path]
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except FileNotFoundError:
                continue
            if resolved.exists() and resolved.is_file() and resolved not in seen:
                seen.add(resolved)
                out.append(resolved)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure page-cache residency for Kimi model/expert files with mincore().")
    ap.add_argument("paths", nargs="+", type=Path, help="Files or directories to inspect.")
    ap.add_argument("--chunk-mib", type=int, default=256)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    files = discover(args.paths)
    rows = []
    chunk = args.chunk_mib * MIB
    for path in files:
        cached, rounded_size = cached_bytes(path, chunk)
        size = path.stat().st_size
        rows.append(
            {
                "label": label_for(path),
                "path": str(path),
                "size_bytes": size,
                "rounded_size_bytes": rounded_size,
                "cached_bytes": cached,
                "cached_gib": f"{cached / GIB:.6f}",
                "cached_pct": f"{(100.0 * cached / size) if size else 0.0:.3f}",
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        fields = ["label", "path", "size_bytes", "rounded_size_bytes", "cached_bytes", "cached_gib", "cached_pct"]
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    by_label: dict[str, int] = {}
    for row in rows:
        by_label[row["label"]] = by_label.get(row["label"], 0) + int(row["cached_bytes"])
    for label, value in sorted(by_label.items(), key=lambda item: -item[1]):
        print(f"{label}\t{value}\t{value / GIB:.3f} GiB")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
