#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


MAGIC = b"GGMLMOEPACKv1\0\0\0"
HEADER_STRUCT = struct.Struct("<16sIIQQ")
ENTRY_STRUCT = struct.Struct("<128siIQQ")
COPY_CHUNK = 16 * 1024 * 1024


@dataclass(frozen=True)
class PackEntry:
    tensor: str
    expert_idx: int
    offset: int
    nbytes: int
    ordinal: int


@dataclass(frozen=True)
class ProfileEntry:
    tensor: str
    expert_idx: int
    count: int
    expert_bytes: int
    rank: int


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def tensor_sort_key(tensor: str) -> tuple[int, int, int, str]:
    match = re.search(r"(?:^|\.)blk\.(\d+)\.", tensor)
    layer = int(match.group(1)) if match else 1_000_000
    if ".ffn_up_exps." in tensor:
        kind = 0
    elif ".ffn_gate_exps." in tensor:
        kind = 1
    elif ".ffn_down_exps." in tensor:
        kind = 2
    elif ".ffn_gate_up_exps." in tensor:
        kind = 3
    else:
        kind = 9
    return layer, kind, len(tensor), tensor


def read_pack_index(path: Path) -> tuple[list[tuple[str, int]], dict[tuple[str, int], PackEntry]]:
    order: list[tuple[str, int]] = []
    index: dict[tuple[str, int], PackEntry] = {}
    with path.open("rb") as f:
        raw = f.read(HEADER_STRUCT.size)
        if len(raw) != HEADER_STRUCT.size:
            raise SystemExit(f"source pack header is truncated: {path}")
        magic, version, _header_size, n_entries, _data_start = HEADER_STRUCT.unpack(raw)
        if magic != MAGIC or version != 1:
            raise SystemExit(f"not a v1 expert pack: {path}")
        for ordinal in range(int(n_entries)):
            raw = f.read(ENTRY_STRUCT.size)
            if len(raw) != ENTRY_STRUCT.size:
                raise SystemExit(f"source pack index is truncated: {path}")
            name_raw, expert_idx, _reserved, offset, nbytes = ENTRY_STRUCT.unpack(raw)
            tensor = name_raw.split(b"\0", 1)[0].decode("utf-8", errors="replace")
            key = (tensor, int(expert_idx))
            order.append(key)
            index[key] = PackEntry(tensor, int(expert_idx), int(offset), int(nbytes), ordinal)
    return order, index


def read_profile(path: Path) -> list[ProfileEntry]:
    rows: list[ProfileEntry] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(ProfileEntry(
                tensor=row["tensor"],
                expert_idx=int(row["expert_idx"]),
                count=int(row.get("count") or "0"),
                expert_bytes=int(row.get("expert_bytes") or "0"),
                rank=int(row.get("rank") or (len(rows) + 1)),
            ))
    if not rows:
        raise SystemExit(f"profile has no rows: {path}")
    return rows


def read_trace_first_order(path: Path | None) -> list[tuple[str, int]]:
    if path is None:
        return []
    order: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tensor = row.get("tensor") or ""
            if not tensor:
                continue
            try:
                expert_idx = int(row.get("expert_idx") or "-1")
            except ValueError:
                continue
            if expert_idx < 0:
                continue
            key = (tensor, expert_idx)
            if key not in seen:
                seen.add(key)
                order.append(key)
    return order


def unique_extend(dst: list[tuple[str, int]], seen: set[tuple[str, int]], keys: list[tuple[str, int]]) -> None:
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        dst.append(key)


def choose_order(
        layout: str,
        source_order: list[tuple[str, int]],
        profile: list[ProfileEntry],
        trace_order: list[tuple[str, int]]) -> list[tuple[str, int]]:
    profile_keys = [(row.tensor, row.expert_idx) for row in profile]
    if layout == "profile-hot-first":
        primary = profile_keys
    elif layout == "trace-order-first":
        primary = trace_order
    elif layout == "layer-kind-expert":
        profile_set = set(profile_keys)
        primary = sorted(profile_set, key=lambda key: (*tensor_sort_key(key[0]), key[1]))
    else:
        raise SystemExit(f"unknown layout: {layout}")

    out: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    unique_extend(out, seen, primary)
    unique_extend(out, seen, profile_keys)
    # Keep deterministic coverage for any source key only when a full-pack
    # reorder is explicitly requested in a future run. The current task builds
    # a hot subset, so do not append source_order here.
    _ = source_order
    return out


def copy_exact(src: BinaryIO, dst: BinaryIO, offset: int, nbytes: int) -> None:
    src.seek(offset)
    remaining = nbytes
    while remaining:
        chunk = src.read(min(COPY_CHUNK, remaining))
        if not chunk:
            raise RuntimeError(f"short read from source pack at offset {offset}")
        dst.write(chunk)
        remaining -= len(chunk)


def hash_slice(path: Path, offset: int, nbytes: int) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        f.seek(offset)
        remaining = nbytes
        while remaining:
            chunk = f.read(min(4 * 1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError(f"short read while hashing {path} at {offset}")
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def sample_indices(n: int, samples: int) -> list[int]:
    if n <= 0 or samples <= 0:
        return []
    points = {0, n - 1}
    if samples > 2:
        for i in range(1, samples - 1):
            points.add((i * (n - 1)) // (samples - 1))
    return sorted(points)


def write_reordered_pack(
        source_pack: Path,
        output: Path,
        manifest: Path,
        keys: list[tuple[str, int]],
        index: dict[tuple[str, int], PackEntry],
        alignment: int,
        force: bool,
        hash_samples: int) -> dict[str, object]:
    if output.resolve() == source_pack.resolve():
        raise SystemExit("refusing to overwrite source pack")
    if output.exists() and not force:
        raise SystemExit(f"output exists, use --force to replace: {output}")
    if output.with_suffix(output.suffix + ".tmp").exists():
        raise SystemExit(f"temporary output exists, remove it manually first: {output}.tmp")

    entries: list[PackEntry] = []
    missing: list[tuple[str, int]] = []
    for key in keys:
        entry = index.get(key)
        if entry is None:
            missing.append(key)
            continue
        if len(entry.tensor.encode("utf-8")) > 127:
            raise SystemExit(f"tensor name too long for pack index: {entry.tensor}")
        entries.append(entry)
    if not entries:
        raise SystemExit("no selected entries are present in source pack")

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    index_size = ENTRY_STRUCT.size * len(entries)
    data_start = align_up(HEADER_STRUCT.size + index_size, alignment)
    payload_bytes = 0
    written: list[PackEntry] = []
    started = time.time()

    with source_pack.open("rb") as src, tmp.open("wb") as out:
        out.write(b"\0" * data_start)
        for i, entry in enumerate(entries, 1):
            pad = align_up(out.tell(), alignment) - out.tell()
            if pad:
                out.write(b"\0" * pad)
            dst_offset = out.tell()
            copy_exact(src, out, entry.offset, entry.nbytes)
            payload_bytes += entry.nbytes
            written.append(PackEntry(entry.tensor, entry.expert_idx, dst_offset, entry.nbytes, entry.ordinal))
            tail = align_up(out.tell(), alignment) - out.tell()
            if tail:
                out.write(b"\0" * tail)
            if i % 512 == 0:
                elapsed = max(time.time() - started, 1e-6)
                print(
                    f"copied {i}/{len(entries)} entries "
                    f"payload_gib={payload_bytes / 2**30:.2f} rate_mib_s={payload_bytes / 2**20 / elapsed:.1f}",
                    flush=True,
                )

        out.seek(0)
        out.write(HEADER_STRUCT.pack(MAGIC, 1, HEADER_STRUCT.size, len(written), data_start))
        for entry in written:
            name = entry.tensor.encode("utf-8")
            out.write(ENTRY_STRUCT.pack(
                name + b"\0" * (128 - len(name)),
                entry.expert_idx,
                0,
                entry.offset,
                entry.nbytes,
            ))

    shutil.move(str(tmp), str(output))

    sample_rows = []
    for idx in sample_indices(len(written), hash_samples):
        src_entry = entries[idx]
        dst_entry = written[idx]
        src_hash = hash_slice(source_pack, src_entry.offset, src_entry.nbytes)
        dst_hash = hash_slice(output, dst_entry.offset, dst_entry.nbytes)
        sample_rows.append({
            "entry_index": idx,
            "tensor": src_entry.tensor,
            "expert_idx": src_entry.expert_idx,
            "nbytes": src_entry.nbytes,
            "source_offset": src_entry.offset,
            "output_offset": dst_entry.offset,
            "sha256_match": src_hash == dst_hash,
            "sha256": src_hash,
        })
        if src_hash != dst_hash:
            raise SystemExit(f"hash mismatch for sample entry {idx}: {src_entry.tensor}:{src_entry.expert_idx}")

    result: dict[str, object] = {
        "source_pack": str(source_pack),
        "output_pack": str(output),
        "entries": len(written),
        "missing": len(missing),
        "payload_bytes": payload_bytes,
        "payload_gib": payload_bytes / 2**30,
        "file_size_bytes": output.stat().st_size,
        "alignment": alignment,
        "data_start": data_start,
        "elapsed_s": time.time() - started,
        "hash_samples": sample_rows,
    }
    manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a physically reordered MoE expert-pack hot subset.")
    parser.add_argument("--source-pack", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--layout", choices=("profile-hot-first", "trace-order-first", "layer-kind-expert"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--align", type=int, default=4096)
    parser.add_argument("--max-rows", type=int, default=0, help="Limit selected rows for debugging; <=0 means all selected rows.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--hash-samples", type=int, default=8)
    args = parser.parse_args()
    if args.align <= 0 or args.align & (args.align - 1):
        raise SystemExit("--align must be a positive power of two")
    if not args.source_pack.is_file():
        raise SystemExit(f"missing source pack: {args.source_pack}")
    if not args.profile.is_file():
        raise SystemExit(f"missing profile: {args.profile}")
    if args.trace is not None and not args.trace.is_file():
        raise SystemExit(f"missing trace: {args.trace}")
    return args


def main() -> int:
    args = parse_args()
    source_order, source_index = read_pack_index(args.source_pack)
    profile = read_profile(args.profile)
    trace_order = read_trace_first_order(args.trace)
    keys = choose_order(args.layout, source_order, profile, trace_order)
    if args.max_rows > 0:
        keys = keys[:args.max_rows]

    result = write_reordered_pack(
        args.source_pack,
        args.output,
        args.manifest,
        keys,
        source_index,
        args.align,
        args.force,
        args.hash_samples,
    )
    result["layout"] = args.layout
    result["profile"] = str(args.profile)
    result["trace"] = str(args.trace) if args.trace else None
    args.manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
