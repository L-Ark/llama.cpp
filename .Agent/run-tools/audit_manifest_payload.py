#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def read_manifest(path: Path, limit: int | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "tensor": row["tensor"],
                "expert": int(row["expert"]),
                "model_offset": int(row["model_offset"]),
                "nbytes": int(row["nbytes"]),
            })
            if limit is not None and len(rows) >= limit:
                break
    return rows


def stream_payload(model: Path, manifest: Path, limit: int | None, chunk_size: int) -> int:
    rows = read_manifest(manifest, limit)
    out_fd = sys.stdout.buffer.fileno()
    total = 0
    with model.open("rb", buffering=0) as f:
        for row in rows:
            f.seek(int(row["model_offset"]))
            remaining = int(row["nbytes"])
            while remaining > 0:
                data = f.read(min(chunk_size, remaining))
                if not data:
                    raise RuntimeError(f"short read at offset={row['model_offset']} remaining={remaining}")
                os.write(out_fd, data)
                total += len(data)
                remaining -= len(data)
    return total


def sampled_block_indices(total_blocks: int, blocks_per_entry: int) -> list[int]:
    if total_blocks <= 0 or blocks_per_entry <= 0:
        return []
    if total_blocks <= blocks_per_entry:
        return list(range(total_blocks))
    if blocks_per_entry == 1:
        return [0]
    return sorted({int(round(i * (total_blocks - 1) / (blocks_per_entry - 1))) for i in range(blocks_per_entry)})


def sample_dedup(
    model: Path,
    manifest: Path,
    entries: int,
    block_size: int,
    blocks_per_entry: int,
    out: Path,
) -> dict[str, object]:
    rows = read_manifest(manifest, entries)
    seen: dict[bytes, int] = {}
    zero_blocks = 0
    total_blocks = 0
    duplicate_blocks = 0
    sampled_bytes = 0
    per_entry = []
    started = time.time()

    with model.open("rb", buffering=0) as f:
        for row in rows:
            nbytes = int(row["nbytes"])
            n_blocks = nbytes // block_size
            indices = sampled_block_indices(n_blocks, blocks_per_entry)
            entry_dups = 0
            entry_zero = 0
            for idx in indices:
                f.seek(int(row["model_offset"]) + idx * block_size)
                data = f.read(block_size)
                if len(data) != block_size:
                    raise RuntimeError(f"short block read tensor={row['tensor']} expert={row['expert']} block={idx}")
                digest = hashlib.blake2b(data, digest_size=12).digest()
                prev = seen.get(digest, 0)
                if prev:
                    duplicate_blocks += 1
                    entry_dups += 1
                seen[digest] = prev + 1
                if not any(data):
                    zero_blocks += 1
                    entry_zero += 1
                total_blocks += 1
                sampled_bytes += block_size
            per_entry.append({
                "tensor": row["tensor"],
                "expert": row["expert"],
                "sampled_blocks": len(indices),
                "duplicate_blocks": entry_dups,
                "zero_blocks": entry_zero,
            })

    result = {
        "manifest": str(manifest),
        "model": str(model),
        "entries_sampled": len(rows),
        "block_size": block_size,
        "blocks_per_entry": blocks_per_entry,
        "sampled_blocks": total_blocks,
        "sampled_bytes": sampled_bytes,
        "unique_sampled_blocks": len(seen),
        "duplicate_blocks": duplicate_blocks,
        "duplicate_ratio": duplicate_blocks / total_blocks if total_blocks else 0.0,
        "zero_blocks": zero_blocks,
        "zero_ratio": zero_blocks / total_blocks if total_blocks else 0.0,
        "elapsed_s": time.time() - started,
        "per_entry_preview": per_entry[:16],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(out)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Stream or sample GGUF byte ranges from an expert offset manifest.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("stream", help="write selected payload bytes to stdout")
    sp.add_argument("--model", required=True, type=Path)
    sp.add_argument("--manifest", required=True, type=Path)
    sp.add_argument("--limit", type=int)
    sp.add_argument("--chunk-size", type=int, default=1024 * 1024)

    dp = sub.add_parser("sample-dedup", help="sample fixed-size blocks and report exact duplicate/zero rates")
    dp.add_argument("--model", required=True, type=Path)
    dp.add_argument("--manifest", required=True, type=Path)
    dp.add_argument("--entries", type=int, default=256)
    dp.add_argument("--block-size", type=int, default=17)
    dp.add_argument("--blocks-per-entry", type=int, default=4096)
    dp.add_argument("--out", required=True, type=Path)

    args = ap.parse_args()
    if args.cmd == "stream":
        stream_payload(args.model, args.manifest, args.limit, args.chunk_size)
        return 0
    if args.cmd == "sample-dedup":
        result = sample_dedup(args.model, args.manifest, args.entries, args.block_size, args.blocks_per_entry, args.out)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
