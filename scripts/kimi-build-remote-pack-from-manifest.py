#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, BinaryIO


PACK_MAGIC = b"GGMLMOEPACKv1\0\0\0"
PACK_HEADER = struct.Struct("<16sIIQQ")
PACK_ENTRY = struct.Struct("<128siIQQ")
ALIGNMENT = 4096
CHUNK_BYTES = 8 * 1024 * 1024


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def git_head(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def nearest_existing_parent(path: Path) -> Path:
    current = path if path.is_dir() else path.parent
    while not current.exists():
        parent = current.parent
        if parent == current:
            raise RuntimeError(f"no existing parent for {path}")
        current = parent
    return current


def int_field(row: dict[str, str], key: str) -> int:
    try:
        return int(row[key])
    except KeyError as exc:
        raise RuntimeError(f"missing required manifest field {key!r}") from exc
    except ValueError as exc:
        raise RuntimeError(f"invalid integer manifest field {key!r}: {row.get(key)!r}") from exc


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise RuntimeError(f"{path}: manifest has no entries")
    return rows


def validate_manifest(rows: list[dict[str, str]]) -> dict[str, Any]:
    data_start = align_up(PACK_HEADER.size + len(rows) * PACK_ENTRY.size)
    expected_min_offset = data_start
    payload_bytes = 0
    mismatch_count = 0
    invalid_range_count = 0
    bad_offset_count = 0
    remote_shards = set()
    selected_tensors = set()
    by_type: dict[str, dict[str, int]] = {}
    by_kind: dict[str, dict[str, int]] = {}
    max_end = data_start

    for idx, row in enumerate(rows):
        entry_idx = int_field(row, "entry_idx")
        if entry_idx != idx:
            raise RuntimeError(f"manifest entry_idx is not contiguous at row {idx}: got {entry_idx}")
        tensor = row["tensor"]
        if len(tensor.encode("utf-8")) >= 128:
            raise RuntimeError(f"tensor name too long for pack v1 index: {tensor}")
        remote_nbytes = int_field(row, "remote_nbytes")
        current_nbytes = int_field(row, "current_nbytes")
        pack_offset = int_field(row, "pack_offset")
        remote_abs_start = int_field(row, "remote_abs_start")
        remote_abs_end = int_field(row, "remote_abs_end")
        if remote_nbytes <= 0:
            raise RuntimeError(f"{tensor}:{row.get('expert_idx')}: remote_nbytes must be positive")
        if remote_abs_end - remote_abs_start + 1 != remote_nbytes:
            invalid_range_count += 1
        if int_field(row, "remote_range_valid") != 1:
            invalid_range_count += 1
        expected_offset = align_up(expected_min_offset)
        if pack_offset != expected_offset or pack_offset % ALIGNMENT != 0:
            bad_offset_count += 1
        expected_min_offset = pack_offset + remote_nbytes
        max_end = max(max_end, expected_min_offset)
        payload_bytes += remote_nbytes
        if current_nbytes != remote_nbytes or int_field(row, "runtime_nbytes_match") != 1:
            mismatch_count += 1
        selected_tensors.add(tensor)
        remote_shards.add(row["remote_shard_name"])
        for bucket in (
            by_type.setdefault(row["remote_type"], {"entries": 0, "bytes": 0}),
            by_kind.setdefault(row["kind"], {"entries": 0, "bytes": 0}),
        ):
            bucket["entries"] += 1
            bucket["bytes"] += remote_nbytes

    return {
        "entries": len(rows),
        "selected_tensors": len(selected_tensors),
        "remote_shards": len(remote_shards),
        "header_bytes": PACK_HEADER.size,
        "entry_bytes": PACK_ENTRY.size,
        "alignment": ALIGNMENT,
        "data_start": data_start,
        "payload_bytes": payload_bytes,
        "estimated_pack_bytes": max_end,
        "runtime_nbytes_mismatch_count": mismatch_count,
        "current_iq3_runtime_compatible": mismatch_count == 0,
        "invalid_remote_range_count": invalid_range_count,
        "bad_pack_offset_count": bad_offset_count,
        "by_remote_type": dict(sorted(by_type.items())),
        "by_kind": dict(sorted(by_kind.items())),
    }


def request_entry(row: dict[str, str]) -> urllib.request.Request:
    return urllib.request.Request(
        row["remote_url"],
        headers={
            "Range": row["remote_range_header"],
            "User-Agent": "kimi-remote-pack-builder/1.0",
        },
    )


def fetch_entry_bytes(row: dict[str, str]) -> bytes:
    expected = int_field(row, "remote_nbytes")
    req = request_entry(row)
    with urllib.request.urlopen(req, timeout=300) as resp:
        status = getattr(resp, "status", resp.getcode())
        if status != 206:
            raise RuntimeError(f"HTTP range request returned status={status}, expected 206")
        data = resp.read(expected + 1)
    if len(data) != expected:
        raise RuntimeError(f"HTTP range length mismatch: got {len(data)}, expected {expected}")
    return data


def copy_entry_to_file(row: dict[str, str], out: BinaryIO) -> str:
    expected = int_field(row, "remote_nbytes")
    remaining = expected
    sha = hashlib.sha256()
    req = request_entry(row)
    with urllib.request.urlopen(req, timeout=600) as resp:
        status = getattr(resp, "status", resp.getcode())
        if status != 206:
            raise RuntimeError(f"HTTP range request returned status={status}, expected 206")
        while remaining > 0:
            chunk = resp.read(min(CHUNK_BYTES, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            sha.update(chunk)
            out.write(chunk)
    if remaining != 0:
        raise RuntimeError(f"short HTTP range read: missing {remaining} bytes")
    return sha.hexdigest()


def run_smoke(rows: list[dict[str, str]], count: int) -> list[dict[str, Any]]:
    out = []
    for row in rows[:count]:
        start = time.time()
        data = fetch_entry_bytes(row)
        elapsed = time.time() - start
        out.append({
            "entry_idx": int_field(row, "entry_idx"),
            "tensor": row["tensor"],
            "expert_idx": int_field(row, "expert_idx"),
            "remote_nbytes": int_field(row, "remote_nbytes"),
            "remote_range_header": row["remote_range_header"],
            "remote_shard_name": row["remote_shard_name"],
            "sha256": hashlib.sha256(data).hexdigest(),
            "seconds": elapsed,
            "bytes_per_second": len(data) / elapsed if elapsed > 0 else 0.0,
        })
    return out


def write_pack(rows: list[dict[str, str]], output_pack: Path, pack_summary: dict[str, Any]) -> dict[str, Any]:
    tmp = output_pack.with_name(output_pack.name + ".tmp")
    if tmp.exists():
        raise RuntimeError(f"temporary output already exists: {tmp}")
    if output_pack.exists():
        raise RuntimeError(f"output pack already exists: {output_pack}")
    output_pack.parent.mkdir(parents=True, exist_ok=True)

    index_bytes = bytearray()
    for row in rows:
        name = row["tensor"].encode("utf-8")
        if len(name) >= 128:
            raise RuntimeError(f"tensor name too long for pack v1 index: {row['tensor']}")
        name_field = name + b"\0" * (128 - len(name))
        index_bytes += PACK_ENTRY.pack(
            name_field,
            int_field(row, "expert_idx"),
            0,
            int_field(row, "pack_offset"),
            int_field(row, "remote_nbytes"),
        )

    sha_rows = []
    with tmp.open("w+b") as f:
        header = PACK_HEADER.pack(
            PACK_MAGIC,
            1,
            PACK_HEADER.size,
            len(rows),
            int(pack_summary["data_start"]),
        )
        f.write(header)
        f.write(index_bytes)
        f.truncate(int(pack_summary["estimated_pack_bytes"]))
        for row in rows:
            f.seek(int_field(row, "pack_offset"))
            start = time.time()
            digest = copy_entry_to_file(row, f)
            elapsed = time.time() - start
            sha_rows.append({
                "entry_idx": int_field(row, "entry_idx"),
                "remote_nbytes": int_field(row, "remote_nbytes"),
                "sha256": digest,
                "seconds": elapsed,
            })
    os.replace(tmp, output_pack)
    return {
        "output_pack": str(output_pack),
        "entries_written": len(rows),
        "bytes_written": int(pack_summary["estimated_pack_bytes"]),
        "payload_sha256_rows": sha_rows[:16],
        "payload_sha256_rows_truncated": len(sha_rows) > 16,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dry-run or build a GGMLMOEPACKv1 directly from a remote manifest."
    )
    parser.add_argument("--manifest-tsv", type=Path, required=True)
    parser.add_argument("--output-pack", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--smoke-entries", type=int, default=0,
                        help="Fetch this many manifest entries into memory for HTTP Range validation.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate inputs without writing output-pack. This is the default unless --execute is set.")
    parser.add_argument("--execute", action="store_true",
                        help="Actually write output-pack. Default is dry-run/no-write.")
    args = parser.parse_args()

    if args.dry_run and args.execute:
        raise RuntimeError("--dry-run and --execute are mutually exclusive")
    if args.smoke_entries < 0:
        raise RuntimeError("--smoke-entries must be non-negative")

    repo_root = Path(__file__).resolve().parents[1]
    rows = load_manifest(args.manifest_tsv)
    pack_summary = validate_manifest(rows)
    target_fs = nearest_existing_parent(args.output_pack)
    disk = shutil.disk_usage(target_fs)
    fits_output_fs = disk.free >= int(pack_summary["estimated_pack_bytes"])
    missing_bytes = max(0, int(pack_summary["estimated_pack_bytes"]) - disk.free)

    if pack_summary["invalid_remote_range_count"] != 0:
        raise RuntimeError("manifest contains invalid remote ranges")
    if pack_summary["bad_pack_offset_count"] != 0:
        raise RuntimeError("manifest contains invalid pack offsets")
    if args.execute and not fits_output_fs:
        raise RuntimeError(
            f"output filesystem lacks space: missing {missing_bytes / (1024 ** 3):.3f} GiB"
        )

    smoke = run_smoke(rows, args.smoke_entries) if args.smoke_entries else []
    write_summary = None
    if args.execute:
        write_summary = write_pack(rows, args.output_pack, pack_summary)

    summary = {
        "repo_head": git_head(repo_root),
        "manifest_tsv": str(args.manifest_tsv),
        "output_pack": str(args.output_pack),
        "dry_run": not args.execute,
        "execute": args.execute,
        "pack_written": write_summary is not None,
        "output_fs_path": str(target_fs),
        "output_fs_free_bytes": disk.free,
        "output_fs_free_gib": disk.free / (1024 ** 3),
        "fits_output_fs": fits_output_fs,
        "missing_bytes": missing_bytes,
        "missing_gib": missing_bytes / (1024 ** 3),
        "pack": pack_summary,
        "smoke_entries_requested": args.smoke_entries,
        "smoke_entries_completed": len(smoke),
        "smoke_bytes": sum(int(item["remote_nbytes"]) for item in smoke),
        "smoke": smoke,
        "write_summary": write_summary,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"entries={pack_summary['entries']}")
    print(f"selected_tensors={pack_summary['selected_tensors']}")
    print(f"payload_gib={pack_summary['payload_bytes'] / (1024 ** 3):.3f}")
    print(f"estimated_pack_gib={pack_summary['estimated_pack_bytes'] / (1024 ** 3):.3f}")
    print(f"current_iq3_runtime_compatible={int(pack_summary['current_iq3_runtime_compatible'])}")
    print(f"invalid_remote_range_count={pack_summary['invalid_remote_range_count']}")
    print(f"bad_pack_offset_count={pack_summary['bad_pack_offset_count']}")
    print(f"fits_output_fs={int(fits_output_fs)}")
    print(f"missing_gib={summary['missing_gib']:.3f}")
    print(f"execute={int(args.execute)}")
    print(f"pack_written={int(write_summary is not None)}")
    print(f"smoke_entries_completed={len(smoke)}")
    print(f"smoke_bytes={summary['smoke_bytes']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
