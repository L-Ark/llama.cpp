#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import struct
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any


PACK_HEADER = struct.Struct("<16sIIQQ")
PACK_ENTRY = struct.Struct("<128siIQQ")
PACK_HEADER_SIZE = PACK_HEADER.size
PACK_ENTRY_SIZE = PACK_ENTRY.size
ALIGNMENT = 4096
KIND_ORDER = ("up", "gate", "down")


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


def int_field(row: dict[str, str], key: str) -> int:
    try:
        return int(row[key])
    except KeyError as exc:
        raise RuntimeError(f"missing required manifest field {key!r}") from exc
    except ValueError as exc:
        raise RuntimeError(f"invalid integer manifest field {key!r}: {row.get(key)!r}") from exc


def load_manifest(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        raise RuntimeError(f"{path}: manifest has no rows")
    required = {
        "entry_idx",
        "tensor",
        "expert_idx",
        "kind",
        "remote_nbytes",
        "current_nbytes",
        "pack_offset",
        "remote_abs_start",
        "remote_abs_end",
        "remote_range_valid",
        "runtime_nbytes_match",
    }
    missing = sorted(required - set(fieldnames))
    if missing:
        raise RuntimeError(f"{path}: missing required fields: {', '.join(missing)}")
    return rows, fieldnames


def estimate_layout(rows: list[dict[str, str]]) -> tuple[int, int]:
    offset = align_up(PACK_HEADER_SIZE + len(rows) * PACK_ENTRY_SIZE)
    payload = 0
    max_end = offset
    for row in rows:
        offset = align_up(offset)
        nbytes = int_field(row, "remote_nbytes")
        payload += nbytes
        max_end = max(max_end, offset + nbytes)
        offset += nbytes
    return payload, max_end


def validate_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    payload = 0
    invalid_ranges = 0
    bad_offsets = 0
    mismatches = 0
    selected_tensors = set()
    remote_shards = set()
    by_type: dict[str, dict[str, int]] = {}

    expected_offset = align_up(PACK_HEADER_SIZE + len(rows) * PACK_ENTRY_SIZE)
    max_end = expected_offset
    for idx, row in enumerate(rows):
        entry_idx = int_field(row, "entry_idx")
        if entry_idx != idx:
            raise RuntimeError(f"partition entry_idx is not contiguous at row {idx}: got {entry_idx}")
        tensor = row["tensor"]
        if len(tensor.encode("utf-8")) >= 128:
            raise RuntimeError(f"tensor name too long for pack v1 index: {tensor}")

        nbytes = int_field(row, "remote_nbytes")
        current_nbytes = int_field(row, "current_nbytes")
        pack_offset = int_field(row, "pack_offset")
        remote_abs_start = int_field(row, "remote_abs_start")
        remote_abs_end = int_field(row, "remote_abs_end")
        if nbytes <= 0:
            raise RuntimeError(f"{tensor}:{row.get('expert_idx')}: remote_nbytes must be positive")
        if remote_abs_end - remote_abs_start + 1 != nbytes:
            invalid_ranges += 1
        if int_field(row, "remote_range_valid") != 1:
            invalid_ranges += 1
        expected_offset = align_up(expected_offset)
        if pack_offset != expected_offset or pack_offset % ALIGNMENT != 0:
            bad_offsets += 1
        if current_nbytes != nbytes or int_field(row, "runtime_nbytes_match") != 1:
            mismatches += 1
        payload += nbytes
        max_end = max(max_end, pack_offset + nbytes)
        expected_offset = pack_offset + nbytes
        selected_tensors.add(tensor)
        if "remote_shard_name" in row:
            remote_shards.add(row["remote_shard_name"])
        bucket = by_type.setdefault(row.get("remote_type", ""), {"entries": 0, "bytes": 0})
        bucket["entries"] += 1
        bucket["bytes"] += nbytes

    return {
        "entries": len(rows),
        "selected_tensors": len(selected_tensors),
        "remote_shards": len(remote_shards),
        "payload_bytes": payload,
        "estimated_pack_bytes": max_end,
        "runtime_nbytes_mismatch_count": mismatches,
        "current_iq3_runtime_compatible": mismatches == 0,
        "invalid_remote_range_count": invalid_ranges,
        "bad_pack_offset_count": bad_offsets,
        "by_remote_type": dict(sorted(by_type.items())),
    }


def partition_rows_by_kind(rows: list[dict[str, str]]) -> OrderedDict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        kind = row["kind"]
        grouped.setdefault(kind, []).append(row)

    ordered = OrderedDict()
    for kind in KIND_ORDER:
        if kind in grouped:
            ordered[kind] = grouped.pop(kind)
    for kind in sorted(grouped):
        ordered[kind] = grouped[kind]
    return ordered


def relayout_partition(
        name: str,
        rows: list[dict[str, str]],
        fieldnames: list[str]) -> tuple[list[dict[str, str]], list[str]]:
    out_fields = list(fieldnames)
    for extra in ("source_entry_idx", "source_pack_offset", "partition"):
        if extra not in out_fields:
            out_fields.append(extra)

    offset = align_up(PACK_HEADER_SIZE + len(rows) * PACK_ENTRY_SIZE)
    planned: list[dict[str, str]] = []
    for idx, row in enumerate(rows):
        offset = align_up(offset)
        new_row = dict(row)
        new_row["source_entry_idx"] = row["entry_idx"]
        new_row["source_pack_offset"] = row["pack_offset"]
        new_row["partition"] = name
        new_row["entry_idx"] = str(idx)
        new_row["pack_offset"] = str(offset)
        planned.append(new_row)
        offset += int_field(row, "remote_nbytes")
    return planned, out_fields


def write_manifest(path: Path, rows: list[dict[str, str]], fieldnames: list[str], force: bool) -> None:
    if path.exists() and not force:
        raise RuntimeError(f"refusing to overwrite existing manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_builder_commands(path: Path, partitions: list[dict[str, Any]], force: bool) -> None:
    if path.exists() and not force:
        raise RuntimeError(f"refusing to overwrite existing command file: {path}")
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'RUN="${RUN:?set RUN to the output run directory}"',
        'REPO="${REPO:-/root/lfz/llama.cpp-vendor-kimi}"',
        'cd "$REPO"',
    ]
    for part in partitions:
        stem = part["name"]
        manifest = part["manifest_tsv"]
        lines.extend([
            f"python3 scripts/kimi-build-remote-pack-from-manifest.py \\",
            f"  --dry-run \\",
            f"  --manifest-tsv {manifest} \\",
            f"  --output-pack \"$RUN/{stem}.expert-pack\" \\",
            f"  --summary-json \"$RUN/{stem}.builder-summary.json\" \\",
            f"  --smoke-entries 1",
            f"test ! -e \"$RUN/{stem}.expert-pack\"",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Partition a remote GGMLMOEPACKv1 manifest into independently buildable manifests."
    )
    parser.add_argument("--manifest-tsv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--by-kind", action="store_true", help="Partition by manifest kind field.")
    parser.add_argument("--prefix", default="", help="Optional output manifest prefix.")
    parser.add_argument("--force", action="store_true", help="Allow overwriting generated manifests/summaries.")
    args = parser.parse_args()

    if not args.by_kind:
        raise RuntimeError("only --by-kind is implemented")

    rows, fieldnames = load_manifest(args.manifest_tsv)
    for idx, row in enumerate(rows):
        if int_field(row, "entry_idx") != idx:
            raise RuntimeError(f"source manifest entry_idx is not contiguous at row {idx}")

    repo_root = Path(__file__).resolve().parents[1]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(args.out_dir)
    source_payload, source_estimated = estimate_layout(rows)

    partitions = []
    total_entries = 0
    total_payload = 0
    total_estimated = 0
    for name, group in partition_rows_by_kind(rows).items():
        relaid, out_fields = relayout_partition(name, group, fieldnames)
        summary = validate_rows(relaid)
        manifest_name = f"{args.prefix}{name}.manifest.tsv"
        manifest_path = args.out_dir / manifest_name
        write_manifest(manifest_path, relaid, out_fields, args.force)
        fits_individual = disk.free >= int(summary["estimated_pack_bytes"])
        part = {
            "name": name,
            "manifest_tsv": str(manifest_path),
            "entries": summary["entries"],
            "selected_tensors": summary["selected_tensors"],
            "remote_shards": summary["remote_shards"],
            "payload_bytes": summary["payload_bytes"],
            "payload_gib": summary["payload_bytes"] / (1024 ** 3),
            "estimated_pack_bytes": summary["estimated_pack_bytes"],
            "estimated_pack_gib": summary["estimated_pack_bytes"] / (1024 ** 3),
            "fits_output_fs_individually": fits_individual,
            "runtime_nbytes_mismatch_count": summary["runtime_nbytes_mismatch_count"],
            "current_iq3_runtime_compatible": summary["current_iq3_runtime_compatible"],
            "invalid_remote_range_count": summary["invalid_remote_range_count"],
            "bad_pack_offset_count": summary["bad_pack_offset_count"],
            "by_remote_type": summary["by_remote_type"],
        }
        partitions.append(part)
        total_entries += int(summary["entries"])
        total_payload += int(summary["payload_bytes"])
        total_estimated += int(summary["estimated_pack_bytes"])

    commands_path = args.out_dir / f"{args.prefix}builder-dry-run-commands.sh"
    write_builder_commands(commands_path, partitions, args.force)

    summary = {
        "repo_head": git_head(repo_root),
        "manifest_tsv": str(args.manifest_tsv),
        "strategy": "by_kind",
        "out_dir": str(args.out_dir),
        "output_fs_free_bytes": disk.free,
        "output_fs_free_gib": disk.free / (1024 ** 3),
        "source_entries": len(rows),
        "source_payload_bytes": source_payload,
        "source_payload_gib": source_payload / (1024 ** 3),
        "source_estimated_pack_bytes": source_estimated,
        "source_estimated_pack_gib": source_estimated / (1024 ** 3),
        "partition_count": len(partitions),
        "partition_entries": total_entries,
        "partition_payload_bytes": total_payload,
        "partition_payload_gib": total_payload / (1024 ** 3),
        "partition_estimated_pack_bytes_sum": total_estimated,
        "partition_estimated_pack_gib_sum": total_estimated / (1024 ** 3),
        "all_partitions_fit_output_fs_simultaneously": disk.free >= total_estimated,
        "all_partitions_missing_bytes": max(0, total_estimated - disk.free),
        "all_partitions_missing_gib": max(0, total_estimated - disk.free) / (1024 ** 3),
        "partitions": partitions,
        "builder_dry_run_commands": str(commands_path),
    }
    if total_entries != len(rows):
        raise RuntimeError(f"partition entry count mismatch: {total_entries} != {len(rows)}")
    if total_payload != source_payload:
        raise RuntimeError(f"partition payload mismatch: {total_payload} != {source_payload}")
    if args.summary_json.exists() and not args.force:
        raise RuntimeError(f"refusing to overwrite existing summary: {args.summary_json}")
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"partitions={len(partitions)}")
    print(f"entries={total_entries}")
    print(f"payload_gib={total_payload / (1024 ** 3):.3f}")
    print(f"estimated_pack_gib_sum={total_estimated / (1024 ** 3):.3f}")
    print(f"all_fit_simultaneously={int(summary['all_partitions_fit_output_fs_simultaneously'])}")
    print(f"missing_gib={summary['all_partitions_missing_gib']:.3f}")
    for part in partitions:
        print(
            f"partition={part['name']} entries={part['entries']} "
            f"payload_gib={part['payload_gib']:.3f} "
            f"estimated_pack_gib={part['estimated_pack_gib']:.3f} "
            f"fits_individual={int(part['fits_output_fs_individually'])} "
            f"current_iq3_runtime_compatible={int(part['current_iq3_runtime_compatible'])}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
