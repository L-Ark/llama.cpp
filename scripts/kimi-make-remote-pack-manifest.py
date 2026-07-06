#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import struct
import subprocess
import sys
import urllib.parse
from pathlib import Path
from typing import Any


PACK_MAGIC = b"GGMLMOEPACKv1\0\0\0"
PACK_HEADER = struct.Struct("<16sIIQQ")
PACK_ENTRY = struct.Struct("<128siIQQ")
ALIGNMENT = 4096


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


def resolve_url(repo: str, revision: str, rfilename: str) -> str:
    quoted_repo = urllib.parse.quote(repo, safe="/")
    quoted_rev = urllib.parse.quote(revision, safe="")
    quoted_file = urllib.parse.quote(rfilename, safe="/")
    return f"https://huggingface.co/{quoted_repo}/resolve/{quoted_rev}/{quoted_file}"


def nearest_existing_parent(path: Path) -> Path:
    current = path if path.is_dir() else path.parent
    while not current.exists():
        parent = current.parent
        if parent == current:
            raise RuntimeError(f"no existing parent for {path}")
        current = parent
    return current


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def int_field(row: dict[str, Any], key: str) -> int:
    try:
        return int(row[key])
    except KeyError as exc:
        raise RuntimeError(f"missing required field {key!r}") from exc
    except ValueError as exc:
        raise RuntimeError(f"invalid integer field {key!r}: {row.get(key)!r}") from exc


def metadata_maps(summary: dict[str, Any]) -> tuple[dict[str, int], dict[str, int]]:
    data_offsets: dict[str, int] = {}
    shard_sizes: dict[str, int] = {}
    for item in summary.get("remote_metadata", []):
        source = item.get("source")
        if not source:
            continue
        data_offsets[str(source)] = int(item["data_offset"])
    for item in summary.get("remote_files", []):
        name = item.get("rfilename")
        if not name:
            continue
        shard_sizes[str(name)] = int(item["size"])
    return data_offsets, shard_sizes


def build_manifest_rows(rows: list[dict[str, str]], summary: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    hf_repo = str(summary["hf_repo"])
    revision = str(summary.get("hf_revision", "main"))
    data_offsets, shard_sizes = metadata_maps(summary)
    if not data_offsets:
        raise RuntimeError("plan JSON contains no remote_metadata data offsets")
    if not shard_sizes:
        raise RuntimeError("plan JSON contains no remote_files sizes")

    data_start = align_up(PACK_HEADER.size + len(rows) * PACK_ENTRY.size)
    pack_offset = data_start
    payload_bytes = 0
    padding_bytes = data_start - (PACK_HEADER.size + len(rows) * PACK_ENTRY.size)
    mismatch_count = 0
    invalid_range_count = 0
    manifest: list[dict[str, Any]] = []
    by_type: dict[str, dict[str, int]] = {}
    by_kind: dict[str, dict[str, int]] = {}

    for entry_idx, row in enumerate(rows):
        tensor = row["tensor"]
        name_len = len(tensor.encode("utf-8"))
        if name_len >= 128:
            raise RuntimeError(f"tensor name too long for pack v1 index: {tensor}")
        expert_idx = int_field(row, "expert_idx")
        current_nbytes = int_field(row, "current_nbytes")
        remote_nbytes = int_field(row, "remote_nbytes")
        if remote_nbytes <= 0:
            raise RuntimeError(f"{tensor}:{expert_idx}: remote_nbytes must be positive")
        shard_name = row["remote_shard_name"]
        if shard_name not in data_offsets:
            raise RuntimeError(f"{tensor}:{expert_idx}: no data_offset for shard {shard_name}")
        if shard_name not in shard_sizes:
            raise RuntimeError(f"{tensor}:{expert_idx}: no shard size for {shard_name}")

        remote_tensor_offset = int_field(row, "remote_tensor_offset")
        remote_abs_start = data_offsets[shard_name] + remote_tensor_offset + expert_idx * remote_nbytes
        remote_abs_end = remote_abs_start + remote_nbytes - 1
        shard_size = shard_sizes[shard_name]
        range_valid = 0 <= remote_abs_start <= remote_abs_end < shard_size
        if not range_valid:
            invalid_range_count += 1

        aligned_offset = align_up(pack_offset)
        padding_bytes += aligned_offset - pack_offset
        pack_offset = aligned_offset
        runtime_nbytes_match = current_nbytes == remote_nbytes
        if not runtime_nbytes_match:
            mismatch_count += 1

        out = {
            "entry_idx": entry_idx,
            "tensor": tensor,
            "expert_idx": expert_idx,
            "kind": row["kind"],
            "layer": int_field(row, "layer"),
            "remote_type": row["remote_type"],
            "current_nbytes": current_nbytes,
            "remote_nbytes": remote_nbytes,
            "runtime_nbytes_match": int(runtime_nbytes_match),
            "remote_shard_name": shard_name,
            "remote_url": resolve_url(hf_repo, revision, shard_name),
            "remote_abs_start": remote_abs_start,
            "remote_abs_end": remote_abs_end,
            "remote_range_header": f"bytes={remote_abs_start}-{remote_abs_end}",
            "remote_range_valid": int(range_valid),
            "pack_offset": pack_offset,
            "source_pack": row.get("source_pack", ""),
        }
        manifest.append(out)

        payload_bytes += remote_nbytes
        pack_offset += remote_nbytes
        for bucket in (by_type.setdefault(row["remote_type"], {"entries": 0, "bytes": 0}),
                       by_kind.setdefault(row["kind"], {"entries": 0, "bytes": 0})):
            bucket["entries"] += 1
            bucket["bytes"] += remote_nbytes

    return manifest, {
        "alignment": ALIGNMENT,
        "header_bytes": PACK_HEADER.size,
        "entry_bytes": PACK_ENTRY.size,
        "data_start": data_start,
        "payload_bytes": payload_bytes,
        "padding_bytes": padding_bytes,
        "estimated_pack_bytes": pack_offset,
        "runtime_nbytes_mismatch_count": mismatch_count,
        "current_iq3_runtime_compatible": mismatch_count == 0,
        "invalid_remote_range_count": invalid_range_count,
        "by_remote_type": dict(sorted(by_type.items())),
        "by_kind": dict(sorted(by_kind.items())),
    }


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "entry_idx",
        "tensor",
        "expert_idx",
        "kind",
        "layer",
        "remote_type",
        "current_nbytes",
        "remote_nbytes",
        "runtime_nbytes_match",
        "remote_shard_name",
        "remote_url",
        "remote_abs_start",
        "remote_abs_end",
        "remote_range_header",
        "remote_range_valid",
        "pack_offset",
        "source_pack",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a non-destructive selected remote expert-pack manifest from a hot-key remote plan."
    )
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--plan-tsv", type=Path, required=True)
    parser.add_argument("--output-pack", type=Path, required=True,
                        help="Intended future pack path; not created.")
    parser.add_argument("--manifest-tsv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    summary = json.loads(args.plan_json.read_text(encoding="utf-8"))
    rows = read_tsv(args.plan_tsv)
    if not rows:
        raise RuntimeError("plan TSV has no selected entries")

    manifest, pack_summary = build_manifest_rows(rows, summary)
    target_fs = nearest_existing_parent(args.output_pack)
    disk = shutil.disk_usage(target_fs)
    fits_output_fs = disk.free >= pack_summary["estimated_pack_bytes"]
    missing_bytes = max(0, pack_summary["estimated_pack_bytes"] - disk.free)

    out_summary = {
        "repo_head": git_head(repo_root),
        "plan_json": str(args.plan_json),
        "plan_tsv": str(args.plan_tsv),
        "output_pack": str(args.output_pack),
        "output_fs_path": str(target_fs),
        "output_fs_free_bytes": disk.free,
        "output_fs_free_gib": disk.free / (1024 ** 3),
        "fits_output_fs": fits_output_fs,
        "missing_bytes": missing_bytes,
        "missing_gib": missing_bytes / (1024 ** 3),
        "entries": len(manifest),
        "selected_tensors": len({row["tensor"] for row in manifest}),
        "remote_shards": len({row["remote_shard_name"] for row in manifest}),
        "pack": pack_summary,
        "hf_repo": summary.get("hf_repo"),
        "hf_revision": summary.get("hf_revision"),
        "remote_prefix": summary.get("remote_prefix"),
        "direct_range_temp_upper_bound_bytes": summary.get("direct_range_temp_upper_bound_bytes"),
        "one_shard_streaming_temp_bytes": summary.get("selected_summary", {}).get("one_shard_streaming_temp_bytes"),
    }

    write_manifest(args.manifest_tsv, manifest)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(out_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"entries={len(manifest)}")
    print(f"selected_tensors={out_summary['selected_tensors']}")
    print(f"remote_shards={out_summary['remote_shards']}")
    print(f"payload_gib={pack_summary['payload_bytes'] / (1024 ** 3):.3f}")
    print(f"estimated_pack_gib={pack_summary['estimated_pack_bytes'] / (1024 ** 3):.3f}")
    print(f"runtime_nbytes_mismatch_count={pack_summary['runtime_nbytes_mismatch_count']}")
    print(f"current_iq3_runtime_compatible={int(pack_summary['current_iq3_runtime_compatible'])}")
    print(f"invalid_remote_range_count={pack_summary['invalid_remote_range_count']}")
    print(f"output_fs_free_gib={out_summary['output_fs_free_gib']:.3f}")
    print(f"fits_output_fs={int(fits_output_fs)}")
    print(f"missing_gib={out_summary['missing_gib']:.3f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
