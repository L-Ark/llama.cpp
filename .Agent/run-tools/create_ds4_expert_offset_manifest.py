#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/lfz/ik_llama/gguf-py")
from gguf import GGUFReader


DEFAULT_MODEL = "/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf"


def read_profile(path: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.rstrip("\n")
            if not line:
                continue
            cols = line.split("\t")
            if len(cols) < 2:
                raise ValueError(f"{path}:{lineno}: expected at least tensor and expert columns")
            tensor = cols[0]
            expert = int(cols[1])
            residual_ms = float(cols[2]) if len(cols) >= 3 and cols[2] else None
            touches = int(cols[3]) if len(cols) >= 4 and cols[3] else None
            profile_nbytes = int(cols[4]) if len(cols) >= 5 and cols[4] else None
            entries.append({
                "tensor": tensor,
                "expert": expert,
                "residual_ms": residual_ms,
                "touches": touches,
                "profile_nbytes": profile_nbytes,
            })
    return entries


def write_manifest(
    *,
    model: Path,
    profile: Path,
    output: Path,
    summary_out: Path | None,
    expected_entries: int | None,
    expected_payload_bytes: int | None,
) -> dict[str, object]:
    profile_entries = read_profile(profile)
    if expected_entries is not None and len(profile_entries) != expected_entries:
        raise RuntimeError(f"entry count mismatch: got {len(profile_entries)}, expected {expected_entries}")

    reader = GGUFReader(str(model), "r")
    tensors = {t.name: t for t in reader.tensors}

    rows: list[dict[str, object]] = []
    total_bytes = 0
    total_residual_ms = 0.0
    missing: list[str] = []
    size_mismatches: list[dict[str, object]] = []

    for item in profile_entries:
        name = str(item["tensor"])
        expert = int(item["expert"])
        tensor = tensors.get(name)
        if tensor is None:
            missing.append(f"{name}:{expert}")
            continue

        shape = [int(x) for x in tensor.shape.tolist()]
        if len(shape) < 3:
            raise RuntimeError(f"tensor {name} has unexpected shape {shape}")
        n_expert = shape[2]
        if expert < 0 or expert >= n_expert:
            raise RuntimeError(f"expert index out of range for {name}: {expert} not in [0, {n_expert})")

        expert_bytes = int(tensor.n_bytes) // n_expert
        if expert_bytes * n_expert != int(tensor.n_bytes):
            raise RuntimeError(f"tensor {name} n_bytes={int(tensor.n_bytes)} is not divisible by experts={n_expert}")
        profile_nbytes = item["profile_nbytes"]
        if profile_nbytes is not None and int(profile_nbytes) != expert_bytes:
            size_mismatches.append({
                "tensor": name,
                "expert": expert,
                "profile_nbytes": int(profile_nbytes),
                "computed_nbytes": expert_bytes,
            })

        residual_ms = item["residual_ms"]
        if residual_ms is not None:
            total_residual_ms += float(residual_ms)
        model_offset = int(tensor.data_offset) + expert * expert_bytes
        rows.append({
            "tensor": name,
            "expert": expert,
            "model_offset": model_offset,
            "nbytes": expert_bytes,
            "shape": "x".join(str(x) for x in shape),
            "n_expert": n_expert,
            "residual_ms": "" if residual_ms is None else f"{float(residual_ms):.6f}",
            "touches": "" if item["touches"] is None else int(item["touches"]),
        })
        total_bytes += expert_bytes

    if missing:
        raise RuntimeError(f"missing tensors: {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    if size_mismatches:
        raise RuntimeError(f"size mismatches: {size_mismatches[:8]}{' ...' if len(size_mismatches) > 8 else ''}")
    if expected_payload_bytes is not None and total_bytes != expected_payload_bytes:
        raise RuntimeError(f"payload bytes mismatch: got {total_bytes}, expected {expected_payload_bytes}")

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["tensor", "expert", "model_offset", "nbytes", "shape", "n_expert", "residual_ms", "touches"],
        )
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(output)

    summary = {
        "model": str(model),
        "profile": str(profile),
        "manifest": str(output),
        "entries": len(rows),
        "payload_bytes": total_bytes,
        "payload_mib": total_bytes / (1024 * 1024),
        "residual_ms_sum": round(total_residual_ms, 6),
        "missing": 0,
        "size_mismatches": 0,
    }

    if summary_out is not None:
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        tmp_summary = summary_out.with_suffix(summary_out.suffix + ".tmp")
        with tmp_summary.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write("\n")
        tmp_summary.replace(summary_out)

    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Create a small GGUF expert offset manifest from a DS4 tensor/expert profile.")
    ap.add_argument("--model", default=DEFAULT_MODEL, type=Path)
    ap.add_argument("--profile", required=True, type=Path)
    ap.add_argument("-o", "--output", required=True, type=Path)
    ap.add_argument("--summary-out", type=Path)
    ap.add_argument("--expected-entries", type=int)
    ap.add_argument("--expected-payload-bytes", type=int)
    args = ap.parse_args()

    summary = write_manifest(
        model=args.model,
        profile=args.profile,
        output=args.output,
        summary_out=args.summary_out,
        expected_entries=args.expected_entries,
        expected_payload_bytes=args.expected_payload_bytes,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
