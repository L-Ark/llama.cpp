#!/usr/bin/env python3
"""Summarize Kimi activation dump sample runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DIRECT_READS_RE = re.compile(r"direct_reads=(\d+)")


def read_metrics(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def direct_reads(metrics: dict[str, Any]) -> int:
    match = DIRECT_READS_RE.search(str(metrics.get("expert_pack_0", "")))
    return int(match.group(1)) if match else -1


def summarize_activation_csv(path: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    tensors: Counter[str] = Counter()
    if not path.exists():
        return {"records": 0, "role_counts": {}, "tensor_count": 0, "top_tensors": []}
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            role = row.get("role", "")
            mode = row.get("mode", "")
            tensor = row.get("tensor", "")
            counts[f"{role},{mode}"] += 1
            if tensor:
                tensors[tensor] += 1
    return {
        "records": sum(counts.values()),
        "role_counts": dict(sorted(counts.items())),
        "tensor_count": len(tensors),
        "top_tensors": tensors.most_common(8),
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Kimi multi-prompt activation sample",
        "",
        "This is a dev-only activation sample for representation screening. It is not a SOTA run.",
        "",
        f"- root: `{result['root']}`",
        f"- prompts: `{len(result['prompts'])}`",
        f"- total activation records: `{result['total_activation_records']}`",
        f"- all quality passed: `{result['all_quality_passed']}`",
        f"- all direct_reads zero: `{result['all_direct_reads_zero']}`",
        f"- max memory peak: `{result['max_memory_peak']}`",
        "",
        "## Prompts",
        "",
        "| prompt | quality | token rate | decode | TTFT | memory peak | direct reads | activation records | role counts |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in result["prompts"]:
        lines.append(
            f"| `{row['prompt_id']}` | `{row['quality']}` | `{row['token_rate']:.2f}` | "
            f"`{row['decode_ms']:.2f} ms / {row['decode_runs']}` | `{row['ttft_ms']:.2f} ms` | "
            f"`{row['memory_peak']}` | `{row['direct_reads']}` | `{row['activation']['records']}` | "
            f"`{row['activation']['role_counts']}` |"
        )
    lines += [
        "",
        "## Decision",
        "",
        result["decision"],
        "",
        "## Reproduce",
        "",
        "```bash",
        result["reproduce_command"],
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for run_dir in sorted(p for p in args.root.iterdir() if p.is_dir()):
        metrics = read_metrics(run_dir)
        if not metrics:
            continue
        activation = summarize_activation_csv(run_dir / "act" / "activations.csv")
        rows.append({
            "prompt_id": str(metrics.get("prompt_id", run_dir.name)),
            "prompt": str(metrics.get("prompt", "")),
            "quality": str(metrics.get("quality", "missing")),
            "quality_reason": str(metrics.get("quality_reason", "")),
            "output": str(metrics.get("output", "")),
            "token_rate": float(metrics.get("token_rate", 0.0) or 0.0),
            "decode_ms": float(metrics.get("decode_ms", 0.0) or 0.0),
            "decode_runs": int(metrics.get("decode_runs", 0) or 0),
            "ttft_ms": float(metrics.get("ttft_ms", 0.0) or 0.0),
            "memory_peak": int(metrics.get("memory.peak", 0) or 0),
            "direct_reads": direct_reads(metrics),
            "activation": activation,
            "run_dir": str(run_dir),
        })

    all_quality = all(row["quality"] == "pass" for row in rows) and bool(rows)
    all_direct_zero = all(row["direct_reads"] == 0 for row in rows) and bool(rows)
    total_records = sum(row["activation"]["records"] for row in rows)
    max_memory = max((row["memory_peak"] for row in rows), default=0)
    expected_records = 72 * len(rows)
    accepted = all_quality and all_direct_zero and total_records >= expected_records and max_memory < 16 * 1024**3
    decision = (
        "Accept this dev-only activation sample for prompt-general representation screening."
        if accepted else
        "Reject or repair this activation sample before representation screening."
    )
    result = {
        "kind": "kimi_activation_sample_summary",
        "root": str(args.root),
        "prompts": rows,
        "total_activation_records": total_records,
        "expected_min_records": expected_records,
        "all_quality_passed": all_quality,
        "all_direct_reads_zero": all_direct_zero,
        "max_memory_peak": max_memory,
        "accepted_for_dev_representation_screening": accepted,
        "decision": decision,
        "reproduce_command": " ".join(__import__("sys").argv),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.out_md, result)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
