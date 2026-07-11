#!/usr/bin/env python3
"""Analyze v2 shadow CSV batchability under simple scheduler coalescing policies."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path


LAYER_RE = re.compile(r"blk\.(\d+)\.")


def layer_of(tensor: str) -> int:
    match = LAYER_RE.search(tensor or "")
    return int(match.group(1)) if match else -1


def bucket(value: int) -> str:
    if value <= 0:
        return "0"
    if value == 1:
        return "1"
    if value <= 4:
        return "2-4"
    if value <= 8:
        return "5-8"
    if value <= 16:
        return "9-16"
    if value <= 32:
        return "17-32"
    return "gt32"


def hist(values: list[int]) -> dict[str, int]:
    out = {key: 0 for key in ["1", "2-4", "5-8", "9-16", "17-32", "gt32"]}
    for value in values:
        out[bucket(value)] += 1
    return out


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((len(ordered) - 1) * q)))
    return ordered[idx]


def read_staged_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            staged = int(row["staged"])
            if staged <= 0:
                continue
            rows.append({
                "seq": int(row["seq"]),
                "phase": row["phase"],
                "role": row["role"],
                "tensor": row["tensor"],
                "layer": layer_of(row["tensor"]),
                "jobs": int(row.get("iouring_jobs") or staged),
                "bytes": int(row["direct_bytes"]),
                "saved": int(row["staged_saved_bytes"]),
            })
    return rows


def adjacent_same_layer_groups(rows: list[dict]) -> list[dict]:
    groups: list[dict] = []
    for row in rows:
        if (
            groups
            and groups[-1]["layer"] == row["layer"]
            and groups[-1]["phase"].startswith("decode")
            and row["phase"].startswith("decode")
        ):
            group = groups[-1]
            group["rows"] += 1
            group["jobs"] += row["jobs"]
            group["bytes"] += row["bytes"]
            group["saved"] += row["saved"]
            group["roles"].add(row["role"])
        else:
            groups.append({
                "layer": row["layer"],
                "phase": row["phase"],
                "rows": 1,
                "jobs": row["jobs"],
                "bytes": row["bytes"],
                "saved": row["saved"],
                "roles": {row["role"]},
            })
    return groups


def summarize_jobs(label: str, jobs: list[int]) -> str:
    if not jobs:
        return f"{label}: batches=0 jobs=0"
    return (
        f"{label}: batches={len(jobs)} jobs={sum(jobs)} "
        f"avg_jobs={statistics.mean(jobs):.2f} "
        f"p50={percentile(jobs, 0.50)} p90={percentile(jobs, 0.90)} "
        f"p99={percentile(jobs, 0.99)} hist={hist(jobs)}"
    )


def analyze(name: str, path: Path) -> str:
    rows = read_staged_rows(path)
    current_jobs = [row["jobs"] for row in rows]
    groups = adjacent_same_layer_groups(rows)
    group_jobs = [group["jobs"] for group in groups]
    multi = [group for group in groups if group["rows"] > 1]
    role_sets: dict[str, int] = {}
    for group in multi:
        key = "+".join(sorted(group["roles"]))
        role_sets[key] = role_sets.get(key, 0) + 1

    current_batches = len(current_jobs)
    merged_batches = len(groups)
    batch_reduction = current_batches - merged_batches
    batch_reduction_pct = (batch_reduction / current_batches * 100.0) if current_batches else 0.0
    groups_ge8 = sum(1 for value in group_jobs if value >= 8)
    groups_ge8_pct = (groups_ge8 / len(groups) * 100.0) if groups else 0.0
    multi_saved_gib = sum(group["saved"] for group in multi) / (1024.0 ** 3)

    lines = [
        f"## {name}",
        "",
        f"- source_csv: `{path}`",
        f"- staged_rows: `{len(rows)}`",
        f"- current: `{summarize_jobs('current', current_jobs)}`",
        f"- adjacent_same_layer: `{summarize_jobs('same_layer', group_jobs)}`",
        f"- multi_groups: `{len(multi)}`",
        f"- multi_group_jobs: `{sum(group['jobs'] for group in multi)}`",
        f"- multi_group_saved_gib: `{multi_saved_gib:.3f}`",
        f"- batches_reduction: `{batch_reduction}` / `{current_batches}` = `{batch_reduction_pct:.1f}%`",
        f"- groups_ge8: `{groups_ge8}` / `{len(groups)}` = `{groups_ge8_pct:.1f}%`",
        f"- multi_role_sets: `{role_sets}`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", action="append", required=True, help="name=path entry")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    sections = ["# Kimi v2 Shadow Coalesce Analysis", ""]
    for item in args.csv:
        if "=" not in item:
            raise ValueError(f"--csv must be name=path, got {item}")
        name, path = item.split("=", 1)
        sections.append(analyze(name, Path(path)))
    report = "\n".join(sections).rstrip() + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
