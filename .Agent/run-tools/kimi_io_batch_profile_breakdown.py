#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
from collections import defaultdict
from typing import Any


ROLE_RE = re.compile(r"ffn_(up|gate|down)_exps")
LAYER_RE = re.compile(r"blk\.(\d+)\.")


def role_of(name: str) -> str:
    match = ROLE_RE.search(name or "")
    return match.group(1) if match else "other"


def layer_of(name: str) -> int | None:
    match = LAYER_RE.search(name or "")
    return int(match.group(1)) if match else None


def fnum(value: object) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def inum(value: object) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def fmt(value: float) -> str:
    return f"{value:.3f}"


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def add_bucket(acc: dict[str, float], row: dict[str, str]) -> None:
    jobs = inum(row.get("jobs"))
    read_jobs = inum(row.get("read_jobs"))
    inflight_avg = fnum(row.get("inflight_avg"))

    acc["batches"] += 1
    acc["jobs"] += jobs
    acc["read_jobs"] += read_jobs
    acc["ram_or_prefetch_jobs"] += inum(row.get("ram_or_prefetch_jobs"))
    acc["wait_calls"] += inum(row.get("wait_calls"))
    acc["submit_calls"] += inum(row.get("submit_calls"))
    acc["cqes"] += inum(row.get("cqes"))
    acc["slot_wait_ms"] += fnum(row.get("slot_wait_ms"))
    acc["submit_ms"] += fnum(row.get("submit_ms"))
    acc["wait_ms"] += fnum(row.get("wait_ms"))
    acc["enqueue_ms"] += fnum(row.get("enqueue_ms"))
    acc["wall_ms"] += fnum(row.get("wall_ms"))
    acc["weighted_inflight_sum"] += inflight_avg * max(read_jobs, 1)
    acc["weighted_inflight_den"] += max(read_jobs, 1)
    acc["max_inflight"] = max(acc["max_inflight"], fnum(row.get("inflight_max")))
    acc["max_jobs"] = max(acc["max_jobs"], jobs)

    if jobs <= 1:
        acc["batches_jobs_1"] += 1
    if 2 <= jobs <= 4:
        acc["batches_jobs_2_4"] += 1
    if 5 <= jobs <= 8:
        acc["batches_jobs_5_8"] += 1
    if jobs > 8:
        acc["batches_jobs_gt8"] += 1
    if read_jobs <= 4:
        acc["batches_read_jobs_le4"] += 1
    if read_jobs <= 8:
        acc["batches_read_jobs_le8"] += 1


def finish_bucket(key: tuple[str, ...], acc: dict[str, float]) -> dict[str, Any]:
    batches = acc["batches"] or 1.0
    read_jobs = acc["read_jobs"] or 1.0
    inflight_den = acc["weighted_inflight_den"] or 1.0
    return {
        "key": key,
        "batches": int(acc["batches"]),
        "jobs": int(acc["jobs"]),
        "read_jobs": int(acc["read_jobs"]),
        "ram_or_prefetch_jobs": int(acc["ram_or_prefetch_jobs"]),
        "avg_jobs_per_batch": acc["jobs"] / batches,
        "avg_read_jobs_per_batch": acc["read_jobs"] / batches,
        "small_read_batch_rate_le4": acc["batches_read_jobs_le4"] / batches,
        "small_read_batch_rate_le8": acc["batches_read_jobs_le8"] / batches,
        "jobs_1_rate": acc["batches_jobs_1"] / batches,
        "jobs_2_4_rate": acc["batches_jobs_2_4"] / batches,
        "jobs_5_8_rate": acc["batches_jobs_5_8"] / batches,
        "jobs_gt8_rate": acc["batches_jobs_gt8"] / batches,
        "weighted_inflight_avg": acc["weighted_inflight_sum"] / inflight_den,
        "max_inflight": int(acc["max_inflight"]),
        "max_jobs": int(acc["max_jobs"]),
        "wait_ms": acc["wait_ms"],
        "wall_ms": acc["wall_ms"],
        "submit_ms": acc["submit_ms"],
        "enqueue_ms": acc["enqueue_ms"],
        "slot_wait_ms": acc["slot_wait_ms"],
        "wait_ms_per_read_job": acc["wait_ms"] / read_jobs,
        "wall_ms_per_batch": acc["wall_ms"] / batches,
    }


def summarize(paths: list[pathlib.Path], min_jobs: int | None = None, max_jobs: int | None = None) -> dict[str, Any]:
    by_prompt: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    by_role: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    by_layer_role: dict[tuple[str, str, int], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    total: dict[str, float] = defaultdict(float)

    for path in paths:
        prompt = path.parent.name
        for row in read_csv(path):
            jobs = inum(row.get("jobs"))
            if min_jobs is not None and jobs < min_jobs:
                continue
            if max_jobs is not None and jobs > max_jobs:
                continue
            op = row.get("op", "")
            first = row.get("first_tensor", "")
            role = role_of(first)
            layer = layer_of(first)
            add_bucket(total, row)
            add_bucket(by_prompt[prompt], row)
            add_bucket(by_role[(op, role)], row)
            if layer is not None:
                add_bucket(by_layer_role[(op, role, layer)], row)

    prompt_rows = [
        {"prompt": key, **finish_bucket((key,), acc)}
        for key, acc in sorted(by_prompt.items())
    ]
    role_rows = [
        {"op": key[0], "role": key[1], **finish_bucket(key, acc)}
        for key, acc in sorted(by_role.items(), key=lambda item: -item[1]["wait_ms"])
    ]
    layer_rows = [
        {"op": key[0], "role": key[1], "layer": key[2], **finish_bucket(tuple(map(str, key)), acc)}
        for key, acc in sorted(by_layer_role.items(), key=lambda item: -item[1]["wait_ms"])
    ]
    return {
        "inputs": [str(path) for path in paths],
        "filters": {
            "min_jobs": min_jobs,
            "max_jobs": max_jobs,
        },
        "total": finish_bucket(("total",), total),
        "prompts": prompt_rows,
        "by_role": role_rows,
        "by_layer_role": layer_rows,
    }


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_md(path: pathlib.Path, report: dict[str, Any], top_n: int) -> None:
    total = report["total"]
    lines = [
        "# Kimi IO batch-profile breakdown",
        "",
        "This report summarizes `GGML_MOE_IO_BATCH_PROFILE_OUT` rows. It measures",
        "how much independent read work the runtime actually exposes to io_uring.",
        "",
        "## Filters",
        "",
        f"- min jobs: `{report.get('filters', {}).get('min_jobs')}`",
        f"- max jobs: `{report.get('filters', {}).get('max_jobs')}`",
        "",
        "## Aggregate",
        "",
        f"- Batches: `{total['batches']}`",
        f"- Read jobs: `{total['read_jobs']}`",
        f"- Avg read jobs/batch: `{fmt(total['avg_read_jobs_per_batch'])}`",
        f"- Weighted inflight avg: `{fmt(total['weighted_inflight_avg'])}`",
        f"- Max inflight: `{total['max_inflight']}`",
        f"- Wait: `{fmt(total['wait_ms'])} ms`",
        f"- Wall: `{fmt(total['wall_ms'])} ms`",
        f"- Wait/read job: `{fmt(total['wait_ms_per_read_job'])} ms`",
        f"- Read batches <=4 jobs: `{fmt(total['small_read_batch_rate_le4'])}`",
        f"- Read batches <=8 jobs: `{fmt(total['small_read_batch_rate_le8'])}`",
        "",
        "## By Prompt",
        "",
        "| prompt | batches | read jobs | avg read jobs/batch | inflight avg | wait ms | wall ms | read batches <=4 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["prompts"]:
        lines.append(
            f"| `{row['prompt']}` | {row['batches']} | {row['read_jobs']} | "
            f"{fmt(row['avg_read_jobs_per_batch'])} | {fmt(row['weighted_inflight_avg'])} | "
            f"{fmt(row['wait_ms'])} | {fmt(row['wall_ms'])} | {fmt(row['small_read_batch_rate_le4'])} |"
        )

    lines += [
        "",
        "## By Op/Role",
        "",
        "| op | role | batches | read jobs | avg read jobs/batch | inflight avg | wait ms | wall ms | read batches <=4 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["by_role"]:
        lines.append(
            f"| `{row['op']}` | `{row['role']}` | {row['batches']} | {row['read_jobs']} | "
            f"{fmt(row['avg_read_jobs_per_batch'])} | {fmt(row['weighted_inflight_avg'])} | "
            f"{fmt(row['wait_ms'])} | {fmt(row['wall_ms'])} | {fmt(row['small_read_batch_rate_le4'])} |"
        )

    lines += [
        "",
        f"## Top {top_n} Layer/Role Buckets By Wait",
        "",
        "| op | role | layer | batches | read jobs | avg read jobs/batch | inflight avg | wait ms | wall ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["by_layer_role"][:top_n]:
        lines.append(
            f"| `{row['op']}` | `{row['role']}` | {row['layer']} | {row['batches']} | "
            f"{row['read_jobs']} | {fmt(row['avg_read_jobs_per_batch'])} | "
            f"{fmt(row['weighted_inflight_avg'])} | {fmt(row['wait_ms'])} | {fmt(row['wall_ms'])} |"
        )

    lines += [
        "",
        "## Interpretation Rules",
        "",
        "- If avg read jobs/batch and weighted inflight are far below the pure IO",
        "  bench depth, the runtime is not exposing enough continuous independent",
        "  read work.",
        "- If wait is concentrated in `runtime_load gate/up`, earlier admission or",
        "  lower-byte up/gate representation is more promising than another down",
        "  overlap change.",
        "- If wait is concentrated in `current_down_overlap`, down prefetch depth or",
        "  down placement should be investigated, but only if endpoint TTFT/RAM",
        "  gates remain valid.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Kimi io_uring batch profile rows.")
    parser.add_argument("--input", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--out-json", type=pathlib.Path, required=True)
    parser.add_argument("--out-md", type=pathlib.Path, required=True)
    parser.add_argument("--out-role-csv", type=pathlib.Path, required=True)
    parser.add_argument("--out-layer-csv", type=pathlib.Path, required=True)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--min-jobs", type=int, default=None)
    parser.add_argument("--max-jobs", type=int, default=None)
    args = parser.parse_args()

    report = summarize(args.input, min_jobs=args.min_jobs, max_jobs=args.max_jobs)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_md(args.out_md, report, args.top_n)
    common_fields = [
        "op", "role", "batches", "jobs", "read_jobs", "avg_read_jobs_per_batch",
        "weighted_inflight_avg", "max_inflight", "wait_ms", "wall_ms",
        "wait_ms_per_read_job", "small_read_batch_rate_le4", "small_read_batch_rate_le8",
    ]
    write_csv(args.out_role_csv, report["by_role"], common_fields)
    write_csv(args.out_layer_csv, report["by_layer_role"], ["op", "role", "layer", *common_fields[2:]])
    print(args.out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
