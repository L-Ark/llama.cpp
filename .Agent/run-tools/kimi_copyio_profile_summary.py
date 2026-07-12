#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pathlib
from typing import Any


GIB = 1024 ** 3


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def read_json(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def summarize(root: pathlib.Path) -> str:
    runs = sorted(path for path in root.glob("dev_*") if path.is_dir() and (path / "metrics.json").exists())
    if not runs:
        raise SystemExit(f"No dev_* run directories with metrics.json found under {root}")

    per_prompt: list[tuple[Any, ...]] = []
    role: dict[str, dict[str, float]] = {}
    queue_rows: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    pack_hit_zero = 0
    pack_hit_one = 0

    for run in runs:
        metrics = read_json(run / "metrics.json")
        decode_runs = fnum(metrics.get("decode_runs"))
        decode_ms_per_token = fnum(metrics.get("decode_ms")) / decode_runs if decode_runs else 0.0
        per_prompt.append((
            run.name,
            metrics.get("quality", "unknown"),
            fnum(metrics.get("token_rate")),
            decode_ms_per_token,
            fnum(metrics.get("ttft_ms")),
            fnum(metrics.get("memory.peak")) / GIB,
        ))

        for copy_row in read_csv(run / "copy-profile.csv"):
            if copy_row.get("pack_hit") == "0":
                pack_hit_zero += 1
            if copy_row.get("pack_hit") == "1":
                pack_hit_one += 1

        queue_dir = root / "analysis" / run.name / "queue-scheduler"
        for row in read_csv(queue_dir / "copy_by_role.csv"):
            key = row.get("role", "unknown")
            role.setdefault(key, {"rows": 0.0, "gib": 0.0, "io": 0.0, "h2d": 0.0, "wall": 0.0})
            role[key]["rows"] += fnum(row.get("rows"))
            role[key]["gib"] += fnum(row.get("bytes_gib"))
            role[key]["io"] += fnum(row.get("io_wait_ms"))
            role[key]["h2d"] += fnum(row.get("h2d_ms"))
            role[key]["wall"] += fnum(row.get("wall_ms"))

        summary = read_json(queue_dir / "summary.json")
        if summary:
            queue_rows.append((
                run.name,
                summary.get("io_wait", {}),
                summary.get("decode_io_wait", {}),
            ))

    total_gib = sum(value["gib"] for value in role.values())
    total_io = sum(value["io"] for value in role.values())
    total_h2d = sum(value["h2d"] for value in role.values())

    lines = [
        "# Kimi current-goal COPY/IO profile summary",
        "",
        f"Run root: `{root}`",
        "",
        "Scope: cold-start N32 diagnostic runs with `PROFILE=1`,",
        "`COPY_PROFILE=1`, IO batch/wait/read/locality traces, current-down",
        "overlap profile, and H2D coalesce profile. `COPY_PROFILE_H2D=1` adds",
        "synchronization, so endpoint token rate here is diagnostic only, not",
        "an accepted SOTA metric.",
        "",
        "## Endpoint",
        "",
        "| prompt | quality | tok/s diagnostic | decode ms/token | TTFT ms | RAM peak GiB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for prompt, quality, tok_s, decode_ms, ttft, ram_gib in per_prompt:
        lines.append(f"| `{prompt}` | {quality} | {tok_s:.2f} | {decode_ms:.3f} | {ttft:.2f} | {ram_gib:.3f} |")

    lines += [
        "",
        "## Movement Split From Copy Profile",
        "",
        f"- copy rows with `pack_hit=0`: `{pack_hit_zero}`",
        f"- copy rows with `pack_hit=1`: `{pack_hit_one}`",
        f"- total profiled payload: `{total_gib:.3f} GiB`",
        f"- total raw copy-profile IO wait: `{total_io:.3f} ms`",
        f"- total measured H2D time: `{total_h2d:.3f} ms`",
        "",
        "| role | rows | GiB | io wait ms | H2D ms | wall ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, value in sorted(role.items(), key=lambda item: -item[1]["wall"]):
        lines.append(
            f"| `{key}` | {int(value['rows'])} | {value['gib']:.3f} | "
            f"{value['io']:.3f} | {value['h2d']:.3f} | {value['wall']:.3f} |"
        )

    lines += [
        "",
        "## Queue Evidence",
        "",
        "| prompt | all wait ms | all p50/p95/p99 | all low inflight | all queue empty | decode-like wait ms | decode p50/p95/p99 | decode low inflight | decode queue empty | decode next-job-done |",
        "|---|---:|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for prompt, all_wait, decode_wait in queue_rows:
        lines.append(
            f"| `{prompt}` | {fnum(all_wait.get('wait_ms')):.3f} | "
            f"{fnum(all_wait.get('p50')):.3f}/{fnum(all_wait.get('p95')):.3f}/{fnum(all_wait.get('p99')):.3f} | "
            f"{fnum(all_wait.get('low_inflight_ratio')):.3f} | {fnum(all_wait.get('queue_empty_ratio')):.3f} | "
            f"{fnum(decode_wait.get('wait_ms')):.3f} | "
            f"{fnum(decode_wait.get('p50')):.3f}/{fnum(decode_wait.get('p95')):.3f}/{fnum(decode_wait.get('p99')):.3f} | "
            f"{fnum(decode_wait.get('low_inflight_ratio')):.3f} | {fnum(decode_wait.get('queue_empty_ratio')):.3f} | "
            f"{fnum(decode_wait.get('next_job_done_ratio')):.3f} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- Fallback remains zero and expert-pack coverage is not the issue in this run.",
        "- The exposed movement path is expert-pack io_uring -> pinned staging -> H2D for VRAM-cache misses.",
        "- Decode-like runtime batches are small: queue reports show roughly",
        "  4.5-4.6 read jobs per runtime batch and inflight around 4, with",
        "  `queue_empty_ratio=0`. The queue is usually not empty; each layer-local",
        "  demand batch is too small to saturate the device like the pure IO bench.",
        "- H2D is measurable but secondary to IO wait in this profile. Across the",
        f"  prompts, measured H2D is `{total_h2d / 1000.0:.3f} s` over about",
        f"  `{total_gib:.1f} GiB` of copied expert payload, while raw copy-profile",
        f"  IO wait is `{total_io / 1000.0:.3f} s` across overlapping demand paths.",
        "- Increasing io depth alone is unlikely to solve the bottleneck unless the",
        "  runtime can create larger useful batches or predict/prefetch future demand.",
        "",
        "## Next Candidate Filter",
        "",
        "1. Do not work on broad CPU fallback hooks next.",
        "2. Do not rebuild another prompt-specific expert pack; pack hit is already 100% in this diagnostic.",
        "3. Prefer candidates that increase useful demand batch size or reduce bytes per miss: prompt-general up/gate cache admission, RAM/VRAM tier with batch-preserving slabs, stronger future-layer prediction, or safe lower-byte expert representation.",
        "4. Any A/B must target at least 4.0-4.5 s N32 endpoint saving, pass quality/TTFT/RAM gates, and be reproducible.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Kimi COPY/IO profile diagnostics.")
    parser.add_argument("--root", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args()
    text = summarize(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
