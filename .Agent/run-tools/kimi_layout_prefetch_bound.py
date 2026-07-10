#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
from collections import defaultdict


def fnum(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def inum(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_csv(path: pathlib.Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def read_metrics(run_dir: pathlib.Path):
    path = run_dir / "metrics.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(errors="replace"))
    except json.JSONDecodeError:
        return {}


def quantile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[idx]


def pearson(xs, ys):
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0.0 or den_y == 0.0:
        return 0.0
    return num / (den_x * den_y)


def fmt(value):
    return f"{value:.3f}"


def write_csv(path: pathlib.Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def gap_bucket(gap_over_read):
    if gap_over_read <= 1:
        return "00 <=1"
    if gap_over_read <= 4:
        return "01 1-4"
    if gap_over_read <= 16:
        return "02 4-16"
    if gap_over_read <= 64:
        return "03 16-64"
    return "04 >64"


def load_batches(run_dir: pathlib.Path, decode_read_max_jobs: int):
    io_rows = read_csv(run_dir / "io-batch-profile.csv")
    locality_rows = {inum(r.get("seq")): r for r in read_csv(run_dir / "io-locality-profile.csv")}
    batches = []
    for row in io_rows:
        seq = inum(row.get("seq"))
        read_jobs = inum(row.get("read_jobs"))
        if read_jobs <= 0 or read_jobs > decode_read_max_jobs:
            continue
        loc = locality_rows.get(seq, {})
        read_bytes = fnum(loc.get("read_bytes"))
        gap_bytes = fnum(loc.get("gap_bytes"))
        span_bytes = fnum(loc.get("span_bytes"))
        wait_ms = fnum(row.get("wait_ms"))
        batches.append({
            "seq": seq,
            "op": row.get("op", ""),
            "first_tensor": row.get("first_tensor", ""),
            "read_jobs": read_jobs,
            "wait_ms": wait_ms,
            "wait_per_job": wait_ms / read_jobs if read_jobs else 0.0,
            "wall_ms": fnum(row.get("wall_ms")),
            "inflight_avg": fnum(row.get("inflight_avg")),
            "inflight_max": fnum(row.get("inflight_max")),
            "read_bytes": read_bytes,
            "read_mib": read_bytes / 1024**2,
            "gap_over_read": gap_bytes / read_bytes if read_bytes > 0 else 0.0,
            "span_over_read": span_bytes / read_bytes if read_bytes > 0 else 0.0,
            "adjacent_pairs": inum(loc.get("adjacent_pairs")),
            "source_switches": inum(loc.get("source_switches")),
            "same_tensor": inum(loc.get("same_tensor")),
        })
    batches.sort(key=lambda b: b["seq"])
    return batches


def layout_bins(batches):
    grouped = defaultdict(list)
    for b in batches:
        grouped[(b["op"], gap_bucket(b["gap_over_read"]))].append(b)
    rows = []
    for (op, bucket), items in grouped.items():
        waits = [b["wait_per_job"] for b in items]
        jobs = sum(b["read_jobs"] for b in items)
        wait_ms = sum(b["wait_ms"] for b in items)
        rows.append({
            "op": op,
            "gap_bucket": bucket,
            "rows": len(items),
            "read_jobs": jobs,
            "wait_ms": fmt(wait_ms),
            "wait_per_job_avg": fmt(wait_ms / jobs if jobs else 0.0),
            "wait_per_job_p50": fmt(quantile(waits, 0.50)),
            "wait_per_job_p10": fmt(quantile(waits, 0.10)),
            "gap_over_read_avg": fmt(sum(b["gap_over_read"] for b in items) / len(items)),
            "adjacent_pairs_per_job": fmt(sum(b["adjacent_pairs"] for b in items) / jobs if jobs else 0.0),
        })
    rows.sort(key=lambda r: (r["op"], r["gap_bucket"]))
    return rows


def batch_size_bound(batches, decode_runs):
    by_jobs = defaultdict(list)
    for b in batches:
        by_jobs[(b["op"], b["read_jobs"])].append(b)
    rows = []
    total_current = 0.0
    total_ideal_p10 = 0.0
    total_ideal_min = 0.0
    for (op, read_jobs), items in sorted(by_jobs.items()):
        waits_per_job = [b["wait_per_job"] for b in items]
        p10 = quantile(waits_per_job, 0.10)
        best = min(waits_per_job) if waits_per_job else 0.0
        current = sum(b["wait_ms"] for b in items)
        ideal_p10 = p10 * read_jobs * len(items)
        ideal_min = best * read_jobs * len(items)
        total_current += current
        total_ideal_p10 += ideal_p10
        total_ideal_min += ideal_min
        rows.append({
            "op": op,
            "read_jobs": read_jobs,
            "rows": len(items),
            "current_wait_ms": fmt(current),
            "current_wait_per_job": fmt(current / (read_jobs * len(items))),
            "p10_wait_per_job": fmt(p10),
            "min_wait_per_job": fmt(best),
            "ideal_p10_wait_ms": fmt(ideal_p10),
            "ideal_min_wait_ms": fmt(ideal_min),
            "p10_saving_ms": fmt(max(current - ideal_p10, 0.0)),
            "min_saving_ms": fmt(max(current - ideal_min, 0.0)),
        })
    summary = {
        "current_wait_ms": total_current,
        "ideal_p10_wait_ms": total_ideal_p10,
        "ideal_min_wait_ms": total_ideal_min,
        "p10_saving_ms": max(total_current - total_ideal_p10, 0.0),
        "min_saving_ms": max(total_current - total_ideal_min, 0.0),
        "p10_saving_ms_per_token": max(total_current - total_ideal_p10, 0.0) / decode_runs if decode_runs else 0.0,
        "min_saving_ms_per_token": max(total_current - total_ideal_min, 0.0) / decode_runs if decode_runs else 0.0,
    }
    return rows, summary


def future_prefetch_bound(batches, decode_runs, windows):
    rows = []
    for window in windows:
        current = 0.0
        ideal = 0.0
        groups = 0
        jobs_sum = 0
        for i in range(0, len(batches), window):
            group = batches[i:i + window]
            if not group:
                continue
            waits = [b["wait_ms"] for b in group]
            jobs = sum(b["read_jobs"] for b in group)
            current += sum(waits)
            ideal += max(waits)
            jobs_sum += jobs
            groups += 1
        saving = max(current - ideal, 0.0)
        rows.append({
            "future_window_batches": window,
            "groups": groups,
            "avg_jobs_per_window": fmt(jobs_sum / groups if groups else 0.0),
            "current_wait_ms": fmt(current),
            "optimistic_wait_ms": fmt(ideal),
            "saving_ms": fmt(saving),
            "saving_ms_per_token": fmt(saving / decode_runs if decode_runs else 0.0),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description="Estimate Kimi expert-pack layout and future-prefetch upper bounds.")
    ap.add_argument("--input", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--decode-read-max-jobs", type=int, default=8)
    args = ap.parse_args()

    run_dir = args.input
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    metrics = read_metrics(run_dir)
    decode_runs = inum(metrics.get("decode_runs"))
    batches = load_batches(run_dir, args.decode_read_max_jobs)
    if not batches:
        raise SystemExit("No decode-like IO batches found")

    bins = layout_bins(batches)
    size_rows, size_summary = batch_size_bound(batches, decode_runs)
    future_rows = future_prefetch_bound(batches, decode_runs, [2, 3, 4, 8, 16])

    wait_per_job = [b["wait_per_job"] for b in batches]
    log_gap = [math.log1p(max(b["gap_over_read"], 0.0)) for b in batches]
    gap_wait_corr = pearson(log_gap, wait_per_job)
    total_wait = sum(b["wait_ms"] for b in batches)
    total_jobs = sum(b["read_jobs"] for b in batches)
    total_read_gib = sum(b["read_bytes"] for b in batches) / 1024**3
    total_wall = sum(b["wall_ms"] for b in batches)

    write_csv(out / "layout_bins.csv", bins, list(bins[0].keys()) if bins else [])
    write_csv(out / "batch_size_bound.csv", size_rows, list(size_rows[0].keys()) if size_rows else [])
    write_csv(out / "future_prefetch_bound.csv", future_rows, list(future_rows[0].keys()) if future_rows else [])

    summary = {
        "run_dir": str(run_dir),
        "decode_runs": decode_runs,
        "decode_ms": metrics.get("decode_ms"),
        "token_rate": metrics.get("token_rate"),
        "batches": len(batches),
        "read_jobs": total_jobs,
        "read_gib": total_read_gib,
        "wait_ms": total_wait,
        "wait_ms_per_token": total_wait / decode_runs if decode_runs else 0.0,
        "avg_read_jobs_per_batch": total_jobs / len(batches),
        "avg_wait_per_job": total_wait / total_jobs if total_jobs else 0.0,
        "gap_wait_corr_log": gap_wait_corr,
        "batch_size_bound": size_summary,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Kimi Layout / Future-Prefetch Bound",
        "",
        f"Input: `{run_dir}`",
        "",
        "## Baseline Decode-Like IO",
        "",
        f"- quality: `{metrics.get('quality', 'unknown')}`",
        f"- token rate: `{metrics.get('token_rate', 'unknown')}`",
        f"- decode: `{metrics.get('decode_ms', 'unknown')} ms / {decode_runs} runs`",
        f"- decode-like batches: `{len(batches)}`",
        f"- read jobs: `{total_jobs}`",
        f"- average read jobs/batch: `{total_jobs / len(batches):.3f}`",
        f"- read GiB: `{total_read_gib:.3f}`",
        f"- wait ms: `{total_wait:.3f}`",
        f"- wait ms/token: `{total_wait / decode_runs if decode_runs else 0.0:.3f}`",
        f"- wait/read job: `{total_wait / total_jobs if total_jobs else 0.0:.3f} ms`",
        f"- log(gap/read) vs wait/read-job Pearson: `{gap_wait_corr:.3f}`",
        "",
        "## Layout-Only Bound",
        "",
        "This bound replaces each batch's wait/read-job with the best observed",
        "wait/read-job for the same op and batch size. It is an optimistic",
        "layout/refill bound, not a measured speedup.",
        "",
        f"- p10 same-size saving: `{size_summary['p10_saving_ms']:.3f} ms`, `{size_summary['p10_saving_ms_per_token']:.3f} ms/token`",
        f"- min same-size saving: `{size_summary['min_saving_ms']:.3f} ms`, `{size_summary['min_saving_ms_per_token']:.3f} ms/token`",
        "",
        "| op | read jobs | rows | current wait ms | current wait/job | p10 wait/job | p10 saving ms | min saving ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in size_rows:
        lines.append(
            f"| `{row['op']}` | {row['read_jobs']} | {row['rows']} | {row['current_wait_ms']} | "
            f"{row['current_wait_per_job']} | {row['p10_wait_per_job']} | {row['p10_saving_ms']} | {row['min_saving_ms']} |"
        )
    lines += [
        "",
        "## Locality Bins",
        "",
        "| op | gap bucket | rows | read jobs | wait/read avg | wait/read p50 | wait/read p10 | gap/read avg | adjacent/job |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in bins:
        lines.append(
            f"| `{row['op']}` | `{row['gap_bucket']}` | {row['rows']} | {row['read_jobs']} | "
            f"{row['wait_per_job_avg']} | {row['wait_per_job_p50']} | {row['wait_per_job_p10']} | "
            f"{row['gap_over_read_avg']} | {row['adjacent_pairs_per_job']} |"
        )
    lines += [
        "",
        "## Future-Layer Prefetch Bound",
        "",
        "This is an optimistic upper bound. It assumes the next K decode-like",
        "batches can be exposed early and their waits overlap perfectly, so each",
        "window costs `max(wait)` instead of `sum(wait)`.",
        "",
        "| future window batches | groups | avg jobs/window | current wait ms | optimistic wait ms | saving ms | saving ms/token |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in future_rows:
        lines.append(
            f"| {row['future_window_batches']} | {row['groups']} | {row['avg_jobs_per_window']} | "
            f"{row['current_wait_ms']} | {row['optimistic_wait_ms']} | {row['saving_ms']} | {row['saving_ms_per_token']} |"
        )
    lines += [
        "",
        "## Decision Rules",
        "",
        "- If layout-only p10 saving is far below `100 ms/token`, do not start with",
        "  pack relayout as the next implementation.",
        "- If future-prefetch bound is large, implementation still requires a",
        "  prompt-general predictor and must be validated on held-out prompts only",
        "  after dev passes.",
        "- If neither bound is large enough, reaching `>5 tok/s` likely requires",
        "  smaller expert representation or a different compute/storage format.",
        "",
        "This report is offline analysis only and makes no SOTA claim.",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
