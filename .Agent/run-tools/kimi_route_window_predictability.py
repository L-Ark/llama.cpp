#!/usr/bin/env python3
"""Estimate ordered-window expert demand predictability from route traces.

The input route-trace files do not expose clean token boundaries, so this script
uses fixed event windows. It reports whether recent expert demand could have
predicted upcoming expert movement well enough to justify a runtime prefetcher.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_events(path: Path) -> list[tuple[str, int]]:
    if "/test_" in str(path) or path.name.startswith("test_"):
        raise SystemExit(f"refusing to read held-out test path: {path}")
    events: list[tuple[str, int]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row['tensor']}#{row['expert_idx']}"
            events.append((key, int(row["expert_bytes"])))
    return events


def chunks(events: list[tuple[str, int]], size: int) -> list[list[tuple[str, int]]]:
    return [events[i : i + size] for i in range(0, len(events), size) if events[i : i + size]]


def summarize_windows(windows: list[list[tuple[str, int]]], depth: int) -> dict[str, Any]:
    totals = defaultdict(float)
    per_window: list[dict[str, float | int]] = []

    for i, actual in enumerate(windows):
        if i == 0:
            continue
        history = windows[max(0, i - depth) : i]
        predicted: dict[str, int] = {}
        for win in history:
            for key, nbytes in win:
                predicted[key] = max(predicted.get(key, 0), nbytes)

        actual_unique: dict[str, int] = {}
        actual_bytes = 0
        covered_bytes = 0
        covered_events = 0
        for key, nbytes in actual:
            actual_bytes += nbytes
            actual_unique[key] = max(actual_unique.get(key, 0), nbytes)
            if key in predicted:
                covered_events += 1
                covered_bytes += nbytes

        false_bytes = sum(nbytes for key, nbytes in predicted.items() if key not in actual_unique)
        predicted_bytes = sum(predicted.values())

        row = {
            "window_index": i,
            "events": len(actual),
            "actual_bytes": actual_bytes,
            "covered_bytes": covered_bytes,
            "covered_events": covered_events,
            "predicted_unique": len(predicted),
            "actual_unique": len(actual_unique),
            "predicted_bytes": predicted_bytes,
            "false_bytes": false_bytes,
            "byte_coverage": covered_bytes / actual_bytes if actual_bytes else 0.0,
            "event_coverage": covered_events / len(actual) if actual else 0.0,
            "false_prefetch_ratio": false_bytes / actual_bytes if actual_bytes else 0.0,
            "net_byte_multiplier": (actual_bytes + false_bytes) / actual_bytes if actual_bytes else 1.0,
        }
        per_window.append(row)
        for key, value in row.items():
            if key == "window_index":
                continue
            totals[key] += float(value)

    n = len(per_window)
    if n == 0:
        return {
            "windows_evaluated": 0,
            "byte_coverage": 0.0,
            "event_coverage": 0.0,
            "false_prefetch_ratio": 0.0,
            "net_byte_multiplier": 1.0,
            "actual_bytes": 0,
            "covered_bytes": 0,
            "false_bytes": 0,
        }

    actual_bytes = totals["actual_bytes"]
    covered_bytes = totals["covered_bytes"]
    false_bytes = totals["false_bytes"]
    return {
        "windows_evaluated": n,
        "actual_bytes": int(actual_bytes),
        "covered_bytes": int(covered_bytes),
        "false_bytes": int(false_bytes),
        "byte_coverage": covered_bytes / actual_bytes if actual_bytes else 0.0,
        "event_coverage": totals["covered_events"] / totals["events"] if totals["events"] else 0.0,
        "false_prefetch_ratio": false_bytes / actual_bytes if actual_bytes else 0.0,
        "net_byte_multiplier": (actual_bytes + false_bytes) / actual_bytes if actual_bytes else 1.0,
        "avg_predicted_unique": totals["predicted_unique"] / n,
        "avg_actual_unique": totals["actual_unique"] / n,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces-glob", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--window", type=int, action="append", default=[])
    parser.add_argument("--depth", type=int, action="append", default=[])
    args = parser.parse_args()

    windows_to_test = args.window or [256, 512, 1024, 1536, 2048]
    depths = args.depth or [1, 2, 4]
    paths = sorted(Path(".").glob(args.traces_glob))
    if not paths:
        raise SystemExit(f"no trace files matched: {args.traces_glob}")

    rows: list[dict[str, Any]] = []
    for path in paths:
        events = read_events(path)
        prompt = path.parent.name
        for window_size in windows_to_test:
            trace_windows = chunks(events, window_size)
            for depth in depths:
                stats = summarize_windows(trace_windows, depth)
                rows.append({
                    "prompt": prompt,
                    "path": str(path),
                    "window_size": window_size,
                    "history_depth": depth,
                    "trace_events": len(events),
                    **stats,
                })

    aggregate: dict[str, dict[str, Any]] = {}
    for window_size in windows_to_test:
        for depth in depths:
            selected = [r for r in rows if r["window_size"] == window_size and r["history_depth"] == depth]
            actual = sum(int(r["actual_bytes"]) for r in selected)
            covered = sum(int(r["covered_bytes"]) for r in selected)
            false = sum(int(r["false_bytes"]) for r in selected)
            min_cov = min(float(r["byte_coverage"]) for r in selected)
            key = f"window{window_size}_depth{depth}"
            aggregate[key] = {
                "window_size": window_size,
                "history_depth": depth,
                "prompts": len(selected),
                "actual_bytes": actual,
                "covered_bytes": covered,
                "false_bytes": false,
                "byte_coverage": covered / actual if actual else 0.0,
                "false_prefetch_ratio": false / actual if actual else 0.0,
                "net_byte_multiplier": (actual + false) / actual if actual else 1.0,
                "min_prompt_byte_coverage": min_cov,
            }

    passing = [
        item for item in aggregate.values()
        if item["byte_coverage"] >= 0.60
        and item["false_prefetch_ratio"] <= 0.25
        and item["net_byte_multiplier"] <= 1.25
        and item["min_prompt_byte_coverage"] >= 0.35
    ]
    best = max(aggregate.values(), key=lambda x: (x["byte_coverage"], -x["false_prefetch_ratio"]))

    out = {
        "note": "dev-only fixed-window route-trace predictability; not token-level router accuracy",
        "traces_glob": args.traces_glob,
        "trace_count": len(paths),
        "gate": {
            "byte_coverage_min": 0.60,
            "false_prefetch_ratio_max": 0.25,
            "net_byte_multiplier_max": 1.25,
            "min_prompt_byte_coverage_min": 0.35,
        },
        "best_by_coverage": best,
        "passing_settings": passing,
        "aggregate": dict(sorted(aggregate.items())),
    }

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({
        "best_by_coverage": best,
        "passing_settings": passing,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
