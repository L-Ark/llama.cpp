#!/usr/bin/env python3
"""Analyze Kimi route-detail traces for per-layer expert persistence.

This is a dev-only feasibility tool. It estimates whether previous decode calls
for the same layer/kind can predict the next call's expert set with low false
prefetch overhead.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "seq": int(row["seq"]),
                "call": int(row["call"]),
                "mode": row["mode"],
                "kind": row["kind"],
                "tensor": row["tensor"],
                "layer": int(row["layer"]),
                "active_index": int(row["active_index"]),
                "expert_idx": int(row["expert_idx"]),
                "dst_id": int(row["dst_id"]),
                "flat_dst_id": int(row["flat_dst_id"]),
                "token_id": int(row["token_id"]),
                "n_active": int(row["n_active"]),
                "expert_bytes": int(row["expert_bytes"]),
                "src0_type": int(row["src0_type"]),
                "ne01": int(row["ne01"]),
                "ne00": int(row["ne00"]),
            })
    return rows


def group_calls(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, int, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["mode"] != "decode":
            continue
        key = (row["mode"], row["kind"], row["layer"], row["tensor"])
        call = row["call"]
        slot = grouped[key].setdefault(call, {
            "call": call,
            "mode": row["mode"],
            "kind": row["kind"],
            "layer": row["layer"],
            "tensor": row["tensor"],
            "experts": {},
        })
        slot["experts"][row["expert_idx"]] = row["expert_bytes"]

    return {key: [calls[c] for c in sorted(calls)] for key, calls in grouped.items()}


def score_calls(calls: list[dict[str, Any]], depth: int) -> dict[str, Any]:
    hist: deque[dict[int, int]] = deque(maxlen=depth)
    actual_bytes = 0
    covered_bytes = 0
    false_bytes = 0
    actual_events = 0
    covered_events = 0
    transitions = 0
    exact_repeats = 0
    jaccard_sum = 0.0

    for call in calls:
        actual: dict[int, int] = call["experts"]
        if not hist:
            hist.append(actual)
            continue
        predicted: dict[int, int] = {}
        for prev in hist:
            for expert, nbytes in prev.items():
                predicted[expert] = max(predicted.get(expert, 0), nbytes)

        actual_set = set(actual)
        pred_set = set(predicted)
        inter = actual_set & pred_set
        union = actual_set | pred_set

        transitions += 1
        if actual_set == pred_set:
            exact_repeats += 1
        jaccard_sum += len(inter) / len(union) if union else 0.0
        actual_events += len(actual)
        covered_events += len(inter)
        actual_bytes += sum(actual.values())
        covered_bytes += sum(actual[e] for e in inter)
        false_bytes += sum(predicted[e] for e in pred_set - actual_set)
        hist.append(actual)

    return {
        "transitions": transitions,
        "actual_events": actual_events,
        "covered_events": covered_events,
        "actual_bytes": actual_bytes,
        "covered_bytes": covered_bytes,
        "false_bytes": false_bytes,
        "event_coverage": covered_events / actual_events if actual_events else 0.0,
        "byte_coverage": covered_bytes / actual_bytes if actual_bytes else 0.0,
        "false_prefetch_ratio": false_bytes / actual_bytes if actual_bytes else 0.0,
        "net_byte_multiplier": (actual_bytes + false_bytes) / actual_bytes if actual_bytes else 1.0,
        "exact_repeat_rate": exact_repeats / transitions if transitions else 0.0,
        "avg_jaccard": jaccard_sum / transitions if transitions else 0.0,
    }


def combine(items: list[dict[str, Any]]) -> dict[str, Any]:
    total = {
        "groups": len(items),
        "transitions": 0,
        "actual_events": 0,
        "covered_events": 0,
        "actual_bytes": 0,
        "covered_bytes": 0,
        "false_bytes": 0,
        "exact_repeat_weighted": 0.0,
        "jaccard_weighted": 0.0,
    }
    for item in items:
        transitions = item["transitions"]
        for key in ("transitions", "actual_events", "covered_events", "actual_bytes", "covered_bytes", "false_bytes"):
            total[key] += item[key]
        total["exact_repeat_weighted"] += item["exact_repeat_rate"] * transitions
        total["jaccard_weighted"] += item["avg_jaccard"] * transitions

    transitions = total["transitions"]
    actual_events = total["actual_events"]
    actual_bytes = total["actual_bytes"]
    return {
        "groups": total["groups"],
        "transitions": transitions,
        "actual_events": actual_events,
        "covered_events": total["covered_events"],
        "actual_bytes": actual_bytes,
        "covered_bytes": total["covered_bytes"],
        "false_bytes": total["false_bytes"],
        "event_coverage": total["covered_events"] / actual_events if actual_events else 0.0,
        "byte_coverage": total["covered_bytes"] / actual_bytes if actual_bytes else 0.0,
        "false_prefetch_ratio": total["false_bytes"] / actual_bytes if actual_bytes else 0.0,
        "net_byte_multiplier": (actual_bytes + total["false_bytes"]) / actual_bytes if actual_bytes else 1.0,
        "exact_repeat_rate": total["exact_repeat_weighted"] / transitions if transitions else 0.0,
        "avg_jaccard": total["jaccard_weighted"] / transitions if transitions else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detail-csv", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--depth", type=int, action="append", default=[])
    args = parser.parse_args()

    depths = args.depth or [1, 2, 4]
    rows = read_rows(Path(args.detail_csv))
    grouped = group_calls(rows)

    per_group_rows: list[dict[str, Any]] = []
    aggregate: dict[str, Any] = {}
    for depth in depths:
        scored: list[dict[str, Any]] = []
        for (mode, kind, layer, tensor), calls in grouped.items():
            stats = score_calls(calls, depth)
            row = {
                "depth": depth,
                "mode": mode,
                "kind": kind,
                "layer": layer,
                "tensor": tensor,
                "calls": len(calls),
                **stats,
            }
            per_group_rows.append(row)
            if stats["transitions"] > 0:
                scored.append(row)

        aggregate[f"depth{depth}_all"] = combine(scored)
        for kind in sorted({r["kind"] for r in scored}):
            aggregate[f"depth{depth}_{kind}"] = combine([r for r in scored if r["kind"] == kind])

    passing = {
        key: value for key, value in aggregate.items()
        if value["byte_coverage"] >= 0.60
        and value["false_prefetch_ratio"] <= 0.25
        and value["net_byte_multiplier"] <= 1.25
    }
    best = max(aggregate.items(), key=lambda kv: (kv[1]["byte_coverage"], -kv[1]["false_prefetch_ratio"]))

    out = {
        "note": "dev-only route-detail same-layer consecutive-call stability; not SOTA",
        "detail_csv": args.detail_csv,
        "rows": len(rows),
        "groups": len(grouped),
        "gate": {
            "byte_coverage_min": 0.60,
            "false_prefetch_ratio_max": 0.25,
            "net_byte_multiplier_max": 1.25,
        },
        "best_by_coverage": {"name": best[0], **best[1]},
        "passing_settings": passing,
        "aggregate": aggregate,
    }

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_group_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_group_rows)

    print(json.dumps(out["best_by_coverage"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
