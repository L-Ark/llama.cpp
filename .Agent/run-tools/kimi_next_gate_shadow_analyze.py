#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_values(raw):
    if raw is None or raw == "":
        return []
    return [int(x) for x in raw.split("|") if x != ""]


def pct(num, den):
    return 0.0 if den == 0 else 100.0 * num / den


def load_rows(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["call"] = int(row["call"])
            row["source_layer"] = int(row["source_layer"])
            row["target_layer"] = int(row["target_layer"])
            row["record_us"] = int(row.get("record_us") or 0)
            row["values_list"] = parse_values(row.get("values"))
            rows.append(row)
    return rows


def summarize(rows, expert_bytes):
    actual_by_call_layer = {}
    preds = []
    record_us_by_call = defaultdict(int)

    for row in rows:
        call = row["call"]
        record_us_by_call[call] = max(record_us_by_call[call], row["record_us"])
        key = (call, row["target_layer"])
        if row["kind"] == "actual":
            actual_by_call_layer[key] = row
        elif row["kind"] == "pred":
            preds.append(row)

    totals = {
        "pairs": 0,
        "actual_experts": 0,
        "predicted_experts": 0,
        "hit_experts": 0,
        "false_experts": 0,
        "missed_experts": 0,
        "actual_bytes": 0,
        "predicted_bytes": 0,
        "hit_bytes": 0,
        "false_prefetch_bytes": 0,
        "missed_bytes": 0,
        "record_calls": len(record_us_by_call),
        "record_us_total": sum(record_us_by_call.values()),
        "record_us_avg": 0.0,
        "record_us_max": max(record_us_by_call.values()) if record_us_by_call else 0,
    }
    if record_us_by_call:
        totals["record_us_avg"] = totals["record_us_total"] / len(record_us_by_call)

    by_layer = defaultdict(lambda: {
        "pairs": 0,
        "actual_experts": 0,
        "predicted_experts": 0,
        "hit_experts": 0,
        "false_experts": 0,
        "missed_experts": 0,
        "actual_bytes": 0,
        "predicted_bytes": 0,
        "hit_bytes": 0,
        "false_prefetch_bytes": 0,
        "missed_bytes": 0,
    })

    unmatched = []
    for pred in preds:
        key = (pred["call"], pred["target_layer"])
        actual = actual_by_call_layer.get(key)
        if actual is None:
            unmatched.append(pred)
            continue

        pred_set = set(pred["values_list"])
        actual_set = set(actual["values_list"])
        hits = pred_set & actual_set
        false = pred_set - actual_set
        missed = actual_set - pred_set

        layer = pred["target_layer"]
        for dst in (totals, by_layer[layer]):
            dst["pairs"] += 1
            dst["actual_experts"] += len(actual_set)
            dst["predicted_experts"] += len(pred_set)
            dst["hit_experts"] += len(hits)
            dst["false_experts"] += len(false)
            dst["missed_experts"] += len(missed)
            dst["actual_bytes"] += len(actual_set) * expert_bytes
            dst["predicted_bytes"] += len(pred_set) * expert_bytes
            dst["hit_bytes"] += len(hits) * expert_bytes
            dst["false_prefetch_bytes"] += len(false) * expert_bytes
            dst["missed_bytes"] += len(missed) * expert_bytes

    totals["unmatched_predictions"] = len(unmatched)

    def finalize(d):
        d["expert_recall"] = 0.0 if d["actual_experts"] == 0 else d["hit_experts"] / d["actual_experts"]
        d["expert_precision"] = 0.0 if d["predicted_experts"] == 0 else d["hit_experts"] / d["predicted_experts"]
        d["byte_recall"] = 0.0 if d["actual_bytes"] == 0 else d["hit_bytes"] / d["actual_bytes"]
        d["false_over_actual_bytes"] = 0.0 if d["actual_bytes"] == 0 else d["false_prefetch_bytes"] / d["actual_bytes"]
        return d

    finalize(totals)
    layers = {str(layer): finalize(stats) for layer, stats in sorted(by_layer.items())}
    return {"summary": totals, "layers": layers}


def write_markdown(result, out_path, source_path, expert_bytes):
    s = result["summary"]
    lines = [
        "# Kimi next-gate shadow profiler report",
        "",
        f"- Source CSV: `{source_path}`",
        f"- Expert bytes used for byte metrics: `{expert_bytes}`",
        f"- Matched prediction pairs: `{s['pairs']}`",
        f"- Unmatched predictions: `{s['unmatched_predictions']}`",
        f"- Expert recall: `{pct(s['hit_experts'], s['actual_experts']):.2f}%`",
        f"- Expert precision: `{pct(s['hit_experts'], s['predicted_experts']):.2f}%`",
        f"- Byte recall: `{pct(s['hit_bytes'], s['actual_bytes']):.2f}%`",
        f"- False-prefetch bytes: `{s['false_prefetch_bytes']}`",
        f"- False/actual bytes: `{s['false_over_actual_bytes']:.4f}`",
        f"- Shadow record overhead: total `{s['record_us_total']}` us, avg `{s['record_us_avg']:.1f}` us/call, max `{s['record_us_max']}` us",
        "",
        "| target layer | pairs | expert recall | precision | false/actual bytes | false bytes | missed bytes |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer, stats in result["layers"].items():
        lines.append(
            f"| {layer} | {stats['pairs']} | "
            f"{pct(stats['hit_experts'], stats['actual_experts']):.2f}% | "
            f"{pct(stats['hit_experts'], stats['predicted_experts']):.2f}% | "
            f"{stats['false_over_actual_bytes']:.4f} | "
            f"{stats['false_prefetch_bytes']} | {stats['missed_bytes']} |"
        )
    out_path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Analyze Kimi next-layer gate shadow profiler CSV")
    ap.add_argument("shadow_csv", type=Path)
    ap.add_argument("--expert-bytes", type=int, default=1,
                    help="bytes per selected expert for byte recall accounting; default counts experts")
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--md-out", type=Path)
    args = ap.parse_args()

    rows = load_rows(args.shadow_csv)
    result = summarize(rows, args.expert_bytes)

    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.md_out:
        write_markdown(result, args.md_out, args.shadow_csv, args.expert_bytes)
    if not args.json_out and not args.md_out:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
