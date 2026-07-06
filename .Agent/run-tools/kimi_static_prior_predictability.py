#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from kimi_expert_predictability_from_trace import Call, empty_bucket, finalize, load_calls  # noqa: E402


def parse_int_list(text: str) -> list[int]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"top-k must be positive: {value}")
        values.append(value)
    return sorted(set(values))


def train_priors(train_calls: list[Call]) -> dict[str, list[int]]:
    counts: dict[str, collections.Counter[int]] = {}
    for call in train_calls:
        counter = counts.setdefault(call.tensor, collections.Counter())
        for expert in call.experts:
            counter[expert] += 1
    return {
        tensor: [expert for expert, _ in counter.most_common()]
        for tensor, counter in counts.items()
    }


def add_prediction(bucket: dict, actual: Call, predicted: set[int] | None) -> None:
    bucket["calls"] += 1
    actual_count = len(actual.experts)
    bucket["actual_events"] += actual_count
    bucket["actual_bytes"] += actual_count * actual.expert_bytes
    if predicted is None:
        return
    bucket["predictable_calls"] += 1
    predicted_count = len(predicted)
    hit_count = len(actual.experts & predicted)
    bucket["predicted_events"] += predicted_count
    bucket["hit_events"] += hit_count
    bucket["predicted_bytes"] += predicted_count * actual.expert_bytes
    bucket["hit_bytes"] += hit_count * actual.expert_bytes


def evaluate_calls(calls: list[Call], priors: dict[str, list[int]], top_k: int) -> dict:
    total = empty_bucket()
    by_kind: dict[str, dict] = {}
    for call in calls:
        prior = priors.get(call.tensor)
        predicted = set(prior[:top_k]) if prior else None
        add_prediction(total, call, predicted)
        add_prediction(by_kind.setdefault(call.kind, empty_bucket()), call, predicted)
    return {
        "total": finalize(total),
        "by_kind": {kind: finalize(bucket) for kind, bucket in sorted(by_kind.items())},
    }


def aggregate_fold_results(folds: list[dict], top_k: int) -> dict:
    total = empty_bucket()
    by_kind: dict[str, dict] = {}
    for fold in folds:
        result = fold["by_top_k"][str(top_k)]
        for key in total:
            total[key] += result["total"][key]
        for kind, bucket in result["by_kind"].items():
            dst = by_kind.setdefault(kind, empty_bucket())
            for key in dst:
                dst[key] += bucket[key]
    return {
        "total": finalize(total),
        "by_kind": {kind: finalize(bucket) for kind, bucket in sorted(by_kind.items())},
    }


def write_csv(path: Path, report: dict) -> None:
    fields = [
        "scope",
        "prompt",
        "top_k",
        "kind",
        "calls",
        "predictable_call_ratio",
        "actual_events",
        "recall",
        "precision",
        "byte_recall",
        "predicted_to_actual_byte_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for top_k in report["top_k_values"]:
            aggregate = report["aggregate_by_top_k"][str(top_k)]
            rows = [("aggregate", "all", "all", aggregate["total"])]
            rows.extend(("aggregate", "all", kind, bucket) for kind, bucket in aggregate["by_kind"].items())
            for fold in report["folds"]:
                result = fold["by_top_k"][str(top_k)]
                rows.append(("fold", fold["heldout_prompt"], "all", result["total"]))
                rows.extend(("fold", fold["heldout_prompt"], kind, bucket) for kind, bucket in result["by_kind"].items())
            for scope, prompt, kind, bucket in rows:
                writer.writerow(
                    {
                        "scope": scope,
                        "prompt": prompt,
                        "top_k": top_k,
                        "kind": kind,
                        "calls": bucket["calls"],
                        "predictable_call_ratio": bucket["predictable_call_ratio"],
                        "actual_events": bucket["actual_events"],
                        "recall": bucket["recall"],
                        "precision": bucket["precision"],
                        "byte_recall": bucket["byte_recall"],
                        "predicted_to_actual_byte_ratio": bucket["predicted_to_actual_byte_ratio"],
                    }
                )


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# Kimi static expert-prior predictability",
        "",
        f"- Input traces: `{len(report['input_traces'])}`",
        f"- Folds: `{len(report['folds'])}`",
        "",
        "Method: leave-one-prompt-out; train per-tensor top-K experts on the other dev prompts.",
        "",
        "| top K | recall | precision | byte recall | predicted/actual bytes |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for top_k in report["top_k_values"]:
        bucket = report["aggregate_by_top_k"][str(top_k)]["total"]
        lines.append(
            f"| {top_k} | {bucket['recall']:.4f} | {bucket['precision']:.4f} | "
            f"{bucket['byte_recall']:.4f} | {bucket['predicted_to_actual_byte_ratio']:.4f} |"
        )
    lines.extend(["", "## By Kind At Largest K", ""])
    largest = str(max(report["top_k_values"]))
    lines.extend([
        "| kind | recall | precision | byte recall | predicted/actual bytes |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for kind, bucket in report["aggregate_by_top_k"][largest]["by_kind"].items():
        lines.append(
            f"| {kind} | {bucket['recall']:.4f} | {bucket['precision']:.4f} | "
            f"{bucket['byte_recall']:.4f} | {bucket['predicted_to_actual_byte_ratio']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Leave-one-prompt-out static top-K expert-prior analysis.")
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--top-k", default="8,16,32,64,128")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    top_k_values = parse_int_list(args.top_k)
    calls_by_prompt = {path.parent.name: load_calls(path) for path in args.trace}
    folds = []
    for heldout_prompt, heldout_calls in sorted(calls_by_prompt.items()):
        train_calls = [
            call
            for prompt, calls in calls_by_prompt.items()
            if prompt != heldout_prompt
            for call in calls
        ]
        priors = train_priors(train_calls)
        fold = {
            "heldout_prompt": heldout_prompt,
            "train_prompts": sorted(prompt for prompt in calls_by_prompt if prompt != heldout_prompt),
            "train_calls": len(train_calls),
            "heldout_calls": len(heldout_calls),
            "trained_tensors": len(priors),
            "by_top_k": {},
        }
        for top_k in top_k_values:
            fold["by_top_k"][str(top_k)] = evaluate_calls(heldout_calls, priors, top_k)
        folds.append(fold)

    aggregate_by_top_k = {
        str(top_k): aggregate_fold_results(folds, top_k)
        for top_k in top_k_values
    }
    report = {
        "input_traces": [str(path) for path in args.trace],
        "method": "leave_one_prompt_out_static_tensor_topk",
        "top_k_values": top_k_values,
        "folds": folds,
        "aggregate_by_top_k": aggregate_by_top_k,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.out_csv, report)
    write_markdown(args.out_md, report)

    print(f"traces={len(args.trace)}")
    print(f"folds={len(folds)}")
    for top_k in top_k_values:
        bucket = aggregate_by_top_k[str(top_k)]["total"]
        print(
            f"top_k={top_k} recall={bucket['recall']:.6f} "
            f"precision={bucket['precision']:.6f} "
            f"byte_recall={bucket['byte_recall']:.6f} "
            f"predicted_to_actual_byte_ratio={bucket['predicted_to_actual_byte_ratio']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
