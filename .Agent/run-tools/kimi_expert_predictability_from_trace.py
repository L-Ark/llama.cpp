#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


LAYER_RE = re.compile(r"blk\.(\d+)\.ffn_(up|gate|down)_exps\.weight")


@dataclass
class Call:
    tensor: str
    experts: set[int]
    expert_bytes: int

    @property
    def kind(self) -> str:
        match = LAYER_RE.search(self.tensor)
        return match.group(2) if match else "other"


def load_calls(path: Path) -> list[Call]:
    calls: list[Call] = []
    current_tensor = ""
    current_experts: set[int] = set()
    current_bytes = 0

    def flush() -> None:
        nonlocal current_tensor, current_experts, current_bytes
        if current_tensor:
            calls.append(Call(current_tensor, set(current_experts), current_bytes))
        current_tensor = ""
        current_experts = set()
        current_bytes = 0

    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            tensor = row.get("tensor", "")
            if not tensor:
                continue
            if tensor != current_tensor:
                flush()
                current_tensor = tensor
            current_experts.add(int(row["expert_idx"]))
            current_bytes = int(row["expert_bytes"])
    flush()
    return calls


def empty_bucket() -> dict:
    return {
        "calls": 0,
        "predictable_calls": 0,
        "actual_events": 0,
        "predicted_events": 0,
        "hit_events": 0,
        "actual_bytes": 0,
        "predicted_bytes": 0,
        "hit_bytes": 0,
    }


def add_metrics(bucket: dict, actual: Call, predicted: set[int] | None) -> None:
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


def finalize(bucket: dict) -> dict:
    actual_events = bucket["actual_events"]
    predicted_events = bucket["predicted_events"]
    actual_bytes = bucket["actual_bytes"]
    predicted_bytes = bucket["predicted_bytes"]
    out = dict(bucket)
    out["recall"] = bucket["hit_events"] / actual_events if actual_events else 0.0
    out["precision"] = bucket["hit_events"] / predicted_events if predicted_events else 0.0
    out["byte_recall"] = bucket["hit_bytes"] / actual_bytes if actual_bytes else 0.0
    out["predicted_to_actual_byte_ratio"] = predicted_bytes / actual_bytes if actual_bytes else 0.0
    out["predictable_call_ratio"] = bucket["predictable_calls"] / bucket["calls"] if bucket["calls"] else 0.0
    return out


def analyze_trace(path: Path) -> dict:
    calls = load_calls(path)
    previous_by_tensor: dict[str, set[int]] = {}
    total = empty_bucket()
    by_kind: dict[str, dict] = {}

    for call in calls:
        predicted = previous_by_tensor.get(call.tensor)
        add_metrics(total, call, predicted)
        add_metrics(by_kind.setdefault(call.kind, empty_bucket()), call, predicted)
        previous_by_tensor[call.tensor] = set(call.experts)

    return {
        "trace": str(path),
        "prompt": path.parent.name,
        "calls": len(calls),
        "unique_tensors": len({call.tensor for call in calls}),
        "total": finalize(total),
        "by_kind": {kind: finalize(bucket) for kind, bucket in sorted(by_kind.items())},
    }


def aggregate(results: list[dict]) -> dict:
    total = empty_bucket()
    by_kind: dict[str, dict] = {}
    for result in results:
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


def write_csv(path: Path, results: list[dict], agg: dict) -> None:
    fields = [
        "scope",
        "prompt",
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
        rows = [("aggregate", "all", "all", agg["total"])]
        rows.extend(("aggregate", "all", kind, bucket) for kind, bucket in agg["by_kind"].items())
        for result in results:
            rows.append(("prompt", result["prompt"], "all", result["total"]))
            rows.extend(("prompt", result["prompt"], kind, bucket) for kind, bucket in result["by_kind"].items())
        for scope, prompt, kind, bucket in rows:
            writer.writerow(
                {
                    "scope": scope,
                    "prompt": prompt,
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
    total = report["aggregate"]["total"]
    lines = [
        "# Kimi expert route predictability",
        "",
        f"- Input traces: `{len(report['input_traces'])}`",
        f"- Aggregate calls: `{total['calls']}`",
        "",
        "Predictor: previous call for the same tensor predicts the current active expert set.",
        "",
        "## Aggregate",
        "",
        "| kind | calls | predictable calls | recall | precision | byte recall | predicted/actual bytes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    rows = [("all", total), *report["aggregate"]["by_kind"].items()]
    for kind, bucket in rows:
        lines.append(
            f"| {kind} | {bucket['calls']} | {bucket['predictable_call_ratio']:.4f} | "
            f"{bucket['recall']:.4f} | {bucket['precision']:.4f} | "
            f"{bucket['byte_recall']:.4f} | {bucket['predicted_to_actual_byte_ratio']:.4f} |"
        )
    lines.extend(["", "## Per Prompt", "", "| prompt | calls | recall | precision | byte recall |", "| --- | ---: | ---: | ---: | ---: |"])
    for result in report["results"]:
        bucket = result["total"]
        lines.append(
            f"| {result['prompt']} | {bucket['calls']} | {bucket['recall']:.4f} | "
            f"{bucket['precision']:.4f} | {bucket['byte_recall']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Kimi expert route predictability from route-trace CSV files.")
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    results = [analyze_trace(path) for path in args.trace]
    agg = aggregate(results)
    report = {
        "input_traces": [str(path) for path in args.trace],
        "predictor": "previous_call_same_tensor",
        "aggregate": agg,
        "results": results,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.out_csv, results, agg)
    write_markdown(args.out_md, report)

    total = agg["total"]
    print(f"traces={len(results)}")
    print(f"calls={total['calls']}")
    print(f"predictable_call_ratio={total['predictable_call_ratio']:.6f}")
    print(f"recall={total['recall']:.6f}")
    print(f"precision={total['precision']:.6f}")
    print(f"byte_recall={total['byte_recall']:.6f}")
    print(f"predicted_to_actual_byte_ratio={total['predicted_to_actual_byte_ratio']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
