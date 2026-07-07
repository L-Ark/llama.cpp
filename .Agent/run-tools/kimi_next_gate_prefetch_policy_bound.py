#!/usr/bin/env python3
"""Evaluate bounded next-gate prefetch policies from shadow CSVs.

This is an offline, dev-only upper-bound screen. It uses next-gate shadow rows
to estimate whether a bounded runtime prefetch policy can keep false bytes low
while retaining enough useful expert coverage to justify implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_csv_ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def parse_csv_floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def parse_values(raw: str | None) -> list[int]:
    if raw is None or raw == "":
        return []
    return [int(x) for x in raw.split("|") if x != ""]


def load_prompt(path: Path) -> list[dict[str, Any]]:
    actual: dict[tuple[int, int], set[int]] = {}
    pred: dict[tuple[int, int], list[int]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            call = int(row["call"])
            target_layer = int(row["target_layer"])
            key = (call, target_layer)
            values = parse_values(row.get("values"))
            if row["kind"] == "actual":
                actual[key] = set(values)
            elif row["kind"] == "pred":
                pred[key] = values

    pairs = []
    for key, pred_values in pred.items():
        actual_set = actual.get(key)
        if actual_set is None:
            continue
        pairs.append({
            "call": key[0],
            "target_layer": key[1],
            "pred": pred_values,
            "actual": actual_set,
        })
    return pairs


def add_stats(dst: dict[str, int], pair: dict[str, Any], cap: int) -> None:
    pred_values = pair["pred"][:cap]
    pred_set = set(pred_values)
    actual_set = pair["actual"]
    hits = pred_set & actual_set
    false = pred_set - actual_set
    missed = actual_set - pred_set
    dst["pairs"] += 1
    dst["actual"] += len(actual_set)
    dst["pred"] += len(pred_set)
    dst["hit"] += len(hits)
    dst["false"] += len(false)
    dst["missed"] += len(missed)


def empty_stats() -> dict[str, int]:
    return {"pairs": 0, "actual": 0, "pred": 0, "hit": 0, "false": 0, "missed": 0}


def finalize(stats: dict[str, int]) -> dict[str, Any]:
    actual = stats["actual"]
    pred = stats["pred"]
    out: dict[str, Any] = dict(stats)
    out["recall"] = stats["hit"] / actual if actual else 0.0
    out["precision"] = stats["hit"] / pred if pred else 0.0
    out["false_over_actual"] = stats["false"] / actual if actual else 0.0
    out["pred_over_actual"] = pred / actual if actual else 0.0
    return out


def stats_for_pairs(pairs: list[dict[str, Any]], cap: int, layers: set[int] | None = None) -> dict[str, Any]:
    stats = empty_stats()
    by_layer: dict[int, dict[str, int]] = defaultdict(empty_stats)
    for pair in pairs:
        layer = pair["target_layer"]
        if layers is not None and layer not in layers:
            continue
        add_stats(stats, pair, cap)
        add_stats(by_layer[layer], pair, cap)
    return {
        "summary": finalize(stats),
        "layers": {str(layer): finalize(s) for layer, s in sorted(by_layer.items())},
    }


def choose_layers(
        train_pairs: list[dict[str, Any]],
        cap: int,
        min_precision: float,
        max_false_over_actual: float,
        min_pairs: int) -> set[int]:
    by_layer: dict[int, dict[str, int]] = defaultdict(empty_stats)
    for pair in train_pairs:
        add_stats(by_layer[pair["target_layer"]], pair, cap)
    selected = set()
    for layer, stats in by_layer.items():
        row = finalize(stats)
        if row["pairs"] < min_pairs:
            continue
        if row["precision"] >= min_precision and row["false_over_actual"] <= max_false_over_actual:
            selected.add(layer)
    return selected


def evaluate_policy(
        prompt_pairs: dict[str, list[dict[str, Any]]],
        cap: int,
        min_precision: float | None,
        max_false_over_actual: float,
        min_pairs: int) -> dict[str, Any]:
    per_prompt = {}
    agg = empty_stats()
    selected_counts = []
    selected_union: set[int] = set()
    for prompt, test_pairs in sorted(prompt_pairs.items()):
        if min_precision is None:
            selected = None
        else:
            train_pairs = [
                pair
                for train_prompt, pairs in prompt_pairs.items()
                if train_prompt != prompt
                for pair in pairs
            ]
            selected = choose_layers(train_pairs, cap, min_precision, max_false_over_actual, min_pairs)
            selected_counts.append(len(selected))
            selected_union.update(selected)
        result = stats_for_pairs(test_pairs, cap, selected)
        per_prompt[prompt] = {
            "selected_layers": "all" if selected is None else sorted(selected),
            **result["summary"],
        }
        for key in ("pairs", "actual", "pred", "hit", "false", "missed"):
            agg[key] += result["summary"][key]
    summary = finalize(agg)
    summary["selected_layers_mean"] = (
        sum(selected_counts) / len(selected_counts) if selected_counts else "all"
    )
    summary["selected_layers_union"] = "all" if min_precision is None else sorted(selected_union)
    return {"summary": summary, "prompts": per_prompt}


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Kimi next-gate bounded prefetch policy bound",
        "",
        "This is an offline dev-only screen. It does not implement runtime prefetch and does not claim SOTA.",
        "",
        f"- source root: `{result['source_root']}`",
        f"- prompts: `{len(result['prompts'])}`",
        f"- caps: `{result['caps']}`",
        f"- min precisions: `{result['min_precisions']}`",
        f"- max false/actual gate for layer selection: `{result['max_false_over_actual']}`",
        "",
        "## Policy Summary",
        "",
        "| policy | pairs | selected layers | pred/actual | recall | precision | false/actual |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, data in result["policies"].items():
        s = data["summary"]
        selected = s["selected_layers_mean"]
        if selected != "all":
            selected = f"{float(selected):.1f}"
        lines.append(
            f"| `{name}` | `{s['pairs']}` | `{selected}` | `{s['pred_over_actual']:.4f}` | "
            f"`{s['recall']:.4f}` | `{s['precision']:.4f}` | `{s['false_over_actual']:.4f}` |"
        )

    lines += [
        "",
        "## Decision Notes",
        "",
        "- `cap1_all` is the strict low-extra-byte bound; it cannot exceed about 12.5% recall because each actual set has 8 experts.",
        "- A runtime candidate needs low false bytes and enough recall to hide exposed IO. Low false bytes alone is not enough.",
        "- These metrics are optimistic because they do not subtract experts already resident in VRAM.",
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
    ap = argparse.ArgumentParser(description="Bound next-gate prefetch policies from shadow CSVs.")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--caps", default="1,2,4,8")
    ap.add_argument("--min-precisions", default="0.75,0.8,0.85,0.9")
    ap.add_argument("--max-false-over-actual", type=float, default=0.15)
    ap.add_argument("--min-layer-pairs", type=int, default=64)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    args = ap.parse_args()

    prompt_pairs = {}
    for csv_path in sorted(args.root.glob("*/next-gate-shadow.csv")):
        prompt_pairs[csv_path.parent.name] = load_prompt(csv_path)
    if not prompt_pairs:
        raise RuntimeError(f"no next-gate-shadow.csv files under {args.root}")

    caps = parse_csv_ints(args.caps)
    min_precisions = parse_csv_floats(args.min_precisions)
    policies = {}
    for cap in caps:
        policies[f"cap{cap}_all"] = evaluate_policy(
            prompt_pairs, cap, None, args.max_false_over_actual, args.min_layer_pairs)
        for min_precision in min_precisions:
            policies[f"cap{cap}_loo_layer_prec{min_precision:.2f}"] = evaluate_policy(
                prompt_pairs, cap, min_precision, args.max_false_over_actual, args.min_layer_pairs)

    result = {
        "kind": "kimi_next_gate_prefetch_policy_bound",
        "source_root": str(args.root),
        "prompts": sorted(prompt_pairs),
        "caps": caps,
        "min_precisions": min_precisions,
        "max_false_over_actual": args.max_false_over_actual,
        "min_layer_pairs": args.min_layer_pairs,
        "policies": policies,
        "reproduce_command": " ".join(["python3", *(__import__("sys").argv)]),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.out_md, result)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
