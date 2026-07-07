#!/usr/bin/env python3
"""Prompt-agnostic predictor bound from Kimi route-detail traces.

This is an offline dev-only diagnostic. It refuses held-out test paths and uses
only route-detail.csv files that expose layer, kind, call, and expert ids.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
from dataclasses import dataclass


@dataclass
class Step:
    prompt_id: str
    token_idx: int
    layer: int
    group: str
    experts: dict[int, int]

    @property
    def actual_bytes(self) -> int:
        return sum(self.experts.values())


def refuse_heldout(path: pathlib.Path) -> None:
    text = str(path)
    if "/test_" in text or path.name.startswith("test_"):
        raise SystemExit(f"refusing to read held-out test path: {path}")


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def group_for_kind(kind: str) -> str:
    return "down" if kind == "down" else "upgate"


def load_metrics(run_dir: pathlib.Path) -> dict:
    path = run_dir / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_detail(run_dir: pathlib.Path) -> tuple[str, dict, list[Step]]:
    refuse_heldout(run_dir)
    metrics = load_metrics(run_dir)
    prompt_id = metrics.get("prompt_id", run_dir.name)
    path = run_dir / "route-detail.csv"
    if not path.exists():
        return prompt_id, metrics, []

    grouped: dict[tuple[int, str, int], collections.Counter[int]] = collections.defaultdict(collections.Counter)
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            if row["mode"] != "decode":
                continue
            layer = int(row["layer"])
            if layer < 0:
                continue
            group = group_for_kind(row["kind"])
            call = int(row["call"])
            expert = int(row["expert_idx"])
            grouped[(layer, group, call)][expert] += int(row["expert_bytes"])

    seen = collections.Counter()
    steps = []
    for layer, group, call in sorted(grouped):
        key = (layer, group)
        token_idx = seen[key]
        seen[key] += 1
        steps.append(Step(
            prompt_id=prompt_id,
            token_idx=token_idx,
            layer=layer,
            group=group,
            experts=dict(grouped[(layer, group, call)]),
        ))
    return prompt_id, metrics, steps


def build_catalog(all_steps: list[Step]):
    samples: dict[tuple[int, str, int], list[int]] = collections.defaultdict(list)
    fallback: dict[tuple[int, str], list[int]] = collections.defaultdict(list)
    for step in all_steps:
        for expert, nbytes in step.experts.items():
            samples[(step.layer, step.group, expert)].append(nbytes)
            fallback[(step.layer, step.group)].append(nbytes)
    exact = {key: int(percentile(vals, 0.5)) for key, vals in samples.items()}
    fallback_median = {key: int(percentile(vals, 0.5)) for key, vals in fallback.items()}
    return exact, fallback_median


def predicted_bytes(layer: int, group: str, experts: set[int], exact_catalog, fallback_catalog) -> int:
    fallback = fallback_catalog.get((layer, group), 0)
    return sum(exact_catalog.get((layer, group, expert), fallback) for expert in experts)


def top_k(counter: collections.Counter[int], k: int) -> set[int]:
    if k <= 0:
        return set()
    return {expert for expert, _ in counter.most_common(k)}


def leave_one_out_global(steps_by_prompt: dict[str, list[Step]], prompt_id: str):
    out: dict[tuple[int, str], collections.Counter[int]] = collections.defaultdict(collections.Counter)
    for other_prompt, steps in steps_by_prompt.items():
        if other_prompt == prompt_id:
            continue
        for step in steps:
            out[(step.layer, step.group)].update(step.experts.keys())
    return out


def make_row(name: str, k: int, prompt_id: str, group: str, counts: collections.Counter):
    actual_bytes = counts["actual_bytes"]
    actual_events = counts["actual_events"]
    return {
        "predictor": name,
        "k": k,
        "prompt_id": prompt_id,
        "group": group,
        "steps": int(counts["steps"]),
        "actual_events": int(actual_events),
        "pred_events": int(counts["pred_events"]),
        "event_recall_pct": 100.0 * counts["hit_events"] / actual_events if actual_events else 0.0,
        "byte_recall_pct": 100.0 * counts["hit_bytes"] / actual_bytes if actual_bytes else 0.0,
        "full_step_pct": 100.0 * counts["full_steps"] / counts["steps"] if counts["steps"] else 0.0,
        "pred_over_actual_bytes": counts["pred_bytes"] / actual_bytes if actual_bytes else 0.0,
        "actual_gib": actual_bytes / 1024**3,
        "pred_gib": counts["pred_bytes"] / 1024**3,
        "hit_gib": counts["hit_bytes"] / 1024**3,
        "wasted_gib": counts["wasted_bytes"] / 1024**3,
        "empty_prediction_pct": 100.0 * counts["empty_predictions"] / counts["steps"] if counts["steps"] else 0.0,
    }


def evaluate_predictor(name: str, steps_by_prompt, k: int, exact_catalog, fallback_catalog):
    totals: dict[tuple[str, str], collections.Counter] = collections.defaultdict(collections.Counter)
    for prompt_id, steps in steps_by_prompt.items():
        global_counts = leave_one_out_global(steps_by_prompt, prompt_id)
        history: dict[tuple[int, str], collections.Counter[int]] = collections.defaultdict(collections.Counter)
        previous: dict[tuple[int, str], set[int]] = {}

        for step in steps:
            key = (step.layer, step.group)
            actual = set(step.experts)
            if name == "previous_same_layer":
                predicted = set(previous.get(key, set()))
            elif name == "history_lfu":
                predicted = top_k(history[key], k)
            elif name == "dev_global_lfu":
                predicted = top_k(global_counts[key], k)
            elif name == "hybrid_lfu":
                score = collections.Counter()
                score.update(global_counts[key])
                if history[key]:
                    max_global = max(score.values()) if score else 1
                    max_hist = max(history[key].values())
                    hist_scale = max_global / max_hist if max_hist else 1.0
                    for expert, count in history[key].items():
                        score[expert] += count * hist_scale
                predicted = top_k(score, k)
            else:
                raise ValueError(name)

            hit = actual & predicted
            pred_bytes = predicted_bytes(step.layer, step.group, predicted, exact_catalog, fallback_catalog)
            wasted_bytes = predicted_bytes(step.layer, step.group, predicted - actual, exact_catalog, fallback_catalog)

            row = totals[(prompt_id, step.group)]
            row["steps"] += 1
            row["actual_events"] += len(actual)
            row["actual_bytes"] += step.actual_bytes
            row["pred_events"] += len(predicted)
            row["pred_bytes"] += pred_bytes
            row["hit_events"] += len(hit)
            row["hit_bytes"] += sum(step.experts[expert] for expert in hit)
            row["wasted_bytes"] += wasted_bytes
            row["full_steps"] += 1 if actual and actual <= predicted else 0
            row["empty_predictions"] += 1 if not predicted else 0

            history[key].update(actual)
            previous[key] = actual

    per_prompt = [
        make_row(name, k, prompt_id, group, counts)
        for (prompt_id, group), counts in sorted(totals.items())
    ]
    aggregate_counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for (_, group), counts in totals.items():
        aggregate_counts[group].update(counts)
    aggregate = [
        make_row(name, k, "ALL_DEV", group, counts)
        for group, counts in sorted(aggregate_counts.items())
    ]
    return aggregate, per_prompt


def write_report(out: pathlib.Path, aggregate: list[dict], per_prompt: list[dict], inputs: list[pathlib.Path]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "inputs": [str(path) for path in inputs],
        "aggregate": aggregate,
        "per_prompt": per_prompt,
    }
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Kimi route-detail predictor bound",
        "",
        "This is an offline dev-trace diagnostic. It does not use held-out test prompts.",
        "",
        "## Aggregate",
        "",
        "| predictor | k | group | steps | byte recall | full steps | pred/actual bytes | hit GiB | wasted GiB |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(aggregate, key=lambda r: (r["group"], r["predictor"], r["k"])):
        lines.append(
            f"| {row['predictor']} | {row['k']} | {row['group']} | {row['steps']} | "
            f"{row['byte_recall_pct']:.1f}% | {row['full_step_pct']:.1f}% | "
            f"{row['pred_over_actual_bytes']:.2f}x | {row['hit_gib']:.2f} | "
            f"{row['wasted_gib']:.2f} |"
        )

    lines.extend([
        "",
        "## Per Prompt",
        "",
        "| prompt | predictor | k | group | byte recall | full steps | pred/actual bytes | hit GiB | wasted GiB |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(per_prompt, key=lambda r: (r["prompt_id"], r["group"], r["predictor"], r["k"])):
        lines.append(
            f"| `{row['prompt_id']}` | {row['predictor']} | {row['k']} | {row['group']} | "
            f"{row['byte_recall_pct']:.1f}% | {row['full_step_pct']:.1f}% | "
            f"{row['pred_over_actual_bytes']:.2f}x | {row['hit_gib']:.2f} | "
            f"{row['wasted_gib']:.2f} |"
        )

    lines.extend([
        "",
        "Interpretation gates:",
        "",
        "- Runtime predictor work is justified only if slow dev prompts reach at least 65% byte recall,",
        "  at most 1.35x predicted bytes, and at least 40% full-step coverage for the bottleneck group.",
        "- Otherwise prediction cannot plausibly close the gap from the current ~1.4 tok/s SOTA to 5 tok/s.",
        "",
    ])
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline prompt-agnostic route-detail predictor bound for Kimi dev traces.")
    parser.add_argument("--runs-root", action="append", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--ks", default="8,16,32,64", help="Comma-separated K values for LFU predictors.")
    args = parser.parse_args()

    steps_by_prompt: dict[str, list[Step]] = {}
    inputs: list[pathlib.Path] = []
    for root in args.runs_root:
        refuse_heldout(root)
        for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            prompt_id, _, steps = parse_detail(run_dir)
            if not steps:
                continue
            steps_by_prompt[prompt_id] = steps
            inputs.append(run_dir)

    if len(steps_by_prompt) < 2:
        raise SystemExit(f"need at least two dev prompt traces, found {len(steps_by_prompt)}")

    all_steps = [step for steps in steps_by_prompt.values() for step in steps]
    exact_catalog, fallback_catalog = build_catalog(all_steps)
    ks = [int(part) for part in args.ks.split(",") if part.strip()]

    aggregate: list[dict] = []
    per_prompt: list[dict] = []
    agg, pp = evaluate_predictor("previous_same_layer", steps_by_prompt, 0, exact_catalog, fallback_catalog)
    aggregate.extend(agg)
    per_prompt.extend(pp)
    for k in ks:
        for predictor in ("history_lfu", "dev_global_lfu", "hybrid_lfu"):
            agg, pp = evaluate_predictor(predictor, steps_by_prompt, k, exact_catalog, fallback_catalog)
            aggregate.extend(agg)
            per_prompt.extend(pp)

    write_report(args.out, aggregate, per_prompt, inputs)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
