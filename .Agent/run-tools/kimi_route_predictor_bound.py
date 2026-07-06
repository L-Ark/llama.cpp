#!/usr/bin/env python3
import argparse
import collections
import csv
import json
import pathlib
import re
from dataclasses import dataclass


ROLE_RE = re.compile(r"blk\.(\d+)\.ffn_(up|gate|down)_exps\.weight")


@dataclass
class Step:
    prompt_id: str
    token_idx: int
    layer: int
    group: str
    experts: dict

    @property
    def actual_bytes(self) -> int:
        return sum(self.experts.values())


def percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def role_group(role):
    return "down" if role == "down" else "upgate"


def load_metrics(run_dir):
    path = run_dir / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_trace(run_dir):
    metrics = load_metrics(run_dir)
    prompt_id = metrics.get("prompt_id", run_dir.name)
    path = run_dir / "route-trace.csv"
    if not path.exists():
        return prompt_id, metrics, []

    raw_groups = []
    current_key = None
    current = None
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            match = ROLE_RE.search(row["tensor"])
            if not match:
                continue
            layer = int(match.group(1))
            group = role_group(match.group(2))
            key = (layer, group)
            if key != current_key:
                if current is not None:
                    raw_groups.append(current)
                current_key = key
                current = {
                    "layer": layer,
                    "group": group,
                    "experts": collections.defaultdict(int),
                }
            expert = int(row["expert_idx"])
            current["experts"][expert] += int(row["expert_bytes"])
    if current is not None:
        raw_groups.append(current)

    seen = collections.Counter()
    steps = []
    for group in raw_groups:
        key = (group["layer"], group["group"])
        token_idx = seen[key]
        seen[key] += 1
        steps.append(Step(
            prompt_id=prompt_id,
            token_idx=token_idx,
            layer=group["layer"],
            group=group["group"],
            experts=dict(group["experts"]),
        ))
    return prompt_id, metrics, steps


def build_catalog(all_steps):
    samples = collections.defaultdict(list)
    for step in all_steps:
        for expert, nbytes in step.experts.items():
            samples[(step.layer, step.group, expert)].append(nbytes)

    exact = {}
    fallback = collections.defaultdict(list)
    for key, vals in samples.items():
        exact[key] = int(percentile(vals, 0.5))
        fallback[(key[0], key[1])].extend(vals)

    fallback_median = {}
    for key, vals in fallback.items():
        fallback_median[key] = int(percentile(vals, 0.5))
    return exact, fallback_median


def predicted_bytes(layer, group, experts, exact_catalog, fallback_catalog):
    fallback = fallback_catalog.get((layer, group), 0)
    total = 0
    for expert in experts:
        total += exact_catalog.get((layer, group, expert), fallback)
    return total


def top_k(counter, k):
    if k <= 0:
        return set()
    return {expert for expert, _ in counter.most_common(k)}


def leave_one_out_global(steps_by_prompt, prompt_id):
    out = collections.defaultdict(collections.Counter)
    for other_prompt, steps in steps_by_prompt.items():
        if other_prompt == prompt_id:
            continue
        for step in steps:
            out[(step.layer, step.group)].update(step.experts.keys())
    return out


def evaluate_predictor(name, steps_by_prompt, k, exact_catalog, fallback_catalog):
    totals = collections.defaultdict(lambda: collections.Counter())
    per_prompt = []

    for prompt_id, steps in steps_by_prompt.items():
        global_counts = leave_one_out_global(steps_by_prompt, prompt_id)
        history = collections.defaultdict(collections.Counter)
        previous = {}

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
            actual_bytes = step.actual_bytes
            hit_bytes = sum(step.experts[expert] for expert in hit)
            pred_bytes = predicted_bytes(
                step.layer, step.group, predicted, exact_catalog, fallback_catalog
            )
            wasted_bytes = predicted_bytes(
                step.layer, step.group, predicted - actual, exact_catalog, fallback_catalog
            )

            row = totals[(prompt_id, step.group)]
            row["steps"] += 1
            row["actual_events"] += len(actual)
            row["actual_bytes"] += actual_bytes
            row["pred_events"] += len(predicted)
            row["pred_bytes"] += pred_bytes
            row["hit_events"] += len(hit)
            row["hit_bytes"] += hit_bytes
            row["wasted_bytes"] += wasted_bytes
            row["full_steps"] += 1 if actual and actual <= predicted else 0
            row["empty_predictions"] += 1 if not predicted else 0

            history[key].update(actual)
            previous[key] = actual

    for (prompt_id, group), counts in sorted(totals.items()):
        per_prompt.append(make_row(name, k, prompt_id, group, counts))

    aggregate_counts = collections.defaultdict(collections.Counter)
    for (_, group), counts in totals.items():
        aggregate_counts[group].update(counts)
    aggregate = [
        make_row(name, k, "ALL_DEV", group, counts)
        for group, counts in sorted(aggregate_counts.items())
    ]
    return aggregate, per_prompt


def pct(num, den):
    return 100.0 * num / den if den else 0.0


def gib(nbytes):
    return nbytes / 1024**3


def make_row(name, k, prompt_id, group, counts):
    return {
        "predictor": name,
        "k": k,
        "prompt_id": prompt_id,
        "group": group,
        "steps": int(counts["steps"]),
        "actual_events": int(counts["actual_events"]),
        "pred_events": int(counts["pred_events"]),
        "event_recall_pct": pct(counts["hit_events"], counts["actual_events"]),
        "byte_recall_pct": pct(counts["hit_bytes"], counts["actual_bytes"]),
        "full_step_pct": pct(counts["full_steps"], counts["steps"]),
        "pred_over_actual_bytes": counts["pred_bytes"] / counts["actual_bytes"] if counts["actual_bytes"] else 0.0,
        "actual_gib": gib(counts["actual_bytes"]),
        "pred_gib": gib(counts["pred_bytes"]),
        "hit_gib": gib(counts["hit_bytes"]),
        "wasted_gib": gib(counts["wasted_bytes"]),
        "empty_prediction_pct": pct(counts["empty_predictions"], counts["steps"]),
    }


def write_report(out, aggregate, per_prompt, inputs):
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "inputs": [str(p) for p in inputs],
        "aggregate": aggregate,
        "per_prompt": per_prompt,
    }
    out.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Kimi route predictor bound",
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
    out.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Offline prompt-agnostic route predictor bound for Kimi dev traces.")
    parser.add_argument("--runs-root", action="append", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--ks", default="8,16,32", help="Comma-separated K values for LFU predictors.")
    args = parser.parse_args()

    steps_by_prompt = {}
    metrics_by_prompt = {}
    inputs = []
    for root in args.runs_root:
        for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            prompt_id, metrics, steps = parse_trace(run_dir)
            if not steps:
                continue
            steps_by_prompt[prompt_id] = steps
            metrics_by_prompt[prompt_id] = metrics
            inputs.append(run_dir)

    if len(steps_by_prompt) < 2:
        raise SystemExit(f"need at least two dev prompt traces, found {len(steps_by_prompt)}")

    exact_catalog, fallback_catalog = build_catalog(
        [step for steps in steps_by_prompt.values() for step in steps]
    )

    ks = [int(part) for part in args.ks.split(",") if part.strip()]
    aggregate = []
    per_prompt = []

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


if __name__ == "__main__":
    raise SystemExit(main())
