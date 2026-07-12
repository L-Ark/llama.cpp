#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass


TRACE_RE = re.compile(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight")
ROLES = ("gate", "up", "down")


@dataclass
class Segment:
    layer: int
    role: str
    experts: list[int]
    expert_bytes: int


@dataclass
class Visit:
    layer: int
    role_experts: dict[str, set[int]]
    role_counts: dict[str, int]

    @property
    def experts(self) -> set[int]:
        out: set[int] = set()
        for experts in self.role_experts.values():
            out.update(experts)
        return out

    @property
    def max_role_count(self) -> int:
        return max(self.role_counts.values()) if self.role_counts else 0


@dataclass
class Trace:
    name: str
    path: pathlib.Path
    tokens: list[dict[int, Visit]]
    rows: int
    segments: int
    prompt_like_groups: int
    incomplete_groups: int


@dataclass
class Case:
    layer: int
    target_layer: int
    prefix: tuple[frozenset[int], ...]
    source: frozenset[int]
    target: frozenset[int]


def fmt(value: float) -> str:
    return f"{value:.4f}"


def read_segments(path: pathlib.Path) -> tuple[list[Segment], dict[tuple[int, str], int], int]:
    segments: list[Segment] = []
    bytes_by_layer_role: dict[tuple[int, str], int] = {}
    rows = 0
    current_key: tuple[int, str] | None = None
    current_experts: list[int] = []
    current_bytes = 0

    def flush() -> None:
        nonlocal current_key, current_experts, current_bytes
        if current_key is None:
            return
        layer, role = current_key
        segments.append(Segment(layer, role, current_experts, current_bytes))
        current_key = None
        current_experts = []
        current_bytes = 0

    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            rows += 1
            match = TRACE_RE.search(row.get("tensor", ""))
            if not match:
                continue
            layer = int(match.group(1))
            role = match.group(2)
            key = (layer, role)
            expert_idx = int(row.get("expert_idx") or 0)
            expert_bytes = int(float(row.get("expert_bytes") or 0))
            bytes_by_layer_role[key] = max(bytes_by_layer_role.get(key, 0), expert_bytes)
            if key != current_key:
                flush()
                current_key = key
                current_bytes = expert_bytes
            current_experts.append(expert_idx)
    flush()
    return segments, bytes_by_layer_role, rows


def segments_to_visits(segments: list[Segment]) -> list[Visit]:
    visits: list[Visit] = []
    current_layer: int | None = None
    role_experts: dict[str, set[int]] = {}
    role_counts: dict[str, int] = {}

    def flush() -> None:
        nonlocal current_layer, role_experts, role_counts
        if current_layer is None:
            return
        visits.append(Visit(current_layer, role_experts, role_counts))
        current_layer = None
        role_experts = {}
        role_counts = {}

    for segment in segments:
        if current_layer is None:
            current_layer = segment.layer
        if segment.layer != current_layer or segment.role in role_experts:
            flush()
            current_layer = segment.layer
        role_experts[segment.role] = set(segment.experts)
        role_counts[segment.role] = len(segment.experts)
    flush()
    return visits


def parse_trace(path: pathlib.Path, layers: int, max_decode_experts: int) -> tuple[Trace, dict[tuple[int, str], int]]:
    segments, bytes_by_layer_role, rows = read_segments(path)
    visits = segments_to_visits(segments)
    groups: list[list[Visit]] = []
    current: list[Visit] = []
    prev_layer = 0
    for visit in visits:
        if current and visit.layer <= prev_layer:
            groups.append(current)
            current = []
        current.append(visit)
        prev_layer = visit.layer
    if current:
        groups.append(current)

    tokens: list[dict[int, Visit]] = []
    prompt_like = 0
    incomplete = 0
    for group in groups:
        by_layer = {visit.layer: visit for visit in group}
        complete = len(by_layer) == layers and all(layer in by_layer for layer in range(1, layers + 1))
        max_role_count = max((visit.max_role_count for visit in group), default=0)
        if complete and max_role_count <= max_decode_experts:
            tokens.append(by_layer)
        elif max_role_count > max_decode_experts:
            prompt_like += 1
        else:
            incomplete += 1
    return (
        Trace(path.parent.name, path, tokens, rows, len(segments), prompt_like, incomplete),
        bytes_by_layer_role,
    )


def bundle_bytes(bytes_by_layer_role: dict[tuple[int, str], int], layer: int, bundle: str) -> int:
    if bundle == "all":
        roles = ROLES
    elif bundle == "upgate":
        roles = ("up", "gate")
    elif bundle == "down":
        roles = ("down",)
    else:
        raise ValueError(bundle)
    return sum(bytes_by_layer_role.get((layer, role), 0) for role in roles)


def make_prefix(token: dict[int, Visit], layer: int, width: int) -> tuple[frozenset[int], ...]:
    start = max(1, layer - width + 1)
    return tuple(frozenset(token[pos].experts) for pos in range(start, layer + 1))


def build_cases(trace: Trace, horizon: int, width: int, layers: int) -> list[Case]:
    cases: list[Case] = []
    for token in trace.tokens:
        for layer in range(1, layers - horizon + 1):
            target_layer = layer + horizon
            cases.append(
                Case(
                    layer=layer,
                    target_layer=target_layer,
                    prefix=make_prefix(token, layer, width),
                    source=frozenset(token[layer].experts),
                    target=frozenset(token[target_layer].experts),
                )
            )
    return cases


def jaccard(a: frozenset[int], b: frozenset[int]) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


def prefix_similarity(a: tuple[frozenset[int], ...], b: tuple[frozenset[int], ...]) -> float:
    width = min(len(a), len(b))
    if width == 0:
        return 0.0
    score = 0.0
    denom = 0.0
    for idx in range(1, width + 1):
        # Most recent layer receives the largest weight.
        weight = float(idx)
        score += weight * jaccard(a[-idx], b[-idx])
        denom += weight
    return score / denom if denom else 0.0


def select_neighbors(
    query: Case,
    train_by_target_layer: dict[int, list[Case]],
    neighbors: int,
    min_similarity: float,
) -> list[tuple[float, Case]]:
    scored: list[tuple[float, Case]] = []
    for candidate in train_by_target_layer.get(query.target_layer, []):
        sim = prefix_similarity(query.prefix, candidate.prefix)
        if sim >= min_similarity:
            scored.append((sim, candidate))
    scored.sort(key=lambda item: (-item[0], len(item[1].target), item[1].layer))
    return scored[:neighbors]


def top_k(counter: Counter[int], budget: int) -> set[int]:
    if budget <= 0:
        return set()
    return {expert for expert, _ in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:budget]}


def predict_neighbors(
    query: Case,
    train_by_target_layer: dict[int, list[Case]],
    global_by_target_layer: dict[int, Counter[int]],
    budget: int,
    neighbors: int,
    min_similarity: float,
    fallback_weight: float,
) -> set[int]:
    scores: Counter[int] = Counter()
    for sim, case in select_neighbors(query, train_by_target_layer, neighbors, min_similarity):
        weight = max(sim, 0.001)
        for expert in case.target:
            scores[expert] += weight
    if fallback_weight > 0:
        for expert, count in global_by_target_layer.get(query.target_layer, Counter()).items():
            scores[expert] += fallback_weight * math.log1p(count)
    return top_k(scores, budget)


def evaluate(
    traces: list[Trace],
    bytes_by_layer_role: dict[tuple[int, str], int],
    horizons: list[int],
    widths: list[int],
    budgets: list[int],
    neighbors_values: list[int],
    min_similarity_values: list[float],
    bundles: list[str],
    layers: int,
    fallback_weight: float,
) -> tuple[list[dict], list[dict]]:
    aggregate_rows: list[dict] = []
    prompt_rows: list[dict] = []

    for horizon in horizons:
        for width in widths:
            per_prompt_cases = {
                trace.name: build_cases(trace, horizon, width, layers)
                for trace in traces
            }
            for budget in budgets:
                for neighbors in neighbors_values:
                    for min_similarity in min_similarity_values:
                        totals_by_bundle = {bundle: Counter() for bundle in bundles}
                        totals_by_prompt_bundle: dict[tuple[str, str], Counter] = defaultdict(Counter)
                        for heldout in traces:
                            train_cases: list[Case] = []
                            for other in traces:
                                if other.name != heldout.name:
                                    train_cases.extend(per_prompt_cases[other.name])
                            train_by_target_layer: dict[int, list[Case]] = defaultdict(list)
                            global_by_target_layer: dict[int, Counter[int]] = defaultdict(Counter)
                            for case in train_cases:
                                train_by_target_layer[case.target_layer].append(case)
                                global_by_target_layer[case.target_layer].update(case.target)

                            for query in per_prompt_cases[heldout.name]:
                                predicted = predict_neighbors(
                                    query,
                                    train_by_target_layer,
                                    global_by_target_layer,
                                    budget,
                                    neighbors,
                                    min_similarity,
                                    fallback_weight,
                                )
                                actual = set(query.target)
                                hit = predicted & actual
                                for bundle in bundles:
                                    per_expert_bytes = bundle_bytes(bytes_by_layer_role, query.target_layer, bundle)
                                    actual_bytes = len(actual) * per_expert_bytes
                                    pred_bytes = len(predicted) * per_expert_bytes
                                    hit_bytes = len(hit) * per_expert_bytes
                                    wasted_bytes = len(predicted - actual) * per_expert_bytes
                                    for counter in (
                                        totals_by_bundle[bundle],
                                        totals_by_prompt_bundle[(heldout.name, bundle)],
                                    ):
                                        counter["steps"] += 1
                                        counter["actual_events"] += len(actual)
                                        counter["pred_events"] += len(predicted)
                                        counter["hit_events"] += len(hit)
                                        counter["actual_bytes"] += actual_bytes
                                        counter["pred_bytes"] += pred_bytes
                                        counter["hit_bytes"] += hit_bytes
                                        counter["wasted_bytes"] += wasted_bytes
                                        counter["full_steps"] += 1 if actual and actual <= predicted else 0
                                        counter["empty_predictions"] += 1 if not predicted else 0

                        for bundle, counts in totals_by_bundle.items():
                            aggregate_rows.append(make_row(
                                prompt_id="ALL_DEV",
                                bundle=bundle,
                                horizon=horizon,
                                width=width,
                                budget=budget,
                                neighbors=neighbors,
                                min_similarity=min_similarity,
                                counts=counts,
                            ))
                        for (prompt_id, bundle), counts in sorted(totals_by_prompt_bundle.items()):
                            prompt_rows.append(make_row(
                                prompt_id=prompt_id,
                                bundle=bundle,
                                horizon=horizon,
                                width=width,
                                budget=budget,
                                neighbors=neighbors,
                                min_similarity=min_similarity,
                                counts=counts,
                            ))
    return aggregate_rows, prompt_rows


def pct(num: float, den: float) -> float:
    return 100.0 * num / den if den else 0.0


def gib(nbytes: float) -> float:
    return nbytes / 1024**3


def make_row(
    prompt_id: str,
    bundle: str,
    horizon: int,
    width: int,
    budget: int,
    neighbors: int,
    min_similarity: float,
    counts: Counter,
) -> dict:
    actual_bytes = counts["actual_bytes"]
    pred_bytes = counts["pred_bytes"]
    return {
        "predictor": "prefix_knn",
        "prompt_id": prompt_id,
        "bundle": bundle,
        "horizon": horizon,
        "width": width,
        "budget": budget,
        "neighbors": neighbors,
        "min_similarity": min_similarity,
        "steps": int(counts["steps"]),
        "actual_events": int(counts["actual_events"]),
        "pred_events": int(counts["pred_events"]),
        "event_recall_pct": pct(counts["hit_events"], counts["actual_events"]),
        "byte_recall_pct": pct(counts["hit_bytes"], actual_bytes),
        "full_step_pct": pct(counts["full_steps"], counts["steps"]),
        "pred_over_actual_bytes": pred_bytes / actual_bytes if actual_bytes else 0.0,
        "actual_gib": gib(actual_bytes),
        "pred_gib": gib(pred_bytes),
        "hit_gib": gib(counts["hit_bytes"]),
        "wasted_gib": gib(counts["wasted_bytes"]),
        "empty_prediction_pct": pct(counts["empty_predictions"], counts["steps"]),
    }


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    out: pathlib.Path,
    aggregate_rows: list[dict],
    prompt_rows: list[dict],
    trace_paths: list[pathlib.Path],
    gates: dict[str, float],
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(
        json.dumps(
            {
                "inputs": [str(path) for path in trace_paths],
                "gates": gates,
                "aggregate": aggregate_rows,
                "per_prompt": prompt_rows,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    write_csv(out.with_suffix(".aggregate.csv"), aggregate_rows)
    write_csv(out.with_suffix(".per_prompt.csv"), prompt_rows)

    passing = [
        row for row in aggregate_rows
        if row["bundle"] == "all"
        and row["byte_recall_pct"] >= gates["byte_recall_pct"]
        and row["pred_over_actual_bytes"] <= gates["pred_over_actual_bytes"]
        and row["full_step_pct"] >= gates["full_step_pct"]
    ]
    best_all = sorted(
        [row for row in aggregate_rows if row["bundle"] == "all"],
        key=lambda row: (
            -row["byte_recall_pct"],
            row["pred_over_actual_bytes"],
            -row["full_step_pct"],
        ),
    )[:12]
    efficient = sorted(
        [row for row in aggregate_rows if row["bundle"] == "all" and row["pred_over_actual_bytes"] <= gates["pred_over_actual_bytes"]],
        key=lambda row: (-row["byte_recall_pct"], -row["full_step_pct"]),
    )[:12]

    lines = [
        "# Kimi Prefix-Neighbor Future Predictor Admission",
        "",
        "This is an offline dev-trace diagnostic. It uses leave-one-dev-prompt-out",
        "evaluation and does not inspect held-out/test prompts.",
        "",
        "## Method",
        "",
        "- For each token and layer, build a prefix signature from the active expert",
        "  sets in the most recent already-computed layers.",
        "- For a target future layer, search the other dev prompts for nearest prefix",
        "  signatures using weighted Jaccard similarity.",
        "- Vote over the neighbor target experts, with a small global-hot fallback,",
        "  then prefetch a bounded budget of experts.",
        "- This is stronger than route LFU/co-occurrence, but still runtime-feasible",
        "  only after the current layer routing is known.",
        "",
        "## Admission Gate",
        "",
        f"- all-role byte recall >= `{gates['byte_recall_pct']:.1f}%`",
        f"- predicted/actual bytes <= `{gates['pred_over_actual_bytes']:.2f}x`",
        f"- full-step coverage >= `{gates['full_step_pct']:.1f}%`",
        "",
        f"Passing all-role rows: `{len(passing)}`",
        "",
        "## Best All-Role Rows By Recall",
        "",
        "| horizon | width | budget | neighbors | min sim | byte recall | full steps | pred/actual | pred GiB | hit GiB | wasted GiB |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best_all:
        lines.append(
            f"| {row['horizon']} | {row['width']} | {row['budget']} | {row['neighbors']} | "
            f"{row['min_similarity']:.2f} | {row['byte_recall_pct']:.2f}% | "
            f"{row['full_step_pct']:.2f}% | {row['pred_over_actual_bytes']:.2f}x | "
            f"{row['pred_gib']:.2f} | {row['hit_gib']:.2f} | {row['wasted_gib']:.2f} |"
        )
    lines.extend([
        "",
        "## Best Rows Under Byte-Budget Gate",
        "",
        "| horizon | width | budget | neighbors | min sim | byte recall | full steps | pred/actual | pred GiB | hit GiB | wasted GiB |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in efficient:
        lines.append(
            f"| {row['horizon']} | {row['width']} | {row['budget']} | {row['neighbors']} | "
            f"{row['min_similarity']:.2f} | {row['byte_recall_pct']:.2f}% | "
            f"{row['full_step_pct']:.2f}% | {row['pred_over_actual_bytes']:.2f}x | "
            f"{row['pred_gib']:.2f} | {row['hit_gib']:.2f} | {row['wasted_gib']:.2f} |"
        )
    lines.extend([
        "",
        "## Decision",
        "",
    ])
    if passing:
        lines.extend([
            "- At least one prefix-neighbor row passes the offline admission gate.",
            "- Before runtime implementation, freeze one row, compute projected endpoint",
            "  saving against the current N32 control, and then validate on held-out",
            "  prompts exactly once.",
        ])
    else:
        lines.extend([
            "- No prefix-neighbor row passes the offline admission gate.",
            "- This rejects route-ID prefix similarity as a standalone future-prefetch",
            "  signal. A runtime prefetch prototype still needs a stronger signal,",
            "  such as router logits, hidden-state features, or a draft/router model.",
            "- The fallback path remains lower-byte expert representation or a storage",
            "  format that reduces bytes rather than trying more route-only policies.",
        ])
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline prefix-neighbor future-expert predictor admission.")
    parser.add_argument("--runs-root", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--layers", type=int, default=60)
    parser.add_argument("--max-decode-experts", type=int, default=8)
    parser.add_argument("--horizons", default="1,2,3")
    parser.add_argument("--widths", default="1,2,3")
    parser.add_argument("--budgets", default="32")
    parser.add_argument("--neighbors", default="9")
    parser.add_argument("--min-similarities", default="0.0")
    parser.add_argument("--bundles", default="all,upgate,down")
    parser.add_argument("--fallback-weight", type=float, default=0.05)
    parser.add_argument("--gate-byte-recall-pct", type=float, default=65.0)
    parser.add_argument("--gate-pred-over-actual", type=float, default=1.35)
    parser.add_argument("--gate-full-step-pct", type=float, default=40.0)
    args = parser.parse_args()

    trace_paths = sorted(args.runs_root.glob("dev_*/route-trace.csv"))
    if not trace_paths:
        raise SystemExit(f"no dev route traces under {args.runs_root}")

    traces: list[Trace] = []
    bytes_by_layer_role: dict[tuple[int, str], int] = {}
    for path in trace_paths:
        trace, per_trace_bytes = parse_trace(path, args.layers, args.max_decode_experts)
        traces.append(trace)
        for key, value in per_trace_bytes.items():
            bytes_by_layer_role[key] = max(bytes_by_layer_role.get(key, 0), value)

    horizons = [int(value) for value in args.horizons.split(",") if value]
    widths = [int(value) for value in args.widths.split(",") if value]
    budgets = [int(value) for value in args.budgets.split(",") if value]
    neighbors_values = [int(value) for value in args.neighbors.split(",") if value]
    min_similarity_values = [float(value) for value in args.min_similarities.split(",") if value]
    bundles = [value for value in args.bundles.split(",") if value]
    aggregate_rows, prompt_rows = evaluate(
        traces,
        bytes_by_layer_role,
        horizons,
        widths,
        budgets,
        neighbors_values,
        min_similarity_values,
        bundles,
        args.layers,
        args.fallback_weight,
    )
    gates = {
        "byte_recall_pct": args.gate_byte_recall_pct,
        "pred_over_actual_bytes": args.gate_pred_over_actual,
        "full_step_pct": args.gate_full_step_pct,
    }
    write_report(args.out, aggregate_rows, prompt_rows, trace_paths, gates)
    passing = [
        row for row in aggregate_rows
        if row["bundle"] == "all"
        and row["byte_recall_pct"] >= gates["byte_recall_pct"]
        and row["pred_over_actual_bytes"] <= gates["pred_over_actual_bytes"]
        and row["full_step_pct"] >= gates["full_step_pct"]
    ]
    print(f"traces={len(traces)}")
    print(f"aggregate_rows={len(aggregate_rows)}")
    print(f"passing_all_role_rows={len(passing)}")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
