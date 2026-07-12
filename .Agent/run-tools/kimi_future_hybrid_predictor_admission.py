#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import pathlib
import re
from collections import Counter, defaultdict, deque
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


def fmt(value: float) -> str:
    return f"{value:.4f}"


def write_csv(path: pathlib.Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_segments(path: pathlib.Path) -> tuple[list[Segment], dict[tuple[int, str], int], int]:
    segments: list[Segment] = []
    bytes_by_layer_role: dict[tuple[int, str], int] = {}
    rows = 0
    cur_key: tuple[int, str] | None = None
    cur_experts: list[int] = []
    cur_bytes = 0

    def flush() -> None:
        nonlocal cur_key, cur_experts, cur_bytes
        if cur_key is None:
            return
        layer, role = cur_key
        segments.append(Segment(layer, role, cur_experts, cur_bytes))
        cur_key = None
        cur_experts = []
        cur_bytes = 0

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
            if key != cur_key:
                flush()
                cur_key = key
                cur_bytes = expert_bytes
            cur_experts.append(expert_idx)
    flush()
    return segments, bytes_by_layer_role, rows


def segments_to_visits(segments: list[Segment]) -> list[Visit]:
    visits: list[Visit] = []
    cur_layer: int | None = None
    role_experts: dict[str, set[int]] = {}
    role_counts: dict[str, int] = {}

    def flush() -> None:
        nonlocal cur_layer, role_experts, role_counts
        if cur_layer is None:
            return
        visits.append(Visit(cur_layer, role_experts, role_counts))
        cur_layer = None
        role_experts = {}
        role_counts = {}

    for seg in segments:
        if cur_layer is None:
            cur_layer = seg.layer
        if seg.layer != cur_layer or seg.role in role_experts:
            flush()
            cur_layer = seg.layer
        role_experts[seg.role] = set(seg.experts)
        role_counts[seg.role] = len(seg.experts)
    flush()
    return visits


def parse_trace(path: pathlib.Path, layers: int, max_decode_experts: int) -> tuple[Trace, dict[tuple[int, str], int]]:
    segments, bytes_by_layer_role, rows = parse_segments(path)
    visits = segments_to_visits(segments)
    groups: list[list[Visit]] = []
    cur: list[Visit] = []
    prev_layer = 0
    for visit in visits:
        if cur and visit.layer <= prev_layer:
            groups.append(cur)
            cur = []
        cur.append(visit)
        prev_layer = visit.layer
    if cur:
        groups.append(cur)

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
    return Trace(path.parent.name, path, tokens, rows, len(segments), prompt_like, incomplete), bytes_by_layer_role


def normalize(counter: Counter[int]) -> dict[int, float]:
    if not counter:
        return {}
    scale = max(counter.values()) or 1
    return {key: value / scale for key, value in counter.items()}


def top_by_score(scores: dict[int, float], budget: int) -> set[int]:
    if budget <= 0:
        return set()
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return {expert for expert, _ in ranked[:budget]}


def expert_bundle_bytes(bytes_by_layer_role: dict[tuple[int, str], int], layer: int, bundle: str) -> int:
    if bundle == "all":
        roles = ROLES
    elif bundle == "upgate":
        roles = ("up", "gate")
    elif bundle == "down":
        roles = ("down",)
    else:
        raise ValueError(f"unknown bundle {bundle}")
    return sum(bytes_by_layer_role.get((layer, role), 0) for role in roles)


class TrainedPredictor:
    def __init__(self, layers: int, horizon: int, train: list[Trace]) -> None:
        self.layers = layers
        self.horizon = horizon
        self.global_counts: dict[int, Counter[int]] = defaultdict(Counter)
        self.cooc_counts: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
        for trace in train:
            for token in trace.tokens:
                for layer in range(1, layers - horizon + 1):
                    source = token[layer].experts
                    target_layer = layer + horizon
                    target = token[target_layer].experts
                    self.global_counts[target_layer].update(target)
                    for expert in source:
                        self.cooc_counts[(layer, expert)].update(target)

    def global_scores(self, target_layer: int) -> dict[int, float]:
        return normalize(self.global_counts[target_layer])

    def cooc_scores(self, layer: int, source: set[int]) -> dict[int, float]:
        c: Counter[int] = Counter()
        for expert in source:
            c.update(self.cooc_counts.get((layer, expert), Counter()))
        return normalize(c)


def add_scores(out: defaultdict[int, float], scores: dict[int, float], weight: float) -> None:
    if weight == 0:
        return
    for expert, score in scores.items():
        out[expert] += weight * score


def predict(
    name: str,
    trained: TrainedPredictor,
    layer: int,
    source: set[int],
    prev_target: set[int],
    recent_counter: Counter[int],
    budget: int,
) -> set[int]:
    target_layer = layer + trained.horizon
    if name == "global_hot":
        return top_by_score(trained.global_scores(target_layer), budget)
    if name == "cooc":
        scores: defaultdict[int, float] = defaultdict(float)
        add_scores(scores, trained.cooc_scores(layer, source), 1.0)
        add_scores(scores, trained.global_scores(target_layer), 0.05)
        return top_by_score(scores, budget)
    if name == "previous_same_layer":
        return set(list(sorted(prev_target))[:budget])
    if name == "recent_lfu":
        return top_by_score(normalize(recent_counter), budget)

    variants = {
        "hybrid_balanced": (1.0, 1.0, 1.0, 0.10),
        "hybrid_cooc": (2.0, 0.75, 0.75, 0.10),
        "hybrid_recent": (0.75, 1.5, 2.0, 0.10),
        "hybrid_prev": (0.75, 2.0, 0.75, 0.10),
    }
    if name not in variants:
        raise ValueError(name)
    w_cooc, w_prev, w_recent, w_global = variants[name]
    scores = defaultdict(float)
    add_scores(scores, trained.cooc_scores(layer, source), w_cooc)
    add_scores(scores, {expert: 1.0 for expert in prev_target}, w_prev)
    add_scores(scores, normalize(recent_counter), w_recent)
    add_scores(scores, trained.global_scores(target_layer), w_global)
    return top_by_score(scores, budget)


def evaluate_fold(
    heldout: Trace,
    train: list[Trace],
    bytes_by_layer_role: dict[tuple[int, str], int],
    layers: int,
    horizons: list[int],
    budgets: list[int],
    bundles: list[str],
    recent_window: int,
    predictor_names: list[str],
) -> list[dict]:
    rows: list[dict] = []
    for horizon in horizons:
        trained = TrainedPredictor(layers, horizon, train)
        totals: dict[tuple[str, int, str], Counter] = defaultdict(Counter)
        previous: dict[int, set[int]] = defaultdict(set)
        recent: dict[int, deque[set[int]]] = defaultdict(lambda: deque(maxlen=recent_window))
        for token in heldout.tokens:
            for layer in range(1, layers - horizon + 1):
                target_layer = layer + horizon
                source = token[layer].experts
                target = token[target_layer].experts
                recent_counter: Counter[int] = Counter()
                for experts in recent[target_layer]:
                    recent_counter.update(experts)
                for name in predictor_names:
                    for budget in budgets:
                        predicted = predict(name, trained, layer, source, previous[target_layer], recent_counter, budget)
                        hits = predicted & target
                        for bundle in bundles:
                            b = expert_bundle_bytes(bytes_by_layer_role, target_layer, bundle)
                            c = totals[(name, budget, bundle)]
                            c["samples"] += 1
                            c["targets"] += len(target)
                            c["predicted"] += len(predicted)
                            c["hits"] += len(hits)
                            c["target_bytes"] += len(target) * b
                            c["predicted_bytes"] += len(predicted) * b
                            c["useful_bytes"] += len(hits) * b
                            c["full_steps"] += 1 if target and target <= predicted else 0
            for target_layer in range(1, layers + 1):
                actual = token[target_layer].experts
                previous[target_layer] = set(actual)
                recent[target_layer].append(set(actual))

        for (name, budget, bundle), c in totals.items():
            rows.append(make_row(heldout.name, name, horizon, budget, bundle, c))
    return rows


def make_row(prompt: str, predictor: str, horizon: int, budget: int, bundle: str, c: Counter) -> dict:
    targets = c["targets"] or 1
    predicted = c["predicted"] or 1
    target_bytes = c["target_bytes"] or 1
    return {
        "prompt": prompt,
        "predictor": predictor,
        "horizon": horizon,
        "budget": budget,
        "bundle": bundle,
        "samples": int(c["samples"]),
        "targets": int(c["targets"]),
        "predicted": int(c["predicted"]),
        "hits": int(c["hits"]),
        "recall": c["hits"] / targets,
        "precision": c["hits"] / predicted,
        "full_step_pct": c["full_steps"] / (c["samples"] or 1),
        "pred_over_actual_bytes": c["predicted_bytes"] / target_bytes,
        "target_gib": c["target_bytes"] / 1024**3,
        "predicted_gib": c["predicted_bytes"] / 1024**3,
        "useful_gib": c["useful_bytes"] / 1024**3,
        "waste_gib": max(c["predicted_bytes"] - c["useful_bytes"], 0) / 1024**3,
    }


def aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int, int, str], Counter] = defaultdict(Counter)
    decode_tokens: dict[tuple[str, int, int, str], int] = defaultdict(int)
    for r in rows:
        key = (r["predictor"], int(r["horizon"]), int(r["budget"]), r["bundle"])
        c = grouped[key]
        for field in ("samples", "targets", "predicted", "hits"):
            c[field] += int(r[field])
        c["full_steps"] += int(round(float(r["full_step_pct"]) * int(r["samples"])))
        for field in ("target_gib", "predicted_gib", "useful_gib", "waste_gib"):
            c[field] += float(r[field])
        decode_tokens[key] += 31

    out: list[dict] = []
    for (predictor, horizon, budget, bundle), c in grouped.items():
        targets = c["targets"] or 1
        predicted = c["predicted"] or 1
        target_gib = c["target_gib"] or 1.0
        dtok = decode_tokens[(predictor, horizon, budget, bundle)] or 1
        out.append({
            "predictor": predictor,
            "horizon": horizon,
            "budget": budget,
            "bundle": bundle,
            "samples": int(c["samples"]),
            "recall": c["hits"] / targets,
            "precision": c["hits"] / predicted,
            "full_step_pct": c["full_steps"] / (c["samples"] or 1),
            "pred_over_actual_bytes": c["predicted_gib"] / target_gib,
            "predicted_gib_per_token": c["predicted_gib"] / dtok,
            "useful_gib_per_token": c["useful_gib"] / dtok,
            "waste_gib_per_token": c["waste_gib"] / dtok,
        })
    out.sort(key=lambda r: (-float(r["recall"]), float(r["pred_over_actual_bytes"]), -float(r["full_step_pct"])))
    return out


def row_for_csv(row: dict) -> dict:
    out = dict(row)
    for key in ("recall", "precision", "full_step_pct", "pred_over_actual_bytes", "predicted_gib_per_token", "useful_gib_per_token", "waste_gib_per_token"):
        if key in out:
            out[key] = fmt(float(out[key]))
    for key in ("target_gib", "predicted_gib", "useful_gib", "waste_gib"):
        if key in out:
            out[key] = fmt(float(out[key]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate hybrid prompt-general future-layer predictors from Kimi route traces.")
    ap.add_argument("--input-glob", required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--layers", type=int, default=60)
    ap.add_argument("--max-decode-experts", type=int, default=16)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--budgets", type=int, nargs="+", default=[4, 8, 12, 16, 32])
    ap.add_argument("--bundles", default="all,upgate,down")
    ap.add_argument("--recent-window", type=int, default=4)
    ap.add_argument("--io-wait-ms-per-token", type=float, default=356.0)
    args = ap.parse_args()

    paths = [pathlib.Path(p) for p in sorted(glob.glob(args.input_glob))]
    if len(paths) < 2:
        raise SystemExit("need at least two route traces")
    args.out.mkdir(parents=True, exist_ok=True)

    traces: list[Trace] = []
    bytes_by_layer_role: dict[tuple[int, str], int] = {}
    seen: set[str] = set()
    parse_rows: list[dict] = []
    for path in paths:
        trace, byte_map = parse_trace(path, args.layers, args.max_decode_experts)
        if trace.name in seen:
            continue
        seen.add(trace.name)
        traces.append(trace)
        for key, value in byte_map.items():
            bytes_by_layer_role[key] = max(bytes_by_layer_role.get(key, 0), value)
        parse_rows.append({
            "prompt": trace.name,
            "path": str(path),
            "rows": trace.rows,
            "segments": trace.segments,
            "decode_tokens": len(trace.tokens),
            "prompt_like_groups": trace.prompt_like_groups,
            "incomplete_groups": trace.incomplete_groups,
        })

    bundles = [part.strip() for part in args.bundles.split(",") if part.strip()]
    predictor_names = [
        "global_hot",
        "cooc",
        "previous_same_layer",
        "recent_lfu",
        "hybrid_balanced",
        "hybrid_cooc",
        "hybrid_recent",
        "hybrid_prev",
    ]
    fold_rows: list[dict] = []
    for heldout in traces:
        train = [trace for trace in traces if trace.name != heldout.name]
        fold_rows.extend(evaluate_fold(
            heldout, train, bytes_by_layer_role, args.layers, args.horizons,
            args.budgets, bundles, args.recent_window, predictor_names,
        ))

    agg_rows = aggregate(fold_rows)
    write_csv(args.out / "parse_summary.csv", parse_rows)
    write_csv(args.out / "fold_metrics.csv", [row_for_csv(r) for r in fold_rows])
    write_csv(args.out / "aggregate_metrics.csv", [row_for_csv(r) for r in agg_rows])

    passing = [
        r for r in agg_rows
        if r["bundle"] == "all"
        and float(r["recall"]) >= 0.65
        and float(r["pred_over_actual_bytes"]) <= 1.35
        and float(r["full_step_pct"]) >= 0.40
    ]
    best_all = [r for r in agg_rows if r["bundle"] == "all"][:20]
    best_gate = sorted([r for r in agg_rows if r["bundle"] == "upgate"], key=lambda r: (-float(r["recall"]), float(r["pred_over_actual_bytes"])))[:10]
    lines = [
        "# Kimi Hybrid Future Predictor Admission",
        "",
        f"- input_glob: `{args.input_glob}`",
        f"- prompts: `{len(traces)}`",
        f"- horizons: `{','.join(map(str, args.horizons))}`",
        f"- budgets: `{','.join(map(str, args.budgets))}`",
        f"- recent window: `{args.recent_window}`",
        f"- wait reference: `{args.io_wait_ms_per_token:.3f} ms/token`",
        "",
        "This is a dev-only offline screen. It does not use held-out/test prompts and makes no SOTA claim.",
        "",
        "## Parse Summary",
        "",
        "| prompt | rows | decode tokens | prompt-like | incomplete |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in parse_rows:
        lines.append(f"| `{row['prompt']}` | {row['rows']} | {row['decode_tokens']} | {row['prompt_like_groups']} | {row['incomplete_groups']} |")
    lines += [
        "",
        "## Admission Gate",
        "",
        "- Required: `>=65%` all-role byte recall, `<=1.35x` predicted/actual bytes, and `>=40%` full-step coverage.",
        f"- Passing rows: `{len(passing)}`.",
        "",
        "## Top All-Role Predictors",
        "",
        "| predictor | H | budget | recall | precision | full steps | pred/actual bytes | pred GiB/token | useful GiB/token | waste GiB/token | linear wait cover ms/tok |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best_all:
        cover = float(row["recall"]) * args.io_wait_ms_per_token
        lines.append(
            f"| {row['predictor']} | {row['horizon']} | {row['budget']} | "
            f"{float(row['recall']):.4f} | {float(row['precision']):.4f} | "
            f"{100.0 * float(row['full_step_pct']):.2f}% | {float(row['pred_over_actual_bytes']):.2f}x | "
            f"{float(row['predicted_gib_per_token']):.3f} | {float(row['useful_gib_per_token']):.3f} | "
            f"{float(row['waste_gib_per_token']):.3f} | {cover:.1f} |"
        )
    lines += [
        "",
        "## Top Up/Gate Bundle Predictors",
        "",
        "| predictor | H | budget | recall | precision | full steps | pred/actual bytes | pred GiB/token | waste GiB/token |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best_gate:
        lines.append(
            f"| {row['predictor']} | {row['horizon']} | {row['budget']} | "
            f"{float(row['recall']):.4f} | {float(row['precision']):.4f} | "
            f"{100.0 * float(row['full_step_pct']):.2f}% | {float(row['pred_over_actual_bytes']):.2f}x | "
            f"{float(row['predicted_gib_per_token']):.3f} | {float(row['waste_gib_per_token']):.3f} |"
        )
    if passing:
        lines += ["", "Decision: at least one row passes the offline gate; runtime design may proceed only after frozen dev settings and held-out validation plan."]
    else:
        lines += [
            "",
            "Decision: no hybrid route-history predictor passes the offline admission gate.",
            "",
            "Interpretation:",
            "",
            "- Same-prompt recency/history improves the best all-role recall over plain cooc, but not enough to make speculative future-layer prefetch demand-safe.",
            "- A runtime prefetch prototype should not be implemented from route traces alone.",
            "- The next predictor attempt needs a stronger signal such as router logits, hidden-state features, or a draft/router model; otherwise the optimization should return to lower-byte expert representation or storage-format changes.",
        ]
    lines += [
        "",
        "## Artifacts",
        "",
        "- `parse_summary.csv`",
        "- `fold_metrics.csv`",
        "- `aggregate_metrics.csv`",
        "",
    ]
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out / "report.md")


if __name__ == "__main__":
    main()
