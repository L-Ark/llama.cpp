#!/usr/bin/env python3
"""Kimi hidden-vector + route-score future expert admission.

This is a dev-only offline screen. It joins filtered activation dumps
(`role=up, active_slot=0`) with route-score traces. The route-score trace
provides the complete top-k expert labels for each decode token/layer; the
activation dump provides one runtime-available hidden vector per token/layer.
Evaluation is leave-one-prompt-out.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


LAYER_RE = re.compile(r"blk\.(\d+)\.")


@dataclass
class TokenRoute:
    call: int
    pos: int
    ids_by_layer: dict[int, tuple[int, ...]]


@dataclass
class ActivationToken:
    layers: dict[int, np.ndarray]


@dataclass
class Sample:
    prompt: str
    token_index: int
    pos: int
    source_layer: int
    target_layer: int
    feature: np.ndarray
    source_ids: tuple[int, ...]
    target_ids: tuple[int, ...]


def parse_csv_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("|") if part)


def parse_int_list(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def fmt(value: float) -> str:
    return f"{value:.6f}"


def layer_of(tensor: str) -> int:
    match = LAYER_RE.search(tensor or "")
    return int(match.group(1)) if match else -1


def compress_feature(values: np.ndarray, feature_dim: int) -> np.ndarray:
    vec = values.astype(np.float32, copy=False)
    if feature_dim > 0 and feature_dim < vec.size:
        if vec.size % feature_dim == 0:
            vec = vec.reshape(feature_dim, vec.size // feature_dim).mean(axis=1)
        else:
            idx = np.linspace(0, vec.size - 1, feature_dim, dtype=np.int64)
            vec = vec[idx]
    vec = vec.astype(np.float32, copy=True)
    vec -= float(vec.mean())
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= norm
    return vec


def load_route_tokens(run: Path, layers: int) -> tuple[list[TokenRoute], dict[str, Any]]:
    path = run / "route-score-trace.csv"
    grouped: dict[int, dict[int, tuple[int, ...]]] = defaultdict(dict)
    pos_by_call: dict[int, int] = {}
    raw_rows = 0
    prompt_like = 0
    if not path.exists():
        raise RuntimeError(f"missing {path}")
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            raw_rows += 1
            try:
                call = int(row["call"])
                layer = int(row["layer"])
                n_tokens = int(row["n_tokens"])
                pos = int(row["pos"])
            except (KeyError, ValueError):
                continue
            if n_tokens != 1 or layer < 1 or layer > layers:
                prompt_like += 1
                continue
            ids = parse_csv_ints(row.get("ids", ""))
            if not ids:
                continue
            grouped[call][layer] = ids
            pos_by_call[call] = pos

    tokens: list[TokenRoute] = []
    incomplete = 0
    for call in sorted(grouped):
        by_layer = grouped[call]
        if len(by_layer) == layers and all(layer in by_layer for layer in range(1, layers + 1)):
            tokens.append(TokenRoute(call=call, pos=pos_by_call.get(call, -1), ids_by_layer=by_layer))
        else:
            incomplete += 1
    return tokens, {
        "route_score_rows": raw_rows,
        "route_tokens": len(tokens),
        "route_prompt_like_rows": prompt_like,
        "route_incomplete_calls": incomplete,
    }


def load_activation_tokens(run: Path, layers: int, feature_dim: int) -> tuple[list[ActivationToken], dict[str, Any]]:
    csv_path = run / "activations.csv"
    bin_path = run / "activations.f32"
    if not csv_path.exists() or not bin_path.exists():
        raise RuntimeError(f"missing activation dump under {run}")
    bin_values = np.memmap(bin_path, dtype=np.float32, mode="r")
    groups: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    prev_layer = -1
    role_counts: Counter[str] = Counter()
    slot_counts: Counter[str] = Counter()
    raw_rows = 0

    with csv_path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            raw_rows += 1
            role_counts[row.get("role", "")] += 1
            slot_counts[row.get("active_slot", "")] += 1
            if row.get("mode") != "decode":
                continue
            if row.get("role") != "up":
                continue
            if int(row.get("active_slot", "-1")) != 0:
                continue
            layer = layer_of(row.get("tensor", ""))
            if layer < 1 or layer > layers:
                continue
            record = {
                "layer": layer,
                "offset_floats": int(row["offset_bytes"]) // 4,
                "ne00": int(row["ne00"]),
            }
            if cur and layer <= prev_layer:
                groups.append(cur)
                cur = []
            cur.append(record)
            prev_layer = layer
    if cur:
        groups.append(cur)

    tokens: list[ActivationToken] = []
    incomplete = 0
    for group in groups:
        by_layer: dict[int, np.ndarray] = {}
        for record in group:
            start = record["offset_floats"]
            end = start + record["ne00"]
            by_layer[record["layer"]] = compress_feature(np.asarray(bin_values[start:end]), feature_dim)
        if len(by_layer) == layers and all(layer in by_layer for layer in range(1, layers + 1)):
            tokens.append(ActivationToken(layers=by_layer))
        else:
            incomplete += 1
    return tokens, {
        "activation_rows": raw_rows,
        "activation_tokens": len(tokens),
        "activation_incomplete_groups": incomplete,
        "activation_roles": dict(role_counts),
        "activation_slots": dict(slot_counts),
        "activation_f32_bytes": bin_path.stat().st_size,
    }


def load_prompt(run: Path, layers: int, feature_dim: int) -> tuple[list[Sample], dict[str, Any]]:
    routes, route_meta = load_route_tokens(run, layers)
    acts, act_meta = load_activation_tokens(run, layers, feature_dim)
    paired = min(len(routes), len(acts))
    samples: list[Sample] = []
    for token_index in range(paired):
        route = routes[token_index]
        act = acts[token_index]
        for layer in range(1, layers + 1):
            samples.append(
                Sample(
                    prompt=run.name,
                    token_index=token_index,
                    pos=route.pos,
                    source_layer=layer,
                    target_layer=layer,
                    feature=act.layers[layer],
                    source_ids=route.ids_by_layer[layer],
                    target_ids=route.ids_by_layer[layer],
                )
            )
    meta = {
        "prompt": run.name,
        **route_meta,
        **act_meta,
        "paired_tokens": paired,
        "sample_rows": len(samples),
    }
    return samples, meta


def make_future_samples(samples: list[Sample], horizon: int, layers: int) -> list[Sample]:
    by_token_layer = {(s.token_index, s.source_layer): s for s in samples}
    out: list[Sample] = []
    for sample in samples:
        target_layer = sample.source_layer + horizon
        if target_layer > layers:
            continue
        target = by_token_layer.get((sample.token_index, target_layer))
        if target is None:
            continue
        out.append(
            Sample(
                prompt=sample.prompt,
                token_index=sample.token_index,
                pos=sample.pos,
                source_layer=sample.source_layer,
                target_layer=target_layer,
                feature=sample.feature,
                source_ids=sample.source_ids,
                target_ids=target.source_ids,
            )
        )
    return out


def static_predict(train: list[Sample], target_layer: int, budget: int) -> list[int]:
    counts: Counter[int] = Counter()
    for sample in train:
        if sample.target_layer == target_layer:
            counts.update(sample.target_ids)
    return [expert for expert, _ in counts.most_common(budget)]


def fill_static(pred: list[int], train: list[Sample], target_layer: int, budget: int) -> list[int]:
    seen = set(pred)
    for expert in static_predict(train, target_layer, 384):
        if expert not in seen:
            pred.append(expert)
            seen.add(expert)
            if len(pred) >= budget:
                return pred
    for expert in range(384):
        if expert not in seen:
            pred.append(expert)
            seen.add(expert)
            if len(pred) >= budget:
                break
    return pred


def source_route_predict(sample: Sample, train: list[Sample], budget: int) -> list[int]:
    pred = list(sample.source_ids)[:budget]
    return fill_static(pred, train, sample.target_layer, budget)


def build_knn_index(train: list[Sample]) -> dict[int, tuple[np.ndarray, list[Sample]]]:
    by_layer: dict[int, list[Sample]] = defaultdict(list)
    for sample in train:
        by_layer[sample.target_layer].append(sample)
    index = {}
    for layer, rows in by_layer.items():
        mat = np.stack([row.feature for row in rows]).astype(np.float32, copy=False)
        index[layer] = (mat, rows)
    return index


def hidden_knn_predict(sample: Sample, train: list[Sample], index: dict[int, tuple[np.ndarray, list[Sample]]], budget: int, neighbors: int) -> list[int]:
    bundle = index.get(sample.target_layer)
    if bundle is None:
        return fill_static([], train, sample.target_layer, budget)
    mat, rows = bundle
    if mat.shape[0] == 0:
        return fill_static([], train, sample.target_layer, budget)
    sims = mat @ sample.feature
    k = min(neighbors, sims.shape[0])
    if k <= 0:
        return fill_static([], train, sample.target_layer, budget)
    if k == sims.shape[0]:
        top = np.argsort(-sims)
    else:
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
    votes: Counter[int] = Counter()
    for idx in top[:k]:
        weight = max(float(sims[idx]) + 1.0, 1e-6)
        for expert in rows[int(idx)].target_ids:
            votes[expert] += weight
    pred = [expert for expert, _ in votes.most_common(budget)]
    return fill_static(pred, train, sample.target_layer, budget)


def score_prediction(actual: set[int], pred: list[int], counter: Counter[str]) -> None:
    pred_set = set(pred)
    counter["samples"] += 1
    counter["actual"] += len(actual)
    counter["predicted"] += len(pred_set)
    counter["hits"] += len(actual & pred_set)
    counter["full_steps"] += int(bool(actual) and actual <= pred_set)


def evaluate(samples_by_prompt: dict[str, list[Sample]], budgets: list[int], neighbors: list[int]) -> list[dict[str, Any]]:
    prompts = sorted(samples_by_prompt)
    rows: list[dict[str, Any]] = []
    policy_names = ["static_target_layer", "source_route_plus_static"] + [f"hidden_knn{k}" for k in neighbors]
    keys = [(policy, budget) for policy in policy_names for budget in budgets]
    totals: dict[tuple[str, int], Counter[str]] = {key: Counter() for key in keys}
    per_prompt_rows: dict[tuple[str, int], list[tuple[str, dict[str, int]]]] = {key: [] for key in keys}
    max_neighbors = max(neighbors) if neighbors else 0
    max_budget = max(budgets) if budgets else 0

    for test_prompt in prompts:
        test = samples_by_prompt[test_prompt]
        train = [sample for prompt, prompt_samples in samples_by_prompt.items() if prompt != test_prompt for sample in prompt_samples]
        prompt_counters: dict[tuple[str, int], Counter[str]] = {key: Counter() for key in keys}
        static_cache: dict[tuple[int, int], list[int]] = {}

        for sample in test:
            actual = set(sample.target_ids)
            for budget in budgets:
                static_key = (sample.target_layer, budget)
                pred_static = static_cache.get(static_key)
                if pred_static is None:
                    pred_static = fill_static([], train, sample.target_layer, budget)
                    static_cache[static_key] = pred_static
                score_prediction(actual, pred_static[:budget], prompt_counters[("static_target_layer", budget)])

                pred_source = source_route_predict(sample, train, budget)
                score_prediction(actual, pred_source[:budget], prompt_counters[("source_route_plus_static", budget)])

        if max_neighbors > 0:
            index = build_knn_index(train)
            test_by_layer: dict[int, list[Sample]] = defaultdict(list)
            for sample in test:
                test_by_layer[sample.target_layer].append(sample)
            for target_layer, layer_samples in test_by_layer.items():
                bundle = index.get(target_layer)
                if bundle is None:
                    for sample in layer_samples:
                        actual = set(sample.target_ids)
                        for neighbor in neighbors:
                            for budget in budgets:
                                pred = fill_static([], train, target_layer, budget)
                                score_prediction(actual, pred[:budget], prompt_counters[(f"hidden_knn{neighbor}", budget)])
                    continue
                train_mat, train_rows = bundle
                if train_mat.shape[0] == 0:
                    continue
                test_mat = np.stack([sample.feature for sample in layer_samples]).astype(np.float32, copy=False)
                sims = test_mat @ train_mat.T
                k = min(max_neighbors, sims.shape[1])
                if k == sims.shape[1]:
                    top_idx = np.argsort(-sims, axis=1)[:, :k]
                else:
                    top_idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
                    row_order = np.argsort(-np.take_along_axis(sims, top_idx, axis=1), axis=1)
                    top_idx = np.take_along_axis(top_idx, row_order, axis=1)
                for row_i, sample in enumerate(layer_samples):
                    actual = set(sample.target_ids)
                    for neighbor in neighbors:
                        votes: Counter[int] = Counter()
                        for idx in top_idx[row_i, : min(neighbor, k)]:
                            weight = max(float(sims[row_i, idx]) + 1.0, 1e-6)
                            for expert in train_rows[int(idx)].target_ids:
                                votes[expert] += weight
                        ordered = [expert for expert, _ in votes.most_common(max_budget)]
                        pred_full = fill_static(ordered, train, target_layer, max_budget)
                        for budget in budgets:
                            score_prediction(actual, pred_full[:budget], prompt_counters[(f"hidden_knn{neighbor}", budget)])

        for key, counter in prompt_counters.items():
            totals[key].update(counter)
            per_prompt_rows[key].append((test_prompt, dict(counter)))

    for policy, budget in keys:
        total = totals[(policy, budget)]
        actual = total["actual"] or 1
        predicted = total["predicted"] or 1
        samples = total["samples"] or 1
        rows.append(
            {
                "policy": policy,
                "budget": budget,
                "samples": int(total["samples"]),
                "actual": int(total["actual"]),
                "predicted": int(total["predicted"]),
                "hits": int(total["hits"]),
                "recall": total["hits"] / actual,
                "precision": total["hits"] / predicted,
                "pred_over_actual": total["predicted"] / actual,
                "full_step_pct": total["full_steps"] / samples,
                "per_prompt": per_prompt_rows[(policy, budget)],
            }
        )
    rows.sort(key=lambda r: (r["recall"], r["full_step_pct"], -r["pred_over_actual"]), reverse=True)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "horizon",
        "policy",
        "budget",
        "samples",
        "actual",
        "predicted",
        "hits",
        "recall",
        "precision",
        "pred_over_actual",
        "full_step_pct",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})


def write_md(path: Path, result: dict[str, Any]) -> None:
    rows = result["rows"]
    gate = result["admission"]
    passing = [
        row for row in rows
        if row["recall"] >= gate["min_recall"]
        and row["pred_over_actual"] <= gate["max_pred_over_actual"]
        and row["full_step_pct"] >= gate["min_full_step_pct"]
    ]
    lines = [
        "# Kimi hidden-vector future expert admission",
        "",
        "This is a dev-only offline screen. It does not change runtime behavior or claim SOTA.",
        "",
        "## Corpus",
        "",
        f"- root: `{result['root']}`",
        f"- prompts: `{len(result['prompts'])}`",
        f"- feature dim: `{result['feature_dim']}`",
        f"- horizons: `{','.join(str(x) for x in result['horizons'])}`",
        f"- budgets: `{','.join(str(x) for x in result['budgets'])}`",
        "",
        "| prompt | route tokens | activation tokens | paired tokens | activation rows | f32 MiB | samples |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for prompt in result["prompts"]:
        lines.append(
            f"| `{prompt['prompt']}` | `{prompt['route_tokens']}` | `{prompt['activation_tokens']}` | "
            f"`{prompt['paired_tokens']}` | `{prompt['activation_rows']}` | "
            f"`{prompt['activation_f32_bytes'] / 1024**2:.1f}` | `{prompt['sample_rows']}` |"
        )
    lines += [
        "",
        "## Admission Gate",
        "",
        f"- recall >= `{gate['min_recall']}`",
        f"- predicted/actual bytes <= `{gate['max_pred_over_actual']}`",
        f"- full-step coverage >= `{gate['min_full_step_pct']}`",
        f"- passing rows: `{len(passing)}`",
        "",
        "## Top Rows",
        "",
        "| H | policy | budget | recall | precision | pred/actual | full steps | samples |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows[:24]:
        lines.append(
            f"| `{row['horizon']}` | `{row['policy']}` | `{row['budget']}` | `{fmt(row['recall'])}` | "
            f"`{fmt(row['precision'])}` | `{fmt(row['pred_over_actual'])}` | "
            f"`{fmt(row['full_step_pct'])}` | `{row['samples']}` |"
        )
    lines += ["", "## Decision", ""]
    if passing:
        best = passing[0]
        lines.append(
            f"At least one hidden-vector row passes the offline gate. Best passing row: H=`{best['horizon']}`, "
            f"policy=`{best['policy']}`, budget=`{best['budget']}`, recall=`{fmt(best['recall'])}`, "
            f"pred/actual=`{fmt(best['pred_over_actual'])}`, full-step=`{fmt(best['full_step_pct'])}`."
        )
        lines.append("")
        lines.append("Do not enable runtime by default yet; freeze the policy and run a held-out validation plan first.")
    else:
        best = rows[0] if rows else None
        if best:
            lines.append(
                f"Reject hidden-vector future prefetch as the next runtime path: best row H=`{best['horizon']}` "
                f"`{best['policy']}` budget `{best['budget']}` reaches recall `{fmt(best['recall'])}` and "
                f"full-step `{fmt(best['full_step_pct'])}` with pred/actual `{fmt(best['pred_over_actual'])}`."
            )
        else:
            lines.append("Reject: no evaluable rows were produced.")
        lines.append("")
        lines.append("This means filtered hidden vectors improve neither complete-batch coverage nor byte efficiency enough to justify runtime prefetch reads.")
    lines += [
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
    parser = argparse.ArgumentParser(description="Evaluate hidden-vector future expert prediction from filtered Kimi traces.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=60)
    parser.add_argument("--horizons", default="1,2,3")
    parser.add_argument("--budgets", default="8,16,32")
    parser.add_argument("--neighbors", default="1,3,5")
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--min-recall", type=float, default=0.65)
    parser.add_argument("--max-pred-over-actual", type=float, default=1.35)
    parser.add_argument("--min-full-step-pct", type=float, default=0.40)
    args = parser.parse_args()

    horizons = parse_int_list(args.horizons)
    budgets = parse_int_list(args.budgets)
    neighbors = parse_int_list(args.neighbors)
    prompt_dirs = sorted(path for path in args.root.iterdir() if path.is_dir() and path.name.startswith("dev_"))
    samples_by_prompt: dict[str, list[Sample]] = {}
    prompt_meta = []
    for prompt_dir in prompt_dirs:
        samples, meta = load_prompt(prompt_dir, args.layers, args.feature_dim)
        samples_by_prompt[prompt_dir.name] = samples
        prompt_meta.append(meta)

    all_rows: list[dict[str, Any]] = []
    for horizon in horizons:
        future_by_prompt = {
            prompt: make_future_samples(samples, horizon, args.layers)
            for prompt, samples in samples_by_prompt.items()
        }
        rows = evaluate(future_by_prompt, budgets, neighbors)
        for row in rows:
            row["horizon"] = horizon
        all_rows.extend(rows)
    all_rows.sort(key=lambda r: (r["recall"], r["full_step_pct"], -r["pred_over_actual"]), reverse=True)

    result = {
        "kind": "kimi_hidden_route_future_admission",
        "root": str(args.root),
        "prompts": prompt_meta,
        "layers": args.layers,
        "feature_dim": args.feature_dim,
        "horizons": horizons,
        "budgets": budgets,
        "neighbors": neighbors,
        "admission": {
            "min_recall": args.min_recall,
            "max_pred_over_actual": args.max_pred_over_actual,
            "min_full_step_pct": args.min_full_step_pct,
        },
        "rows": all_rows,
        "reproduce_command": " ".join(sys.argv),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    write_csv(args.out_dir / "metrics.csv", all_rows)
    write_md(args.out_dir / "report.md", result)
    print(f"wrote {args.out_dir / 'report.json'}")
    print(f"wrote {args.out_dir / 'metrics.csv'}")
    print(f"wrote {args.out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
