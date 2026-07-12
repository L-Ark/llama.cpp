#!/usr/bin/env python3
"""Kimi hidden-feature future-expert admission.

This is a dev-only offline screen. It tests whether activation-dump vectors
carry enough prompt-general signal to predict the next routed expert set before
adding any runtime prefetch reads.

The activation dump currently records one vector per active expert matvec. For
each up/gate group, this tool uses the active_slot=0 up vector as a
runtime-available feature and predicts the expert IDs of the group at a future
execution horizon. Evaluation is leave-one-prompt-out.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


LAYER_RE = re.compile(r"blk\.(\d+)\.")


def parse_csv_ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def layer_of(tensor: str) -> int:
    match = LAYER_RE.search(tensor)
    if not match:
        return -1
    return int(match.group(1))


def fmt(x: float) -> str:
    return f"{x:.6f}"


def load_vector(bin_path: Path, row: dict[str, Any]) -> np.ndarray:
    nbytes = int(row["nbytes"])
    expected = int(row["ne00"]) * 4
    if nbytes != expected:
        raise RuntimeError(
            f"unexpected vector byte count for record {row['record_id']}: "
            f"nbytes={nbytes} expected={expected}"
        )
    with bin_path.open("rb") as f:
        f.seek(int(row["offset_bytes"]))
        raw = f.read(nbytes)
    if len(raw) != nbytes:
        raise RuntimeError(f"short activation read for record {row['record_id']}")
    return np.frombuffer(raw, dtype=np.float32).copy()


def activation_paths(prompt_root: Path) -> tuple[Path, Path]:
    if (prompt_root / "activations.csv").exists():
        return prompt_root / "activations.csv", prompt_root / "activations.f32"
    if (prompt_root / "act" / "activations.csv").exists():
        return prompt_root / "act" / "activations.csv", prompt_root / "act" / "activations.f32"
    raise RuntimeError(f"no activation dump found under {prompt_root}")


def load_prompt(prompt_root: Path, max_groups: int) -> dict[str, Any]:
    csv_path, bin_path = activation_paths(prompt_root)
    by_group: dict[tuple[int, int], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    role_counts: Counter[str] = Counter()
    token_ids: Counter[int] = Counter()
    layers: Counter[int] = Counter()
    dims: Counter[str] = Counter()

    with csv_path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            if row.get("mode") != "decode":
                continue
            for key in (
                "record_id",
                "call",
                "type",
                "expert_idx",
                "active_slot",
                "dst_id",
                "token_id",
                "ne00",
                "ne01",
                "expert_bytes",
                "offset_bytes",
                "nbytes",
            ):
                row[key] = int(row[key])
            layer = layer_of(row["tensor"])
            if layer < 0:
                continue
            role = row["role"]
            role_counts[role] += 1
            token_ids[row["token_id"]] += 1
            layers[layer] += 1
            dims[f"{role}:{row['ne00']}x{row['ne01']}:{row['nbytes']}"] += 1
            by_group[(row["call"], layer)][role].append(row)

    groups: list[dict[str, Any]] = []
    complete_upgate = 0
    down_groups = 0
    for (call, layer), role_rows in sorted(by_group.items(), key=lambda item: (item[0][0], item[0][1])):
        up = sorted(role_rows.get("up", []), key=lambda r: r["active_slot"])
        gate = sorted(role_rows.get("gate", []), key=lambda r: r["active_slot"])
        down = sorted(role_rows.get("down", []), key=lambda r: r["active_slot"])
        if len(up) >= 8 and len(gate) >= 8:
            complete_upgate += 1
        if len(down) >= 8:
            down_groups += 1
        if len(up) < 8:
            continue
        expert_ids = [int(row["expert_idx"]) for row in up[:8]]
        feature_row = up[0]
        vec = load_vector(bin_path, feature_row).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        groups.append(
            {
                "prompt": prompt_root.name,
                "call": int(call),
                "layer": int(layer),
                "token_id": int(feature_row["token_id"]),
                "expert_ids": expert_ids,
                "feature": vec,
                "up_bytes": int(up[0]["expert_bytes"]),
                "gate_bytes": int(gate[0]["expert_bytes"]) if gate else int(up[0]["expert_bytes"]),
            }
        )
        if max_groups > 0 and len(groups) >= max_groups:
            break

    return {
        "prompt": prompt_root.name,
        "csv": str(csv_path),
        "bin": str(bin_path),
        "groups": groups,
        "role_counts": dict(role_counts),
        "token_ids": dict(token_ids),
        "layers": sorted(layers),
        "layer_count": len(layers),
        "dims": dict(dims),
        "complete_upgate_groups": complete_upgate,
        "down_groups": down_groups,
    }


def make_samples(prompt: dict[str, Any], horizon: int) -> list[dict[str, Any]]:
    groups = prompt["groups"]
    out = []
    for idx, group in enumerate(groups):
        target_idx = idx + horizon
        if target_idx >= len(groups):
            continue
        target = groups[target_idx]
        out.append(
            {
                "prompt": prompt["prompt"],
                "idx": idx,
                "horizon": horizon,
                "layer": group["layer"],
                "target_layer": target["layer"],
                "feature": group["feature"],
                "source_ids": tuple(group["expert_ids"]),
                "target_ids": tuple(target["expert_ids"]),
            }
        )
    return out


def static_predict(train: list[dict[str, Any]], target_layer: int, budget: int) -> list[int]:
    counts: Counter[int] = Counter()
    for sample in train:
        if sample["target_layer"] == target_layer:
            counts.update(sample["target_ids"])
    return [expert for expert, _ in counts.most_common(budget)]


def fill_static(pred: list[int], train: list[dict[str, Any]], target_layer: int, budget: int) -> list[int]:
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


def knn_predict(
    sample: dict[str, Any],
    train: list[dict[str, Any]],
    budget: int,
    neighbors: int,
    same_source_layer: bool,
) -> list[int]:
    candidates = [
        other
        for other in train
        if other["target_layer"] == sample["target_layer"]
        and (not same_source_layer or other["layer"] == sample["layer"])
    ]
    if not candidates:
        return fill_static([], train, sample["target_layer"], budget)

    q = sample["feature"]
    sims = [(float(np.dot(q, other["feature"])), other) for other in candidates]
    sims.sort(key=lambda item: item[0], reverse=True)
    votes: Counter[int] = Counter()
    for sim, other in sims[:neighbors]:
        # Shift cosine into positive vote space while preserving rank.
        weight = max(sim + 1.0, 1e-6)
        for expert in other["target_ids"]:
            votes[expert] += weight
    pred = [expert for expert, _ in votes.most_common(budget)]
    return fill_static(pred, train, sample["target_layer"], budget)


def source_route_predict(sample: dict[str, Any], train: list[dict[str, Any]], budget: int) -> list[int]:
    pred = list(sample["source_ids"])[:budget]
    return fill_static(pred, train, sample["target_layer"], budget)


def evaluate(samples_by_prompt: dict[str, list[dict[str, Any]]], budgets: list[int], neighbors: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prompts = sorted(samples_by_prompt)
    for budget in budgets:
        policies: list[tuple[str, Any]] = [
            ("static_target_layer", lambda s, tr: fill_static([], tr, s["target_layer"], budget)),
            ("source_route_plus_static", lambda s, tr: source_route_predict(s, tr, budget)),
        ]
        for k in neighbors:
            policies.append((f"hidden_knn{k}_same_layer", lambda s, tr, kk=k: knn_predict(s, tr, budget, kk, True)))
            policies.append((f"hidden_knn{k}_target_layer", lambda s, tr, kk=k: knn_predict(s, tr, budget, kk, False)))

        for policy_name, policy_fn in policies:
            total = Counter()
            per_prompt = []
            for test_prompt in prompts:
                test = samples_by_prompt[test_prompt]
                train = [s for prompt, rows_for_prompt in samples_by_prompt.items() if prompt != test_prompt for s in rows_for_prompt]
                prompt_counter = Counter()
                for sample in test:
                    actual = set(sample["target_ids"])
                    pred = policy_fn(sample, train)[:budget]
                    pred_set = set(pred)
                    prompt_counter["samples"] += 1
                    prompt_counter["actual"] += len(actual)
                    prompt_counter["predicted"] += len(pred_set)
                    prompt_counter["hits"] += len(actual & pred_set)
                    prompt_counter["full_steps"] += int(bool(actual) and actual <= pred_set)
                total.update(prompt_counter)
                per_prompt.append((test_prompt, dict(prompt_counter)))
            actual = total["actual"] or 1
            predicted = total["predicted"] or 1
            samples = total["samples"] or 1
            rows.append(
                {
                    "policy": policy_name,
                    "budget": budget,
                    "samples": int(total["samples"]),
                    "actual": int(total["actual"]),
                    "predicted": int(total["predicted"]),
                    "hits": int(total["hits"]),
                    "recall": total["hits"] / actual,
                    "precision": total["hits"] / predicted,
                    "pred_over_actual": total["predicted"] / actual,
                    "full_step_pct": total["full_steps"] / samples,
                    "per_prompt": per_prompt,
                }
            )
    rows.sort(key=lambda r: (r["recall"], r["full_step_pct"], -r["pred_over_actual"]), reverse=True)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["horizon", "policy", "budget", "samples", "actual", "predicted", "hits", "recall", "precision", "pred_over_actual", "full_step_pct"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})


def write_md(path: Path, result: dict[str, Any]) -> None:
    rows = result["rows"]
    passing = [
        row
        for row in rows
        if row["recall"] >= result["admission"]["min_recall"]
        and row["pred_over_actual"] <= result["admission"]["max_pred_over_actual"]
        and row["full_step_pct"] >= result["admission"]["min_full_step_pct"]
    ]
    lines = [
        "# Kimi hidden-feature future-expert admission",
        "",
        "This is a dev-only offline screen. It does not change runtime behavior or claim SOTA.",
        "",
        "## Goal",
        "",
        "Test whether activation-dump vectors provide a stronger prompt-general signal for future expert-ID prefetch than route history or router scores.",
        "",
        "## Corpus",
        "",
        f"- root: `{result['root']}`",
        f"- prompts: `{len(result['prompts'])}`",
        f"- horizons: `{','.join(str(x) for x in result['horizons'])}`",
        f"- budgets: `{','.join(str(x) for x in result['budgets'])}`",
        "",
        "| prompt | groups | layers | token ids | complete upgate groups | down groups |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for prompt in result["prompts"]:
        token_ids = ",".join(str(x) for x in sorted(int(k) for k in prompt["token_ids"].keys()))
        lines.append(
            f"| `{prompt['prompt']}` | `{prompt['group_count']}` | `{prompt['layer_count']}` | "
            f"`{token_ids}` | `{prompt['complete_upgate_groups']}` | `{prompt['down_groups']}` |"
        )
    lines += [
        "",
        "Important corpus limitation:",
        "",
        "- the current activation dumps have `token_id=0` for all records, so this screen uses execution `call` order rather than true token/layer sequence labels;",
        "- the dump was produced for representation/surrogate screening, not a full future-prefetch admission corpus;",
        "- a passing row here would still require a fresh full N96 trace with explicit hidden/router labels before runtime reads.",
        "",
        "## Admission Gate",
        "",
        f"- recall >= `{result['admission']['min_recall']}`",
        f"- predicted/actual bytes <= `{result['admission']['max_pred_over_actual']}`",
        f"- full-step coverage >= `{result['admission']['min_full_step_pct']}`",
        f"- passing rows: `{len(passing)}`",
        "",
        "## Top Rows",
        "",
        "| H | policy | budget | recall | precision | pred/actual | full steps | samples |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows[:20]:
        lines.append(
            f"| `{row['horizon']}` | `{row['policy']}` | `{row['budget']}` | `{fmt(row['recall'])}` | "
            f"`{fmt(row['precision'])}` | `{fmt(row['pred_over_actual'])}` | "
            f"`{fmt(row['full_step_pct'])}` | `{row['samples']}` |"
        )
    lines += [
        "",
        "## Decision",
        "",
    ]
    if passing:
        best = passing[0]
        lines.append(
            "A hidden-feature predictor passed this small dev-only screen. Do not enable runtime prefetch yet; first collect a full cold-start N96 hidden/router trace and rerun admission with true token/layer labels."
        )
        lines.append(
            f"Best passing row: `{best['policy']}` budget `{best['budget']}`, recall `{fmt(best['recall'])}`, pred/actual `{fmt(best['pred_over_actual'])}`, full-step `{fmt(best['full_step_pct'])}`."
        )
    else:
        best = rows[0] if rows else None
        if best:
            lines.append(
                f"Reject hidden-feature future-expert prefetch from the existing activation corpus: best row horizon `{best['horizon']}` `{best['policy']}` budget `{best['budget']}` reaches recall `{fmt(best['recall'])}` and full-step `{fmt(best['full_step_pct'])}` with pred/actual `{fmt(best['pred_over_actual'])}`, so it does not pass admission."
            )
        else:
            lines.append("Reject: no evaluable rows were produced.")
        lines.append("")
        lines.append("Next predictor work must add new instrumentation or a real draft/router model, not reuse this limited activation corpus as proof.")
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
    parser = argparse.ArgumentParser(description="Evaluate hidden-feature future-expert prediction from Kimi activation dumps.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--horizons", default="1,2,3")
    parser.add_argument("--budgets", default="8,16,32")
    parser.add_argument("--neighbors", default="1,3,5")
    parser.add_argument("--max-groups-per-prompt", type=int, default=0)
    parser.add_argument("--min-recall", type=float, default=0.65)
    parser.add_argument("--max-pred-over-actual", type=float, default=1.35)
    parser.add_argument("--min-full-step-pct", type=float, default=0.40)
    args = parser.parse_args()

    prompt_roots = sorted(p for p in args.root.iterdir() if p.is_dir())
    prompts = [load_prompt(path, args.max_groups_per_prompt) for path in prompt_roots]
    horizons = parse_csv_ints(args.horizons)
    budgets = parse_csv_ints(args.budgets)
    neighbors = parse_csv_ints(args.neighbors)

    all_rows: list[dict[str, Any]] = []
    for horizon in horizons:
        samples_by_prompt = {prompt["prompt"]: make_samples(prompt, horizon) for prompt in prompts}
        rows = evaluate(samples_by_prompt, budgets, neighbors)
        for row in rows:
            row["horizon"] = horizon
        all_rows.extend(rows)
    all_rows.sort(key=lambda r: (r["recall"], r["full_step_pct"], -r["pred_over_actual"]), reverse=True)

    compact_prompts = []
    for prompt in prompts:
        compact_prompts.append(
            {
                "prompt": prompt["prompt"],
                "csv": prompt["csv"],
                "group_count": len(prompt["groups"]),
                "layer_count": prompt["layer_count"],
                "layers": prompt["layers"],
                "token_ids": prompt["token_ids"],
                "role_counts": prompt["role_counts"],
                "dims": prompt["dims"],
                "complete_upgate_groups": prompt["complete_upgate_groups"],
                "down_groups": prompt["down_groups"],
            }
        )

    result = {
        "kind": "kimi_hidden_feature_future_expert_admission",
        "root": str(args.root),
        "prompts": compact_prompts,
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
    json_path = args.out_dir / "report.json"
    csv_path = args.out_dir / "metrics.csv"
    md_path = args.out_dir / "report.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    write_csv(csv_path, all_rows)
    write_md(md_path, result)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
