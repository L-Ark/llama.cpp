#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROUTE_OPS = {"cpu_up_route", "cpu_gate_route", "cpu_down_route"}
LAYER_RE = re.compile(r"blk\.(\d+)\.")


def parse_budget_list(value: str) -> list[int]:
    budgets = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        budgets.append(int(float(raw) * 1024 * 1024))
    if not budgets:
        raise SystemExit("--budgets-mib must contain at least one value")
    return sorted(set(budgets))


def tensor_kind(tensor: str) -> str:
    if "ffn_up_exps" in tensor:
        return "up"
    if "ffn_gate_exps" in tensor:
        return "gate"
    if "ffn_down_exps" in tensor:
        return "down"
    return "other"


def tensor_layer(tensor: str) -> int:
    match = LAYER_RE.search(tensor)
    return int(match.group(1)) if match else -1


def read_trace(path: Path) -> list[dict[str, Any]]:
    raw_rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f))

    if raw_rows and "op" in raw_rows[0]:
        start = next((i for i, row in enumerate(raw_rows) if row.get("op") == "mark_submit"), 0)
        end = next((i for i, row in enumerate(raw_rows[start + 1 :], start + 1) if row.get("op") == "mark_first_token"), len(raw_rows))
        raw_rows = raw_rows[start:end]

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(raw_rows):
        op = row.get("op") or ""
        if op and op not in ROUTE_OPS:
            continue
        tensor = row.get("tensor") or ""
        expert_raw = row.get("expert_idx") or ""
        bytes_raw = row.get("expert_bytes") or ""
        if not tensor or not expert_raw or not bytes_raw:
            continue
        try:
            expert_idx = int(expert_raw)
            expert_bytes = int(bytes_raw)
        except ValueError:
            continue
        rows.append(
            {
                "seq": int(row.get("seq") or index),
                "tensor": tensor,
                "expert_idx": expert_idx,
                "expert_bytes": expert_bytes,
                "layer": tensor_layer(tensor),
                "kind": tensor_kind(tensor),
            }
        )
    return rows


def route_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["tensor"]), int(row["expert_idx"])


def unique_route_bytes(rows: list[dict[str, Any]]) -> dict[tuple[str, int], int]:
    result: dict[tuple[str, int], int] = {}
    for row in rows:
        key = route_key(row)
        result[key] = max(result.get(key, 0), int(row["expert_bytes"]))
    return result


def top_budget(keys_to_bytes: dict[tuple[str, int], int], scores: Counter[tuple[str, int]], budget: int) -> set[tuple[str, int]]:
    selected: set[tuple[str, int]] = set()
    used = 0
    ordered = sorted(
        keys_to_bytes,
        key=lambda key: (scores.get(key, 0), keys_to_bytes[key], key[0], key[1]),
        reverse=True,
    )
    for key in ordered:
        size = keys_to_bytes[key]
        if used + size > budget:
            continue
        selected.add(key)
        used += size
    return selected


def coverage(keys_to_bytes: dict[tuple[str, int], int], selected: set[tuple[str, int]]) -> dict[str, Any]:
    total = sum(keys_to_bytes.values())
    hit = sum(size for key, size in keys_to_bytes.items() if key in selected)
    return {
        "hit_bytes": hit,
        "total_bytes": total,
        "hit_gib": hit / float(1024**3),
        "total_gib": total / float(1024**3),
        "byte_recall": (hit / total) if total else 0.0,
        "selected_routes": len(selected),
        "total_routes": len(keys_to_bytes),
    }


def oracle_scores(keys_to_bytes: dict[tuple[str, int], int]) -> Counter[tuple[str, int]]:
    return Counter({key: size for key, size in keys_to_bytes.items()})


def profile_scores(train_rows: list[dict[str, Any]]) -> Counter[tuple[str, int]]:
    scores: Counter[tuple[str, int]] = Counter()
    for row in train_rows:
        scores[route_key(row)] += int(row["expert_bytes"])
    return scores


def adjacent_expert_id_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_layer_kind: dict[tuple[int, str], set[int]] = {}
    bytes_by_layer_kind: dict[tuple[int, str], dict[int, int]] = {}
    for row in rows:
        key = (int(row["layer"]), str(row["kind"]))
        by_layer_kind.setdefault(key, set()).add(int(row["expert_idx"]))
        bytes_by_layer_kind.setdefault(key, {})[int(row["expert_idx"])] = max(
            bytes_by_layer_kind.setdefault(key, {}).get(int(row["expert_idx"]), 0),
            int(row["expert_bytes"]),
        )

    hits = 0
    total = 0
    hit_bytes = 0
    total_bytes = 0
    for (layer, kind), experts in by_layer_kind.items():
        prev = by_layer_kind.get((layer - 1, kind), set())
        total += len(experts)
        hits += len(experts & prev)
        sizes = bytes_by_layer_kind[(layer, kind)]
        total_bytes += sum(sizes.values())
        hit_bytes += sum(size for expert, size in sizes.items() if expert in prev)

    return {
        "expert_id_recall": (hits / total) if total else 0.0,
        "byte_recall": (hit_bytes / total_bytes) if total_bytes else 0.0,
        "hit_bytes": hit_bytes,
        "total_bytes": total_bytes,
    }


def adjacent_causal_predictor_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_layer_kind: dict[tuple[int, str], set[int]] = {}
    bytes_by_layer_kind: dict[tuple[int, str], dict[int, int]] = {}
    kinds_by_layer: dict[int, set[str]] = {}
    layers: set[int] = set()
    for row in rows:
        layer = int(row["layer"])
        kind = str(row["kind"])
        if layer < 0 or kind == "other":
            continue
        key = (layer, kind)
        layers.add(layer)
        kinds_by_layer.setdefault(layer, set()).add(kind)
        by_layer_kind.setdefault(key, set()).add(int(row["expert_idx"]))
        bytes_by_layer_kind.setdefault(key, {})[int(row["expert_idx"])] = max(
            bytes_by_layer_kind.setdefault(key, {}).get(int(row["expert_idx"]), 0),
            int(row["expert_bytes"]),
        )

    predicted_routes = 0
    predicted_hit_routes = 0
    target_routes = 0
    predicted_bytes = 0
    predicted_hit_bytes = 0
    target_bytes = 0
    for layer in sorted(layers):
        if layer - 1 not in layers:
            continue
        for kind in sorted(kinds_by_layer.get(layer, set())):
            actual = by_layer_kind.get((layer, kind), set())
            previous = by_layer_kind.get((layer - 1, kind), set())
            sizes = bytes_by_layer_kind.get((layer, kind), {})
            if not actual:
                continue
            predicted = previous
            hits = actual & predicted
            predicted_routes += len(predicted)
            predicted_hit_routes += len(hits)
            target_routes += len(actual)
            avg_size = (sum(sizes.values()) / len(sizes)) if sizes else 0.0
            predicted_bytes += int(round(avg_size * len(predicted)))
            predicted_hit_bytes += sum(sizes.get(expert, int(avg_size)) for expert in hits)
            target_bytes += sum(sizes.values())

    return {
        "route_precision": (predicted_hit_routes / predicted_routes) if predicted_routes else 0.0,
        "route_recall": (predicted_hit_routes / target_routes) if target_routes else 0.0,
        "byte_precision": (predicted_hit_bytes / predicted_bytes) if predicted_bytes else 0.0,
        "byte_recall": (predicted_hit_bytes / target_bytes) if target_bytes else 0.0,
        "predicted_gib": predicted_bytes / float(1024**3),
        "hit_gib": predicted_hit_bytes / float(1024**3),
        "target_gib": target_bytes / float(1024**3),
        "predicted_routes": predicted_routes,
        "hit_routes": predicted_hit_routes,
        "target_routes": target_routes,
    }


def analyze(paths: list[Path], budgets: list[int]) -> dict[str, Any]:
    traces = [{"path": str(path), "rows": read_trace(path)} for path in paths]
    reports = []
    for i, trace in enumerate(traces):
        rows = trace["rows"]
        keys_to_bytes = unique_route_bytes(rows)
        trace_report: dict[str, Any] = {
            "path": trace["path"],
            "route_events": len(rows),
            "unique_routes": len(keys_to_bytes),
            "working_set_gib": sum(keys_to_bytes.values()) / float(1024**3),
            "adjacent_expert_id_predictor": adjacent_expert_id_scores(rows),
            "adjacent_causal_predictor": adjacent_causal_predictor_scores(rows),
            "budgets": [],
        }
        train_rows = [row for j, other in enumerate(traces) if j != i for row in other["rows"]]
        for budget in budgets:
            item: dict[str, Any] = {"budget_mib": budget / float(1024 * 1024)}
            item["oracle"] = coverage(keys_to_bytes, top_budget(keys_to_bytes, oracle_scores(keys_to_bytes), budget))
            if train_rows:
                scores = profile_scores(train_rows)
                item["profile_holdout"] = coverage(keys_to_bytes, top_budget(keys_to_bytes, scores, budget))
            trace_report["budgets"].append(item)
        reports.append(trace_report)

    recommendation = "collect_more_traces"
    if len(traces) == 1:
        recommendation = "collect_more_traces_before_runtime_prefetch"
    else:
        best = 0.0
        best_budget = None
        for report in reports:
            for item in report["budgets"]:
                holdout = item.get("profile_holdout")
                if holdout and holdout["byte_recall"] > best:
                    best = float(holdout["byte_recall"])
                    best_budget = item["budget_mib"]
        recommendation = (
            f"implement_profile_prefetch_budget_{best_budget:g}_mib" if best >= 0.25
            else "reject_static_profile_prefetch_try_model_based_route_prediction"
        )

    return {
        "schema": "moe_route_predictability_v1",
        "trace_count": len(traces),
        "budgets_mib": [budget / float(1024 * 1024) for budget in budgets],
        "recommendation": recommendation,
        "reports": reports,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate whether route traces justify speculative expert prefetch.")
    parser.add_argument("traces", type=Path, nargs="+", help="Route trace CSV or TTFT trace CSV with cpu_*_route events.")
    parser.add_argument("--budgets-mib", default="128,256,512,1024,2048")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = analyze(args.traces, parse_budget_list(args.budgets_mib))
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
