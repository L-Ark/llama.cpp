#!/usr/bin/env python3
"""Offline FineMoE-style route-prefix prediction bound for Kimi traces.

The input route_trace.csv does not carry token ids, but accepted Kimi traces
have monotonically increasing layer groups with a layer reset between decode
segments. This tool reconstructs those segments and evaluates whether prior
route prefixes can predict future-layer expert tensors well enough to justify a
runtime prefetch/cache-protection prototype.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TENSOR_RE = re.compile(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight")


def parse_trace(path: Path) -> list[list[dict[str, Any]]]:
    segments: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    last_layer: int | None = None
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            m = TENSOR_RE.search(row["tensor"])
            if not m:
                continue
            layer = int(m.group(1))
            role = m.group(2)
            if last_layer is not None and layer < last_layer:
                if cur:
                    segments.append(cur)
                    cur = []
            cur.append(
                {
                    "seq": int(row["seq"]),
                    "layer": layer,
                    "role": role,
                    "expert_idx": int(row["expert_idx"]),
                    "expert_bytes": int(row["expert_bytes"]),
                    "tensor": row["tensor"],
                    "key": f"{row['tensor']}#{row['expert_idx']}",
                }
            )
            last_layer = layer
    if cur:
        segments.append(cur)
    return segments


def layer_key_map(segment: list[dict[str, Any]]) -> dict[int, dict[str, int]]:
    out: dict[int, dict[str, int]] = defaultdict(dict)
    for ev in segment:
        out[ev["layer"]][ev["key"]] = ev["expert_bytes"]
    return dict(out)


def keys_for_layers(layer_map: dict[int, dict[str, int]], start: int, end: int) -> dict[str, int]:
    out: dict[str, int] = {}
    for layer in range(start, end + 1):
        out.update(layer_map.get(layer, {}))
    return out


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def choose_nearest_history(history: list[dict[int, dict[str, int]]], prefix_keys: set[str]) -> tuple[int, float]:
    best_idx = -1
    best_score = -1.0
    for idx, hist in enumerate(history):
        hist_prefix = set(keys_for_layers(hist, 1, max_prefix_layer(hist, prefix_keys)).keys())
        score = jaccard(prefix_keys, hist_prefix)
        if score > best_score:
            best_idx = idx
            best_score = score
    return best_idx, best_score


def max_prefix_layer(layer_map: dict[int, dict[str, int]], prefix_keys: set[str]) -> int:
    # The caller always provides a prefix from layers 1..L. Recovering L from
    # keys is ambiguous, so this function is only a placeholder for empty maps.
    if not layer_map or not prefix_keys:
        return 0
    return max(layer_map)


def choose_nearest_history_for_layer(
    history: list[dict[int, dict[str, int]]],
    prefix_keys: set[str],
    prefix_layer: int,
) -> tuple[int, float]:
    best_idx = -1
    best_score = -1.0
    for idx, hist in enumerate(history):
        hist_prefix = set(keys_for_layers(hist, 1, prefix_layer).keys())
        score = jaccard(prefix_keys, hist_prefix)
        if score > best_score:
            best_idx = idx
            best_score = score
    return best_idx, best_score


def global_hotset_predict(
    history: list[dict[int, dict[str, int]]],
    start_layer: int,
    end_layer: int,
    actual: dict[str, int],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    sizes: dict[str, int] = {}
    actual_count_by_layer: Counter[int] = Counter()
    key_layer: dict[str, int] = {}
    for key in actual:
        m = TENSOR_RE.search(key.split("#", 1)[0])
        if m:
            actual_count_by_layer[int(m.group(1))] += 1
    for hist in history:
        for layer in range(start_layer, end_layer + 1):
            for key, size in hist.get(layer, {}).items():
                counts[key] += 1
                sizes[key] = size
                key_layer[key] = layer
    out: dict[str, int] = {}
    for layer in range(start_layer, end_layer + 1):
        n = actual_count_by_layer[layer]
        candidates = [key for key, _ in counts.most_common() if key_layer.get(key) == layer]
        for key in candidates[:n]:
            out[key] = sizes[key]
    return out


def score_prediction(pred: dict[str, int], actual: dict[str, int]) -> dict[str, float]:
    actual_bytes = sum(actual.values())
    pred_bytes = sum(pred.values())
    useful = sum(actual[key] for key in pred.keys() & actual.keys())
    false_positive = sum(size for key, size in pred.items() if key not in actual)
    return {
        "actual_bytes": float(actual_bytes),
        "predicted_bytes": float(pred_bytes),
        "useful_bytes": float(useful),
        "false_positive_bytes": float(false_positive),
        "byte_recall": useful / actual_bytes if actual_bytes else 0.0,
        "precision_bytes": useful / pred_bytes if pred_bytes else 0.0,
        "useful_to_false_positive": useful / false_positive if false_positive else math.inf,
    }


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    sums = {
        "actual_bytes": sum(r["actual_bytes"] for r in rows),
        "predicted_bytes": sum(r["predicted_bytes"] for r in rows),
        "useful_bytes": sum(r["useful_bytes"] for r in rows),
        "false_positive_bytes": sum(r["false_positive_bytes"] for r in rows),
    }
    recalls = [r["byte_recall"] for r in rows]
    precisions = [r["precision_bytes"] for r in rows]
    useful = sums["useful_bytes"]
    fp = sums["false_positive_bytes"]
    return {
        **sums,
        "byte_recall": useful / sums["actual_bytes"] if sums["actual_bytes"] else 0.0,
        "precision_bytes": useful / sums["predicted_bytes"] if sums["predicted_bytes"] else 0.0,
        "useful_to_false_positive": useful / fp if fp else math.inf,
        "window_count": len(rows),
        "mean_window_recall": statistics.fmean(recalls),
        "p50_window_recall": statistics.median(recalls),
        "mean_window_precision": statistics.fmean(precisions),
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Kimi FineMoE route-prefix bound",
        "",
        f"- generated_at: `{result['generated_at']}`",
        f"- trace: `{result['trace']}`",
        f"- segments_used: `{result['segments_used']}`",
        f"- max_prefetch_distance: `{result['max_distance']}`",
        "",
        "| distance | prefix predictor recall | prefix useful/fp | global recall | global useful/fp | windows |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["by_distance"]:
        p = row["prefix_predictor"]
        g = row["global_hotset"]
        lines.append(
            "| {d} | {pr:.4f} | {pur} | {gr:.4f} | {gur} | {n} |".format(
                d=row["distance"],
                pr=p["byte_recall"],
                pur="inf" if math.isinf(p["useful_to_false_positive"]) else f"{p['useful_to_false_positive']:.3f}",
                gr=g["byte_recall"],
                gur="inf" if math.isinf(g["useful_to_false_positive"]) else f"{g['useful_to_false_positive']:.3f}",
                n=p["window_count"],
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            result["decision"],
            "",
            "## Reproduce",
            "",
            "```bash",
            result["reproduce_command"],
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    parser.add_argument("--max-distance", type=int, default=8)
    parser.add_argument("--min-segment-layers", type=int, default=50)
    args = parser.parse_args()

    raw_segments = parse_trace(args.trace)
    maps = [layer_key_map(seg) for seg in raw_segments]
    maps = [m for m in maps if len(m) >= args.min_segment_layers]
    max_layer = max(max(m) for m in maps)

    by_distance: list[dict[str, Any]] = []
    for distance in range(1, args.max_distance + 1):
        prefix_rows: list[dict[str, float]] = []
        global_rows: list[dict[str, float]] = []
        scores: list[float] = []
        for seg_idx in range(1, len(maps)):
            history = maps[:seg_idx]
            current = maps[seg_idx]
            for prefix_layer in range(1, max_layer - distance + 1):
                actual = keys_for_layers(current, prefix_layer + 1, prefix_layer + distance)
                if not actual:
                    continue
                prefix = set(keys_for_layers(current, 1, prefix_layer).keys())
                hist_idx, score = choose_nearest_history_for_layer(history, prefix, prefix_layer)
                if hist_idx < 0:
                    continue
                pred = keys_for_layers(history[hist_idx], prefix_layer + 1, prefix_layer + distance)
                gpred = global_hotset_predict(history, prefix_layer + 1, prefix_layer + distance, actual)
                prefix_rows.append(score_prediction(pred, actual))
                global_rows.append(score_prediction(gpred, actual))
                scores.append(score)

        by_distance.append(
            {
                "distance": distance,
                "prefix_predictor": aggregate(prefix_rows),
                "global_hotset": aggregate(global_rows),
                "mean_prefix_similarity": statistics.fmean(scores) if scores else 0.0,
                "p50_prefix_similarity": statistics.median(scores) if scores else 0.0,
            }
        )

    best = max(by_distance, key=lambda row: row["prefix_predictor"].get("byte_recall", 0.0))
    best_recall = best["prefix_predictor"]["byte_recall"]
    best_ratio = best["prefix_predictor"]["useful_to_false_positive"]
    if best_recall >= 0.25 and best_ratio >= 3.0:
        decision = (
            "Phase 0 passes the coarse prediction bound. A default-off dry-run "
            "runtime predictor/protection logger is worth designing next."
        )
    else:
        decision = (
            "Phase 0 does not justify runtime FineMoE prefetch/protection yet. "
            "The prefix predictor does not recover enough future expert bytes "
            "with a strong useful-to-false-positive ratio on this accepted trace."
        )

    result = {
        "kind": "kimi_finemoe_route_prefix_bound",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "trace": str(args.trace),
        "raw_segments": len(raw_segments),
        "segments_used": len(maps),
        "max_layer": max_layer,
        "max_distance": args.max_distance,
        "reproduce_command": " ".join(__import__("sys").argv),
        "by_distance": by_distance,
        "decision": decision,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.out_md, result)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
