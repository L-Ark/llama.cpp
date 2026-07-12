#!/usr/bin/env python3
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def parse_pipe_ints(value: str) -> list[int]:
    if not value:
        return []
    return [int(x) for x in value.split("|") if x]


def parse_pipe_floats(value: str) -> list[float]:
    if not value:
        return []
    return [float(x) for x in value.split("|") if x]


def entropy(values: list[float]) -> float:
    if not values:
        return 0.0
    total = sum(max(v, 0.0) for v in values)
    if total <= 0.0:
        return 0.0
    h = 0.0
    for v in values:
        p = max(v, 0.0) / total
        if p > 0.0:
            h -= p * math.log(p)
    return h


def ordered_top(ids: list[int], scores: list[float], n: int) -> list[int]:
    pairs = list(zip(ids, scores if len(scores) == len(ids) else [0.0] * len(ids)))
    pairs.sort(key=lambda x: (-x[1], x[0]))
    return [expert for expert, _ in pairs[:n]]


def union_ordered(*seqs: list[int]) -> list[int]:
    out = []
    seen = set()
    for seq in seqs:
        for value in seq:
            if value in seen:
                continue
            seen.add(value)
            out.append(value)
    return out


def load_prompt(prompt_dir: Path) -> list[dict]:
    path = prompt_dir / "route-score-trace.csv"
    rows = []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                n_tokens = int(row["n_tokens"])
                layer = int(row["layer"])
                call = int(row["call"])
            except (KeyError, ValueError):
                continue
            if n_tokens != 1 or layer < 0:
                continue
            ids = parse_pipe_ints(row.get("ids", ""))
            scores = parse_pipe_floats(row.get("scores", ""))
            weights = parse_pipe_floats(row.get("weights", ""))
            if not ids:
                continue
            margin = 0.0
            if len(scores) >= 2:
                sorted_scores = sorted(scores, reverse=True)
                margin = sorted_scores[0] - sorted_scores[1]
            rows.append({
                "prompt": prompt_dir.name,
                "call": call,
                "layer": layer,
                "ids": ids,
                "scores": scores,
                "weights": weights,
                "margin": margin,
                "entropy": entropy(scores),
            })
    return rows


def predict(policy: str, row: dict, by_key: dict[tuple[int, int], dict]) -> list[int] | None:
    call = row["call"]
    layer = row["layer"]
    prev_token = by_key.get((call - 1, layer))
    prev_layer = by_key.get((call, layer - 1))

    if policy == "prev_token_top2":
        return ordered_top(prev_token["ids"], prev_token["scores"], 2) if prev_token else None
    if policy == "prev_token_top4":
        return ordered_top(prev_token["ids"], prev_token["scores"], 4) if prev_token else None
    if policy == "prev_token_top8":
        return ordered_top(prev_token["ids"], prev_token["scores"], 8) if prev_token else None
    if policy == "prev_layer_top2":
        return ordered_top(prev_layer["ids"], prev_layer["scores"], 2) if prev_layer else None
    if policy == "prev_layer_top4":
        return ordered_top(prev_layer["ids"], prev_layer["scores"], 4) if prev_layer else None
    if policy == "prev_layer_top8":
        return ordered_top(prev_layer["ids"], prev_layer["scores"], 8) if prev_layer else None
    if policy == "hybrid_layer4_token4":
        if not prev_layer and not prev_token:
            return None
        return union_ordered(
            ordered_top(prev_layer["ids"], prev_layer["scores"], 4) if prev_layer else [],
            ordered_top(prev_token["ids"], prev_token["scores"], 4) if prev_token else [],
        )
    if policy == "hybrid_layer8_token8":
        if not prev_layer and not prev_token:
            return None
        return union_ordered(
            ordered_top(prev_layer["ids"], prev_layer["scores"], 8) if prev_layer else [],
            ordered_top(prev_token["ids"], prev_token["scores"], 8) if prev_token else [],
        )
    raise ValueError(policy)


def evaluate(rows_by_prompt: dict[str, list[dict]], policies: list[str]) -> list[dict]:
    results = []
    for policy in policies:
        total_rows = 0
        eligible_rows = 0
        actual_total = 0
        pred_total = 0
        hit_total = 0
        full_cover = 0
        for rows in rows_by_prompt.values():
            by_key = {(r["call"], r["layer"]): r for r in rows}
            for row in rows:
                total_rows += 1
                actual = set(row["ids"])
                actual_total += len(actual)
                pred = predict(policy, row, by_key)
                if pred is None:
                    continue
                eligible_rows += 1
                pred_set = set(pred)
                hits = len(actual & pred_set)
                hit_total += hits
                pred_total += len(pred_set)
                if actual <= pred_set:
                    full_cover += 1
        recall = hit_total / actual_total if actual_total else 0.0
        eligible_recall = hit_total / (eligible_rows * 8) if eligible_rows else 0.0
        precision = hit_total / pred_total if pred_total else 0.0
        pred_actual = pred_total / actual_total if actual_total else 0.0
        full_step = full_cover / total_rows if total_rows else 0.0
        full_step_eligible = full_cover / eligible_rows if eligible_rows else 0.0
        results.append({
            "policy": policy,
            "rows": total_rows,
            "eligible_rows": eligible_rows,
            "recall": recall,
            "eligible_recall": eligible_recall,
            "precision": precision,
            "pred_actual": pred_actual,
            "full_step": full_step,
            "full_step_eligible": full_step_eligible,
        })
    return results


def margin_bins(rows_by_prompt: dict[str, list[dict]]) -> list[dict]:
    pairs = []
    for rows in rows_by_prompt.values():
        by_key = {(r["call"], r["layer"]): r for r in rows}
        for row in rows:
            nxt = by_key.get((row["call"] + 1, row["layer"]))
            if not nxt:
                continue
            actual = set(nxt["ids"])
            prev = set(row["ids"])
            hits = len(actual & prev)
            pairs.append({
                "margin": row["margin"],
                "entropy": row["entropy"],
                "recall": hits / len(actual) if actual else 0.0,
            })
    if not pairs:
        return []
    pairs.sort(key=lambda x: x["margin"])
    bins = []
    for i in range(4):
        lo = i * len(pairs) // 4
        hi = (i + 1) * len(pairs) // 4
        part = pairs[lo:hi]
        if not part:
            continue
        bins.append({
            "bin": i + 1,
            "rows": len(part),
            "margin_min": part[0]["margin"],
            "margin_max": part[-1]["margin"],
            "avg_margin": sum(x["margin"] for x in part) / len(part),
            "avg_entropy": sum(x["entropy"] for x in part) / len(part),
            "next_token_same_layer_recall": sum(x["recall"] for x in part) / len(part),
        })
    return bins


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, root: Path, results: list[dict], bins: list[dict], rows_by_prompt: dict[str, list[dict]]) -> None:
    total_rows = sum(len(rows) for rows in rows_by_prompt.values())
    prompts = ", ".join(sorted(rows_by_prompt))
    best = sorted(results, key=lambda r: (r["full_step"], r["recall"], -r["pred_actual"]), reverse=True)[0] if results else None
    with path.open("w", encoding="utf-8") as f:
        f.write("# Kimi Route Score Predictor Offline Analysis\n\n")
        f.write(f"Trace root: `{root}`\n\n")
        f.write(f"Prompts: `{prompts}`\n\n")
        f.write(f"Decode route-score rows: `{total_rows}`\n\n")
        f.write("## Policy Results\n\n")
        f.write("| policy | rows | eligible | recall | precision | pred/actual | full-step |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in results:
            f.write(
                f"| {r['policy']} | {r['rows']} | {r['eligible_rows']} | "
                f"{r['recall']:.4f} | {r['precision']:.4f} | {r['pred_actual']:.3f} | {r['full_step']:.4f} |\n"
            )
        f.write("\n")
        if best:
            f.write("## Decision\n\n")
            f.write(
                f"Best full-step policy is `{best['policy']}` with recall `{best['recall']:.4f}`, "
                f"pred/actual `{best['pred_actual']:.3f}`, and full-step cover `{best['full_step']:.4f}`.\n\n"
            )
            if best["full_step"] < 0.40 or best["pred_actual"] > 1.35 or best["recall"] < 0.65:
                f.write(
                    "This does not pass the runtime prefetch gate. The next runtime A/B should not be built from "
                    "these simple score/history policies unless a stronger hidden-state or draft-router signal is added.\n\n"
                )
            else:
                f.write("This passes the exploratory gate and can be converted into a default-off runtime A/B.\n\n")
        f.write("## Margin Bins\n\n")
        f.write("| bin | rows | margin min | margin max | avg margin | avg entropy | next-token same-layer recall |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|\n")
        for b in bins:
            f.write(
                f"| {b['bin']} | {b['rows']} | {b['margin_min']:.6f} | {b['margin_max']:.6f} | "
                f"{b['avg_margin']:.6f} | {b['avg_entropy']:.6f} | {b['next_token_same_layer_recall']:.4f} |\n"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    prompt_dirs = [p for p in sorted(args.root.iterdir()) if (p / "route-score-trace.csv").exists()]
    rows_by_prompt = {p.name: load_prompt(p) for p in prompt_dirs}
    rows_by_prompt = {k: v for k, v in rows_by_prompt.items() if v}
    if not rows_by_prompt:
        raise SystemExit(f"no route-score rows found under {args.root}")

    policies = [
        "prev_token_top2",
        "prev_token_top4",
        "prev_token_top8",
        "prev_layer_top2",
        "prev_layer_top4",
        "prev_layer_top8",
        "hybrid_layer4_token4",
        "hybrid_layer8_token8",
    ]
    results = evaluate(rows_by_prompt, policies)
    bins = margin_bins(rows_by_prompt)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "policy-results.csv", results)
    write_csv(args.out_dir / "margin-bins.csv", bins)
    write_markdown(args.out_dir / "analysis.md", args.root, results, bins, rows_by_prompt)


if __name__ == "__main__":
    main()
