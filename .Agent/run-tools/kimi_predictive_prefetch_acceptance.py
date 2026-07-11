#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import csv
import re
from pathlib import Path


TENSOR_RE = re.compile(r"blk\.(\d+)\.ffn_(up|gate|down)_exps\.weight")


def pct(num: int | float, den: int | float) -> float:
    return 100.0 * float(num) / float(den) if den else 0.0


def iter_groups(route_trace: Path) -> list[tuple[str, tuple[int, ...]]]:
    groups: list[tuple[str, tuple[int, ...]]] = []
    cur_tensor: str | None = None
    cur_experts: list[int] = []

    with route_trace.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            tensor = row.get("tensor", "")
            if not tensor:
                continue
            expert = int(row.get("expert_idx", "0") or 0)
            if cur_tensor is not None and tensor != cur_tensor:
                groups.append((cur_tensor, tuple(cur_experts)))
                cur_experts = []
            cur_tensor = tensor
            cur_experts.append(expert)

    if cur_tensor is not None:
        groups.append((cur_tensor, tuple(cur_experts)))
    return groups


def load_decode_sequences(route_root: Path, role: str, max_active: int, exclude_re: re.Pattern[str]) -> dict[tuple[str, int], list[tuple[int, ...]]]:
    sequences: dict[tuple[str, int], list[tuple[int, ...]]] = collections.defaultdict(list)
    run_dirs = sorted(p for p in route_root.iterdir() if p.is_dir() and (p / "route-trace.csv").exists())
    if not run_dirs:
        raise SystemExit(f"No route-trace.csv files found under {route_root}")

    for run_dir in run_dirs:
        if exclude_re.search(run_dir.name):
            raise SystemExit(f"Refusing held-out/test-looking route trace: {run_dir}")
        for tensor, experts in iter_groups(run_dir / "route-trace.csv"):
            match = TENSOR_RE.search(tensor)
            if not match or len(experts) > max_active:
                continue
            layer = int(match.group(1))
            tensor_role = match.group(2)
            if tensor_role != role:
                continue
            sequences[(run_dir.name, layer)].append(experts)
    return sequences


def eval_previous(sequences: dict[tuple[str, int], list[tuple[int, ...]]]) -> tuple[int, int, int, int]:
    hits = predicted = actual = exact = 0
    for seq in sequences.values():
        for prev, cur in zip(seq, seq[1:]):
            cur_set = set(cur)
            hits += sum(1 for expert in prev if expert in cur_set)
            predicted += len(prev)
            actual += len(cur)
            exact += int(set(prev) == cur_set)
    return hits, predicted, actual, exact


def eval_previous_topn(sequences: dict[tuple[str, int], list[tuple[int, ...]]], n: int) -> tuple[int, int, int]:
    hits = predicted = actual = 0
    for seq in sequences.values():
        for prev, cur in zip(seq, seq[1:]):
            pred = prev[:n]
            cur_set = set(cur)
            hits += sum(1 for expert in pred if expert in cur_set)
            predicted += len(pred)
            actual += len(cur)
    return hits, predicted, actual


def eval_window(sequences: dict[tuple[str, int], list[tuple[int, ...]]], window: int, topn: int) -> tuple[int, int, int, int]:
    hits = predicted = actual = pairs = 0
    for seq in sequences.values():
        for i in range(window, len(seq)):
            counts: collections.Counter[int] = collections.Counter()
            recent: dict[int, int] = {}
            for age, experts in enumerate(seq[i - window:i], start=1):
                for expert in experts:
                    counts[expert] += 1
                    recent[expert] = age
            pred = sorted(counts, key=lambda e: (-counts[e], -recent[e], e))[:topn]
            cur_set = set(seq[i])
            hits += sum(1 for expert in pred if expert in cur_set)
            predicted += len(pred)
            actual += len(seq[i])
            pairs += 1
    return hits, predicted, actual, pairs


def layer_window_rows(sequences: dict[tuple[str, int], list[tuple[int, ...]]], window: int, topn: int) -> list[dict[str, object]]:
    by_layer: dict[int, list[tuple[int, ...]]] = collections.defaultdict(list)
    for (_prompt, layer), seq in sequences.items():
        marker = (-1,)
        by_layer[layer].extend(seq + [marker])

    rows: list[dict[str, object]] = []
    for layer, merged in by_layer.items():
        split: list[list[tuple[int, ...]]] = [[]]
        for item in merged:
            if item == (-1,):
                split.append([])
            else:
                split[-1].append(item)
        scoped = {(str(i), layer): seq for i, seq in enumerate(split) if seq}
        hits, predicted, actual, pairs = eval_window(scoped, window, topn)
        if predicted:
            rows.append({
                "layer": layer,
                "pairs": pairs,
                "hits": hits,
                "predicted": predicted,
                "actual": actual,
                "precision_pct": pct(hits, predicted),
                "recall_pct": pct(hits, actual),
            })
    rows.sort(key=lambda r: (float(r["precision_pct"]), float(r["recall_pct"])), reverse=True)
    return rows


def write_outputs(
    out_csv: Path,
    out_md: Path,
    route_root: Path,
    role: str,
    max_active: int,
    sequences: dict[tuple[str, int], list[tuple[int, ...]]],
    windows: list[int],
    topns: list[int],
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for n in range(1, max(topns) + 1):
        hits, predicted, actual = eval_previous_topn(sequences, n)
        rows.append({
            "mode": "previous_topn",
            "window": 1,
            "topn": n,
            "hits": hits,
            "predicted": predicted,
            "actual": actual,
            "pairs": "",
            "precision_pct": f"{pct(hits, predicted):.3f}",
            "recall_pct": f"{pct(hits, actual):.3f}",
        })
    for window in windows:
        for topn in topns:
            hits, predicted, actual, pairs = eval_window(sequences, window, topn)
            rows.append({
                "mode": "window_frequency",
                "window": window,
                "topn": topn,
                "hits": hits,
                "predicted": predicted,
                "actual": actual,
                "pairs": pairs,
                "precision_pct": f"{pct(hits, predicted):.3f}",
                "recall_pct": f"{pct(hits, actual):.3f}",
            })

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["mode", "window", "topn", "hits", "predicted", "actual", "pairs", "precision_pct", "recall_pct"],
        )
        writer.writeheader()
        writer.writerows(rows)

    prompts = sorted({prompt for prompt, _layer in sequences})
    layers = sorted({layer for _prompt, layer in sequences})
    prev_hits, prev_pred, prev_actual, prev_exact = eval_previous(sequences)
    prev_pairs = sum(max(len(seq) - 1, 0) for seq in sequences.values())
    lines = [
        "# Kimi Predictive Prefetch Acceptance",
        "",
        f"- route root: `{route_root}`",
        f"- role: `{role}`",
        f"- max active filter: `{max_active}`",
        f"- prompts: `{len(prompts)}`",
        f"- layers: `{len(layers)}`",
        f"- layer/prompt sequences: `{len(sequences)}`",
        "",
        "## Previous Token Same-Layer Top-8",
        "",
        f"- precision: `{pct(prev_hits, prev_pred):.2f}%`",
        f"- recall: `{pct(prev_hits, prev_actual):.2f}%`",
        f"- exact set match: `{pct(prev_exact, prev_pairs):.2f}%`",
        "",
        "## Sliding Window Frequency",
        "",
        "| window | top-n | precision | recall | predicted per layer-token |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["mode"] != "window_frequency":
            continue
        predicted = int(row["predicted"])
        pairs = int(row["pairs"])
        lines.append(
            f"| {row['window']} | {row['topn']} | {row['precision_pct']}% | "
            f"{row['recall_pct']}% | {predicted / pairs if pairs else 0:.2f} |"
        )

    best_window = 4 if 4 in windows else windows[0]
    best_topn = 1 if 1 in topns else topns[0]
    lines.extend([
        "",
        f"## Top Layers For Window {best_window}, Top-{best_topn}",
        "",
        "| layer | precision | recall | pairs |",
        "|---:|---:|---:|---:|",
    ])
    for row in layer_window_rows(sequences, best_window, best_topn)[:20]:
        lines.append(
            f"| {row['layer']} | {float(row['precision_pct']):.2f}% | "
            f"{float(row['recall_pct']):.2f}% | {row['pairs']} |"
        )

    lines.extend([
        "",
        "Interpretation:",
        "",
        "- Full previous-token top-8 prediction is too wasteful for direct prefetch.",
        "- A small sliding-window top-1/top-2 predictor may be worth testing only",
        "  if runtime can throttle planned prefetch and avoid increasing total IO.",
        "- This report is an offline acceptance screen, not a SOTA result.",
        "",
    ])
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate Kimi route-trace predictive-prefetch acceptance.")
    ap.add_argument("--route-root", required=True, type=Path)
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--out-md", required=True, type=Path)
    ap.add_argument("--role", default="gate", choices=["gate", "up", "down"])
    ap.add_argument("--max-active", type=int, default=8)
    ap.add_argument("--windows", default="2,4,8,16")
    ap.add_argument("--topns", default="1,2,4,8,12,16")
    ap.add_argument("--exclude-prompt-regex", default=r"heldout|test_")
    args = ap.parse_args()

    windows = [int(x) for x in args.windows.split(",") if x.strip()]
    topns = [int(x) for x in args.topns.split(",") if x.strip()]
    if not windows or not topns:
        raise SystemExit("windows and topns must not be empty")

    sequences = load_decode_sequences(
        args.route_root,
        role=args.role,
        max_active=args.max_active,
        exclude_re=re.compile(args.exclude_prompt_regex),
    )
    write_outputs(args.out_csv, args.out_md, args.route_root, args.role, args.max_active, sequences, windows, topns)
    print(args.out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
