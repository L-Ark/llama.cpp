#!/usr/bin/env python3
"""Offline upgate/down split sweep with scaled expert entry bytes.

This is a dev-only route-trace model. It keeps total VRAM cache bytes fixed,
varies the upgate/down split, and measures dev-global LFU miss bytes for
candidate expert byte ratios.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import pathlib
import re
import time


ROLE_RE = re.compile(r"ffn_(up|gate|down)_exps")
SLOT_RE = re.compile(r"slots=(\d+) slot=([0-9.]+) MiB")


def role_group(tensor: str) -> str | None:
    match = ROLE_RE.search(tensor)
    if not match:
        return None
    return "down" if match.group(1) == "down" else "upgate"


def parse_capacity(metrics: dict, group: str) -> int:
    value = str(metrics.get("vram_down_0" if group == "down" else "vram_upgate_0", ""))
    match = SLOT_RE.search(value)
    if not match:
        raise RuntimeError(f"cannot parse cache capacity for {group}: {value}")
    return int(int(match.group(1)) * float(match.group(2)) * 1024 * 1024)


def scaled_size(size: int, ratio: float) -> int:
    return max(1, int(math.ceil(size * ratio)))


def load_prompt(run_dir: pathlib.Path):
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    events = {"upgate": [], "down": []}
    bytes_by_key = {"upgate": {}, "down": {}}
    with (run_dir / "route-trace.csv").open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            group = role_group(row["tensor"])
            if group is None:
                continue
            key = (row["tensor"], int(row["expert_idx"]))
            size = int(row["expert_bytes"])
            events[group].append((key, size))
            bytes_by_key[group][key] = size
    return metrics, events, bytes_by_key


def build_lfu(counter: collections.Counter, bytes_by_key: dict, capacity: int, ratio: float) -> set:
    used = 0
    hot = set()
    for key, _ in counter.most_common():
        size = scaled_size(bytes_by_key[key], ratio)
        if used + size > capacity:
            continue
        hot.add(key)
        used += size
    return hot


def replay(events: list[tuple[tuple[str, int], int]], hotset: set, ratio: float) -> tuple[int, int, int]:
    hits = misses = miss_bytes = 0
    for key, size in events:
        if key in hotset:
            hits += 1
        else:
            misses += 1
            miss_bytes += scaled_size(size, ratio)
    return hits, misses, miss_bytes


def hit_pct(hits: int, misses: int) -> float:
    return 100.0 * hits / max(1, hits + misses)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=pathlib.Path, required=True)
    parser.add_argument("--ratios", default="1.0,0.505,0.4")
    parser.add_argument("--pcts", default="40,45,50,55,60,62,65,70,75,80")
    parser.add_argument("--current-pct", type=int, default=62)
    parser.add_argument("--out-json", type=pathlib.Path, required=True)
    parser.add_argument("--out-md", type=pathlib.Path, required=True)
    args = parser.parse_args()

    ratios = [float(x) for x in args.ratios.split(",") if x.strip()]
    pcts = [int(x) for x in args.pcts.split(",") if x.strip()]

    prompts = []
    global_counts = {"upgate": collections.Counter(), "down": collections.Counter()}
    global_bytes = {"upgate": {}, "down": {}}
    for run_dir in sorted(p for p in args.runs_root.iterdir() if p.is_dir() and (p / "route-trace.csv").exists()):
        metrics, events, bytes_by_key = load_prompt(run_dir)
        prompts.append((run_dir, metrics, events, bytes_by_key))
        for group in ("upgate", "down"):
            global_counts[group].update(key for key, _ in events[group])
            global_bytes[group].update(bytes_by_key[group])
    if not prompts:
        raise SystemExit(f"no route traces under {args.runs_root}")

    first_metrics = prompts[0][1]
    total_capacity = parse_capacity(first_metrics, "upgate") + parse_capacity(first_metrics, "down")

    rows = []
    for ratio in ratios:
        for pct in pcts:
            up_capacity = int(total_capacity * pct / 100.0)
            down_capacity = max(0, total_capacity - up_capacity)
            hot = {
                "upgate": build_lfu(global_counts["upgate"], global_bytes["upgate"], up_capacity, ratio),
                "down": build_lfu(global_counts["down"], global_bytes["down"], down_capacity, ratio),
            }
            totals = collections.defaultdict(float)
            for _, metrics, events, _ in prompts:
                decode_runs = int(metrics.get("decode_runs", 0)) or 1
                for group in ("upgate", "down"):
                    hits, misses, miss_bytes = replay(events[group], hot[group], ratio)
                    totals[f"{group}_hits"] += hits
                    totals[f"{group}_misses"] += misses
                    totals[f"{group}_miss_gib_per_token_sum"] += miss_bytes / 1024**3 / decode_runs
            n_prompts = len(prompts)
            up_hits = int(totals["upgate_hits"])
            up_misses = int(totals["upgate_misses"])
            down_hits = int(totals["down_hits"])
            down_misses = int(totals["down_misses"])
            rows.append({
                "ratio": ratio,
                "upgate_pct": pct,
                "up_capacity_gib": up_capacity / 1024**3,
                "down_capacity_gib": down_capacity / 1024**3,
                "upgate_hit_pct": hit_pct(up_hits, up_misses),
                "down_hit_pct": hit_pct(down_hits, down_misses),
                "upgate_miss_gib_per_token": totals["upgate_miss_gib_per_token_sum"] / n_prompts,
                "down_miss_gib_per_token": totals["down_miss_gib_per_token_sum"] / n_prompts,
                "total_miss_gib_per_token": (
                    totals["upgate_miss_gib_per_token_sum"] +
                    totals["down_miss_gib_per_token_sum"]) / n_prompts,
            })

    best_by_ratio = {}
    current_by_ratio = {}
    for ratio in ratios:
        ratio_rows = [row for row in rows if row["ratio"] == ratio]
        best_by_ratio[str(ratio)] = min(ratio_rows, key=lambda r: r["total_miss_gib_per_token"])
        current_rows = [row for row in ratio_rows if row["upgate_pct"] == args.current_pct]
        if current_rows:
            current_by_ratio[str(ratio)] = current_rows[0]

    result = {
        "kind": "kimi_quant_split_sweep_bound",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runs_root": str(args.runs_root),
        "ratios": ratios,
        "pcts": pcts,
        "current_pct": args.current_pct,
        "prompts": len(prompts),
        "total_capacity_gib": total_capacity / 1024**3,
        "best_by_ratio": best_by_ratio,
        "current_by_ratio": current_by_ratio,
        "rows": rows,
        "reproduce_command": " ".join([
            ".Agent/run-tools/kimi_quant_split_sweep_bound.py",
            f"--runs-root {args.runs_root}",
            f"--ratios {args.ratios}",
            f"--pcts {args.pcts}",
            f"--current-pct {args.current_pct}",
            f"--out-json {args.out_json}",
            f"--out-md {args.out_md}",
        ]),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# Kimi quant split-pool sweep bound",
        "",
        "This is a dev-only offline route-trace simulation. It does not change runtime behavior or claim SOTA.",
        "",
        f"- generated_at: `{result['generated_at']}`",
        f"- runs_root: `{args.runs_root}`",
        f"- prompts: `{len(prompts)}`",
        f"- total cache capacity: `{result['total_capacity_gib']:.3f} GiB`",
        "",
        "## Best Versus Current Split",
        "",
        "| ratio | current pct | current miss GiB/tok | best pct | best miss GiB/tok | relative gain | best up hit | best down hit |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ratio in ratios:
        key = str(ratio)
        best = best_by_ratio[key]
        current = current_by_ratio.get(key)
        cur_miss = current["total_miss_gib_per_token"] if current else float("nan")
        gain = (cur_miss - best["total_miss_gib_per_token"]) / cur_miss if cur_miss > 0 else 0.0
        lines.append(
            f"| {ratio:.3f} | {args.current_pct} | {cur_miss:.3f} | {best['upgate_pct']} | "
            f"{best['total_miss_gib_per_token']:.3f} | {100.0 * gain:.2f}% | "
            f"{best['upgate_hit_pct']:.1f}% | {best['down_hit_pct']:.1f}% |"
        )
    lines += [
        "",
        "## Full Sweep",
        "",
        "| ratio | upgate pct | up GiB | down GiB | up hit | down hit | up miss GiB/tok | down miss GiB/tok | total miss GiB/tok |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['ratio']:.3f} | {row['upgate_pct']} | {row['up_capacity_gib']:.3f} | "
            f"{row['down_capacity_gib']:.3f} | {row['upgate_hit_pct']:.1f}% | "
            f"{row['down_hit_pct']:.1f}% | {row['upgate_miss_gib_per_token']:.3f} | "
            f"{row['down_miss_gib_per_token']:.3f} | {row['total_miss_gib_per_token']:.3f} |"
        )
    lines += [
        "",
        "## Decision Rule",
        "",
        "- Run a real cold-start split sweep only if current IQ3 ratio `1.0` improves by at least `5%` versus the current split.",
        "- For future byte-reduced representations, use this table to set the first runtime split rather than treating `62` as fixed.",
        "",
        "## Reproduce",
        "",
        "```bash",
        result["reproduce_command"],
        "```",
        "",
    ]
    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print(args.out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
