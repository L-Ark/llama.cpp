#!/usr/bin/env python3
"""Offline bound for lower-quant expert bytes plus larger effective VRAM cache.

This uses existing route traces only. It does not change runtime behavior and
does not use held-out prompts for tuning. The model keeps the same VRAM cache
byte budget, scales each expert entry size by a candidate quant ratio, and
replays prompt-general and oracle cache policies.
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
from dataclasses import dataclass
from typing import Iterable


ROLE_RE = re.compile(r"ffn_(up|gate|down)_exps")
SLOT_RE = re.compile(r"slots=(\d+) slot=([0-9.]+) MiB")


@dataclass(frozen=True)
class Event:
    key: tuple[str, int]
    role: str
    bytes: int


def role_group(tensor: str) -> str | None:
    match = ROLE_RE.search(tensor)
    if not match:
        return None
    role = match.group(1)
    return "down" if role == "down" else "upgate"


def parse_cache_capacity(metrics: dict, group: str) -> int:
    value = str(metrics.get("vram_down_0" if group == "down" else "vram_upgate_0", ""))
    match = SLOT_RE.search(value)
    if not match:
        raise RuntimeError(f"cannot parse cache capacity for {group}: {value}")
    slots = int(match.group(1))
    slot_mib = float(match.group(2))
    return int(slots * slot_mib * 1024 * 1024)


def parse_current_hit(metrics: dict, group: str) -> float:
    value = str(metrics.get("vram_down_0" if group == "down" else "vram_upgate_0", ""))
    match = re.search(r"hit_rate=([0-9.]+)%", value)
    return float(match.group(1)) if match else 0.0


def load_prompt(run_dir: pathlib.Path) -> tuple[dict, dict[str, list[Event]], dict[tuple[str, int], int]]:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    events: dict[str, list[Event]] = {"upgate": [], "down": []}
    bytes_by_key: dict[tuple[str, int], int] = {}
    with (run_dir / "route-trace.csv").open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            group = role_group(row["tensor"])
            if group is None:
                continue
            key = (row["tensor"], int(row["expert_idx"]))
            size = int(row["expert_bytes"])
            events[group].append(Event(key=key, role=group, bytes=size))
            bytes_by_key[key] = size
    return metrics, events, bytes_by_key


def scaled_size(size: int, ratio: float) -> int:
    return max(1, int(math.ceil(size * ratio)))


def build_lfu_hotset(counter: collections.Counter, bytes_by_key: dict[tuple[str, int], int], capacity: int, ratio: float) -> set[tuple[str, int]]:
    used = 0
    hot: set[tuple[str, int]] = set()
    for key, _ in counter.most_common():
        size = scaled_size(bytes_by_key[key], ratio)
        if used + size > capacity:
            continue
        hot.add(key)
        used += size
    return hot


def replay_static(events: Iterable[Event], hotset: set[tuple[str, int]], ratio: float) -> tuple[int, int, int]:
    hits = 0
    misses = 0
    miss_bytes = 0
    for event in events:
        if event.key in hotset:
            hits += 1
        else:
            misses += 1
            miss_bytes += scaled_size(event.bytes, ratio)
    return hits, misses, miss_bytes


def replay_lru(events: list[Event], capacity: int, ratio: float) -> tuple[int, int, int]:
    cache: collections.OrderedDict[tuple[str, int], int] = collections.OrderedDict()
    used = 0
    hits = misses = miss_bytes = 0
    for event in events:
        key = event.key
        size = scaled_size(event.bytes, ratio)
        if key in cache:
            hits += 1
            cache.move_to_end(key)
            continue
        misses += 1
        miss_bytes += size
        while cache and used + size > capacity:
            _, victim_size = cache.popitem(last=False)
            used -= victim_size
        if size <= capacity:
            cache[key] = size
            used += size
    return hits, misses, miss_bytes


def replay_farthest_next(events: list[Event], capacity: int, ratio: float) -> tuple[int, int, int]:
    future: dict[tuple[str, int], collections.deque[int]] = collections.defaultdict(collections.deque)
    for idx, event in enumerate(events):
        future[event.key].append(idx)
    cache: dict[tuple[str, int], int] = {}
    used = 0
    hits = misses = miss_bytes = 0
    never = 10**18
    for event in events:
        key = event.key
        size = scaled_size(event.bytes, ratio)
        future[key].popleft()
        if key in cache:
            hits += 1
            continue
        misses += 1
        miss_bytes += size
        while cache and used + size > capacity:
            victim = max(cache, key=lambda k: future[k][0] if future[k] else never)
            used -= cache.pop(victim)
        if size <= capacity:
            cache[key] = size
            used += size
    return hits, misses, miss_bytes


def hit_pct(hits: int, misses: int) -> float:
    return 100.0 * hits / max(1, hits + misses)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=pathlib.Path, required=True)
    parser.add_argument("--ratios", default="1.0,0.827,0.696,0.563,0.505,0.4,0.3")
    parser.add_argument("--include-farthest-next", action="store_true")
    parser.add_argument("--out-json", type=pathlib.Path, required=True)
    parser.add_argument("--out-md", type=pathlib.Path, required=True)
    args = parser.parse_args()

    ratios = [float(part) for part in args.ratios.split(",") if part.strip()]
    prompts = []
    global_counts: dict[str, collections.Counter] = {"upgate": collections.Counter(), "down": collections.Counter()}
    global_bytes: dict[str, dict[tuple[str, int], int]] = {"upgate": {}, "down": {}}
    for run_dir in sorted(p for p in args.runs_root.iterdir() if p.is_dir() and (p / "route-trace.csv").exists()):
        metrics, events, bytes_by_key = load_prompt(run_dir)
        prompts.append((run_dir, metrics, events, bytes_by_key))
        for group in ("upgate", "down"):
            global_counts[group].update(event.key for event in events[group])
            for event in events[group]:
                global_bytes[group][event.key] = event.bytes

    if not prompts:
        raise SystemExit(f"no prompt route traces under {args.runs_root}")

    rows = []
    summary_rows = []
    for ratio in ratios:
        global_hot = {}
        for group in ("upgate", "down"):
            capacity = parse_cache_capacity(prompts[0][1], group)
            global_hot[group] = build_lfu_hotset(global_counts[group], global_bytes[group], capacity, ratio)

        totals = collections.defaultdict(float)
        for run_dir, metrics, events, bytes_by_key in prompts:
            prompt_id = metrics.get("prompt_id", run_dir.name)
            decode_runs = int(metrics.get("decode_runs", 0)) or 1
            for group in ("upgate", "down"):
                capacity = parse_cache_capacity(metrics, group)
                seq = events[group]
                prompt_counts = collections.Counter(event.key for event in seq)
                prompt_hot = build_lfu_hotset(prompt_counts, bytes_by_key, capacity, ratio)
                gh, gm, gb = replay_static(seq, global_hot[group], ratio)
                ph, pm, pb = replay_static(seq, prompt_hot, ratio)
                lh, lm, lb = replay_lru(seq, capacity, ratio)
                if args.include_farthest_next:
                    bh, bm, bb = replay_farthest_next(seq, capacity, ratio)
                    farthest_hit = hit_pct(bh, bm)
                    farthest_miss = bb / 1024**3 / decode_runs
                else:
                    farthest_hit = None
                    farthest_miss = None
                total_scaled = sum(scaled_size(event.bytes, ratio) for event in seq)
                row = {
                    "ratio": ratio,
                    "prompt_id": prompt_id,
                    "group": group,
                    "events": len(seq),
                    "unique": len(set(event.key for event in seq)),
                    "decode_runs": decode_runs,
                    "capacity_gib": capacity / 1024**3,
                    "current_hit_pct": parse_current_hit(metrics, group),
                    "total_scaled_gib": total_scaled / 1024**3,
                    "global_lfu_hit_pct": hit_pct(gh, gm),
                    "global_lfu_miss_gib_per_token": gb / 1024**3 / decode_runs,
                    "prompt_lfu_hit_pct": hit_pct(ph, pm),
                    "prompt_lfu_miss_gib_per_token": pb / 1024**3 / decode_runs,
                    "lru_hit_pct": hit_pct(lh, lm),
                    "lru_miss_gib_per_token": lb / 1024**3 / decode_runs,
                    "farthest_next_hit_pct": farthest_hit,
                    "farthest_next_miss_gib_per_token": farthest_miss,
                }
                rows.append(row)
                aggregate_items = [
                    ("global_lfu", row["global_lfu_hit_pct"], row["global_lfu_miss_gib_per_token"]),
                    ("prompt_lfu", row["prompt_lfu_hit_pct"], row["prompt_lfu_miss_gib_per_token"]),
                    ("lru", row["lru_hit_pct"], row["lru_miss_gib_per_token"]),
                ]
                if farthest_hit is not None and farthest_miss is not None:
                    aggregate_items.append(("farthest_next", farthest_hit, farthest_miss))
                for prefix, hit, miss in aggregate_items:
                    totals[f"{prefix}_hit_sum"] += hit
                    totals[f"{prefix}_miss_sum"] += miss
                totals["groups"] += 1
        groups = max(1.0, totals["groups"])
        summary_rows.append({
            "ratio": ratio,
            "global_lfu_mean_hit_pct": totals["global_lfu_hit_sum"] / groups,
            "global_lfu_mean_miss_gib_per_token": totals["global_lfu_miss_sum"] / (groups / 2.0),
            "prompt_lfu_mean_hit_pct": totals["prompt_lfu_hit_sum"] / groups,
            "prompt_lfu_mean_miss_gib_per_token": totals["prompt_lfu_miss_sum"] / (groups / 2.0),
            "lru_mean_hit_pct": totals["lru_hit_sum"] / groups,
            "lru_mean_miss_gib_per_token": totals["lru_miss_sum"] / (groups / 2.0),
            "farthest_next_mean_hit_pct": (totals["farthest_next_hit_sum"] / groups) if args.include_farthest_next else None,
            "farthest_next_mean_miss_gib_per_token": (totals["farthest_next_miss_sum"] / (groups / 2.0)) if args.include_farthest_next else None,
        })

    result = {
        "kind": "kimi_quant_scaled_cache_bound",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runs_root": str(args.runs_root),
        "ratios": ratios,
        "include_farthest_next": args.include_farthest_next,
        "prompts": len(prompts),
        "summary": summary_rows,
        "rows": rows,
        "reproduce_command": " ".join([
            ".Agent/run-tools/kimi_quant_scaled_cache_bound.py",
            f"--runs-root {args.runs_root}",
            f"--ratios {args.ratios}",
            f"--out-json {args.out_json}",
            f"--out-md {args.out_md}",
        ]),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# Kimi quant-scaled VRAM cache bound",
        "",
        "This is a dev-only offline route-trace simulation. It does not change runtime behavior or claim SOTA.",
        "",
        f"- generated_at: `{result['generated_at']}`",
        f"- runs_root: `{args.runs_root}`",
        f"- prompts: `{len(prompts)}`",
        "",
        "## Mean Bound",
        "",
        "| ratio | global LFU hit | global LFU miss GiB/tok | prompt LFU hit | prompt LFU miss GiB/tok | LRU hit | LRU miss GiB/tok | farthest-next hit | farthest-next miss GiB/tok |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        farthest_hit = "n/a" if row["farthest_next_mean_hit_pct"] is None else f"{row['farthest_next_mean_hit_pct']:.1f}%"
        farthest_miss = "n/a" if row["farthest_next_mean_miss_gib_per_token"] is None else f"{row['farthest_next_mean_miss_gib_per_token']:.3f}"
        lines.append(
            f"| {row['ratio']:.3f} | {row['global_lfu_mean_hit_pct']:.1f}% | {row['global_lfu_mean_miss_gib_per_token']:.3f} | "
            f"{row['prompt_lfu_mean_hit_pct']:.1f}% | {row['prompt_lfu_mean_miss_gib_per_token']:.3f} | "
            f"{row['lru_mean_hit_pct']:.1f}% | {row['lru_mean_miss_gib_per_token']:.3f} | "
            f"{farthest_hit} | {farthest_miss} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- `global LFU` is prompt-general over the dev set and is the only policy in this table that resembles a deployable static hotset.",
        "- `prompt LFU` and `farthest-next` are oracles. They are upper bounds, not acceptable SOTA policies.",
        "- Miss GiB/token includes both up/gate and down groups after applying the quant byte ratio.",
        "- A `5 tok/s` non-overlap transfer budget at `10.4 GiB/s` and `40 ms` compute floor is about `1.664 GiB/token`.",
        "- A perfect-overlap transfer budget is about `2.08 GiB/token`.",
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
