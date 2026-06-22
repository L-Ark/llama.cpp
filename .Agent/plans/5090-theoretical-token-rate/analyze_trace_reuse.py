#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import simulate_cache_policy as sim


def load_keys(path: Path) -> list[sim.Key]:
    rows: list[sim.Key] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(sim.Key(row["tensor"], int(row["expert_idx"]), int(row["expert_bytes"])))
    return rows


def distance_buckets(dist: int | None) -> str:
    if dist is None:
        return "never"
    if dist <= 4:
        return "1-4"
    if dist <= 16:
        return "5-16"
    if dist <= 64:
        return "17-64"
    if dist <= 256:
        return "65-256"
    if dist <= 1024:
        return "257-1024"
    if dist <= 4096:
        return "1025-4096"
    return "gt4096"


def summarize(args: argparse.Namespace) -> dict[str, object]:
    trace = load_keys(args.trace)
    remaining = Counter(trace)
    next_positions: dict[sim.Key, list[int]] = defaultdict(list)
    for idx in range(len(trace) - 1, -1, -1):
        next_positions[trace[idx]].append(idx)

    profile_rows = sim.load_profile(args.profile)
    profile_counts: Counter[sim.Key] = Counter()
    for key, count in profile_rows:
        profile_counts[key] = count

    run_args = argparse.Namespace(
        upgate_slots=args.upgate_slots,
        down_slots=args.down_slots,
        reserve_pct=args.reserve_pct,
        policy="lfu_lru",
        admit_min_count=1,
        protect_min_count=1,
        profile_after=0,
        ghost_admit_after=1,
        prefix_events=args.prefix_events,
    )
    down = sim.Cache(
        run_args.down_slots, run_args.policy, profile_counts, set(),
        run_args.admit_min_count, run_args.protect_min_count, run_args.profile_after,
        run_args.ghost_admit_after)
    upgate = sim.Cache(
        run_args.upgate_slots, run_args.policy, profile_counts, set(),
        run_args.admit_min_count, run_args.protect_min_count, run_args.profile_after,
        run_args.ghost_admit_after)
    sim.preload_from_profile(down, profile_rows, "down", args.reserve_pct)
    sim.preload_from_profile(upgate, profile_rows, "upgate", args.reserve_pct)

    miss_reuse = Counter()
    miss_reuse_by_cache: dict[str, Counter[str]] = {"upgate": Counter(), "down": Counter()}
    profile_bucket = Counter()
    profile_bucket_by_cache: dict[str, Counter[str]] = {"upgate": Counter(), "down": Counter()}
    prefix_miss_reuse = Counter()
    miss_count = 0
    for idx, key in enumerate(trace):
        positions = next_positions[key]
        while positions and positions[-1] <= idx:
            positions.pop()
        next_idx = positions[-1] if positions else None
        dist = None if next_idx is None else next_idx - idx
        cid = sim.cache_id(key)
        cache = upgate if cid == "upgate" else down
        was_hit = key in cache.entries
        cache.access(key, idx)
        remaining[key] -= 1
        if was_hit:
            continue
        miss_count += 1
        bucket = distance_buckets(dist)
        miss_reuse[bucket] += 1
        miss_reuse_by_cache[cid][bucket] += 1
        if idx < args.prefix_events:
            prefix_miss_reuse[bucket] += 1
        count = profile_counts[key]
        if count == 0:
            pb = "0"
        elif count == 1:
            pb = "1"
        elif count <= 4:
            pb = "2-4"
        elif count <= 8:
            pb = "5-8"
        elif count <= 16:
            pb = "9-16"
        elif count <= 32:
            pb = "17-32"
        else:
            pb = "gt32"
        profile_bucket[pb] += 1
        profile_bucket_by_cache[cid][pb] += 1

    return {
        "trace": str(args.trace),
        "profile": str(args.profile),
        "events": len(trace),
        "misses_replayed": miss_count,
        "reuse_distance_buckets": dict(miss_reuse),
        "reuse_distance_by_cache": {k: dict(v) for k, v in miss_reuse_by_cache.items()},
        "prefix_reuse_distance_buckets": dict(prefix_miss_reuse),
        "profile_count_buckets": dict(profile_bucket),
        "profile_count_by_cache": {k: dict(v) for k, v in profile_bucket_by_cache.items()},
        "final_cache": {
            "upgate": {"hits": upgate.hits, "misses": upgate.misses},
            "down": {"hits": down.hits, "misses": down.misses},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--upgate-slots", type=int, default=2282)
    parser.add_argument("--down-slots", type=int, default=1028)
    parser.add_argument("--reserve-pct", type=int, default=10)
    parser.add_argument("--prefix-events", type=int, default=12624)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    result = summarize(args)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
