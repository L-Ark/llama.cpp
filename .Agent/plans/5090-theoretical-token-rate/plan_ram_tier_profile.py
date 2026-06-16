#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from simulate_cache_policy import Cache, Key, cache_id, load_profile, load_trace, preload_from_profile


def replay_residual_misses(args: argparse.Namespace) -> dict[str, object]:
    profile_rows = load_profile(args.profile)
    trace = load_trace(args.trace)
    profile_counts: Counter[Key] = Counter()
    for key, count in profile_rows:
        profile_counts[key] = count

    down = Cache(
        args.down_slots, args.policy, profile_counts, set(),
        args.admit_min_count, args.protect_min_count, args.profile_after,
        args.ghost_admit_after)
    upgate = Cache(
        args.upgate_slots, args.policy, profile_counts, set(),
        args.admit_min_count, args.protect_min_count, args.profile_after,
        args.ghost_admit_after)
    down_preload = preload_from_profile(down, profile_rows, "down", args.reserve_pct)
    upgate_preload = preload_from_profile(upgate, profile_rows, "upgate", args.reserve_pct)

    residual: Counter[Key] = Counter()
    prefix_residual: Counter[Key] = Counter()
    for idx, key in enumerate(trace):
        cache = upgate if cache_id(key) == "upgate" else down
        if not cache.access(key, idx):
            residual[key] += 1
            if idx < args.prefix_events:
                prefix_residual[key] += 1

    budget_bytes = args.ram_tier_mib * 1024 * 1024
    selected: list[tuple[Key, int]] = []
    used_bytes = 0
    # Equal row sizes within a tensor class make count/byte equivalent, but
    # sorting by density keeps the planner valid if future packs differ.
    ranked = sorted(
        residual.items(),
        key=lambda kv: (kv[1] / max(1, kv[0].expert_bytes), kv[1], -kv[0].expert_bytes),
        reverse=True,
    )
    for key, count in ranked:
        if used_bytes + key.expert_bytes > budget_bytes:
            continue
        selected.append((key, count))
        used_bytes += key.expert_bytes

    selected_keys = {key for key, _count in selected}
    covered_misses = sum(residual[key] for key in selected_keys)
    covered_prefix_misses = sum(prefix_residual[key] for key in selected_keys)
    total_residual_misses = sum(residual.values())
    total_prefix_residual_misses = sum(prefix_residual.values())

    return {
        "trace": str(args.trace),
        "profile": str(args.profile),
        "policy": args.policy,
        "reserve_pct": args.reserve_pct,
        "slots": {"upgate": args.upgate_slots, "down": args.down_slots},
        "preloads": {"upgate": upgate_preload, "down": down_preload},
        "ram_tier_mib": args.ram_tier_mib,
        "selected_entries": len(selected),
        "selected_bytes": used_bytes,
        "selected_mib": round(used_bytes / (1024 * 1024), 3),
        "total_events": len(trace),
        "residual_unique": len(residual),
        "residual_misses": total_residual_misses,
        "covered_misses": covered_misses,
        "covered_miss_pct": round(100.0 * covered_misses / max(1, total_residual_misses), 3),
        "prefix_events": args.prefix_events,
        "prefix_residual_misses": total_prefix_residual_misses,
        "covered_prefix_misses": covered_prefix_misses,
        "covered_prefix_miss_pct": round(100.0 * covered_prefix_misses / max(1, total_prefix_residual_misses), 3),
        "cache_replay": {
            "hits": down.hits + upgate.hits,
            "misses": down.misses + upgate.misses,
            "upgate": {
                "hits": upgate.hits,
                "misses": upgate.misses,
                "resident": len(upgate.entries),
            },
            "down": {
                "hits": down.hits,
                "misses": down.misses,
                "resident": len(down.entries),
            },
        },
        "selected": selected,
    }


def write_profile(path: Path, selected: list[tuple[Key, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cumulative = 0
    with path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["rank", "count", "expert_bytes", "cumulative_bytes", "tensor_base", "expert_idx", "tensor"])
        for rank, (key, count) in enumerate(selected, start=1):
            cumulative += key.expert_bytes
            writer.writerow([rank, count, key.expert_bytes, cumulative, "0x0", key.expert_idx, key.tensor])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--out-profile", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--upgate-slots", type=int, default=2282)
    parser.add_argument("--down-slots", type=int, default=1028)
    parser.add_argument("--reserve-pct", type=int, default=10)
    parser.add_argument("--policy", choices=["lru", "lfu_lru", "profile_lfu_lru", "hybrid_profile_lfu_lru"], default="lfu_lru")
    parser.add_argument("--admit-min-count", type=int, default=1)
    parser.add_argument("--protect-min-count", type=int, default=1)
    parser.add_argument("--profile-after", type=int, default=0)
    parser.add_argument("--ghost-admit-after", type=int, default=1)
    parser.add_argument("--prefix-events", type=int, default=12624)
    parser.add_argument("--ram-tier-mib", type=int, default=3072)
    args = parser.parse_args()

    result = replay_residual_misses(args)
    selected = result.pop("selected")
    write_profile(args.out_profile, selected)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    result["out_profile"] = str(args.out_profile)
    args.out_summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
