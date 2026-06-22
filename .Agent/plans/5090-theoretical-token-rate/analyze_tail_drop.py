#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from simulate_cache_policy import Cache, Key, cache_id, load_profile, load_trace, preload_from_profile


def load_key_set(path: Path | None) -> set[Key]:
    if path is None:
        return set()
    keys: set[Key] = set()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys.add(Key(row["tensor"], int(row["expert_idx"]), int(row["expert_bytes"])))
    return keys


def tensor_group(tensor: str) -> str:
    if ".ffn_up_exps." in tensor:
        return tensor.replace(".ffn_up_exps.", ".ffn_up_gate_exps.")
    if ".ffn_gate_exps." in tensor:
        return tensor.replace(".ffn_gate_exps.", ".ffn_up_gate_exps.")
    return tensor


def rank_in_group(group_counts: dict[str, int], key: Key) -> int:
    group = tensor_group(key.tensor)
    rank = group_counts[group] % 8
    group_counts[group] += 1
    return rank


def analyze(args: argparse.Namespace) -> dict[str, object]:
    profile_rows = load_profile(args.profile)
    trace = load_trace(args.trace)
    ram_keys = load_key_set(args.ram_tier_profile)

    profile_counts: Counter[Key] = Counter()
    for key, count in profile_rows:
        profile_counts[key] = count

    down = Cache(
        args.down_slots,
        args.policy,
        profile_counts,
        set(),
        args.admit_min_count,
        args.protect_min_count,
        args.profile_after,
        args.ghost_admit_after,
    )
    upgate = Cache(
        args.upgate_slots,
        args.policy,
        profile_counts,
        set(),
        args.admit_min_count,
        args.protect_min_count,
        args.profile_after,
        args.ghost_admit_after,
    )
    down_preload = preload_from_profile(down, profile_rows, "down", args.reserve_pct)
    upgate_preload = preload_from_profile(upgate, profile_rows, "upgate", args.reserve_pct)

    group_counts: dict[str, int] = defaultdict(int)
    ranks = {str(i): {"events": 0, "vram_hits": 0, "ram_hits": 0, "ssd_misses": 0} for i in range(8)}
    thresholds: dict[str, dict[str, int]] = {}
    for keep in range(1, 8):
        thresholds[str(keep)] = {
            "dropped_events": 0,
            "dropped_vram_hits": 0,
            "dropped_ram_hits": 0,
            "saved_ssd_misses": 0,
        }

    total_vram_hits = 0
    total_ram_hits = 0
    total_ssd_misses = 0
    for idx, key in enumerate(trace):
        rank = rank_in_group(group_counts, key)
        rank_key = str(rank)
        ranks[rank_key]["events"] += 1

        cache = upgate if cache_id(key) == "upgate" else down
        if cache.access(key, idx):
            status = "vram"
            total_vram_hits += 1
            ranks[rank_key]["vram_hits"] += 1
        elif key in ram_keys:
            status = "ram"
            total_ram_hits += 1
            ranks[rank_key]["ram_hits"] += 1
        else:
            status = "ssd"
            total_ssd_misses += 1
            ranks[rank_key]["ssd_misses"] += 1

        for keep in range(1, 8):
            if rank < keep:
                continue
            bucket = thresholds[str(keep)]
            bucket["dropped_events"] += 1
            if status == "vram":
                bucket["dropped_vram_hits"] += 1
            elif status == "ram":
                bucket["dropped_ram_hits"] += 1
            else:
                bucket["saved_ssd_misses"] += 1

    for keep, bucket in thresholds.items():
        dropped = bucket["dropped_events"]
        bucket["saved_ssd_pct_of_current_ssd"] = round(100.0 * bucket["saved_ssd_misses"] / max(1, total_ssd_misses), 3)
        bucket["dropped_resident_events"] = bucket["dropped_vram_hits"] + bucket["dropped_ram_hits"]
        bucket["resident_drop_pct"] = round(100.0 * bucket["dropped_resident_events"] / max(1, dropped), 3)

    return {
        "trace": str(args.trace),
        "profile": str(args.profile),
        "ram_tier_profile": str(args.ram_tier_profile) if args.ram_tier_profile else None,
        "events": len(trace),
        "slots": {"upgate": args.upgate_slots, "down": args.down_slots},
        "preloads": {"upgate": upgate_preload, "down": down_preload},
        "vram_hits": total_vram_hits,
        "ram_hits": total_ram_hits,
        "ssd_misses": total_ssd_misses,
        "ranks": ranks,
        "keep_first_n_experts": thresholds,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--ram-tier-profile", type=Path)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--upgate-slots", type=int, default=1936)
    parser.add_argument("--down-slots", type=int, default=872)
    parser.add_argument("--reserve-pct", type=int, default=10)
    parser.add_argument(
        "--policy",
        choices=["lru", "lfu_lru", "profile_lfu_lru", "hybrid_profile_lfu_lru"],
        default="lfu_lru",
    )
    parser.add_argument("--admit-min-count", type=int, default=1)
    parser.add_argument("--protect-min-count", type=int, default=1)
    parser.add_argument("--profile-after", type=int, default=0)
    parser.add_argument("--ghost-admit-after", type=int, default=1)
    args = parser.parse_args()

    result = analyze(args)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
