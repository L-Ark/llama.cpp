#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from simulate_cache_policy import Cache, Key, cache_id, load_profile, load_trace, preload_from_profile


def load_key_set(path: Path) -> set[Key]:
    keys: set[Key] = set()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys.add(Key(row["tensor"], int(row["expert_idx"]), int(row["expert_bytes"])))
    return keys


def write_trace(path: Path, events: list[tuple[int, Key]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["seq", "expert_bytes", "tensor_base", "expert_idx", "tensor"])
        for seq, key in events:
            writer.writerow([seq, key.expert_bytes, "0x0", key.expert_idx, key.tensor])


def make_trace(args: argparse.Namespace) -> dict[str, object]:
    profile_rows = load_profile(args.profile)
    trace = load_trace(args.trace)
    ram_keys = load_key_set(args.ram_tier_profile) if args.ram_tier_profile else set()

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

    ssd_events: list[tuple[int, Key]] = []
    vram_hits = 0
    vram_misses = 0
    ram_hits = 0
    prefix_ssd = 0
    for idx, key in enumerate(trace):
        cache = upgate if cache_id(key) == "upgate" else down
        if cache.access(key, idx):
            vram_hits += 1
            continue
        vram_misses += 1
        if key in ram_keys:
            ram_hits += 1
            continue
        ssd_events.append((idx + 1, key))
        if idx < args.prefix_events:
            prefix_ssd += 1

    write_trace(args.out_trace, ssd_events)
    result = {
        "trace": str(args.trace),
        "profile": str(args.profile),
        "ram_tier_profile": str(args.ram_tier_profile) if args.ram_tier_profile else None,
        "out_trace": str(args.out_trace),
        "events": len(trace),
        "ssd_events": len(ssd_events),
        "prefix_events": args.prefix_events,
        "prefix_ssd_events": prefix_ssd,
        "vram_hits": vram_hits,
        "vram_misses": vram_misses,
        "ram_hits": ram_hits,
        "predicted_pack_reads": len(ssd_events),
        "slots": {"upgate": args.upgate_slots, "down": args.down_slots},
        "preloads": {"upgate": upgate_preload, "down": down_preload},
        "policy": args.policy,
        "reserve_pct": args.reserve_pct,
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--ram-tier-profile", type=Path)
    parser.add_argument("--out-trace", type=Path, required=True)
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
    parser.add_argument("--prefix-events", type=int, default=12624)
    args = parser.parse_args()

    print(json.dumps(make_trace(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
