#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import simulate_cache_policy as sim


MIB = 1024 * 1024


def slots_for_budget(total_mib: int, upgate_pct: int) -> tuple[int, int]:
    total_bytes = total_mib * MIB
    upgate_bytes = total_bytes * upgate_pct // 100
    down_bytes = total_bytes - upgate_bytes
    return upgate_bytes // sim.UPGATE_BYTES, down_bytes // sim.DOWN_BYTES


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--total-mib", type=int, default=13498)
    parser.add_argument("--reserve-pct", type=int, default=10)
    parser.add_argument("--prefix-events", type=int, default=12624)
    parser.add_argument("--policy", default="lfu_lru",
                        choices=["lru", "lfu_lru", "profile_lfu_lru", "hybrid_profile_lfu_lru"])
    parser.add_argument("--pct-start", type=int, default=35)
    parser.add_argument("--pct-end", type=int, default=85)
    parser.add_argument("--pct-step", type=int, default=5)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    profile_rows = sim.load_profile(args.profile)
    trace = sim.load_trace(args.trace)
    results = []
    for pct in range(args.pct_start, args.pct_end + 1, args.pct_step):
        upgate_slots, down_slots = slots_for_budget(args.total_mib, pct)
        run_args = argparse.Namespace(
            trace=args.trace,
            profile=args.profile,
            upgate_slots=upgate_slots,
            down_slots=down_slots,
            reserve_pct=args.reserve_pct,
            policy=args.policy,
            admit_min_count=1,
            protect_min_count=1,
            profile_after=0,
            prefix_events=args.prefix_events,
        )
        record = sim.simulate_loaded(run_args, profile_rows, trace)
        record["upgate_pct"] = pct
        record["total_mib"] = args.total_mib
        results.append(record)

    results.sort(key=lambda d: (int(d["misses"]), int(d["prefix_misses"]), -int(d["upgate_pct"])))
    text = json.dumps(results, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
