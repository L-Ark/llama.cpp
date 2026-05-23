#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Entry:
    tensor: str
    expert_idx: int
    count: int
    expert_bytes: int


@dataclass(frozen=True)
class Hotset:
    slots: int
    protected_slots: int
    slot_bytes: int
    routes: int
    bytes: int
    full_hits: int
    full_hit_bytes: int
    protected_hits: int
    protected_hit_bytes: int


def bucket(tensor: str) -> str:
    if ".ffn_up_exps." in tensor or ".ffn_gate_exps." in tensor or ".ffn_gate_up_exps." in tensor:
        return "upgate"
    if ".ffn_down_exps." in tensor:
        return "down"
    return "other"


def load_entries(path: Path) -> list[Entry]:
    with path.open("r", newline="") as f:
        rows = csv.DictReader(f)
        entries = []
        for row in rows:
            entries.append(Entry(
                tensor=row["tensor"],
                expert_idx=int(row["expert_idx"]),
                count=int(row["count"]),
                expert_bytes=int(row["expert_bytes"]),
            ))
    return entries


def static_hotset(entries: list[Entry], budget_bytes: int) -> tuple[int, int, int]:
    used = 0
    hits = 0
    stored = 0
    for e in sorted(entries, key=lambda x: x.count, reverse=True):
        if used + e.expert_bytes > budget_bytes:
            continue
        used += e.expert_bytes
        hits += e.count
        stored += 1
    return hits, used, stored


def hotset_by_slots(entries: list[Entry], n_slots: int) -> tuple[int, int, int]:
    if n_slots <= 0:
        return 0, 0, 0
    hits = 0
    hit_bytes = 0
    stored = 0
    # Cache entries consume one slot, so pick by total avoided bytes per slot.
    for e in sorted(entries, key=lambda x: (x.count * x.expert_bytes, x.count), reverse=True)[:n_slots]:
        hits += e.count
        hit_bytes += e.count * e.expert_bytes
        stored += 1
    return hits, hit_bytes, stored


def cache_hotset(entries: list[Entry], budget_mib: int, reserve_pct: int, protect: bool) -> Hotset:
    routes = sum(e.count for e in entries)
    total_bytes = sum(e.count * e.expert_bytes for e in entries)
    slot_bytes = max((e.expert_bytes for e in entries), default=0)
    slots = 0 if slot_bytes == 0 else (budget_mib * 1024 * 1024) // slot_bytes
    if slots > 16384:
        slots = 16384
    if slots <= 0:
        protected_slots = 0
    elif protect:
        reserve_slots = math.ceil(slots * max(0, min(reserve_pct, 95)) / 100.0)
        protected_slots = max(slots - reserve_slots, 1)
    else:
        protected_slots = slots
    full_hits, full_hit_bytes, _ = hotset_by_slots(entries, slots)
    protected_hits, protected_hit_bytes, _ = hotset_by_slots(entries, protected_slots)
    return Hotset(
        slots=slots,
        protected_slots=protected_slots,
        slot_bytes=slot_bytes,
        routes=routes,
        bytes=total_bytes,
        full_hits=full_hits,
        full_hit_bytes=full_hit_bytes,
        protected_hits=protected_hits,
        protected_hit_bytes=protected_hit_bytes,
    )


def split_estimate(
        entries: list[Entry],
        budget_mib: int,
        upgate_pct: int,
        reserve_pct: int,
        protect: bool,
        upgate_weight: float,
        down_weight: float,
        objective: str) -> dict[str, float | int]:
    upgate_budget = budget_mib * upgate_pct // 100
    down_budget = budget_mib - upgate_budget
    by_bucket = {
        "upgate": [e for e in entries if bucket(e.tensor) == "upgate"],
        "down": [e for e in entries if bucket(e.tensor) == "down"],
    }
    up = cache_hotset(by_bucket["upgate"], upgate_budget, reserve_pct, protect)
    down = cache_hotset(by_bucket["down"], down_budget, reserve_pct, protect)
    use_protected = objective == "protected"
    up_hit_bytes = up.protected_hit_bytes if use_protected else up.full_hit_bytes
    down_hit_bytes = down.protected_hit_bytes if use_protected else down.full_hit_bytes
    up_hits = up.protected_hits if use_protected else up.full_hits
    down_hits = down.protected_hits if use_protected else down.full_hits
    total_routes = up.routes + down.routes
    total_bytes = up.bytes + down.bytes
    miss_bytes = (up.bytes - up_hit_bytes) + (down.bytes - down_hit_bytes)
    weighted_miss = (up.bytes - up_hit_bytes) * upgate_weight + (down.bytes - down_hit_bytes) * down_weight
    return {
        "pct": upgate_pct,
        "upgate_budget_mib": upgate_budget,
        "down_budget_mib": down_budget,
        "upgate_slots": up.slots,
        "down_slots": down.slots,
        "upgate_protected_slots": up.protected_slots,
        "down_protected_slots": down.protected_slots,
        "upgate_hit_pct": 100.0 * up_hits / max(up.routes, 1),
        "down_hit_pct": 100.0 * down_hits / max(down.routes, 1),
        "hit_pct": 100.0 * (up_hits + down_hits) / max(total_routes, 1),
        "miss_gib": miss_bytes / (1024 ** 3),
        "weighted_miss_gib": weighted_miss / (1024 ** 3),
        "routed_gib": total_bytes / (1024 ** 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate hot-expert VRAM cache coverage from GGML_MOE_BATCH_PROFILE_OUT CSV.")
    parser.add_argument("profile", type=Path)
    parser.add_argument("--budget-mib", type=int, default=16 * 1024)
    parser.add_argument("--upgate-pct", type=int, default=63)
    parser.add_argument("--reserve-pct", type=int, default=20, help="Protected-profile reserve percentage; mirrors GGML_MOE_VRAM_PROFILE_RESERVE_PCT.")
    parser.add_argument("--no-protect-profile", dest="protect_profile", action="store_false", help="Disable protected preload slot accounting.")
    parser.add_argument("--sweep", action="store_true", help="Print estimates for a range of up/gate cache percentages and recommend the lowest-cost split.")
    parser.add_argument("--sweep-min", type=int, default=35)
    parser.add_argument("--sweep-max", type=int, default=75)
    parser.add_argument("--sweep-step", type=int, default=5)
    parser.add_argument("--upgate-weight", type=float, default=1.0, help="Relative miss cost for up/gate experts.")
    parser.add_argument("--down-weight", type=float, default=1.0, help="Relative miss cost for down experts.")
    parser.add_argument("--objective", choices=("full", "protected"), default="protected", help="Use all cache slots or only profile-protected slots for the cost estimate.")
    parser.set_defaults(protect_profile=True)
    args = parser.parse_args()

    if args.budget_mib <= 0:
        raise SystemExit("--budget-mib must be positive")
    if args.upgate_pct < 1 or args.upgate_pct > 99:
        raise SystemExit("--upgate-pct must be 1..99")
    if args.sweep_step <= 0:
        raise SystemExit("--sweep-step must be positive")
    if args.sweep_min < 1 or args.sweep_max > 99 or args.sweep_min > args.sweep_max:
        raise SystemExit("--sweep-min/--sweep-max must define a range within 1..99")

    entries = load_entries(args.profile)
    total_routes = sum(e.count for e in entries)
    total_bytes = sum(e.count * e.expert_bytes for e in entries)
    budget = args.budget_mib * 1024 * 1024
    upgate_budget = budget * args.upgate_pct // 100
    down_budget = budget - upgate_budget

    print(f"entries={len(entries)} routes={total_routes} routed_bytes={total_bytes / (1024**3):.2f} GiB")

    hits, used, stored = static_hotset(entries, budget)
    print(f"single_cache budget={args.budget_mib} MiB stored={stored} used={used / (1024**2):.1f} MiB hit_est={100.0 * hits / max(total_routes, 1):.1f}%")

    split_hits = 0
    split_used = 0
    split_stored = 0
    for name, sub_budget in (("upgate", upgate_budget), ("down", down_budget), ("other", 0)):
        sub = [e for e in entries if bucket(e.tensor) == name]
        sub_routes = sum(e.count for e in sub)
        if name == "other":
            print(f"{name}: entries={len(sub)} routes={sub_routes} budget=0 MiB")
            continue
        h, u, s = static_hotset(sub, sub_budget)
        split_hits += h
        split_used += u
        split_stored += s
        print(f"{name}: entries={len(sub)} routes={sub_routes} budget={sub_budget // (1024**2)} MiB stored={s} used={u / (1024**2):.1f} MiB hit_est={100.0 * h / max(sub_routes, 1):.1f}%")

    print(f"split_cache upgate_pct={args.upgate_pct} stored={split_stored} used={split_used / (1024**2):.1f} MiB hit_est={100.0 * split_hits / max(total_routes, 1):.1f}%")

    if args.sweep:
        print()
        print(
            "pct,up_mib,down_mib,up_slots,down_slots,up_protected,down_protected,"
            "up_hit_pct,down_hit_pct,hit_pct,miss_gib,weighted_miss_gib"
        )
        estimates = []
        for pct in range(args.sweep_min, args.sweep_max + 1, args.sweep_step):
            est = split_estimate(
                entries, args.budget_mib, pct, args.reserve_pct, args.protect_profile,
                args.upgate_weight, args.down_weight, args.objective)
            estimates.append(est)
            print(
                f"{est['pct']},{est['upgate_budget_mib']},{est['down_budget_mib']},"
                f"{est['upgate_slots']},{est['down_slots']},"
                f"{est['upgate_protected_slots']},{est['down_protected_slots']},"
                f"{est['upgate_hit_pct']:.2f},{est['down_hit_pct']:.2f},{est['hit_pct']:.2f},"
                f"{est['miss_gib']:.2f},{est['weighted_miss_gib']:.2f}"
            )
        best = min(estimates, key=lambda e: (e["weighted_miss_gib"], e["miss_gib"], e["pct"]))
        print(
            f"recommend upgate_pct={best['pct']} objective={args.objective} "
            f"weighted_miss={best['weighted_miss_gib']:.2f} GiB "
            f"miss={best['miss_gib']:.2f} GiB hit_est={best['hit_pct']:.2f}%"
        )


if __name__ == "__main__":
    main()
