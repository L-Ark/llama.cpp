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
class TraceEvent:
    seq: int
    tensor: str
    expert_idx: int
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


def load_trace(path: Path) -> list[TraceEvent]:
    with path.open("r", newline="") as f:
        rows = csv.DictReader(f)
        events = []
        for row in rows:
            events.append(TraceEvent(
                seq=int(row["seq"]),
                tensor=row["tensor"],
                expert_idx=int(row["expert_idx"]),
                expert_bytes=int(row["expert_bytes"]),
            ))
    return sorted(events, key=lambda e: e.seq)


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


class CacheSim:
    def __init__(self, slots: int, policy: str) -> None:
        self.slots = slots
        self.policy = policy
        self.clock = 1
        self.key: list[tuple[str, int] | None] = [None] * slots
        self.used: list[int] = [0] * slots
        self.slot_hits: list[int] = [0] * slots
        self.pinned: list[bool] = [False] * slots
        self.hits = 0
        self.misses = 0
        self.dropped = 0
        self.miss_bytes = 0

    def preload(self, keys: list[tuple[str, int]]) -> int:
        loaded = 0
        for key in keys:
            if loaded >= self.slots:
                break
            if key in self.key:
                continue
            self.key[loaded] = key
            self.used[loaded] = self.clock
            self.clock += 1
            self.pinned[loaded] = True
            loaded += 1
        return loaded

    def access(self, key: tuple[str, int], nbytes: int) -> bool:
        for i, slot_key in enumerate(self.key):
            if slot_key == key:
                self.hits += 1
                self.used[i] = self.clock
                self.clock += 1
                self.slot_hits[i] += 1
                return True

        self.misses += 1
        self.miss_bytes += nbytes
        slot = self._find_insert_slot()
        if slot < 0:
            self.dropped += 1
            return False
        self.key[slot] = key
        self.used[slot] = self.clock
        self.clock += 1
        self.slot_hits[slot] = 0
        self.pinned[slot] = False
        return False

    def _find_insert_slot(self) -> int:
        for i, key in enumerate(self.key):
            if key is None:
                return i

        victim = -1
        oldest = 2**63 - 1
        lowest_hits = 2**31 - 1
        for i in range(self.slots):
            if self.pinned[i]:
                continue
            if self.policy == "lfu_lru":
                if self.slot_hits[i] < lowest_hits or (self.slot_hits[i] == lowest_hits and self.used[i] < oldest):
                    lowest_hits = self.slot_hits[i]
                    oldest = self.used[i]
                    victim = i
            elif self.used[i] < oldest:
                oldest = self.used[i]
                victim = i
        return victim


def slot_count(entries: list[Entry], events: list[TraceEvent], budget_mib: int) -> tuple[int, int]:
    slot_bytes = max([e.expert_bytes for e in entries] + [e.expert_bytes for e in events] + [0])
    slots = 0 if slot_bytes == 0 else (budget_mib * 1024 * 1024) // slot_bytes
    return min(slots, 16384), slot_bytes


def protected_slot_count(slots: int, reserve_pct: int, protect: bool) -> int:
    if slots <= 0:
        return 0
    if not protect:
        return slots
    reserve_slots = math.ceil(slots * max(0, min(reserve_pct, 95)) / 100.0)
    return max(slots - reserve_slots, 1)


def preload_keys(entries: list[Entry], n_slots: int) -> list[tuple[str, int]]:
    ranked = sorted(entries, key=lambda e: (e.count, e.count * e.expert_bytes, e.tensor), reverse=True)
    return [(e.tensor, e.expert_idx) for e in ranked[:n_slots]]


def simulate_trace(
        profile_entries: list[Entry],
        trace_events: list[TraceEvent],
        budget_mib: int,
        upgate_pct: int,
        reserve_pct: int,
        protect: bool,
        policy: str,
        upgate_weight: float,
        down_weight: float) -> dict[str, float | int]:
    upgate_budget = budget_mib * upgate_pct // 100
    down_budget = budget_mib - upgate_budget
    by_bucket_entries = {
        "upgate": [e for e in profile_entries if bucket(e.tensor) == "upgate"],
        "down": [e for e in profile_entries if bucket(e.tensor) == "down"],
    }
    by_bucket_events = {
        "upgate": [e for e in trace_events if bucket(e.tensor) == "upgate"],
        "down": [e for e in trace_events if bucket(e.tensor) == "down"],
    }

    up_slots, up_slot_bytes = slot_count(by_bucket_entries["upgate"], by_bucket_events["upgate"], upgate_budget)
    down_slots, down_slot_bytes = slot_count(by_bucket_entries["down"], by_bucket_events["down"], down_budget)
    caches = {
        "upgate": CacheSim(up_slots, policy),
        "down": CacheSim(down_slots, policy),
    }
    up_protected = protected_slot_count(up_slots, reserve_pct, protect)
    down_protected = protected_slot_count(down_slots, reserve_pct, protect)
    caches["upgate"].preload(preload_keys(by_bucket_entries["upgate"], up_protected))
    caches["down"].preload(preload_keys(by_bucket_entries["down"], down_protected))

    events = 0
    other_events = 0
    for event in trace_events:
        name = bucket(event.tensor)
        if name not in caches:
            other_events += 1
            continue
        caches[name].access((event.tensor, event.expert_idx), event.expert_bytes)
        events += 1

    up = caches["upgate"]
    down = caches["down"]
    up_total = up.hits + up.misses
    down_total = down.hits + down.misses
    total = up_total + down_total
    miss_bytes = up.miss_bytes + down.miss_bytes
    weighted_miss = up.miss_bytes * upgate_weight + down.miss_bytes * down_weight
    return {
        "pct": upgate_pct,
        "upgate_budget_mib": upgate_budget,
        "down_budget_mib": down_budget,
        "upgate_slots": up_slots,
        "down_slots": down_slots,
        "upgate_protected_slots": up_protected,
        "down_protected_slots": down_protected,
        "upgate_slot_mib": up_slot_bytes / (1024 ** 2),
        "down_slot_mib": down_slot_bytes / (1024 ** 2),
        "upgate_hit_pct": 100.0 * up.hits / max(up_total, 1),
        "down_hit_pct": 100.0 * down.hits / max(down_total, 1),
        "hit_pct": 100.0 * (up.hits + down.hits) / max(total, 1),
        "upgate_miss_gib": up.miss_bytes / (1024 ** 3),
        "down_miss_gib": down.miss_bytes / (1024 ** 3),
        "miss_gib": miss_bytes / (1024 ** 3),
        "weighted_miss_gib": weighted_miss / (1024 ** 3),
        "events": events,
        "other_events": other_events,
        "dropped": up.dropped + down.dropped,
    }


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
    parser.add_argument("--trace", type=Path, help="Optional sequence trace CSV from GGML_MOE_ROUTE_TRACE_OUT.")
    parser.add_argument("--policy", choices=("lru", "lfu_lru"), default="lfu_lru", help="Cache eviction policy for --trace replay.")
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

    if args.trace:
        trace_events = load_trace(args.trace)
        if not trace_events:
            raise SystemExit(f"empty route trace: {args.trace}")
        if args.sweep:
            print()
            print(
                "trace_pct,up_mib,down_mib,up_slots,down_slots,up_protected,down_protected,"
                "up_hit_pct,down_hit_pct,hit_pct,up_miss_gib,down_miss_gib,miss_gib,weighted_miss_gib,dropped"
            )
            trace_estimates = []
            for pct in range(args.sweep_min, args.sweep_max + 1, args.sweep_step):
                est = simulate_trace(
                    entries, trace_events, args.budget_mib, pct, args.reserve_pct,
                    args.protect_profile, args.policy, args.upgate_weight, args.down_weight)
                trace_estimates.append(est)
                print(
                    f"{est['pct']},{est['upgate_budget_mib']},{est['down_budget_mib']},"
                    f"{est['upgate_slots']},{est['down_slots']},"
                    f"{est['upgate_protected_slots']},{est['down_protected_slots']},"
                    f"{est['upgate_hit_pct']:.2f},{est['down_hit_pct']:.2f},{est['hit_pct']:.2f},"
                    f"{est['upgate_miss_gib']:.2f},{est['down_miss_gib']:.2f},"
                    f"{est['miss_gib']:.2f},{est['weighted_miss_gib']:.2f},{est['dropped']}"
                )
            best = min(trace_estimates, key=lambda e: (e["weighted_miss_gib"], e["miss_gib"], e["pct"]))
            print(
                f"trace_recommend upgate_pct={best['pct']} policy={args.policy} "
                f"weighted_miss={best['weighted_miss_gib']:.2f} GiB "
                f"miss={best['miss_gib']:.2f} GiB hit_est={best['hit_pct']:.2f}% "
                f"events={best['events']} other_events={best['other_events']}"
            )
        else:
            est = simulate_trace(
                entries, trace_events, args.budget_mib, args.upgate_pct, args.reserve_pct,
                args.protect_profile, args.policy, args.upgate_weight, args.down_weight)
            print()
            print(
                f"trace_sim upgate_pct={est['pct']} policy={args.policy} "
                f"up_hit={est['upgate_hit_pct']:.2f}% down_hit={est['down_hit_pct']:.2f}% "
                f"hit={est['hit_pct']:.2f}% miss={est['miss_gib']:.2f} GiB "
                f"weighted_miss={est['weighted_miss_gib']:.2f} GiB dropped={est['dropped']}"
            )


if __name__ == "__main__":
    main()
