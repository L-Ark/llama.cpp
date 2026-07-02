#!/usr/bin/env python3
"""Replay MoE route traces against explicit size-class VRAM cache pools."""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Entry:
    count: int
    expert_bytes: int
    expert_idx: int
    tensor: str


@dataclass(frozen=True)
class Event:
    seq: int
    expert_bytes: int
    expert_idx: int
    tensor: str


class CacheSim:
    def __init__(self, slots: int, policy: str, admit_after: int) -> None:
        self.slots = slots
        self.policy = policy
        self.admit_after = admit_after
        self.clock = 1
        self.key: list[tuple[str, int] | None] = [None] * slots
        self.used: list[int] = [0] * slots
        self.slot_hits: list[int] = [0] * slots
        self.pinned: list[bool] = [False] * slots
        self.key_misses: dict[tuple[str, int], int] = {}
        self.hits = 0
        self.misses = 0
        self.miss_bytes = 0
        self.bypassed = 0
        self.dropped = 0
        self.evictions = 0
        self.preloads = 0

    def preload(self, keys: list[tuple[str, int]]) -> None:
        for key in keys:
            if self.preloads >= self.slots:
                break
            if key in self.key:
                continue
            self.key[self.preloads] = key
            self.used[self.preloads] = self.clock
            self.clock += 1
            self.pinned[self.preloads] = True
            self.preloads += 1

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
        miss_count = self.key_misses.get(key, 0) + 1
        self.key_misses[key] = miss_count
        if miss_count < self.admit_after:
            self.bypassed += 1
            return False

        slot = self._find_insert_slot()
        if slot < 0:
            self.dropped += 1
            return False
        if self.key[slot] is not None:
            self.evictions += 1
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
                if self.slot_hits[i] < lowest_hits or (
                    self.slot_hits[i] == lowest_hits and self.used[i] < oldest
                ):
                    lowest_hits = self.slot_hits[i]
                    oldest = self.used[i]
                    victim = i
            elif self.used[i] < oldest:
                oldest = self.used[i]
                victim = i
        return victim


@dataclass
class Pool:
    name: str
    max_mib: float
    budget_mib: int
    slot_bytes: int
    cache: CacheSim

    @property
    def max_bytes(self) -> int:
        return int(self.max_mib * 1024 * 1024 + 0.5)


def load_profile(path: Path) -> list[Entry]:
    out: list[Entry] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            out.append(
                Entry(
                    count=int(row["count"]),
                    expert_bytes=int(row["expert_bytes"]),
                    expert_idx=int(row["expert_idx"]),
                    tensor=row["tensor"],
                )
            )
    return out


def load_trace(path: Path) -> list[Event]:
    out: list[Event] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            out.append(
                Event(
                    seq=int(row["seq"]),
                    expert_bytes=int(row["expert_bytes"]),
                    expert_idx=int(row["expert_idx"]),
                    tensor=row["tensor"],
                )
            )
    return out


def load_fallback(path: Path) -> dict[int, tuple[int, int]]:
    by_size: dict[int, list[int]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            b = int(row["expert_bytes"])
            cur = by_size.setdefault(b, [0, 0])
            cur[0] += int(row["count"])
            cur[1] += int(row["fallback_us"])
    return {k: (v[0], v[1]) for k, v in by_size.items()}


def parse_calibration(stderr_path: Path) -> tuple[float, float, float]:
    text = stderr_path.read_text(errors="ignore")
    stage = re.search(r"calibrate_pinned_stage .*host_stage_gib_s=([0-9.]+) h2d_gib_s=([0-9.]+)", text)
    if stage:
        host_gib_s = float(stage.group(1))
        h2d_gib_s = float(stage.group(2))
    else:
        pin = re.search(
            r"pinned staging: copies=(\d+).*slot=([0-9.]+) MiB .*host_stage=([0-9.]+) ms .*h2d=([0-9.]+) ms",
            text,
        )
        if not pin:
            raise SystemExit(f"cannot parse pinned staging calibration from {stderr_path}")
        copies = int(pin.group(1))
        slot_mib = float(pin.group(2))
        host_ms = float(pin.group(3))
        h2d_ms = float(pin.group(4))
        gib = copies * slot_mib / 1024.0
        host_gib_s = gib / max(host_ms / 1000.0, 1e-9)
        h2d_gib_s = gib / max(h2d_ms / 1000.0, 1e-9)

    down = re.search(
        r"calibrate_down .*stage_ms_per_gib=([0-9.]+)",
        text,
    )
    down_ms_per_gib = float(down.group(1)) if down else 52.84
    return host_gib_s, h2d_gib_s, down_ms_per_gib


def select_pool(event: Event, pools: list[Pool]) -> Pool | None:
    for pool in pools:
        if event.expert_bytes <= pool.max_bytes:
            return pool
    return None


def preload_keys(entries: list[Entry], pool: Pool, n_keys: int) -> list[tuple[str, int]]:
    candidates = [e for e in entries if e.expert_bytes <= pool.max_bytes]
    candidates.sort(key=lambda e: (e.count, e.count * e.expert_bytes, e.tensor), reverse=True)
    return [(e.tensor, e.expert_idx) for e in candidates[:n_keys]]


def simulate(
    name: str,
    class_mib: list[float],
    budgets_mib: list[int],
    entries: list[Entry],
    events: list[Event],
    policy: str,
    admit_after: int,
    preload: str,
    reserve_pct: int,
) -> dict[str, object]:
    pools: list[Pool] = []
    for i, (max_mib, budget_mib) in enumerate(zip(class_mib, budgets_mib)):
        max_bytes = int(max_mib * 1024 * 1024 + 0.5)
        fit_sizes = [e.expert_bytes for e in events if e.expert_bytes <= max_bytes]
        fit_sizes += [e.expert_bytes for e in entries if e.expert_bytes <= max_bytes]
        slot_bytes = max(fit_sizes) if fit_sizes else max_bytes
        slots = int((budget_mib * 1024 * 1024) // slot_bytes) if slot_bytes > 0 else 0
        slots = min(slots, 16384)
        pool = Pool(f"pool{i}", max_mib, budget_mib, slot_bytes, CacheSim(slots, policy, admit_after))
        pools.append(pool)

    # Assign each profile entry to the smallest pool that fits; otherwise larger
    # pools would preload entries that a smaller pool should own.
    owned_entries: dict[str, list[Entry]] = {p.name: [] for p in pools}
    for e in entries:
        dummy = Event(0, e.expert_bytes, e.expert_idx, e.tensor)
        pool = select_pool(dummy, pools)
        if pool:
            owned_entries[pool.name].append(e)

    for pool in pools:
        if preload == "none":
            n_preload = 0
        elif preload == "full":
            n_preload = pool.cache.slots
        else:
            reserve = math.ceil(pool.cache.slots * max(0, min(reserve_pct, 95)) / 100.0)
            n_preload = max(pool.cache.slots - reserve, 1) if pool.cache.slots else 0
        pool.cache.preload(preload_keys(owned_entries[pool.name], pool, n_preload))

    excluded = 0
    excluded_bytes = 0
    for event in events:
        pool = select_pool(event, pools)
        if not pool:
            excluded += 1
            excluded_bytes += event.expert_bytes
            continue
        pool.cache.access((event.tensor, event.expert_idx), event.expert_bytes)

    pool_rows = []
    total_hits = 0
    total_misses = 0
    total_miss_bytes = 0
    for pool in pools:
        c = pool.cache
        total = c.hits + c.misses
        total_hits += c.hits
        total_misses += c.misses
        total_miss_bytes += c.miss_bytes
        pool_rows.append(
            {
                "name": pool.name,
                "max_mib": pool.max_mib,
                "budget_mib": pool.budget_mib,
                "slot_mib": pool.slot_bytes / 2**20,
                "slots": pool.cache.slots,
                "preloads": c.preloads,
                "hits": c.hits,
                "misses": c.misses,
                "hit_pct": 100.0 * c.hits / max(total, 1),
                "miss_gib": c.miss_bytes / 2**30,
                "bypassed": c.bypassed,
                "dropped": c.dropped,
                "evictions": c.evictions,
            }
        )

    total = total_hits + total_misses
    return {
        "name": name,
        "pools": pool_rows,
        "hits": total_hits,
        "misses": total_misses,
        "hit_pct": 100.0 * total_hits / max(total, 1),
        "miss_gib": total_miss_bytes / 2**30,
        "excluded": excluded,
        "excluded_gib": excluded_bytes / 2**30,
    }


def parse_csv_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def parse_csv_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, type=Path)
    ap.add_argument("--trace", required=True, type=Path)
    ap.add_argument("--fallback", required=True, type=Path)
    ap.add_argument("--stderr", required=True, type=Path)
    ap.add_argument("--candidate", action="append", default=[],
                    help="name:class_mib_csv:budget_mib_csv, e.g. A:5.5,7.5:7000,8000")
    ap.add_argument("--policy", choices=("lru", "lfu_lru"), default="lfu_lru")
    ap.add_argument("--admit-after", type=int, default=2)
    ap.add_argument("--preload", choices=("protected", "none", "full"), default="protected")
    ap.add_argument("--reserve-pct", type=int, default=10)
    ap.add_argument("--baseline-decode-ms", type=float, default=117578.26)
    ap.add_argument("--baseline-decode-runs", type=int, default=77)
    args = ap.parse_args()

    entries = load_profile(args.profile)
    events = load_trace(args.trace)
    fallback = load_fallback(args.fallback)
    host_gib_s, h2d_gib_s, down_ms_per_gib = parse_calibration(args.stderr)

    print("size_bytes,size_mib,trace_routes,trace_gib,fallback_count,fallback_ms")
    by_size: dict[int, list[int]] = {}
    for event in events:
        cur = by_size.setdefault(event.expert_bytes, [0, 0])
        cur[0] += 1
        cur[1] += event.expert_bytes
    for b in sorted(set(by_size) | set(fallback)):
        routes, routed_bytes = by_size.get(b, [0, 0])
        fb_count, fb_us = fallback.get(b, (0, 0))
        print(f"{b},{b / 2**20:.3f},{routes},{routed_bytes / 2**30:.3f},{fb_count},{fb_us / 1000.0:.3f}")

    candidates = args.candidate or [
        "single-15000:7.5:15000",
        "two-pool-A:5.5,7.5:7000,8000",
        "two-pool-B:5.5,7.5:8000,7000",
        "two-pool-C:5.5,7.5:9000,6000",
    ]
    results = []
    for spec in candidates:
        name, classes, budgets = spec.split(":", 2)
        class_mib = parse_csv_floats(classes)
        budgets_mib = parse_csv_ints(budgets)
        if len(class_mib) != len(budgets_mib):
            raise SystemExit(f"class/budget length mismatch: {spec}")
        results.append(
            simulate(
                name, class_mib, budgets_mib, entries, events,
                args.policy, args.admit_after, args.preload, args.reserve_pct,
            )
        )

    baseline = results[0]
    baseline_miss = float(baseline["miss_gib"])
    print()
    print(
        "candidate,pool,slot_mib,budget_mib,slots,preloads,hits,misses,hit_pct,"
        "miss_gib,bypassed,dropped,evictions"
    )
    for result in results:
        for pool in result["pools"]:  # type: ignore[index]
            print(
                f"{result['name']},{pool['name']},{pool['slot_mib']:.3f},{pool['budget_mib']},"
                f"{pool['slots']},{pool['preloads']},{pool['hits']},{pool['misses']},"
                f"{pool['hit_pct']:.2f},{pool['miss_gib']:.2f},{pool['bypassed']},"
                f"{pool['dropped']},{pool['evictions']}"
            )

    print()
    print(
        "candidate,hit_pct,miss_gib,delta_miss_gib,expected_host_stage_ms,"
        "expected_h2d_ms,expected_down_saving_ms,decode_upper_tok_s,excluded,excluded_gib"
    )
    for result in results:
        miss = float(result["miss_gib"])
        saved_gib = baseline_miss - miss
        host_ms = 1000.0 * miss / max(host_gib_s, 1e-9)
        h2d_ms = 1000.0 * miss / max(h2d_gib_s, 1e-9)
        down_saving_ms = saved_gib * down_ms_per_gib
        decode_ms = args.baseline_decode_ms - max(down_saving_ms, 0.0)
        tok_s = args.baseline_decode_runs / max(decode_ms / 1000.0, 1e-9)
        print(
            f"{result['name']},{result['hit_pct']:.2f},{miss:.2f},{saved_gib:.2f},"
            f"{host_ms:.0f},{h2d_ms:.0f},{down_saving_ms:.0f},{tok_s:.3f},"
            f"{result['excluded']},{result['excluded_gib']:.2f}"
        )


if __name__ == "__main__":
    main()
