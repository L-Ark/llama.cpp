#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import heapq
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


UPGATE_BYTES = 4_030_464
DOWN_BYTES = 4_816_896


@dataclass(frozen=True)
class Key:
    tensor: str
    expert_idx: int
    expert_bytes: int


class Cache:
    def __init__(
        self,
        slots: int,
        policy: str,
        profile_counts: Counter[Key],
        pinned: set[Key],
        admit_min_count: int,
        protect_min_count: int,
        profile_after: int,
        ghost_admit_after: int,
    ) -> None:
        self.slots = slots
        self.policy = policy
        self.profile_counts = profile_counts
        self.pinned = set(pinned)
        self.admit_min_count = admit_min_count
        self.protect_min_count = protect_min_count
        self.profile_after = profile_after
        self.ghost_admit_after = ghost_admit_after
        self.ghost_counts: Counter[Key] = Counter()
        self.entries: dict[Key, tuple[int, int]] = {}
        self.heap: list[tuple[tuple[int, int, int], int, Key]] = []
        self.clock = 0
        self.hits = 0
        self.misses = 0
        self.bypasses = 0
        self.evictions = 0

    def preload(self, key: Key, pin: bool) -> bool:
        if len(self.entries) >= self.slots:
            return False
        self.clock += 1
        self.entries[key] = (self.clock, 0)
        self._push_heap(key, 0)
        if pin:
            self.pinned.add(key)
        return True

    def access(self, key: Key, event_idx: int) -> bool:
        self.clock += 1
        if key in self.entries:
            used, hits = self.entries[key]
            self.entries[key] = (self.clock, hits + 1)
            self._push_heap(key, event_idx)
            self.hits += 1
            return True

        self.misses += 1
        if self.ghost_admit_after > 1:
            self.ghost_counts[key] += 1
            if self.ghost_counts[key] < self.ghost_admit_after:
                self.bypasses += 1
                return False
        if self.profile_counts[key] < self.admit_min_count:
            self.bypasses += 1
            return False
        if len(self.entries) < self.slots:
            self.entries[key] = (self.clock, 0)
            self._push_heap(key, event_idx)
            return False

        victim = self._victim(event_idx)
        if victim is None:
            self.bypasses += 1
            return False
        self.entries.pop(victim, None)
        self.pinned.discard(victim)
        self.entries[key] = (self.clock, 0)
        self._push_heap(key, event_idx)
        self.evictions += 1
        return False

    def _score(self, key: Key, used: int, hits: int, event_idx: int | None = None) -> tuple[int, int, int]:
        count = self.profile_counts[key]
        if self.policy == "hybrid_profile_lfu_lru" and event_idx is not None and event_idx < self.profile_after:
            return (hits, used, count)
        if self.policy == "lru":
            return (used, 0, count)
        if self.policy == "lfu_lru":
            return (hits, used, count)
        if self.policy in {"profile_lfu_lru", "hybrid_profile_lfu_lru"}:
            return (count, hits, used)
        raise ValueError(f"unknown policy: {self.policy}")

    def _push_heap(self, key: Key, event_idx: int) -> None:
        used, hits = self.entries[key]
        heapq.heappush(self.heap, (self._score(key, used, hits, event_idx), self.clock, key))

    def _victim(self, event_idx: int) -> Key | None:
        while self.heap:
            _score, _serial, key = heapq.heappop(self.heap)
            cur = self.entries.get(key)
            if cur is None:
                continue
            cur_score = self._score(key, cur[0], cur[1], event_idx)
            if cur_score != _score:
                heapq.heappush(self.heap, (cur_score, self.clock, key))
                continue
            if key in self.pinned and self.profile_counts[key] >= self.protect_min_count:
                continue
            return key
        return None


def cache_id(key: Key) -> str:
    return "upgate" if key.expert_bytes <= UPGATE_BYTES else "down"


def load_profile(path: Path) -> list[tuple[Key, int]]:
    rows: list[tuple[Key, int]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = Key(row["tensor"], int(row["expert_idx"]), int(row["expert_bytes"]))
            rows.append((key, int(row["count"])))
    return rows


def load_trace(path: Path) -> list[Key]:
    rows: list[Key] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(Key(row["tensor"], int(row["expert_idx"]), int(row["expert_bytes"])))
    return rows


def preload_from_profile(
    cache: Cache,
    profile_rows: list[tuple[Key, int]],
    cid: str,
    reserve_pct: int,
) -> int:
    budget = max(1, cache.slots - (cache.slots * reserve_pct) // 100)
    loaded = 0
    for key, _count in profile_rows:
        if cache_id(key) != cid:
            continue
        if loaded >= budget:
            break
        if key in cache.entries:
            continue
        if cache.preload(key, pin=True):
            loaded += 1
    return loaded


def simulate(args: argparse.Namespace) -> dict[str, object]:
    profile_rows = load_profile(args.profile)
    trace = load_trace(args.trace)
    return simulate_loaded(args, profile_rows, trace)


def simulate_loaded(
    args: argparse.Namespace,
    profile_rows: list[tuple[Key, int]],
    trace: list[Key],
) -> dict[str, object]:
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

    prefix_hits = 0
    prefix_misses = 0
    for idx, key in enumerate(trace):
        hit = (upgate if cache_id(key) == "upgate" else down).access(key, idx)
        if idx < args.prefix_events:
            if hit:
                prefix_hits += 1
            else:
                prefix_misses += 1

    total_hits = down.hits + upgate.hits
    total_misses = down.misses + upgate.misses
    return {
        "policy": args.policy,
        "reserve_pct": args.reserve_pct,
        "admit_min_count": args.admit_min_count,
        "protect_min_count": args.protect_min_count,
        "profile_after": args.profile_after,
        "ghost_admit_after": args.ghost_admit_after,
        "prefix_events": args.prefix_events,
        "prefix_hits": prefix_hits,
        "prefix_misses": prefix_misses,
        "prefix_hit_rate_pct": round(100.0 * prefix_hits / max(1, prefix_hits + prefix_misses), 3),
        "slots": {"upgate": args.upgate_slots, "down": args.down_slots},
        "preloads": {"upgate": upgate_preload, "down": down_preload},
        "events": len(trace),
        "hits": total_hits,
        "misses": total_misses,
        "hit_rate_pct": round(100.0 * total_hits / (total_hits + total_misses), 3),
        "bypasses": down.bypasses + upgate.bypasses,
        "evictions": down.evictions + upgate.evictions,
        "upgate": {
            "hits": upgate.hits,
            "misses": upgate.misses,
            "hit_rate_pct": round(100.0 * upgate.hits / max(1, upgate.hits + upgate.misses), 3),
            "bypasses": upgate.bypasses,
            "evictions": upgate.evictions,
            "resident": len(upgate.entries),
        },
        "down": {
            "hits": down.hits,
            "misses": down.misses,
            "hit_rate_pct": round(100.0 * down.hits / max(1, down.hits + down.misses), 3),
            "bypasses": down.bypasses,
            "evictions": down.evictions,
            "resident": len(down.entries),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--upgate-slots", type=int, default=2282)
    parser.add_argument("--down-slots", type=int, default=1028)
    parser.add_argument("--reserve-pct", type=int, default=10)
    parser.add_argument("--policy", choices=["lru", "lfu_lru", "profile_lfu_lru", "hybrid_profile_lfu_lru"], default="lfu_lru")
    parser.add_argument("--admit-min-count", type=int, default=1)
    parser.add_argument("--protect-min-count", type=int, default=1)
    parser.add_argument("--profile-after", type=int, default=0)
    parser.add_argument("--ghost-admit-after", type=int, default=1)
    parser.add_argument("--prefix-events", type=int, default=12624)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.sweep:
        profile_rows = load_profile(args.profile)
        trace = load_trace(args.trace)
        results = []
        for policy in ["lfu_lru", "profile_lfu_lru", "hybrid_profile_lfu_lru", "lru"]:
            for admit in [1, 2, 3, 4, 5, 8, 12, 16, 24, 32]:
                for protect in [1, 2, 3, 4, 5, 8, 12, 16, 24, 32]:
                    for ghost_admit_after in [1, 2, 3, 4]:
                        profile_after_values = [0]
                        if policy == "hybrid_profile_lfu_lru":
                            profile_after_values = [4096, 8192, 12624, 24000, 48000]
                        for profile_after in profile_after_values:
                            args.policy = policy
                            args.admit_min_count = admit
                            args.protect_min_count = protect
                            args.profile_after = profile_after
                            args.ghost_admit_after = ghost_admit_after
                            results.append(simulate_loaded(args, profile_rows, trace))
        results.sort(key=lambda d: (int(d["misses"]), -float(d["hit_rate_pct"])))
        text = json.dumps(results, indent=2, sort_keys=True) + "\n"
        if args.out:
            args.out.write_text(text)
        print(text)
        return 0

    print(json.dumps(simulate(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
