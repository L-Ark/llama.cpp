#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import struct
from dataclasses import dataclass
from pathlib import Path


PACK_HEADER = struct.Struct("<16sIIQQ")
PACK_ENTRY = struct.Struct("<128siIQQ")
PACK_MAGIC = b"GGMLMOEPACKv1\0\0\0"
PACK_ALIGN = 4096


@dataclass(frozen=True)
class ProfileEntry:
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
class PackEntry:
    tensor: str
    expert_idx: int
    offset: int
    nbytes: int


class CacheSim:
    def __init__(self, slots: int, policy: str) -> None:
        self.slots = slots
        self.policy = policy
        self.key: list[tuple[str, int] | None] = [None] * slots
        self.used: list[int] = [0] * slots
        self.slot_hits: list[int] = [0] * slots
        self.pinned: list[bool] = [False] * slots
        self.clock = 1
        self.hits = 0
        self.misses = 0
        self.miss_bytes = 0

    def preload(self, keys: list[tuple[str, int]]) -> None:
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

    def contains(self, key: tuple[str, int]) -> bool:
        return key in self.key

    def access(self, key: tuple[str, int], nbytes: int) -> bool:
        for i, slot_key in enumerate(self.key):
            if slot_key == key:
                self.hits += 1
                self.used[i] = self.clock
                self.clock += 1
                if self.slot_hits[i] != (1 << 32) - 1:
                    self.slot_hits[i] += 1
                return True

        self.misses += 1
        self.miss_bytes += nbytes
        slot = self._find_insert_slot()
        if slot >= 0:
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
                        self.slot_hits[i] == lowest_hits and self.used[i] < oldest):
                    lowest_hits = self.slot_hits[i]
                    oldest = self.used[i]
                    victim = i
            elif self.used[i] < oldest:
                oldest = self.used[i]
                victim = i
        return victim


def bucket(tensor: str) -> str:
    if ".ffn_up_exps." in tensor or ".ffn_gate_exps." in tensor or ".ffn_gate_up_exps." in tensor:
        return "upgate"
    if ".ffn_down_exps." in tensor:
        return "down"
    return "other"


def align_up(value: int, alignment: int = PACK_ALIGN) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def load_profile(path: Path) -> list[ProfileEntry]:
    with path.open("r", newline="") as f:
        return [
            ProfileEntry(
                tensor=row["tensor"],
                expert_idx=int(row["expert_idx"]),
                count=int(row["count"]),
                expert_bytes=int(row["expert_bytes"]),
            )
            for row in csv.DictReader(f)
        ]


def load_trace(path: Path) -> list[TraceEvent]:
    with path.open("r", newline="") as f:
        rows = [
            TraceEvent(
                seq=int(row["seq"]),
                tensor=row["tensor"],
                expert_idx=int(row["expert_idx"]),
                expert_bytes=int(row["expert_bytes"]),
            )
            for row in csv.DictReader(f)
        ]
    return sorted(rows, key=lambda e: e.seq)


def load_pack(path: Path) -> tuple[list[tuple[str, int]], dict[tuple[str, int], PackEntry]]:
    order: list[tuple[str, int]] = []
    entries: dict[tuple[str, int], PackEntry] = {}
    with path.open("rb") as f:
        magic, version, _header_size, n_entries, _data_start = PACK_HEADER.unpack(f.read(PACK_HEADER.size))
        if magic != PACK_MAGIC or version != 1:
            raise SystemExit(f"not a v1 MoE expert pack: {path}")
        for _ in range(n_entries):
            name_raw, expert_idx, _reserved, offset, nbytes = PACK_ENTRY.unpack(f.read(PACK_ENTRY.size))
            tensor = name_raw.split(b"\0", 1)[0].decode("utf-8")
            key = (tensor, expert_idx)
            order.append(key)
            entries[key] = PackEntry(tensor=tensor, expert_idx=expert_idx, offset=offset, nbytes=nbytes)
    return order, entries


def preload_keys(entries: list[ProfileEntry], n_slots: int) -> list[tuple[str, int]]:
    ranked = sorted(entries, key=lambda e: (e.count, e.count * e.expert_bytes, e.tensor), reverse=True)
    return [(e.tensor, e.expert_idx) for e in ranked[:n_slots]]


def protected_slot_count(slots: int, reserve_pct: int, protect: bool) -> int:
    if slots <= 0:
        return 0
    if not protect:
        return slots
    reserve = (slots * max(0, min(reserve_pct, 95)) + 99) // 100
    return max(slots - reserve, 1)


def slot_count(entries: list[ProfileEntry], events: list[TraceEvent], budget_mib: int) -> tuple[int, int]:
    slot_bytes = max([e.expert_bytes for e in entries] + [e.expert_bytes for e in events] + [0])
    slots = 0 if slot_bytes == 0 else (budget_mib * 1024 * 1024) // slot_bytes
    return min(slots, 16384), slot_bytes


def build_caches(
        profile: list[ProfileEntry],
        trace: list[TraceEvent],
        budget_mib: int,
        upgate_pct: int,
        reserve_pct: int,
        protect_profile: bool,
        policy: str) -> dict[str, CacheSim]:
    budgets = {
        "upgate": budget_mib * upgate_pct // 100,
        "down": budget_mib - (budget_mib * upgate_pct // 100),
    }
    caches: dict[str, CacheSim] = {}
    for name in ("upgate", "down"):
        entries = [e for e in profile if bucket(e.tensor) == name]
        events = [e for e in trace if bucket(e.tensor) == name]
        slots, _slot_bytes = slot_count(entries, events, budgets[name])
        cache = CacheSim(slots, policy)
        cache.preload(preload_keys(entries, protected_slot_count(slots, reserve_pct, protect_profile)))
        caches[name] = cache
    return caches


def miss_stream(
        profile: list[ProfileEntry],
        trace: list[TraceEvent],
        budget_mib: int,
        upgate_pct: int,
        reserve_pct: int,
        protect_profile: bool,
        policy: str) -> tuple[list[TraceEvent], dict[str, CacheSim]]:
    caches = build_caches(profile, trace, budget_mib, upgate_pct, reserve_pct, protect_profile, policy)
    misses: list[TraceEvent] = []
    for event in trace:
        name = bucket(event.tensor)
        if name not in caches:
            continue
        if not caches[name].access((event.tensor, event.expert_idx), event.expert_bytes):
            misses.append(event)
    return misses, caches


def grouped_tensor_runs(events: list[TraceEvent]) -> list[list[TraceEvent]]:
    runs: list[list[TraceEvent]] = []
    current: list[TraceEvent] = []
    last_tensor: str | None = None
    for event in events:
        if last_tensor is not None and event.tensor != last_tensor:
            runs.append(current)
            current = []
        current.append(event)
        last_tensor = event.tensor
    if current:
        runs.append(current)
    return runs


def coalesce_report(misses: list[TraceEvent], pack: dict[tuple[str, int], PackEntry], thresholds_mib: list[int]) -> None:
    runs = grouped_tensor_runs(misses)
    base_bytes = sum(e.expert_bytes for e in misses)
    print(f"miss_stream,runs={len(runs)},misses={len(misses)},base_gib={base_bytes / 2**30:.2f}")
    print("coalesce,threshold_mib,read_gib,extra_pct,read_calls,saved_calls,extra_gib")
    for threshold_mib in thresholds_mib:
        threshold = threshold_mib * 1024 * 1024
        total = 0
        calls = 0
        saved_calls = 0
        extra = 0
        for run in runs:
            # Accepted runtime splits stage jobs by alternating miss order across two rings.
            for parity in (0, 1):
                jobs = [event for i, event in enumerate(run) if i % 2 == parity]
                if not jobs:
                    continue
                jobs = sorted(jobs, key=lambda e: pack[(e.tensor, e.expert_idx)].offset)
                start = None
                end = None
                count = 0
                useful = 0
                for event in jobs:
                    entry = pack[(event.tensor, event.expert_idx)]
                    if start is None:
                        start = entry.offset
                        end = entry.offset + entry.nbytes
                        count = 1
                        useful = entry.nbytes
                        continue
                    assert end is not None
                    gap = entry.offset - end
                    if 0 <= gap <= threshold:
                        end = max(end, entry.offset + entry.nbytes)
                        count += 1
                        useful += entry.nbytes
                    else:
                        total += end - start
                        calls += 1
                        saved_calls += max(0, count - 1)
                        extra += (end - start) - useful
                        start = entry.offset
                        end = entry.offset + entry.nbytes
                        count = 1
                        useful = entry.nbytes
                assert start is not None and end is not None
                total += end - start
                calls += 1
                saved_calls += max(0, count - 1)
                extra += (end - start) - useful
        extra_pct = 0.0 if base_bytes == 0 else (total / base_bytes - 1.0) * 100.0
        print(
            f"coalesce,{threshold_mib},{total / 2**30:.2f},{extra_pct:.2f},"
            f"{calls},{saved_calls},{extra / 2**30:.2f}")


def assign_offsets(order: list[tuple[str, int]], pack_order: list[tuple[str, int]], pack: dict[tuple[str, int], PackEntry]) -> dict[tuple[str, int], int]:
    offsets: dict[tuple[str, int], int] = {}
    seen: set[tuple[str, int]] = set()
    pos = align_up(PACK_HEADER.size + PACK_ENTRY.size * len(pack_order))
    for key in order + pack_order:
        if key in seen:
            continue
        pos = align_up(pos)
        offsets[key] = pos
        pos += align_up(pack[key].nbytes)
        seen.add(key)
    return offsets


def locality_stats(misses: list[TraceEvent], offsets: dict[tuple[str, int], int]) -> dict[str, float]:
    jumps: list[int] = []
    backward = 0
    last = None
    for event in misses:
        offset = offsets[(event.tensor, event.expert_idx)]
        if last is not None:
            delta = offset - last
            jumps.append(abs(delta))
            if delta < 0:
                backward += 1
        last = offset
    if not jumps:
        return {k: 0.0 for k in ("total_tib", "mean_mib", "p50_mib", "p90_mib", "p99_mib", "backward_pct")}
    jumps_sorted = sorted(jumps)
    n = len(jumps_sorted)
    total = sum(jumps_sorted)
    return {
        "total_tib": total / 2**40,
        "mean_mib": (total / n) / 2**20,
        "p50_mib": jumps_sorted[n // 2] / 2**20,
        "p90_mib": jumps_sorted[(n * 90) // 100] / 2**20,
        "p99_mib": jumps_sorted[(n * 99) // 100] / 2**20,
        "backward_pct": 100.0 * backward / n,
    }


def layout_report(misses: list[TraceEvent], pack_order: list[tuple[str, int]], pack: dict[tuple[str, int], PackEntry]) -> None:
    first_miss_order: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    counts: dict[tuple[str, int], int] = {}
    for event in misses:
        key = (event.tensor, event.expert_idx)
        counts[key] = counts.get(key, 0) + 1
        if key not in seen:
            seen.add(key)
            first_miss_order.append(key)
    freq_order = sorted(counts, key=lambda k: (counts[k], k[0], k[1]), reverse=True)
    layouts = {
        "current": {key: pack[key].offset for key in pack_order},
        "first_miss": assign_offsets(first_miss_order, pack_order, pack),
        "frequency": assign_offsets(freq_order, pack_order, pack),
    }
    print("layout,name,total_tib,mean_mib,p50_mib,p90_mib,p99_mib,backward_pct")
    for name, offsets in layouts.items():
        stats = locality_stats(misses, offsets)
        print(
            f"layout,{name},{stats['total_tib']:.2f},{stats['mean_mib']:.2f},"
            f"{stats['p50_mib']:.2f},{stats['p90_mib']:.2f},{stats['p99_mib']:.2f},"
            f"{stats['backward_pct']:.2f}")


def layer_from_tensor(tensor: str) -> int | None:
    match = re.search(r"\bblk\.(\d+)\.", tensor)
    return int(match.group(1)) if match else None


def previous_route_prediction_report(
        profile: list[ProfileEntry],
        trace: list[TraceEvent],
        budget_mib: int,
        upgate_pct: int,
        reserve_pct: int,
        protect_profile: bool,
        policy: str,
        thresholds: list[float]) -> None:
    runs = grouped_tensor_runs(trace)
    up_calls: list[tuple[int, str, str, tuple[int, ...], tuple[int, ...]]] = []
    for i in range(0, len(runs) - 2, 3):
        up = runs[i]
        gate = runs[i + 1]
        if not up or not gate or ".ffn_up_exps." not in up[0].tensor or ".ffn_gate_exps." not in gate[0].tensor:
            continue
        layer = layer_from_tensor(up[0].tensor)
        if layer is None:
            continue
        up_calls.append((
            layer,
            up[0].tensor,
            gate[0].tensor,
            tuple(e.expert_idx for e in up),
            tuple(e.expert_idx for e in gate),
        ))

    prev: dict[int, set[int]] = {}
    layer_hits: dict[int, list[float]] = {}
    for layer, _up_tensor, _gate_tensor, up_experts, _gate_experts in up_calls:
        current = set(up_experts)
        if layer in prev and current:
            layer_hits.setdefault(layer, []).append(len(current & prev[layer]) / len(current))
        prev[layer] = current
    layer_score = {layer: sum(values) / len(values) for layer, values in layer_hits.items()}

    print("predict_prev_route,threshold,layers,pred_reads,useful_reads,waste_reads,precision,miss_cover,actual_misses")
    for threshold in thresholds:
        caches = build_caches(profile, trace, budget_mib, upgate_pct, reserve_pct, protect_profile, policy)
        upgate_cache = caches["upgate"]
        prev_experts: dict[int, set[int]] = {}
        layers_used: set[int] = set()
        pred_reads = 0
        useful_reads = 0
        actual_misses = 0
        for layer, up_tensor, gate_tensor, up_experts, gate_experts in up_calls:
            predicted: list[tuple[str, int]] = []
            if layer in prev_experts and layer_score.get(layer, 0.0) >= threshold:
                layers_used.add(layer)
                for expert in prev_experts[layer]:
                    predicted.append((up_tensor, expert))
                    predicted.append((gate_tensor, expert))
                predicted = [key for key in predicted if not upgate_cache.contains(key)]
                pred_reads += len(predicted)

            actual_keys = [(up_tensor, e) for e in up_experts] + [(gate_tensor, e) for e in gate_experts]
            actual_miss_keys = {key for key in actual_keys if not upgate_cache.contains(key)}
            useful_reads += sum(1 for key in predicted if key in actual_miss_keys)
            for key in actual_keys:
                if not upgate_cache.access(key, 0):
                    actual_misses += 1
            prev_experts[layer] = set(up_experts)

        precision = useful_reads / pred_reads if pred_reads else 0.0
        miss_cover = useful_reads / actual_misses if actual_misses else 0.0
        print(
            f"predict_prev_route,{threshold:.2f},{len(layers_used)},{pred_reads},"
            f"{useful_reads},{pred_reads - useful_reads},{precision:.3f},"
            f"{miss_cover:.3f},{actual_misses}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze MoE expert-pack read scheduling mechanisms from a route trace.")
    parser.add_argument("--profile", type=Path, required=True, help="Route profile CSV used for protected hotset preload.")
    parser.add_argument("--trace", type=Path, required=True, help="Route trace CSV from GGML_MOE_ROUTE_TRACE_OUT.")
    parser.add_argument("--expert-pack", type=Path, required=True, help="MoE expert pack to analyze.")
    parser.add_argument("--budget-mib", type=int, default=2048)
    parser.add_argument("--upgate-pct", type=int, default=50)
    parser.add_argument("--reserve-pct", type=int, default=20)
    parser.add_argument("--protect-profile", action="store_true")
    parser.add_argument("--policy", choices=("lru", "lfu_lru"), default="lfu_lru")
    parser.add_argument("--coalesce-threshold-mib", type=int, nargs="*", default=[0, 4, 8, 16, 32, 64])
    parser.add_argument("--prediction-threshold", type=float, nargs="*", default=[0.0, 0.25, 0.35, 0.45, 0.5, 0.6])
    args = parser.parse_args()

    profile = load_profile(args.profile)
    trace = load_trace(args.trace)
    pack_order, pack = load_pack(args.expert_pack)
    misses, caches = miss_stream(
        profile,
        trace,
        args.budget_mib,
        args.upgate_pct,
        args.reserve_pct,
        args.protect_profile,
        args.policy,
    )

    print(
        "cache,"
        + ",".join(
            f"{name}_hits={cache.hits},{name}_misses={cache.misses},{name}_miss_gib={cache.miss_bytes / 2**30:.2f}"
            for name, cache in caches.items()
        )
    )
    coalesce_report(misses, pack, args.coalesce_threshold_mib)
    layout_report(misses, pack_order, pack)
    previous_route_prediction_report(
        profile,
        trace,
        args.budget_mib,
        args.upgate_pct,
        args.reserve_pct,
        args.protect_profile,
        args.policy,
        args.prediction_threshold,
    )


if __name__ == "__main__":
    main()
