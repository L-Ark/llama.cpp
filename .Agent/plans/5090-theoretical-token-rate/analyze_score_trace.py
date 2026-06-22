#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from simulate_cache_policy import Cache, Key, cache_id, load_profile, load_trace, preload_from_profile


def load_ram_keys(path: Path | None) -> set[Key]:
    if path is None:
        return set()
    keys: set[Key] = set()
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            keys.add(Key(row["tensor"], int(row["expert_idx"]), int(row["expert_bytes"])))
    return keys


def layer_from_tensor(tensor: str) -> int:
    if not tensor.startswith("blk."):
        return -1
    try:
        return int(tensor.split(".", 2)[1])
    except (IndexError, ValueError):
        return -1


def score_groups(path: Path) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    cur: list[dict[str, object]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            cur.append(row)
            if len(cur) != 8:
                continue
            layer = int(cur[0]["layer"])
            experts = [int(r["expert_idx"]) for r in cur]
            weights = [float(r["weight"]) for r in cur]
            sort_scores = [float(r["sort_score"]) for r in cur]
            groups.append({
                "layer": layer,
                "experts": experts,
                "weights": weights,
                "sort_scores": sort_scores,
                "max_weight": max(weights),
                "max_sort_score": max(sort_scores),
            })
            cur = []
    return groups


def route_groups(trace: list[Key]) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    i = 0
    while i + 24 <= len(trace):
        chunk = trace[i:i + 24]
        layers = {layer_from_tensor(k.tensor) for k in chunk}
        if len(layers) != 1:
            i += 1
            continue
        up = chunk[0:8]
        gate = chunk[8:16]
        down = chunk[16:24]
        if not all(".ffn_up_exps." in k.tensor for k in up) or \
                not all(".ffn_gate_exps." in k.tensor for k in gate) or \
                not all(".ffn_down_exps." in k.tensor for k in down):
            i += 1
            continue
        experts = [k.expert_idx for k in up]
        if [k.expert_idx for k in gate] != experts or [k.expert_idx for k in down] != experts:
            i += 1
            continue
        groups.append({"trace_index": i, "layer": next(iter(layers)), "experts": experts, "events": chunk})
        i += 24
    return groups


def setup_caches(args: argparse.Namespace) -> tuple[Cache, Cache, Counter[Key], int, int]:
    profile_rows = load_profile(args.profile)
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
    return down, upgate, profile_counts, down_preload, upgate_preload


def bucket_init() -> dict[str, int]:
    return {"events": 0, "vram": 0, "ram": 0, "ssd": 0}


def bucket_add(bucket: dict[str, int], status: str) -> None:
    bucket["events"] += 1
    bucket[status] += 1


def pct(num: int, den: int) -> float:
    return round(100.0 * num / max(1, den), 3)


def analyze(args: argparse.Namespace) -> dict[str, object]:
    scores = score_groups(args.score_trace)
    trace = load_trace(args.route_trace)
    routes = route_groups(trace)
    ram_keys = load_ram_keys(args.ram_tier_profile)
    down, upgate, _profile_counts, down_preload, upgate_preload = setup_caches(args)

    score_cursor = 0
    matched_groups = 0
    unmatched_groups = 0
    total = bucket_init()
    by_rank = {str(i): bucket_init() for i in range(8)}
    by_weight_ratio = {name: bucket_init() for name in ["lt_0.05", "0.05_0.10", "0.10_0.20", "0.20_0.40", "0.40_0.70", "ge_0.70"]}
    drop_rules: dict[str, dict[str, int]] = {}
    for keep in range(1, 8):
        drop_rules[f"keep_rank_lt_{keep}"] = {"dropped": 0, "saved_ssd": 0, "dropped_resident": 0}
    for threshold in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]:
        drop_rules[f"weight_ratio_lt_{threshold:.2f}"] = {"dropped": 0, "saved_ssd": 0, "dropped_resident": 0}
    for threshold in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]:
        drop_rules[f"ssd_and_weight_ratio_lt_{threshold:.2f}"] = {"dropped": 0, "saved_ssd": 0, "dropped_resident": 0}

    for event_idx, key in enumerate(trace):
        cache = upgate if cache_id(key) == "upgate" else down
        if cache.access(key, event_idx):
            status = "vram"
        elif key in ram_keys:
            status = "ram"
        else:
            status = "ssd"
        bucket_add(total, status)

    # Re-run caches so group analysis uses the same status sequence.
    down, upgate, _profile_counts, _down_preload, _upgate_preload = setup_caches(args)
    trace_event_idx = 0
    for route in routes:
        route_experts = route["experts"]
        score_match = None
        for j in range(score_cursor, min(len(scores), score_cursor + args.match_window)):
            cand = scores[j]
            if cand["layer"] == route["layer"] and sorted(cand["experts"]) == sorted(route_experts):
                score_match = cand
                score_cursor = j + 1
                break
        if score_match is None:
            unmatched_groups += 1
            for key in route["events"]:
                cache = upgate if cache_id(key) == "upgate" else down
                cache.access(key, trace_event_idx)
                trace_event_idx += 1
            continue
        matched_groups += 1
        rank_by_expert = {expert: rank for rank, expert in enumerate(score_match["experts"])}
        weight_by_expert = {expert: score_match["weights"][rank] for rank, expert in enumerate(score_match["experts"])}
        max_weight = max(float(score_match["max_weight"]), 1e-12)

        for key in route["events"]:
            cache = upgate if cache_id(key) == "upgate" else down
            if cache.access(key, trace_event_idx):
                status = "vram"
            elif key in ram_keys:
                status = "ram"
            else:
                status = "ssd"
            trace_event_idx += 1

            rank = rank_by_expert.get(key.expert_idx, -1)
            if rank >= 0:
                bucket_add(by_rank[str(rank)], status)
            ratio = weight_by_expert.get(key.expert_idx, 0.0) / max_weight
            if ratio < 0.05:
                bucket_add(by_weight_ratio["lt_0.05"], status)
            elif ratio < 0.10:
                bucket_add(by_weight_ratio["0.05_0.10"], status)
            elif ratio < 0.20:
                bucket_add(by_weight_ratio["0.10_0.20"], status)
            elif ratio < 0.40:
                bucket_add(by_weight_ratio["0.20_0.40"], status)
            elif ratio < 0.70:
                bucket_add(by_weight_ratio["0.40_0.70"], status)
            else:
                bucket_add(by_weight_ratio["ge_0.70"], status)

            for keep in range(1, 8):
                if rank >= keep:
                    d = drop_rules[f"keep_rank_lt_{keep}"]
                    d["dropped"] += 1
                    if status == "ssd":
                        d["saved_ssd"] += 1
                    else:
                        d["dropped_resident"] += 1
            for threshold in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]:
                if ratio < threshold:
                    d = drop_rules[f"weight_ratio_lt_{threshold:.2f}"]
                    d["dropped"] += 1
                    if status == "ssd":
                        d["saved_ssd"] += 1
                    else:
                        d["dropped_resident"] += 1
                    d2 = drop_rules[f"ssd_and_weight_ratio_lt_{threshold:.2f}"]
                    d2["dropped"] += 1 if status == "ssd" else 0
                    d2["saved_ssd"] += 1 if status == "ssd" else 0
                    d2["dropped_resident"] += 0

    for bucket in list(by_rank.values()) + list(by_weight_ratio.values()):
        bucket["ssd_pct"] = pct(bucket["ssd"], bucket["events"])
        bucket["resident_pct"] = pct(bucket["vram"] + bucket["ram"], bucket["events"])
    for d in drop_rules.values():
        d["saved_ssd_pct_of_current_ssd"] = pct(d["saved_ssd"], total["ssd"])
        d["resident_drop_pct"] = pct(d["dropped_resident"], d["dropped"])

    return {
        "score_trace": str(args.score_trace),
        "route_trace": str(args.route_trace),
        "profile": str(args.profile),
        "ram_tier_profile": str(args.ram_tier_profile) if args.ram_tier_profile else None,
        "score_groups": len(scores),
        "route_groups": len(routes),
        "matched_groups": matched_groups,
        "unmatched_groups": unmatched_groups,
        "preloads": {"down": down_preload, "upgate": upgate_preload},
        "total": total,
        "by_rank": by_rank,
        "by_weight_ratio": by_weight_ratio,
        "drop_rules": drop_rules,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-trace", type=Path, required=True)
    parser.add_argument("--route-trace", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--ram-tier-profile", type=Path)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--match-window", type=int, default=512)
    parser.add_argument("--upgate-slots", type=int, default=1793)
    parser.add_argument("--down-slots", type=int, default=1000)
    parser.add_argument("--reserve-pct", type=int, default=10)
    parser.add_argument(
        "--policy",
        choices=["lru", "lfu_lru", "profile_lfu_lru", "hybrid_profile_lfu_lru"],
        default="lfu_lru",
    )
    parser.add_argument("--admit-min-count", type=int, default=1)
    parser.add_argument("--protect-min-count", type=int, default=1)
    parser.add_argument("--profile-after", type=int, default=12624)
    parser.add_argument("--ghost-admit-after", type=int, default=1)
    args = parser.parse_args()
    result = analyze(args)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
