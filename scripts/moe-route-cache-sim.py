#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Entry:
    tensor: str
    expert_idx: int
    count: int
    expert_bytes: int


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate hot-expert VRAM cache coverage from GGML_MOE_BATCH_PROFILE_OUT CSV.")
    parser.add_argument("profile", type=Path)
    parser.add_argument("--budget-mib", type=int, default=16 * 1024)
    parser.add_argument("--upgate-pct", type=int, default=63)
    args = parser.parse_args()

    if args.budget_mib <= 0:
        raise SystemExit("--budget-mib must be positive")
    if args.upgate_pct < 1 or args.upgate_pct > 99:
        raise SystemExit("--upgate-pct must be 1..99")

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


if __name__ == "__main__":
    main()
