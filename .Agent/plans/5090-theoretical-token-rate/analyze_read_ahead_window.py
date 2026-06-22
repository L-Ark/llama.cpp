#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Key:
    tensor: str
    expert_idx: int
    expert_bytes: int


def load_trace(path: Path) -> list[tuple[int, Key]]:
    rows: list[tuple[int, Key]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((
                int(row["seq"]),
                Key(row["tensor"], int(row["expert_idx"]), int(row["expert_bytes"])),
            ))
    return rows


def bucket(n: int) -> str:
    if n <= 4:
        return "1-4"
    if n <= 16:
        return "5-16"
    if n <= 64:
        return "17-64"
    if n <= 256:
        return "65-256"
    if n <= 1024:
        return "257-1024"
    return "gt1024"


def simulate(entries: list[tuple[int, Key]], windows: list[int]) -> dict[str, object]:
    results = {}
    for window in windows:
        ready: Counter[Key] = Counter()
        q: deque[Key] = deque()
        hits = 0
        misses = 0
        evictions = 0
        max_ready_bytes = 0
        ready_bytes = 0
        duplicate_prefetches = 0
        for i, (_, key) in enumerate(entries):
            j = i + window
            if j < len(entries):
                future_key = entries[j][1]
                if ready[future_key] > 0:
                    duplicate_prefetches += 1
                ready[future_key] += 1
                q.append(future_key)
                ready_bytes += future_key.expert_bytes
                if ready_bytes > max_ready_bytes:
                    max_ready_bytes = ready_bytes

            if ready[key] > 0:
                ready[key] -= 1
                ready_bytes -= key.expert_bytes
                hits += 1
            else:
                misses += 1

            while len(q) > window:
                old = q.popleft()
                if ready[old] > 0:
                    ready[old] -= 1
                    ready_bytes -= old.expert_bytes
                    evictions += 1

        results[str(window)] = {
            "hits": hits,
            "misses": misses,
            "hit_rate_pct": round(100.0 * hits / len(entries), 2) if entries else 0.0,
            "evictions": evictions,
            "duplicate_prefetches": duplicate_prefetches,
            "max_ready_mib": round(max_ready_bytes / (1024.0 * 1024.0), 2),
        }
    return results


def distance_summary(entries: list[tuple[int, Key]]) -> dict[str, object]:
    next_pos: dict[Key, list[int]] = {}
    for i in range(len(entries) - 1, -1, -1):
        next_pos.setdefault(entries[i][1], []).append(i)
    counts: Counter[str] = Counter()
    by_type: dict[str, Counter[str]] = {"upgate": Counter(), "down": Counter()}
    for i, (_, key) in enumerate(entries):
        positions = next_pos[key]
        while positions and positions[-1] <= i:
            positions.pop()
        if positions:
            dist = positions[-1] - i
            b = bucket(dist)
        else:
            b = "never"
        kind = "down" if "ffn_down" in key.tensor else "upgate"
        counts[b] += 1
        by_type[kind][b] += 1
    return {
        "all": dict(counts),
        "by_type": {k: dict(v) for k, v in by_type.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--windows", default="1,2,4,8,16,32,64,128,256")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    entries = load_trace(args.trace)
    windows = [int(x) for x in args.windows.split(",") if x]
    result = {
        "trace": str(args.trace),
        "events": len(entries),
        "distance_summary": distance_summary(entries),
        "window_sim": simulate(entries, windows),
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
