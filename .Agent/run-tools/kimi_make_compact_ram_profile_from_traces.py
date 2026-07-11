#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


MIB = 1024 * 1024


class Key(tuple):
    __slots__ = ()

    def __new__(cls, tensor: str, expert_idx: int, expert_bytes: int):
        return tuple.__new__(cls, (tensor, expert_idx, expert_bytes))

    @property
    def tensor(self) -> str:
        return self[0]

    @property
    def expert_idx(self) -> int:
        return self[1]

    @property
    def expert_bytes(self) -> int:
        return self[2]

    @property
    def layer(self) -> int:
        match = re.search(r"blk\.(\d+)\.", self.tensor)
        return int(match.group(1)) if match else -1

    @property
    def role(self) -> str:
        if ".ffn_up_exps." in self.tensor:
            return "up"
        if ".ffn_gate_exps." in self.tensor:
            return "gate"
        if ".ffn_down_exps." in self.tensor:
            return "down"
        return "other"


def csv_set(text: str) -> set[str]:
    return {part.strip() for part in text.split(",") if part.strip()}


def prompt_id_for(path: Path) -> str:
    return path.parent.name


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a compact RAM tier profile from multiple dev IO traces.")
    ap.add_argument("--io-read-trace", action="append", type=Path, required=True)
    ap.add_argument("--layers", default="", help="Comma-separated layer ids. Empty means all.")
    ap.add_argument("--roles", default="up,gate,down")
    ap.add_argument("--max-jobs", type=int, default=8)
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--min-prompts", type=int, default=1)
    ap.add_argument("--budget-mib", type=int, required=True)
    ap.add_argument("--out-profile", type=Path, required=True)
    ap.add_argument("--out-report", type=Path, required=True)
    args = ap.parse_args()

    layers = {int(x) for x in csv_set(args.layers)} if args.layers.strip() else set()
    roles = csv_set(args.roles)
    budget = args.budget_mib * MIB

    counts: Counter[Key] = Counter()
    prompt_hits: dict[Key, set[str]] = defaultdict(set)
    prompt_trace_rows: Counter[str] = Counter()
    skipped_jobs = 0
    skipped_filter = 0

    for path in args.io_read_trace:
        prompt = prompt_id_for(path)
        with path.open(newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("jobs") == "jobs" or not row.get("tensor"):
                    continue
                jobs = int(float(row.get("jobs", "0") or 0))
                if args.max_jobs > 0 and jobs > args.max_jobs:
                    skipped_jobs += 1
                    continue
                key = Key(row["tensor"], int(row["expert_idx"]), int(row["nbytes"]))
                if key.role not in roles or (layers and key.layer not in layers):
                    skipped_filter += 1
                    continue
                counts[key] += 1
                prompt_hits[key].add(prompt)
                prompt_trace_rows[prompt] += 1

    candidates = [
        (key, count)
        for key, count in counts.items()
        if count >= args.min_count and len(prompt_hits[key]) >= args.min_prompts
    ]
    candidates.sort(
        key=lambda item: (
            len(prompt_hits[item[0]]),
            item[1] * item[0].expert_bytes,
            item[1],
            -item[0].expert_bytes,
        ),
        reverse=True,
    )

    selected: list[tuple[Key, int]] = []
    used = 0
    for key, count in candidates:
        if used + key.expert_bytes > budget:
            continue
        selected.append((key, count))
        used += key.expert_bytes

    args.out_profile.parent.mkdir(parents=True, exist_ok=True)
    cumulative = 0
    with args.out_profile.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["rank", "count", "expert_bytes", "cumulative_bytes", "tensor_base", "expert_idx", "tensor"])
        for rank, (key, count) in enumerate(selected, 1):
            cumulative += key.expert_bytes
            writer.writerow([rank, count, key.expert_bytes, cumulative, "0x0", key.expert_idx, key.tensor])

    by_layer_role: dict[str, dict[str, float]] = defaultdict(lambda: {"entries": 0, "bytes": 0, "count": 0, "prompts_sum": 0})
    for key, count in selected:
        bucket = by_layer_role[f"blk.{key.layer}.{key.role}"]
        bucket["entries"] += 1
        bucket["bytes"] += key.expert_bytes
        bucket["count"] += count
        bucket["prompts_sum"] += len(prompt_hits[key])

    report = {
        "kind": "kimi_compact_ram_profile_from_traces",
        "inputs": [str(p) for p in args.io_read_trace],
        "prompt_count": len({prompt_id_for(p) for p in args.io_read_trace}),
        "prompt_trace_rows": dict(sorted(prompt_trace_rows.items())),
        "layers": sorted(layers) if layers else "all",
        "roles": sorted(roles),
        "max_jobs": args.max_jobs,
        "min_count": args.min_count,
        "min_prompts": args.min_prompts,
        "budget_mib": args.budget_mib,
        "skipped_jobs": skipped_jobs,
        "skipped_filter": skipped_filter,
        "candidate_entries": len(candidates),
        "selected_entries": len(selected),
        "selected_bytes": used,
        "selected_mib": used / MIB,
        "by_layer_role": by_layer_role,
        "top_entries": [
            {
                "rank": idx + 1,
                "tensor": key.tensor,
                "expert_idx": key.expert_idx,
                "expert_bytes": key.expert_bytes,
                "count": count,
                "prompts": sorted(prompt_hits[key]),
                "prompt_count": len(prompt_hits[key]),
                "traffic_bytes": key.expert_bytes * count,
            }
            for idx, (key, count) in enumerate(selected[:100])
        ],
        "out_profile": str(args.out_profile),
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out_profile} entries={len(selected)} mib={used / MIB:.2f}")
    print(f"wrote {args.out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
