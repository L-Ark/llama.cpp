#!/usr/bin/env python3
"""Screen RAM-tier candidates across dev prompts before runtime A/B.

The GP169 RAM-tier experiments showed that static weighted bytes from one trace
are not enough.  This tool aggregates foreground io-read traces across dev
prompts and scores RAM candidates by:

- prompt coverage;
- actual decode/runtime rows and bytes;
- current VRAM overlap;
- source_idx distribution;
- batch coverage, including RAM-dominant batches.

It intentionally does not claim performance.  Its job is to reject weak RAM
profiles before they reach a cold-start model run.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


MIB = 1024 * 1024


@dataclass(frozen=True)
class Key:
    tensor: str
    expert_idx: int
    expert_bytes: int

    @property
    def stable_id(self) -> tuple[str, int]:
        return (self.tensor, self.expert_idx)

    @property
    def role(self) -> str:
        if ".ffn_down_exps." in self.tensor:
            return "down"
        if ".ffn_up_exps." in self.tensor:
            return "up"
        if ".ffn_gate_exps." in self.tensor:
            return "gate"
        return "other"

    @property
    def layer(self) -> int:
        m = re.search(r"blk\.(\d+)\.", self.tensor)
        return int(m.group(1)) if m else -1


@dataclass
class CandidateStat:
    key: Key
    rows: int = 0
    bytes: int = 0
    prompts: set[str] = field(default_factory=set)
    sources: Counter[int] = field(default_factory=Counter)
    source_bytes: Counter[int] = field(default_factory=Counter)
    batches: set[tuple[str, int]] = field(default_factory=set)
    jobs_hist: Counter[int] = field(default_factory=Counter)

    @property
    def prompt_count(self) -> int:
        return len(self.prompts)


@dataclass
class BatchStat:
    prompt_id: str
    batch_seq: int
    jobs: int
    total_rows: int = 0
    candidate_rows: int = 0
    total_bytes: int = 0
    candidate_bytes: int = 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--io-read-trace", action="append", type=Path, default=[],
                    help="Path to an io-read-trace.csv. Repeatable.")
    ap.add_argument("--input-root", action="append", type=Path, default=[],
                    help="Directory containing */io-read-trace.csv. Repeatable.")
    ap.add_argument("--route-profile", action="append", type=Path, default=[],
                    help="Route profile used to simulate current VRAM entries. Repeatable.")
    ap.add_argument("--out-profile", required=True, type=Path)
    ap.add_argument("--out-report", required=True, type=Path)
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--budget-mib", type=int, required=True)
    ap.add_argument("--roles", default="down")
    ap.add_argument("--max-jobs", type=int, default=8)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--min-prompts", type=int, default=1)
    ap.add_argument("--down-slots", type=int, default=0)
    ap.add_argument("--upgate-slots", type=int, default=0)
    ap.add_argument("--split-max-mib", type=float, default=6.0)
    ap.add_argument("--exclude-vram", action="store_true")
    ap.add_argument("--allowed-source-idx", default="")
    ap.add_argument("--min-batch-hits", type=int, default=4)
    ap.add_argument("--min-batch-hit-pct", type=float, default=75.0)
    ap.add_argument("--exclude-prompt-regex", default=r"heldout|test_")
    ap.add_argument("--top", type=int, default=120)
    return ap.parse_args()


def role_cache_id(key: Key, split_max_mib: float) -> str:
    if key.role in {"up", "gate"} and key.expert_bytes <= split_max_mib * MIB:
        return "upgate"
    return "down"


def prompt_id_for_trace(path: Path) -> str:
    if path.name == "io-read-trace.csv":
        return path.parent.name
    return path.stem


def find_io_traces(args: argparse.Namespace) -> list[Path]:
    paths = list(args.io_read_trace)
    for root in args.input_root:
        if root.is_file() and root.name == "io-read-trace.csv":
            paths.append(root)
        elif root.is_dir():
            paths.extend(sorted(root.glob("*/io-read-trace.csv")))
    seen: set[Path] = set()
    out: list[Path] = []
    exclude_re = re.compile(args.exclude_prompt_regex)
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        prompt = prompt_id_for_trace(path)
        if exclude_re.search(prompt):
            raise SystemExit(f"Refusing held-out/test-looking trace: {path}")
        out.append(path)
    if not out:
        raise SystemExit("No io-read-trace.csv inputs found")
    return out


def load_vram_ids(args: argparse.Namespace) -> set[tuple[str, int]]:
    if not args.route_profile:
        return set()
    limits = {"down": args.down_slots, "upgate": args.upgate_slots}
    if args.exclude_vram and args.down_slots <= 0 and args.upgate_slots <= 0:
        raise SystemExit("--exclude-vram requires --down-slots and/or --upgate-slots")

    aggregate: dict[tuple[str, int], tuple[Key, int]] = {}
    selected: set[tuple[str, int]] = set()
    for path in args.route_profile:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tensor = row.get("tensor", "")
                if not tensor:
                    continue
                key = Key(tensor=tensor, expert_idx=int(row["expert_idx"]), expert_bytes=int(row["expert_bytes"]))
                count = int(row.get("count", "0") or "0")
                stable_id = key.stable_id
                prev = aggregate.get(stable_id)
                aggregate[stable_id] = (key, count + (prev[1] if prev else 0))

    ranked = sorted(
        aggregate.values(),
        key=lambda item: (item[0].expert_bytes * item[1], item[1]),
        reverse=True,
    )
    seen = {"down": 0, "upgate": 0}
    for key, _count in ranked:
        cid = role_cache_id(key, args.split_max_mib)
        if limits.get(cid, 0) <= 0 or seen[cid] >= limits[cid]:
            continue
        stable_id = key.stable_id
        if stable_id in selected:
            continue
        selected.add(stable_id)
        seen[cid] += 1
    return selected


def row_key(row: dict[str, str]) -> Key:
    return Key(tensor=row["tensor"], expert_idx=int(row["expert_idx"]), expert_bytes=int(row["nbytes"]))


def load_stats(
    traces: list[Path],
    *,
    roles: set[str],
    max_jobs: int,
    allowed_sources: set[int],
    vram_ids: set[tuple[str, int]],
    exclude_vram: bool,
) -> tuple[dict[tuple[str, int], CandidateStat], dict[tuple[str, int], BatchStat], dict[str, object]]:
    stats: dict[tuple[str, int], CandidateStat] = {}
    batches: dict[tuple[str, int], BatchStat] = {}
    counters: Counter[str] = Counter()
    source_rows: Counter[int] = Counter()
    source_bytes: Counter[int] = Counter()
    vram_overlap_rows = 0
    vram_overlap_bytes = 0

    for path in traces:
        prompt = prompt_id_for_trace(path)
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("jobs") == "jobs" or not row.get("tensor"):
                    continue
                counters["rows_total"] += 1
                jobs = int(row["jobs"])
                if max_jobs > 0 and jobs > max_jobs:
                    counters["rows_above_max_jobs"] += 1
                    continue
                key = row_key(row)
                if key.role not in roles:
                    counters["rows_role_skipped"] += 1
                    continue
                source_idx = int(row.get("source_idx", "0") or "0")
                nbytes = int(row["nbytes"])
                source_rows[source_idx] += 1
                source_bytes[source_idx] += nbytes
                if allowed_sources and source_idx not in allowed_sources:
                    counters["rows_source_skipped"] += 1
                    continue

                stable_id = key.stable_id
                in_vram = stable_id in vram_ids
                if in_vram:
                    vram_overlap_rows += 1
                    vram_overlap_bytes += nbytes
                    if exclude_vram:
                        counters["rows_vram_skipped"] += 1
                        continue

                batch_id = (prompt, int(row["batch_seq"]))
                batch = batches.get(batch_id)
                if batch is None:
                    batch = BatchStat(prompt_id=prompt, batch_seq=batch_id[1], jobs=jobs)
                    batches[batch_id] = batch
                batch.total_rows += 1
                batch.total_bytes += nbytes

                stat = stats.get(stable_id)
                if stat is None:
                    stat = CandidateStat(key=key)
                    stats[stable_id] = stat
                stat.rows += 1
                stat.bytes += nbytes
                stat.prompts.add(prompt)
                stat.sources[source_idx] += 1
                stat.source_bytes[source_idx] += nbytes
                stat.batches.add(batch_id)
                stat.jobs_hist[jobs] += 1

    metadata = {
        "counters": dict(counters),
        "source_rows": dict(sorted(source_rows.items())),
        "source_bytes": dict(sorted(source_bytes.items())),
        "vram_overlap_rows": vram_overlap_rows,
        "vram_overlap_bytes": vram_overlap_bytes,
    }
    return stats, batches, metadata


def select_candidates(
    stats: dict[tuple[str, int], CandidateStat],
    *,
    budget_mib: int,
    min_count: int,
    min_prompts: int,
    prompt_total: int,
) -> list[CandidateStat]:
    candidates = [
        stat for stat in stats.values()
        if stat.rows >= min_count and stat.prompt_count >= min_prompts
    ]

    def score(stat: CandidateStat) -> tuple[float, int, int]:
        prompt_weight = stat.prompt_count / max(1, prompt_total)
        reuse = stat.bytes / max(1, stat.key.expert_bytes)
        return (reuse * (0.5 + prompt_weight), stat.bytes, stat.rows)

    candidates.sort(key=score, reverse=True)
    budget = budget_mib * MIB
    selected: list[CandidateStat] = []
    used = 0
    for stat in candidates:
        if used + stat.key.expert_bytes > budget:
            continue
        selected.append(stat)
        used += stat.key.expert_bytes
    return selected


def apply_selection_to_batches(
    selected: list[CandidateStat],
    batches: dict[tuple[str, int], BatchStat],
) -> dict[str, object]:
    selected_ids = {stat.key.stable_id for stat in selected}
    # Reconstruct from per-stat batch membership.  We only need selected rows,
    # bytes, and batch coverage summary for offline gating.
    for stat in selected:
        for batch_id in stat.batches:
            batch = batches.get(batch_id)
            if batch is None:
                continue
            # Approximate row/byte contribution by distributing stat bytes over
            # its batch appearances.  Exact row accounting is not needed for the
            # gate; hit count and dominance are conservative enough here.
            batch.candidate_rows += 1
            batch.candidate_bytes += stat.key.expert_bytes

    dominant = 0
    ram_only = 0
    hit_batches = 0
    hit_candidate_bytes = 0
    dominant_candidate_bytes = 0
    ram_only_candidate_bytes = 0
    rows_hist: Counter[int] = Counter()
    pct_hist: Counter[str] = Counter()
    for batch in batches.values():
        if batch.candidate_rows <= 0:
            continue
        hit_batches += 1
        hit_candidate_bytes += batch.candidate_bytes
        rows_hist[batch.candidate_rows] += 1
        pct = 100.0 * batch.candidate_rows / max(1, batch.total_rows)
        if batch.candidate_rows >= ARGS.min_batch_hits and pct >= ARGS.min_batch_hit_pct:
            dominant += 1
            dominant_candidate_bytes += batch.candidate_bytes
        if batch.candidate_rows >= batch.total_rows:
            ram_only += 1
            ram_only_candidate_bytes += batch.candidate_bytes
        if pct >= 75:
            pct_hist[">=75"] += 1
        elif pct >= 50:
            pct_hist["50-75"] += 1
        elif pct >= 25:
            pct_hist["25-50"] += 1
        else:
            pct_hist["<25"] += 1

    return {
        "selected_id_count": len(selected_ids),
        "total_batches": len(batches),
        "hit_batches": hit_batches,
        "ram_dominant_batches": dominant,
        "ram_only_batches": ram_only,
        "hit_candidate_bytes": hit_candidate_bytes,
        "hit_candidate_gib": hit_candidate_bytes / (1024 ** 3),
        "ram_dominant_candidate_bytes": dominant_candidate_bytes,
        "ram_dominant_candidate_gib": dominant_candidate_bytes / (1024 ** 3),
        "ram_only_candidate_bytes": ram_only_candidate_bytes,
        "ram_only_candidate_gib": ram_only_candidate_bytes / (1024 ** 3),
        "candidate_rows_hist": dict(sorted(rows_hist.items())),
        "candidate_pct_hist": dict(sorted(pct_hist.items())),
    }


def write_profile(path: Path, selected: list[CandidateStat]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cumulative = 0
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "count", "expert_bytes", "cumulative_bytes", "tensor_base", "expert_idx", "tensor"])
        for rank, stat in enumerate(selected, 1):
            cumulative += stat.key.expert_bytes
            writer.writerow([rank, stat.rows, stat.key.expert_bytes, cumulative, "0x0", stat.key.expert_idx, stat.key.tensor])


def write_candidate_csv(path: Path, selected: list[CandidateStat]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "rank", "tensor", "expert_idx", "role", "layer", "expert_bytes",
            "rows", "bytes", "prompts", "source_rows", "source_bytes", "jobs_hist",
        ])
        writer.writeheader()
        for rank, stat in enumerate(selected, 1):
            writer.writerow({
                "rank": rank,
                "tensor": stat.key.tensor,
                "expert_idx": stat.key.expert_idx,
                "role": stat.key.role,
                "layer": stat.key.layer,
                "expert_bytes": stat.key.expert_bytes,
                "rows": stat.rows,
                "bytes": stat.bytes,
                "prompts": ",".join(sorted(stat.prompts)),
                "source_rows": json.dumps(dict(sorted(stat.sources.items())), sort_keys=True),
                "source_bytes": json.dumps(dict(sorted(stat.source_bytes.items())), sort_keys=True),
                "jobs_hist": json.dumps(dict(sorted(stat.jobs_hist.items())), sort_keys=True),
            })


def main() -> int:
    global ARGS
    ARGS = parse_args()
    roles = {x.strip() for x in ARGS.roles.split(",") if x.strip()}
    allowed_sources = {int(x) for x in ARGS.allowed_source_idx.split(",") if x.strip()} if ARGS.allowed_source_idx.strip() else set()
    traces = find_io_traces(ARGS)
    vram_ids = load_vram_ids(ARGS)
    stats, batches, metadata = load_stats(
        traces,
        roles=roles,
        max_jobs=ARGS.max_jobs,
        allowed_sources=allowed_sources,
        vram_ids=vram_ids,
        exclude_vram=ARGS.exclude_vram,
    )
    prompts = sorted({prompt_id_for_trace(path) for path in traces})
    selected = select_candidates(
        stats,
        budget_mib=ARGS.budget_mib,
        min_count=ARGS.min_count,
        min_prompts=ARGS.min_prompts,
        prompt_total=len(prompts),
    )
    batch_summary = apply_selection_to_batches(selected, batches)
    selected_bytes = sum(stat.key.expert_bytes for stat in selected)
    selected_weighted_bytes = sum(stat.bytes for stat in selected)
    selected_source_rows: Counter[int] = Counter()
    selected_source_bytes: Counter[int] = Counter()
    for stat in selected:
        selected_source_rows.update(stat.sources)
        selected_source_bytes.update(stat.source_bytes)

    report = {
        "kind": "kimi_ram_candidate_multidev_screen",
        "traces": [str(path) for path in traces],
        "prompts": prompts,
        "roles": sorted(roles),
        "budget_mib": ARGS.budget_mib,
        "max_jobs": ARGS.max_jobs,
        "min_count": ARGS.min_count,
        "min_prompts": ARGS.min_prompts,
        "allowed_source_idx": sorted(allowed_sources),
        "exclude_vram": ARGS.exclude_vram,
        "route_profiles": [str(path) for path in ARGS.route_profile],
        "vram_ids": len(vram_ids),
        "metadata": metadata,
        "selected_entries": len(selected),
        "selected_bytes": selected_bytes,
        "selected_mib": selected_bytes / MIB,
        "selected_weighted_bytes": selected_weighted_bytes,
        "selected_weighted_gib": selected_weighted_bytes / (1024 ** 3),
        "selected_source_rows": dict(sorted(selected_source_rows.items())),
        "selected_source_bytes": dict(sorted(selected_source_bytes.items())),
        "batch_summary": batch_summary,
        "top_entries": [
            {
                "rank": i + 1,
                "tensor": stat.key.tensor,
                "expert_idx": stat.key.expert_idx,
                "role": stat.key.role,
                "layer": stat.key.layer,
                "expert_bytes": stat.key.expert_bytes,
                "rows": stat.rows,
                "bytes": stat.bytes,
                "prompts": sorted(stat.prompts),
                "sources": dict(sorted(stat.sources.items())),
                "source_bytes": dict(sorted(stat.source_bytes.items())),
            }
            for i, stat in enumerate(selected[: ARGS.top])
        ],
    }

    write_profile(ARGS.out_profile, selected)
    write_candidate_csv(ARGS.out_csv, selected)
    ARGS.out_report.parent.mkdir(parents=True, exist_ok=True)
    ARGS.out_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"traces={len(traces)} prompts={len(prompts)} candidates={len(stats)}")
    print(f"selected entries={len(selected)} mib={selected_bytes/MIB:.2f} weighted_gib={selected_weighted_bytes/(1024**3):.3f}")
    print(f"selected_source_rows={dict(sorted(selected_source_rows.items()))}")
    print(
        "batches "
        f"hit={batch_summary['hit_batches']} "
        f"dominant={batch_summary['ram_dominant_batches']} "
        f"only={batch_summary['ram_only_batches']} "
        f"total={batch_summary['total_batches']}"
    )
    print(f"wrote {ARGS.out_profile}")
    print(f"wrote {ARGS.out_csv}")
    print(f"wrote {ARGS.out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
