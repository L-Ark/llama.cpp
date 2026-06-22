#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import struct
from dataclasses import dataclass
from pathlib import Path


ALIGNMENT = 4096


@dataclass(frozen=True)
class Entry:
    tensor: str
    expert_idx: int
    offset: int
    nbytes: int


@dataclass(frozen=True)
class TraceKey:
    tensor: str
    expert_idx: int
    expert_bytes: int


def read_pack_index(path: Path) -> dict[tuple[str, int, int], Entry]:
    out: dict[tuple[str, int, int], Entry] = {}
    with path.open("rb") as f:
        header = f.read(40)
        if len(header) < 40:
            raise ValueError("pack too small")
        magic, version, header_size, n_entries, _data_start = struct.unpack("<16sIIQQ", header)
        if not magic.startswith(b"GGMLMOEPACKv1") or version != 1 or header_size < 40:
            raise ValueError("unsupported expert pack header")
        entry_size = 128 + 4 + 4 + 8 + 8
        for _ in range(n_entries):
            raw = f.read(entry_size)
            if len(raw) != entry_size:
                raise ValueError("truncated expert pack index")
            raw_tensor = raw[:128]
            expert_idx, reserved = struct.unpack_from("<ii", raw, 128)
            offset, nbytes = struct.unpack_from("<QQ", raw, 136)
            if reserved != 0:
                continue
            tensor = raw_tensor.split(b"\0", 1)[0].decode("utf-8", errors="replace")
            entry = Entry(tensor=tensor, expert_idx=expert_idx, offset=offset, nbytes=nbytes)
            out[(tensor, expert_idx, nbytes)] = entry
    return out


def read_trace(path: Path) -> list[TraceKey]:
    rows: list[TraceKey] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(TraceKey(row["tensor"], int(row["expert_idx"]), int(row["expert_bytes"])))
    return rows


def analyze_group(entries: list[Entry]) -> dict[str, int]:
    if not entries:
        return {"jobs": 0, "runs": 0, "coalesced_jobs": 0, "extra_bytes": 0}
    ordered = sorted(entries, key=lambda e: e.offset)
    runs = 1
    coalesced_jobs = 0
    extra_bytes = 0
    run_start = ordered[0].offset
    run_end = ordered[0].offset + ((ordered[0].nbytes + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
    run_jobs = 1
    for entry in ordered[1:]:
        read_sz = ((entry.nbytes + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
        if entry.offset == run_end:
            run_end += read_sz
            run_jobs += 1
            continue
        if run_jobs > 1:
            coalesced_jobs += run_jobs
            extra_bytes += (run_end - run_start) - sum(
                ((e.nbytes + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
                for e in ordered
                if run_start <= e.offset < run_end
            )
        runs += 1
        run_start = entry.offset
        run_end = entry.offset + read_sz
        run_jobs = 1
    if run_jobs > 1:
        coalesced_jobs += run_jobs
        extra_bytes += (run_end - run_start) - sum(
            ((e.nbytes + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
            for e in ordered
            if run_start <= e.offset < run_end
        )
    return {
        "jobs": len(entries),
        "runs": runs,
        "coalesced_jobs": coalesced_jobs,
        "extra_bytes": extra_bytes,
    }


def summarize(args: argparse.Namespace) -> dict[str, object]:
    index = read_pack_index(args.pack)
    trace = read_trace(args.trace)
    groups: list[list[Entry]] = []
    cur: list[Entry] = []
    cur_size: int | None = None
    missing = 0
    for key in trace:
        entry = index.get((key.tensor, key.expert_idx, key.expert_bytes))
        if entry is None:
            missing += 1
            continue
        if cur and (entry.nbytes != cur_size or len(cur) >= args.group_size):
            groups.append(cur)
            cur = []
            cur_size = None
        cur.append(entry)
        cur_size = entry.nbytes
    if cur:
        groups.append(cur)

    total_jobs = 0
    total_runs = 0
    coalesced_jobs = 0
    extra_bytes = 0
    hist = {"1": 0, "2": 0, "3-4": 0, "5-8": 0}
    run_hist = {"1": 0, "2": 0, "3-4": 0, "5-8": 0}
    for group in groups:
        rec = analyze_group(group)
        total_jobs += rec["jobs"]
        total_runs += rec["runs"]
        coalesced_jobs += rec["coalesced_jobs"]
        extra_bytes += rec["extra_bytes"]
        jobs = rec["jobs"]
        runs = rec["runs"]
        hist["1" if jobs == 1 else "2" if jobs == 2 else "3-4" if jobs <= 4 else "5-8"] += 1
        run_hist["1" if runs == 1 else "2" if runs == 2 else "3-4" if runs <= 4 else "5-8"] += 1

    return {
        "pack": str(args.pack),
        "trace": str(args.trace),
        "group_size": args.group_size,
        "trace_events": len(trace),
        "indexed_events": total_jobs,
        "missing_events": missing,
        "groups": len(groups),
        "jobs": total_jobs,
        "coalesced_runs": total_runs,
        "possible_read_reduction": total_jobs - total_runs,
        "possible_read_reduction_pct": round(100.0 * (total_jobs - total_runs) / max(1, total_jobs), 3),
        "coalesced_jobs": coalesced_jobs,
        "coalesced_jobs_pct": round(100.0 * coalesced_jobs / max(1, total_jobs), 3),
        "extra_bytes": extra_bytes,
        "group_hist": hist,
        "run_hist": run_hist,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    result = summarize(args)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
