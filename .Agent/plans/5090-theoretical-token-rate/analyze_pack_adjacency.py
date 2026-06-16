#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import struct
from pathlib import Path


ENTRY_STRUCT = struct.Struct("<128siIQQ")


def load_pack_index(path: Path) -> dict[tuple[str, int, int], tuple[int, int]]:
    with path.open("rb") as f:
        magic = f.read(16)
        version, header_size = struct.unpack("<II", f.read(8))
        n_entries, data_start = struct.unpack("<QQ", f.read(16))
        if not magic.startswith(b"GGMLMOEPACKv1") or version != 1 or header_size < 40 or data_start < header_size:
            raise SystemExit(f"invalid pack header: {path}")
        entries: dict[tuple[str, int, int], tuple[int, int]] = {}
        for _ in range(n_entries):
            raw = f.read(ENTRY_STRUCT.size)
            if len(raw) != ENTRY_STRUCT.size:
                raise SystemExit("short pack index")
            tensor_raw, expert_idx, _reserved, offset, nbytes = ENTRY_STRUCT.unpack(raw)
            tensor = tensor_raw.split(b"\0", 1)[0].decode("utf-8")
            entries[(tensor, expert_idx, nbytes)] = (offset, nbytes)
        return entries


def load_trace(path: Path) -> list[tuple[int, int, str, int]]:
    rows: list[tuple[int, int, str, int]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((int(row["seq"]), int(row["expert_bytes"]), row["tensor"], int(row["expert_idx"])))
    return rows


def tensor_group(tensor: str) -> str:
    if ".ffn_up_exps." in tensor:
        return tensor.replace(".ffn_up_exps.", ".ffn_up_gate.")
    if ".ffn_gate_exps." in tensor:
        return tensor.replace(".ffn_gate_exps.", ".ffn_up_gate.")
    return tensor


def summarize_window(events: list[tuple[int, int, int]], max_gap: int) -> tuple[int, int, int, int]:
    if not events:
        return (0, 0, 0, 0)
    events = sorted(events)
    runs = 1
    merged_events = 1
    best_run = 1
    cur_run = 1
    for (_off_prev, n_prev, _seq_prev), (off, _n, _seq) in zip(events, events[1:]):
        if off <= _off_prev + n_prev + max_gap:
            cur_run += 1
        else:
            if cur_run > 1:
                merged_events += cur_run
            runs += 1
            best_run = max(best_run, cur_run)
            cur_run = 1
    if cur_run > 1:
        merged_events += cur_run
    best_run = max(best_run, cur_run)
    return (len(events), runs, merged_events, best_run)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--max-gap", type=int, default=0)
    args = parser.parse_args()

    pack = load_pack_index(args.pack)
    trace = load_trace(args.trace)
    looked_up: list[tuple[int, int, int, str]] = []
    missing = 0
    for seq, nbytes, tensor, expert_idx in trace:
        entry = pack.get((tensor, expert_idx, nbytes))
        if entry is None:
            missing += 1
            continue
        offset, size = entry
        looked_up.append((seq, offset, size, tensor_group(tensor)))

    total_events = 0
    total_runs = 0
    total_merged_events = 0
    best_run = 0
    for start in range(0, len(looked_up), args.window):
        window = looked_up[start:start + args.window]
        by_group: dict[str, list[tuple[int, int, int]]] = {}
        for seq, offset, size, group in window:
            by_group.setdefault(group, []).append((offset, size, seq))
        for events in by_group.values():
            events_n, runs, merged_events, run_len = summarize_window(events, args.max_gap)
            total_events += events_n
            total_runs += runs
            total_merged_events += merged_events
            best_run = max(best_run, run_len)

    print(f"trace_events={len(trace)} lookup_events={len(looked_up)} missing={missing}")
    print(f"window={args.window} max_gap={args.max_gap}")
    print(f"grouped_events={total_events} grouped_runs={total_runs}")
    if total_events:
        print(f"runs_per_event={total_runs / total_events:.4f}")
        print(f"mergeable_events_pct={100.0 * total_merged_events / total_events:.2f}")
    print(f"best_run={best_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
