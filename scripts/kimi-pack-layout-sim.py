#!/usr/bin/env python3
import argparse
import csv
import itertools
import os
import struct
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


PACK_MAGIC = b"GGMLMOEPACKv1\0\0\0"
PACK_HEADER = struct.Struct("<16sIIQQ")
PACK_ENTRY = struct.Struct("<128siIQQ")
ALIGNMENT = 4096


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def load_pack_entries(paths: list[Path]) -> dict[tuple[str, int], dict]:
    entries = {}
    for source_idx, path in enumerate(paths):
        with path.open("rb") as f:
            header = f.read(PACK_HEADER.size)
            if len(header) != PACK_HEADER.size:
                raise RuntimeError(f"{path}: short pack header")
            magic, version, header_size, n_entries, data_start = PACK_HEADER.unpack(header)
            if magic != PACK_MAGIC or version != 1 or header_size < PACK_HEADER.size or data_start < header_size:
                raise RuntimeError(f"{path}: invalid pack header")
            for _ in range(n_entries):
                raw = f.read(PACK_ENTRY.size)
                if len(raw) != PACK_ENTRY.size:
                    raise RuntimeError(f"{path}: short pack index")
                name_raw, expert_idx, _reserved, offset, nbytes = PACK_ENTRY.unpack(raw)
                name = name_raw.split(b"\0", 1)[0].decode("utf-8")
                key = (name, expert_idx)
                if key in entries:
                    # Runtime lookup also treats duplicate tensor/expert/size keys as invalid.
                    raise RuntimeError(f"duplicate pack key: {name}:{expert_idx}")
                entries[key] = {
                    "tensor": name,
                    "expert_idx": expert_idx,
                    "source_idx": source_idx,
                    "offset": offset,
                    "nbytes": nbytes,
                    "path": str(path),
                }
    return entries


def load_trace(path: Path) -> dict[int, list[dict]]:
    batches: dict[int, list[dict]] = defaultdict(list)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("batch_seq") == "batch_seq":
                continue
            batch_seq = int(row["batch_seq"])
            item = {
                "batch_pos": int(row["batch_pos"]),
                "op": row["op"],
                "jobs": int(row["jobs"]),
                "read_jobs": int(row["read_jobs"]),
                "tensor": row["tensor"],
                "expert_idx": int(row["expert_idx"]),
                "source_idx": int(row["source_idx"]),
                "offset": int(row["offset"]),
                "nbytes": int(row["nbytes"]),
            }
            batches[batch_seq].append(item)
    for items in batches.values():
        items.sort(key=lambda r: r["batch_pos"])
    return dict(sorted(batches.items()))


def pack_entries_by_tensor(pack_entries: dict[tuple[str, int], dict], trace_batches: dict[int, list[dict]]) -> dict[str, dict[int, int]]:
    by_tensor: dict[str, dict[int, int]] = defaultdict(dict)
    for (tensor, expert_idx), entry in pack_entries.items():
        by_tensor[tensor][expert_idx] = entry["nbytes"]
    for items in trace_batches.values():
        for item in items:
            by_tensor[item["tensor"]].setdefault(item["expert_idx"], item["nbytes"])
    return by_tensor


def trace_stats(trace_batches: dict[int, list[dict]]):
    freq = defaultdict(Counter)
    first_use = defaultdict(list)
    seen = defaultdict(set)
    edges = defaultdict(Counter)
    for items in trace_batches.values():
        by_tensor = defaultdict(list)
        for item in items:
            tensor = item["tensor"]
            expert = item["expert_idx"]
            freq[tensor][expert] += 1
            if expert not in seen[tensor]:
                seen[tensor].add(expert)
                first_use[tensor].append(expert)
            by_tensor[tensor].append(expert)
        for tensor, experts in by_tensor.items():
            uniq = sorted(set(experts))
            for a, b in itertools.combinations(uniq, 2):
                edges[tensor][(a, b)] += 1
    return freq, first_use, edges


def greedy_order(experts: list[int], freq: Counter, edges: Counter) -> list[int]:
    remaining = set(experts)
    order = []
    while remaining:
        if not order:
            cur = max(remaining, key=lambda e: (freq[e], -e))
        else:
            last = order[-1]
            def score(e):
                key = (last, e) if last < e else (e, last)
                return (edges[key], freq[e], -e)
            cur = max(remaining, key=score)
        order.append(cur)
        remaining.remove(cur)
    return order


def build_layouts(by_tensor: dict[str, dict[int, int]], freq, first_use, edges):
    layouts = {}
    for name in ["first_use", "frequency", "greedy_pair"]:
        layouts[name] = {}
    for tensor, sizes in by_tensor.items():
        experts = sorted(sizes)
        used_first = first_use.get(tensor, [])
        first_order = used_first + [e for e in experts if e not in set(used_first)]
        layouts["first_use"][tensor] = first_order
        layouts["frequency"][tensor] = sorted(experts, key=lambda e: (-freq[tensor][e], e))
        layouts["greedy_pair"][tensor] = greedy_order(experts, freq[tensor], edges[tensor])
    return layouts


def layout_offsets(by_tensor: dict[str, dict[int, int]], order_by_tensor: dict[str, list[int]]):
    offsets = {}
    for tensor, order in order_by_tensor.items():
        offset = 0
        for expert in order:
            offset = align_up(offset)
            offsets[(tensor, expert)] = (0, offset, by_tensor[tensor][expert])
            offset += by_tensor[tensor][expert]
    return offsets


def current_offsets(trace_batches: dict[int, list[dict]]):
    offsets = {}
    for items in trace_batches.values():
        for item in items:
            offsets[(item["tensor"], item["expert_idx"])] = (
                item["source_idx"], item["offset"], item["nbytes"]
            )
    return offsets


def batch_metrics(items: list[dict], offsets: dict[tuple[str, int], tuple[int, int, int]]) -> dict:
    read_bytes = 0
    groups = defaultdict(list)
    for item in items:
        key = (item["tensor"], item["expert_idx"])
        source_idx, offset, nbytes = offsets[key]
        read_bytes += nbytes
        groups[source_idx].append((offset, nbytes))
    span_bytes = 0
    gap_bytes = 0
    max_gap_bytes = 0
    adjacent_pairs = 0
    for source_items in groups.values():
        source_items.sort()
        source_min = None
        source_end = None
        for offset, nbytes in source_items:
            end = offset + nbytes
            if source_min is None:
                source_min = offset
                source_end = end
                continue
            if offset == source_end:
                adjacent_pairs += 1
                source_end = max(source_end, end)
            elif offset > source_end:
                gap = offset - source_end
                gap_bytes += gap
                max_gap_bytes = max(max_gap_bytes, gap)
                source_end = end
            else:
                source_end = max(source_end, end)
        if source_min is not None:
            span_bytes += source_end - source_min
    return {
        "read_jobs": len(items),
        "read_bytes": read_bytes,
        "span_bytes": span_bytes,
        "gap_bytes": gap_bytes,
        "max_gap_bytes": max_gap_bytes,
        "adjacent_pairs": adjacent_pairs,
        "coalesce_rows": 1 if len(items) > 1 and read_bytes and gap_bytes / read_bytes < 0.25 else 0,
    }


def evaluate(trace_batches: dict[int, list[dict]], offsets: dict[tuple[str, int], tuple[int, int, int]]):
    total = Counter()
    by_op = defaultdict(Counter)
    for items in trace_batches.values():
        metrics = batch_metrics(items, offsets)
        op = items[0]["op"] if items else ""
        for key, value in metrics.items():
            total[key] += value
            by_op[op][key] += value
        total["rows"] += 1
        by_op[op]["rows"] += 1
    return total, by_op


def print_counter(name: str, c: Counter):
    read_bytes = c["read_bytes"] or 1
    read_jobs = c["read_jobs"] or 1
    print(
        f"{name}: rows={c['rows']} read_jobs={c['read_jobs']} "
        f"read_bytes={c['read_bytes']} span/read={c['span_bytes']/read_bytes:.4f} "
        f"gap/read={c['gap_bytes']/read_bytes:.4f} "
        f"adjacent/read_jobs={c['adjacent_pairs']/read_jobs:.4f} "
        f"coalesce_rows={c['coalesce_rows']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate Kimi expert-pack physical layouts from per-read IO traces.")
    parser.add_argument("--trace", required=True, type=Path, help="io-read-trace.csv from GGML_MOE_IO_READ_TRACE_OUT.")
    parser.add_argument("--pack", action="append", type=Path, default=[], help="GGMLMOEPACKv1 file; repeat for overlays.")
    args = parser.parse_args()

    trace_batches = load_trace(args.trace)
    pack_entries = load_pack_entries(args.pack) if args.pack else {}
    by_tensor = pack_entries_by_tensor(pack_entries, trace_batches)
    freq, first_use, edges = trace_stats(trace_batches)
    layouts = build_layouts(by_tensor, freq, first_use, edges)
    offset_sets = {"current_trace": current_offsets(trace_batches)}
    for name, order_by_tensor in layouts.items():
        offset_sets[name] = layout_offsets(by_tensor, order_by_tensor)

    print(f"commit={git_head()}")
    print(f"argv={' '.join(sys.argv)}")
    print(f"trace={args.trace}")
    print("packs=" + ",".join(str(p) for p in args.pack))
    print(f"batches={len(trace_batches)} tensors={len(by_tensor)}")
    for name, offsets in offset_sets.items():
        total, by_op = evaluate(trace_batches, offsets)
        print_counter(name, total)
        for op, counter in sorted(by_op.items()):
            print_counter(f"  {name}/{op}", counter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
