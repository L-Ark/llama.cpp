#!/usr/bin/env python3
"""Offline Kimi expert-pack layout bound from route-detail traces.

This dev-only diagnostic compares the current physical expert layout with
simple per-tensor reorderings learned from dev route-detail traces. It does not
read held-out test prompts and does not generate a runtime pack.
"""

from __future__ import annotations

import argparse
import collections
import csv
import itertools
import json
import pathlib
import struct
from typing import Any


PACK_MAGIC = b"GGMLMOEPACKv1\0\0\0"
PACK_HEADER = struct.Struct("<16sIIQQ")
PACK_ENTRY = struct.Struct("<128siIQQ")
ALIGNMENT = 4096


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def refuse_heldout(path: pathlib.Path) -> None:
    text = str(path)
    if "/test_" in text or path.name.startswith("test_"):
        raise SystemExit(f"refusing to read held-out test path: {path}")


def load_pack_entries(paths: list[pathlib.Path]) -> tuple[dict[tuple[str, int, int], tuple[int, int, int]], int]:
    entries: dict[tuple[str, int, int], tuple[int, int, int]] = {}
    source_idx = 0
    for path in paths:
        with path.open("rb") as f:
            header = f.read(PACK_HEADER.size)
            if len(header) != PACK_HEADER.size:
                raise RuntimeError(f"{path}: short pack header")
            magic, version, header_size, n_entries, data_start = PACK_HEADER.unpack(header)
            if magic != PACK_MAGIC or version != 1 or header_size < PACK_HEADER.size or data_start < header_size:
                raise RuntimeError(f"{path}: invalid expert-pack header")
            for _ in range(n_entries):
                raw = f.read(PACK_ENTRY.size)
                if len(raw) != PACK_ENTRY.size:
                    raise RuntimeError(f"{path}: short pack index")
                name_raw, expert_idx, _reserved, offset, nbytes = PACK_ENTRY.unpack(raw)
                tensor = name_raw.split(b"\0", 1)[0].decode("utf-8")
                entries[(tensor, expert_idx, nbytes)] = (source_idx, offset, nbytes)
        source_idx += 1
    return entries, source_idx


def load_alias_entries(path: pathlib.Path, existing: dict[tuple[str, int, int], tuple[int, int, int]], next_source_idx: int):
    source_ids: dict[str, int] = {}
    loaded = 0
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            tensor = row["tensor"]
            expert_idx = int(row["expert_idx"])
            nbytes = int(row["nbytes"])
            key = (tensor, expert_idx, nbytes)
            if key in existing:
                continue
            source_path = row["source_path"]
            if source_path not in source_ids:
                source_ids[source_path] = next_source_idx + len(source_ids)
            existing[key] = (source_ids[source_path], int(row["offset"]), nbytes)
            loaded += 1
    return loaded, len(source_ids)


def read_detail_steps(run_dir: pathlib.Path) -> tuple[str, list[dict[str, Any]]]:
    refuse_heldout(run_dir)
    prompt_id = run_dir.name
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        try:
            prompt_id = json.loads(metrics_path.read_text(encoding="utf-8")).get("prompt_id", prompt_id)
        except Exception:
            pass
    path = run_dir / "route-detail.csv"
    if not path.exists():
        return prompt_id, []

    grouped: dict[tuple[int, str, int], dict[int, int]] = collections.defaultdict(dict)
    tensor_for_key: dict[tuple[int, str, int], str] = {}
    kind_for_key: dict[tuple[int, str, int], str] = {}
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            if row["mode"] != "decode":
                continue
            layer = int(row["layer"])
            if layer < 0:
                continue
            tensor = row["tensor"]
            kind = row["kind"]
            call = int(row["call"])
            key = (layer, tensor, call)
            expert = int(row["expert_idx"])
            grouped[key][expert] = int(row["expert_bytes"])
            tensor_for_key[key] = tensor
            kind_for_key[key] = kind

    steps = []
    for key in sorted(grouped):
        layer, tensor, call = key
        experts = grouped[key]
        steps.append({
            "prompt_id": prompt_id,
            "layer": layer,
            "tensor": tensor,
            "kind": kind_for_key[key],
            "call": call,
            "experts": experts,
        })
    return prompt_id, steps


def collect_stats(steps: list[dict[str, Any]]):
    sizes: dict[str, dict[int, int]] = collections.defaultdict(dict)
    freq: dict[str, collections.Counter[int]] = collections.defaultdict(collections.Counter)
    first_use: dict[str, list[int]] = collections.defaultdict(list)
    seen: dict[str, set[int]] = collections.defaultdict(set)
    edges: dict[str, collections.Counter[tuple[int, int]]] = collections.defaultdict(collections.Counter)

    for step in steps:
        tensor = step["tensor"]
        experts = step["experts"]
        for expert, nbytes in experts.items():
            sizes[tensor][expert] = max(sizes[tensor].get(expert, 0), nbytes)
            freq[tensor][expert] += 1
            if expert not in seen[tensor]:
                seen[tensor].add(expert)
                first_use[tensor].append(expert)
        for a, b in itertools.combinations(sorted(experts), 2):
            edges[tensor][(a, b)] += 1
    return sizes, freq, first_use, edges


def greedy_order(experts: list[int], freq: collections.Counter[int], edges: collections.Counter[tuple[int, int]]) -> list[int]:
    remaining = set(experts)
    order: list[int] = []
    while remaining:
        if not order:
            cur = max(remaining, key=lambda e: (freq[e], -e))
        else:
            last = order[-1]
            def score(e: int):
                key = (last, e) if last < e else (e, last)
                return (edges[key], freq[e], -e)
            cur = max(remaining, key=score)
        order.append(cur)
        remaining.remove(cur)
    return order


def build_layouts(sizes, freq, first_use, edges):
    layouts: dict[str, dict[tuple[str, int, int], tuple[int, int, int]]] = {}
    orderings: dict[str, dict[str, list[int]]] = {}
    for name in ("first_use", "frequency", "greedy_pair"):
        orderings[name] = {}
    for tensor, by_expert in sizes.items():
        experts = sorted(by_expert)
        used_first = first_use.get(tensor, [])
        first = used_first + [expert for expert in experts if expert not in set(used_first)]
        orderings["first_use"][tensor] = first
        orderings["frequency"][tensor] = sorted(experts, key=lambda e: (-freq[tensor][e], e))
        orderings["greedy_pair"][tensor] = greedy_order(experts, freq[tensor], edges[tensor])

    for name, by_tensor in orderings.items():
        offsets: dict[tuple[str, int, int], tuple[int, int, int]] = {}
        for tensor, order in by_tensor.items():
            offset = 0
            for expert in order:
                offset = align_up(offset)
                nbytes = sizes[tensor][expert]
                offsets[(tensor, expert, nbytes)] = (0, offset, nbytes)
                offset += nbytes
        layouts[name] = offsets
    return layouts


def step_metrics(step: dict[str, Any], offsets: dict[tuple[str, int, int], tuple[int, int, int]]) -> dict[str, int]:
    groups: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    read_bytes = 0
    missing = 0
    for expert, nbytes in step["experts"].items():
        key = (step["tensor"], expert, nbytes)
        if key not in offsets:
            missing += 1
            continue
        source_idx, offset, size = offsets[key]
        read_bytes += size
        groups[source_idx].append((offset, size))

    span_bytes = 0
    gap_bytes = 0
    max_gap_bytes = 0
    adjacent_pairs = 0
    for items in groups.values():
        items.sort()
        source_min = None
        source_end = None
        for offset, size in items:
            end = offset + size
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
        "rows": 1,
        "read_jobs": len(step["experts"]) - missing,
        "missing": missing,
        "read_bytes": read_bytes,
        "span_bytes": span_bytes,
        "gap_bytes": gap_bytes,
        "max_gap_bytes": max_gap_bytes,
        "adjacent_pairs": adjacent_pairs,
        "coalesce_rows": 1 if len(step["experts"]) > 1 and read_bytes and gap_bytes / read_bytes < 0.25 else 0,
    }


def add_counter(dst: collections.Counter, src: dict[str, int]) -> None:
    for key, value in src.items():
        dst[key] += value


def evaluate(steps: list[dict[str, Any]], offsets) -> tuple[collections.Counter, dict[str, collections.Counter]]:
    total: collections.Counter = collections.Counter()
    by_kind: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for step in steps:
        metrics = step_metrics(step, offsets)
        add_counter(total, metrics)
        add_counter(by_kind[step["kind"]], metrics)
    return total, by_kind


def summarize_counter(counter: collections.Counter) -> dict[str, float | int]:
    read_bytes = counter["read_bytes"] or 1
    read_jobs = counter["read_jobs"] or 1
    rows = counter["rows"] or 1
    return {
        "rows": int(counter["rows"]),
        "read_jobs": int(counter["read_jobs"]),
        "missing": int(counter["missing"]),
        "read_gib": counter["read_bytes"] / 1024**3,
        "span_over_read": counter["span_bytes"] / read_bytes,
        "gap_over_read": counter["gap_bytes"] / read_bytes,
        "adjacent_per_job": counter["adjacent_pairs"] / read_jobs,
        "coalesce_row_pct": 100.0 * counter["coalesce_rows"] / rows,
        "max_gap_mib": counter["max_gap_bytes"] / 1024**2,
    }


def write_report(out: pathlib.Path, payload: dict[str, Any]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = payload["rows"]
    lines = [
        "# Kimi route-detail pack layout bound",
        "",
        "This is a dev-only offline diagnostic. It does not use held-out test prompts and does not generate a runtime pack.",
        "",
        "## Summary",
        "",
        f"- dev prompts: `{payload['prompt_count']}`",
        f"- route rows: `{payload['step_count']}`",
        f"- current metadata entries: `{payload['current_entry_count']}`",
        "",
        "| layout | kind | rows | read GiB | span/read | gap/read | adjacent/job | coalesce rows | missing |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['layout']} | {row['kind']} | {row['rows']} | {row['read_gib']:.2f} | "
            f"{row['span_over_read']:.3f} | {row['gap_over_read']:.3f} | "
            f"{row['adjacent_per_job']:.3f} | {row['coalesce_row_pct']:.1f}% | {row['missing']} |"
        )
    lines.extend([
        "",
        "Interpretation:",
        "",
        "- `span/read` and `gap/read` estimate how much physical locality a layout creates for each active tensor batch.",
        "- Better locality is useful only if runtime can exploit it with larger contiguous/coalesced reads or lower per-read overhead.",
        "- If a layout improves locality but does not reduce read bytes or create fewer wait waves, token-rate gains may be small.",
        "",
    ])
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline pack-layout locality bound from route-detail traces.")
    parser.add_argument("--runs-root", action="append", type=pathlib.Path, required=True)
    parser.add_argument("--pack", action="append", type=pathlib.Path, default=[])
    parser.add_argument("--alias-tsv", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    all_steps: list[dict[str, Any]] = []
    prompt_ids: set[str] = set()
    for root in args.runs_root:
        refuse_heldout(root)
        for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            prompt_id, steps = read_detail_steps(run_dir)
            if steps:
                prompt_ids.add(prompt_id)
                all_steps.extend(steps)
    if not all_steps:
        raise SystemExit("no route-detail steps found")

    current_offsets, next_source_idx = load_pack_entries(args.pack)
    if args.alias_tsv:
        load_alias_entries(args.alias_tsv, current_offsets, next_source_idx)

    sizes, freq, first_use, edges = collect_stats(all_steps)
    layouts = {"current": current_offsets}
    layouts.update(build_layouts(sizes, freq, first_use, edges))

    rows = []
    for layout_name, offsets in layouts.items():
        total, by_kind = evaluate(all_steps, offsets)
        row = {"layout": layout_name, "kind": "all", **summarize_counter(total)}
        rows.append(row)
        for kind in sorted(by_kind):
            rows.append({"layout": layout_name, "kind": kind, **summarize_counter(by_kind[kind])})

    payload = {
        "prompt_count": len(prompt_ids),
        "step_count": len(all_steps),
        "current_entry_count": len(current_offsets),
        "packs": [str(path) for path in args.pack],
        "alias_tsv": str(args.alias_tsv) if args.alias_tsv else "",
        "rows": rows,
    }
    write_report(args.out, payload)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
