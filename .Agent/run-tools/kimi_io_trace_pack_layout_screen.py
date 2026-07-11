#!/usr/bin/env python3
"""Pack-layout coalescing screen from foreground io-read traces.

This tool answers a narrow question before any pack rewrite:

If active expert entries were laid out so that entries from the same tensor or
same layer/role were contiguous, how many foreground read jobs could be
coalesced, and how much gap over-read would that require?

It uses actual io-read-trace.csv rows rather than route traces.  It is
diagnostic-only and does not create a pack or change runtime behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import itertools
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


GIB = 1024 ** 3
MIB = 1024 ** 2
TENSOR_RE = re.compile(r"blk\.(\d+)\.ffn_(up|gate|down)_exps\.weight")


@dataclass(frozen=True)
class ReadRow:
    prompt_id: str
    batch_seq: int
    jobs: int
    tensor: str
    layer: int
    role: str
    expert_idx: int
    source_idx: int
    offset: int
    nbytes: int

    @property
    def end(self) -> int:
        return self.offset + self.nbytes


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--io-read-trace", action="append", type=Path, default=[])
    ap.add_argument("--input-root", action="append", type=Path, default=[])
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--out-md", required=True, type=Path)
    ap.add_argument("--roles", default="up,gate,down")
    ap.add_argument("--max-jobs", type=int, default=8)
    ap.add_argument("--max-gap-mib", type=float, default=1.0)
    ap.add_argument("--baseline-metrics", type=Path)
    ap.add_argument("--exclude-prompt-regex", default=r"heldout|test_")
    return ap.parse_args()


def prompt_id_for_trace(path: Path) -> str:
    return path.parent.name if path.name == "io-read-trace.csv" else path.stem


def find_traces(args: argparse.Namespace) -> list[Path]:
    paths = list(args.io_read_trace)
    for root in args.input_root:
        if root.is_file() and root.name == "io-read-trace.csv":
            paths.append(root)
        elif root.is_dir():
            paths.extend(sorted(root.glob("*/io-read-trace.csv")))
    out: list[Path] = []
    seen: set[Path] = set()
    deny = re.compile(args.exclude_prompt_regex)
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if deny.search(prompt_id_for_trace(path)):
            raise SystemExit(f"refusing held-out/test-looking trace: {path}")
        out.append(path)
    if not out:
        raise SystemExit("no io-read-trace.csv inputs")
    return out


def parse_tensor(tensor: str) -> tuple[int, str]:
    m = TENSOR_RE.search(tensor)
    if not m:
        return -1, "other"
    return int(m.group(1)), m.group(2)


def load_rows(paths: list[Path], roles: set[str], max_jobs: int) -> list[ReadRow]:
    rows: list[ReadRow] = []
    for path in paths:
        prompt = prompt_id_for_trace(path)
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("jobs") == "jobs" or not row.get("tensor"):
                    continue
                jobs = int(row["jobs"])
                if max_jobs > 0 and jobs > max_jobs:
                    continue
                layer, role = parse_tensor(row["tensor"])
                if role not in roles:
                    continue
                rows.append(ReadRow(
                    prompt_id=prompt,
                    batch_seq=int(row["batch_seq"]),
                    jobs=jobs,
                    tensor=row["tensor"],
                    layer=layer,
                    role=role,
                    expert_idx=int(row["expert_idx"]),
                    source_idx=int(row["source_idx"]),
                    offset=int(row["offset"]),
                    nbytes=int(row["nbytes"]),
                ))
    return rows


def parse_metrics(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def metric_int(metrics: dict[str, str], pattern: str) -> int:
    text = "\n".join(metrics.values())
    m = re.search(pattern, text)
    return int(m.group(1)) if m else 0


def metric_float(metrics: dict[str, str], key: str) -> float:
    raw = metrics.get(key, "0")
    m = re.search(r"[-+]?[0-9]*\.?[0-9]+", raw)
    return float(m.group(0)) if m else 0.0


def coalesce_extents(items: list[ReadRow], max_gap: int) -> tuple[int, int, int, int]:
    """Return extents, read_bytes, span_bytes, gap_bytes."""
    if not items:
        return 0, 0, 0, 0
    by_source: dict[int, list[ReadRow]] = defaultdict(list)
    for item in items:
        by_source[item.source_idx].append(item)

    extents = 0
    read_bytes = sum(item.nbytes for item in items)
    span_bytes = 0
    gap_bytes = 0
    for group in by_source.values():
        group.sort(key=lambda row: row.offset)
        cur_start = group[0].offset
        cur_end = group[0].end
        extents += 1
        for row in group[1:]:
            if row.offset <= cur_end:
                cur_end = max(cur_end, row.end)
                continue
            gap = row.offset - cur_end
            if gap <= max_gap:
                gap_bytes += gap
                cur_end = row.end
            else:
                span_bytes += cur_end - cur_start
                cur_start = row.offset
                cur_end = row.end
                extents += 1
        span_bytes += cur_end - cur_start
    return extents, read_bytes, span_bytes, gap_bytes


def theoretical_layout_extents(items: list[ReadRow], key_kind: str) -> tuple[int, int, int, int]:
    """Ideal layout: rows with same grouping key become adjacent, zero gap."""
    if not items:
        return 0, 0, 0, 0
    groups: dict[tuple[object, ...], list[ReadRow]] = defaultdict(list)
    for row in items:
        if key_kind == "tensor":
            key = (row.source_idx, row.tensor)
        elif key_kind == "layer_role":
            key = (row.source_idx, row.layer, row.role)
        elif key_kind == "layer":
            key = (row.source_idx, row.layer)
        elif key_kind == "source":
            key = (row.source_idx,)
        else:
            raise RuntimeError(key_kind)
        groups[key].append(row)
    extents = len(groups)
    read_bytes = sum(row.nbytes for row in items)
    return extents, read_bytes, read_bytes, 0


def greedy_pair_order(experts: list[int], freq: Counter[int], edges: Counter[tuple[int, int]]) -> list[int]:
    remaining = set(experts)
    order: list[int] = []
    while remaining:
        if not order:
            cur = max(remaining, key=lambda expert: (freq[expert], -expert))
        else:
            last = order[-1]

            def score(expert: int) -> tuple[int, int, int]:
                edge = (last, expert) if last < expert else (expert, last)
                return (edges[edge], freq[expert], -expert)

            cur = max(remaining, key=score)
        order.append(cur)
        remaining.remove(cur)
    return order


def build_static_layouts(rows: list[ReadRow]) -> dict[str, dict[tuple[str, int], tuple[int, int, int]]]:
    """Build prompt-general per-tensor clustered layouts from observed dev rows.

    The output maps (tensor, expert_idx) to (synthetic_source, offset, nbytes).
    Each tensor gets its own synthetic source. This avoids granting free
    cross-tensor coalescing to a per-tensor clustered pack.
    """
    sizes: dict[str, dict[int, int]] = defaultdict(dict)
    freq: dict[str, Counter[int]] = defaultdict(Counter)
    first_use: dict[str, list[int]] = defaultdict(list)
    seen_first: dict[str, set[int]] = defaultdict(set)
    edges: dict[str, Counter[tuple[int, int]]] = defaultdict(Counter)
    by_batch_tensor: dict[tuple[str, int, str], set[int]] = defaultdict(set)

    for row in rows:
        sizes[row.tensor][row.expert_idx] = max(sizes[row.tensor].get(row.expert_idx, 0), row.nbytes)
        freq[row.tensor][row.expert_idx] += 1
        if row.expert_idx not in seen_first[row.tensor]:
            seen_first[row.tensor].add(row.expert_idx)
            first_use[row.tensor].append(row.expert_idx)
        by_batch_tensor[(row.prompt_id, row.batch_seq, row.tensor)].add(row.expert_idx)

    for (_prompt, _batch, tensor), experts in by_batch_tensor.items():
        for a, b in itertools.combinations(sorted(experts), 2):
            edges[tensor][(a, b)] += 1

    tensor_sources = {tensor: i for i, tensor in enumerate(sorted(sizes))}
    orders: dict[str, dict[str, list[int]]] = {
        "expert_id": {},
        "frequency": {},
        "first_use": {},
        "greedy_pair": {},
    }
    for tensor, by_expert in sizes.items():
        experts = sorted(by_expert)
        first = first_use[tensor] + [expert for expert in experts if expert not in seen_first[tensor]]
        orders["expert_id"][tensor] = experts
        orders["frequency"][tensor] = sorted(experts, key=lambda expert: (-freq[tensor][expert], expert))
        orders["first_use"][tensor] = first
        orders["greedy_pair"][tensor] = greedy_pair_order(experts, freq[tensor], edges[tensor])

    layouts: dict[str, dict[tuple[str, int], tuple[int, int, int]]] = {}
    for name, by_tensor in orders.items():
        layout: dict[tuple[str, int], tuple[int, int, int]] = {}
        for tensor, order in by_tensor.items():
            offset = 0
            source = tensor_sources[tensor]
            for expert in order:
                nbytes = sizes[tensor][expert]
                layout[(tensor, expert)] = (source, offset, nbytes)
                offset += nbytes
        layouts[name] = layout
    return layouts


def coalesce_static_layout(items: list[ReadRow], layout: dict[tuple[str, int], tuple[int, int, int]], max_gap: int) -> tuple[int, int, int, int, int]:
    """Return extents, read_bytes, span_bytes, gap_bytes, missing."""
    mapped: dict[int, list[tuple[int, int]]] = defaultdict(list)
    read_bytes = 0
    missing = 0
    for row in items:
        entry = layout.get((row.tensor, row.expert_idx))
        if entry is None:
            missing += 1
            continue
        source, offset, nbytes = entry
        mapped[source].append((offset, nbytes))
        read_bytes += nbytes

    extents = 0
    span_bytes = 0
    gap_bytes = 0
    for spans in mapped.values():
        if not spans:
            continue
        spans.sort()
        cur_start = spans[0][0]
        cur_end = spans[0][0] + spans[0][1]
        extents += 1
        for offset, nbytes in spans[1:]:
            end = offset + nbytes
            if offset <= cur_end:
                cur_end = max(cur_end, end)
                continue
            gap = offset - cur_end
            if gap <= max_gap:
                gap_bytes += gap
                cur_end = end
            else:
                span_bytes += cur_end - cur_start
                cur_start = offset
                cur_end = end
                extents += 1
        span_bytes += cur_end - cur_start
    return extents, read_bytes, span_bytes, gap_bytes, missing


def summarize(rows: list[ReadRow], max_gap: int) -> dict[str, object]:
    batches: dict[tuple[str, int], list[ReadRow]] = defaultdict(list)
    for row in rows:
        batches[(row.prompt_id, row.batch_seq)].append(row)

    current = Counter()
    ideal = {kind: Counter() for kind in ("tensor", "layer_role", "layer", "source")}
    static_layout_defs = build_static_layouts(rows)
    static_layouts = {kind: Counter() for kind in static_layout_defs}
    batch_hist = Counter()
    for items in batches.values():
        batch_hist[len(items)] += 1
        extents, read_bytes, span_bytes, gap_bytes = coalesce_extents(items, max_gap)
        current["extents"] += extents
        current["read_jobs"] += len(items)
        current["read_bytes"] += read_bytes
        current["span_bytes"] += span_bytes
        current["gap_bytes"] += gap_bytes
        for kind in ideal:
            e, rb, sb, gb = theoretical_layout_extents(items, kind)
            ideal[kind]["extents"] += e
            ideal[kind]["read_jobs"] += len(items)
            ideal[kind]["read_bytes"] += rb
            ideal[kind]["span_bytes"] += sb
            ideal[kind]["gap_bytes"] += gb
        for kind, layout in static_layout_defs.items():
            e, rb, sb, gb, missing = coalesce_static_layout(items, layout, max_gap)
            static_layouts[kind]["extents"] += e
            static_layouts[kind]["read_jobs"] += len(items)
            static_layouts[kind]["read_bytes"] += rb
            static_layouts[kind]["span_bytes"] += sb
            static_layouts[kind]["gap_bytes"] += gb
            static_layouts[kind]["missing"] += missing

    signatures = Counter()
    layer_signatures = Counter()
    for items in batches.values():
        signatures[tuple(sorted((row.tensor, row.expert_idx) for row in items))] += 1
        layer_signatures[tuple(sorted((row.layer, row.role, row.expert_idx) for row in items))] += 1

    return {
        "rows": len(rows),
        "batches": len(batches),
        "batch_hist": dict(sorted(batch_hist.items())),
        "current_offset_coalescing": dict(current),
        "ideal_layouts": {k: dict(v) for k, v in ideal.items()},
        "static_layouts": {k: dict(v) for k, v in static_layouts.items()},
        "batch_signature_reuse": {
            "unique_exact_signatures": len(signatures),
            "repeated_exact_batches": sum(count for count in signatures.values() if count > 1),
            "top_exact_signatures": [
                {"count": count, "signature": [list(item) for item in sig[:16]]}
                for sig, count in signatures.most_common(10)
            ],
            "unique_layer_signatures": len(layer_signatures),
            "repeated_layer_batches": sum(count for count in layer_signatures.values() if count > 1),
            "top_layer_signatures": [
                {"count": count, "signature": [list(item) for item in sig[:16]]}
                for sig, count in layer_signatures.most_common(10)
            ],
        },
    }


def add_bounds(report: dict[str, object], metrics: dict[str, str]) -> None:
    decode_ms = metric_float(metrics, "decode_ms")
    decode_runs = int(metric_float(metrics, "decode_runs"))
    iouring_wait_us = metric_int(metrics, r"iouring_wait_us=(\d+)")
    iouring_reads = metric_int(metrics, r"iouring_reads=(\d+)")
    if not decode_ms or not iouring_reads:
        return
    per_read_wait_ms = (iouring_wait_us / 1000.0) / iouring_reads
    current = report["current_offset_coalescing"]  # type: ignore[index]
    current_extents = float(current["extents"])  # type: ignore[index]
    bounds = {}
    combined_layouts = {}
    for kind, data in report.get("ideal_layouts", {}).items():  # type: ignore[union-attr]
        combined_layouts[f"ideal:{kind}"] = data
    for kind, data in report.get("static_layouts", {}).items():  # type: ignore[union-attr]
        combined_layouts[f"static:{kind}"] = data
    for kind, data in combined_layouts.items():
        extents = float(data["extents"])
        saved_reads = max(0.0, current_extents - extents)
        saved_ms = saved_reads * per_read_wait_ms
        bounded_decode = max(1.0, decode_ms - saved_ms)
        bounds[kind] = {
            "ideal_extents": extents,
            "saved_reads": saved_reads,
            "saved_ms_upper": saved_ms,
            "bounded_decode_ms": bounded_decode,
            "bounded_tok_s": decode_runs / (bounded_decode / 1000.0) if decode_runs else 0.0,
        }
    report["baseline_bound_input"] = {
        "decode_ms": decode_ms,
        "decode_runs": decode_runs,
        "iouring_reads": iouring_reads,
        "iouring_wait_ms": iouring_wait_us / 1000.0,
        "per_read_wait_ms": per_read_wait_ms,
    }
    report["bounds_from_current_offset_extents"] = bounds


def write_md(path: Path, report: dict[str, object]) -> None:
    lines = [
        "# Kimi IO-trace pack layout screen",
        "",
        "This is a dev-only offline screen. It does not rewrite packs or claim SOTA.",
        "",
        f"- traces: `{len(report['traces'])}`",
        f"- rows: `{report['rows']}`",
        f"- batches: `{report['batches']}`",
        f"- max gap: `{report['max_gap_mib']:.2f} MiB`",
        "",
        "## Batch Signature Reuse",
        "",
    ]
    reuse = report["batch_signature_reuse"]
    lines += [
        f"- unique exact signatures: `{reuse['unique_exact_signatures']}`",
        f"- repeated exact batches: `{reuse['repeated_exact_batches']}`",
        f"- unique layer signatures: `{reuse['unique_layer_signatures']}`",
        f"- repeated layer batches: `{reuse['repeated_layer_batches']}`",
        "",
        "## Current Offset Coalescing",
        "",
    ]
    cur = report["current_offset_coalescing"]
    lines += [
        f"- read jobs: `{cur['read_jobs']}`",
        f"- coalesced extents with current offsets: `{cur['extents']}`",
        f"- read bytes: `{cur['read_bytes'] / GIB:.3f} GiB`",
        f"- span bytes with max-gap coalescing: `{cur['span_bytes'] / GIB:.3f} GiB`",
        f"- gap bytes: `{cur['gap_bytes'] / GIB:.3f} GiB`",
        "",
        "## Ideal Layouts",
        "",
        "| layout | extents | read reduction | read GiB | span GiB | gap GiB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for kind, data in report["ideal_layouts"].items():
        lines.append(
            f"| `{kind}` | `{data['extents']}` | `{cur['extents'] - data['extents']}` | "
            f"`{data['read_bytes'] / GIB:.3f}` | `{data['span_bytes'] / GIB:.3f}` | `{data['gap_bytes'] / GIB:.3f}` |"
        )
    lines += [
        "",
        "## Static Per-Tensor Clustered Layouts",
        "",
        "| layout | extents | read reduction | read GiB | span GiB | gap GiB | missing |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for kind, data in report["static_layouts"].items():
        lines.append(
            f"| `{kind}` | `{data['extents']}` | `{cur['extents'] - data['extents']}` | "
            f"`{data['read_bytes'] / GIB:.3f}` | `{data['span_bytes'] / GIB:.3f}` | "
            f"`{data['gap_bytes'] / GIB:.3f}` | `{data.get('missing', 0)}` |"
        )
    if "bounds_from_current_offset_extents" in report:
        lines += [
            "",
            "## Decode Bound",
            "",
            "| layout | saved reads | saved ms upper | bounded decode ms | bounded tok/s |",
            "|---|---:|---:|---:|---:|",
        ]
        for kind, data in report["bounds_from_current_offset_extents"].items():
            lines.append(
                f"| `{kind}` | `{data['saved_reads']:.1f}` | `{data['saved_ms_upper']:.2f}` | "
                f"`{data['bounded_decode_ms']:.2f}` | `{data['bounded_tok_s']:.2f}` |"
            )
    lines += [
        "",
        "## Decision Rule",
        "",
        "- If ideal same-layer/source bounds are small, pack layout cannot be the next runtime path.",
        "- If the bound is large, build a default-off pack-layout A/B and validate N32 before N96.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    roles = {x.strip() for x in args.roles.split(",") if x.strip()}
    traces = find_traces(args)
    rows = load_rows(traces, roles, args.max_jobs)
    report = summarize(rows, int(args.max_gap_mib * MIB))
    report.update({
        "kind": "kimi_io_trace_pack_layout_screen",
        "traces": [str(p) for p in traces],
        "roles": sorted(roles),
        "max_jobs": args.max_jobs,
        "max_gap_mib": args.max_gap_mib,
    })
    add_bounds(report, parse_metrics(args.baseline_metrics))
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_md(args.out_md, report)
    print(f"rows={report['rows']} batches={report['batches']}")
    cur = report["current_offset_coalescing"]
    print(f"current read_jobs={cur['read_jobs']} extents={cur['extents']} span_gib={cur['span_bytes']/GIB:.3f}")
    for kind, data in report["ideal_layouts"].items():
        print(f"{kind}: extents={data['extents']} reduction={cur['extents'] - data['extents']}")
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
