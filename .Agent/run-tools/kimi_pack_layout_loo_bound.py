#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any


GIB = 1024 ** 3
MIB = 1024 ** 2
TENSOR_RE = re.compile(r"blk\.(\d+)\.ffn_(up|gate|down)_exps\.weight")


@dataclass(frozen=True)
class ReadRow:
    prompt: str
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


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def inum(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def prompt_id_for_trace(path: pathlib.Path) -> str:
    return path.parent.name if path.name == "io-read-trace.csv" else path.stem


def discover_traces(paths: list[pathlib.Path], exclude_re: re.Pattern[str]) -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    seen: set[pathlib.Path] = set()
    for path in paths:
        if path.is_file() and path.name == "io-read-trace.csv":
            candidates = [path]
        elif path.is_dir():
            candidates = sorted(path.glob("*/io-read-trace.csv"))
            if (path / "io-read-trace.csv").exists():
                candidates.append(path / "io-read-trace.csv")
        else:
            candidates = []
        for candidate in candidates:
            prompt = prompt_id_for_trace(candidate)
            if exclude_re.search(prompt):
                raise SystemExit(f"Refusing held-out/test-looking trace: {candidate}")
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            out.append(candidate)
    return out


def parse_tensor(tensor: str) -> tuple[int, str]:
    match = TENSOR_RE.search(tensor or "")
    if not match:
        return -1, "other"
    return int(match.group(1)), match.group(2)


def read_trace(path: pathlib.Path, roles: set[str], max_jobs: int) -> list[ReadRow]:
    prompt = prompt_id_for_trace(path)
    rows: list[ReadRow] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            if not row.get("tensor"):
                continue
            jobs = inum(row.get("jobs"))
            if max_jobs > 0 and jobs > max_jobs:
                continue
            layer, role = parse_tensor(row["tensor"])
            if role not in roles:
                continue
            rows.append(ReadRow(
                prompt=prompt,
                batch_seq=inum(row.get("batch_seq")),
                jobs=jobs,
                tensor=row["tensor"],
                layer=layer,
                role=role,
                expert_idx=inum(row.get("expert_idx")),
                source_idx=inum(row.get("source_idx")),
                offset=inum(row.get("offset")),
                nbytes=inum(row.get("nbytes")),
            ))
    return rows


def read_metrics(trace: pathlib.Path) -> dict[str, Any]:
    metrics_json = trace.parent / "metrics.json"
    if metrics_json.exists():
        try:
            data = json.loads(metrics_json.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}
    metrics_txt = trace.parent / "metrics.txt"
    if metrics_txt.exists():
        text = metrics_txt.read_text(encoding="utf-8", errors="replace")
        wait = re.search(r"iouring_wait_us=(\d+)", text)
        reads = re.search(r"iouring_reads=(\d+)", text)
        if wait:
            data["iouring_wait_ms"] = int(wait.group(1)) / 1000.0
        if reads:
            data["iouring_reads"] = int(reads.group(1))
    return data


def coalesce_offset_extents(items: list[ReadRow], max_gap: int) -> tuple[int, int, int]:
    by_source: dict[int, list[ReadRow]] = defaultdict(list)
    read_bytes = 0
    for item in items:
        by_source[item.source_idx].append(item)
        read_bytes += item.nbytes
    extents = 0
    span_bytes = 0
    for group in by_source.values():
        group.sort(key=lambda row: row.offset)
        if not group:
            continue
        cur_start = group[0].offset
        cur_end = group[0].end
        extents += 1
        for row in group[1:]:
            if row.offset <= cur_end:
                cur_end = max(cur_end, row.end)
                continue
            gap = row.offset - cur_end
            if gap <= max_gap:
                cur_end = row.end
            else:
                span_bytes += cur_end - cur_start
                cur_start = row.offset
                cur_end = row.end
                extents += 1
        span_bytes += cur_end - cur_start
    return extents, read_bytes, span_bytes


def greedy_pair_order(experts: list[int], freq: Counter[int], edges: Counter[tuple[int, int]]) -> list[int]:
    remaining = set(experts)
    out: list[int] = []
    while remaining:
        if not out:
            cur = max(remaining, key=lambda expert: (freq[expert], -expert))
        else:
            last = out[-1]
            cur = max(
                remaining,
                key=lambda expert: (
                    edges[(last, expert) if last < expert else (expert, last)],
                    freq[expert],
                    -expert,
                ),
            )
        out.append(cur)
        remaining.remove(cur)
    return out


def build_layouts(train_rows: list[ReadRow], all_rows: list[ReadRow]) -> dict[str, dict[tuple[str, int], tuple[int, int, int]]]:
    sizes: dict[str, dict[int, int]] = defaultdict(dict)
    for row in all_rows:
        sizes[row.tensor][row.expert_idx] = max(sizes[row.tensor].get(row.expert_idx, 0), row.nbytes)

    freq: dict[str, Counter[int]] = defaultdict(Counter)
    first_use: dict[str, list[int]] = defaultdict(list)
    seen_first: dict[str, set[int]] = defaultdict(set)
    batch_tensor: dict[tuple[str, int, str], set[int]] = defaultdict(set)
    edges: dict[str, Counter[tuple[int, int]]] = defaultdict(Counter)
    for row in train_rows:
        freq[row.tensor][row.expert_idx] += 1
        if row.expert_idx not in seen_first[row.tensor]:
            seen_first[row.tensor].add(row.expert_idx)
            first_use[row.tensor].append(row.expert_idx)
        batch_tensor[(row.prompt, row.batch_seq, row.tensor)].add(row.expert_idx)
    for (_prompt, _seq, tensor), experts in batch_tensor.items():
        for a, b in combinations(sorted(experts), 2):
            edges[tensor][(a, b)] += 1

    orders: dict[str, dict[str, list[int]]] = {"expert_id": {}, "frequency": {}, "first_use": {}, "greedy_pair": {}}
    for tensor, by_expert in sizes.items():
        experts = sorted(by_expert)
        seen = set(seen_first.get(tensor, set()))
        unseen = [expert for expert in experts if expert not in seen]
        orders["expert_id"][tensor] = experts
        orders["frequency"][tensor] = sorted(experts, key=lambda expert: (-freq[tensor][expert], expert))
        orders["first_use"][tensor] = first_use.get(tensor, []) + unseen
        orders["greedy_pair"][tensor] = greedy_pair_order(experts, freq[tensor], edges[tensor])

    tensor_sources = {tensor: i for i, tensor in enumerate(sorted(sizes))}
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


def coalesce_layout_extents(items: list[ReadRow], layout: dict[tuple[str, int], tuple[int, int, int]], max_gap: int) -> tuple[int, int, int, int]:
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
    for spans in mapped.values():
        spans.sort()
        if not spans:
            continue
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
                cur_end = end
            else:
                span_bytes += cur_end - cur_start
                cur_start = offset
                cur_end = end
                extents += 1
        span_bytes += cur_end - cur_start
    return extents, read_bytes, span_bytes, missing


def group_batches(rows: list[ReadRow]) -> dict[int, list[ReadRow]]:
    batches: dict[int, list[ReadRow]] = defaultdict(list)
    for row in rows:
        batches[row.batch_seq].append(row)
    return batches


def eval_layout(test_rows: list[ReadRow], layout: dict[tuple[str, int], tuple[int, int, int]], max_gap: int) -> dict[str, Any]:
    current_extents = 0
    current_read_bytes = 0
    current_span_bytes = 0
    layout_extents = 0
    layout_read_bytes = 0
    layout_span_bytes = 0
    missing = 0
    for items in group_batches(test_rows).values():
        e, rb, sb = coalesce_offset_extents(items, max_gap)
        current_extents += e
        current_read_bytes += rb
        current_span_bytes += sb
        le, lrb, lsb, miss = coalesce_layout_extents(items, layout, max_gap)
        layout_extents += le
        layout_read_bytes += lrb
        layout_span_bytes += lsb
        missing += miss
    return {
        "current_extents": current_extents,
        "current_read_bytes": current_read_bytes,
        "current_span_bytes": current_span_bytes,
        "layout_extents": layout_extents,
        "layout_read_bytes": layout_read_bytes,
        "layout_span_bytes": layout_span_bytes,
        "missing": missing,
        "saved_reads": max(0, current_extents - layout_extents),
    }


def write_markdown(path: pathlib.Path, report: dict[str, Any]) -> None:
    lines = [
        "# Kimi Pack Layout Leave-One-Prompt-Out Bound",
        "",
        "This is a dev-only offline screen. It does not rewrite packs or claim SOTA.",
        "",
        f"- traces: `{len(report['traces'])}`",
        f"- roles: `{','.join(report['roles'])}`",
        f"- max jobs: `{report['max_jobs']}`",
        f"- max gap: `{report['max_gap_mib']:.2f} MiB`",
        "",
        "## LOO Summary",
        "",
        "| layout | saved reads | saved ms upper | saved ms/token | bounded tok/s | missing |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["summary_rows"]:
        lines.append(
            f"| `{row['layout']}` | `{row['saved_reads']}` | `{row['saved_ms_upper']:.3f}` | "
            f"`{row['saved_ms_per_token']:.3f}` | `{row['bounded_tok_s']:.3f}` | `{row['missing']}` |"
        )
    lines += [
        "",
        "## Per Prompt",
        "",
        "| train prompts | test prompt | layout | current extents | layout extents | saved reads | saved ms upper | bounded tok/s |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| `{','.join(row['train_prompts'])}` | `{row['test_prompt']}` | `{row['layout']}` | "
            f"`{row['current_extents']}` | `{row['layout_extents']}` | `{row['saved_reads']}` | "
            f"`{row['saved_ms_upper']:.3f}` | `{row['bounded_tok_s']:.3f}` |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
    ]
    for item in report["interpretation"]:
        lines.append(f"- {item}")
    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        report["reproduce_command"],
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", action="append", type=pathlib.Path, required=True)
    parser.add_argument("--out-json", type=pathlib.Path, required=True)
    parser.add_argument("--out-md", type=pathlib.Path, required=True)
    parser.add_argument("--roles", default="up,gate,down")
    parser.add_argument("--max-jobs", type=int, default=8)
    parser.add_argument("--max-gap-mib", type=float, default=1.0)
    parser.add_argument("--baseline-decode-ms", type=float, required=True)
    parser.add_argument("--exclude-prompt-regex", default=r"(^|[_-])(test|holdout|heldout)([_-]|$)")
    args = parser.parse_args()

    roles = {role.strip() for role in args.roles.split(",") if role.strip()}
    traces = discover_traces(args.input_root, re.compile(args.exclude_prompt_regex, re.IGNORECASE))
    if len(traces) < 2:
        raise SystemExit("need at least two dev traces for leave-one-prompt-out")

    rows_by_prompt: dict[str, list[ReadRow]] = {}
    metrics_by_prompt: dict[str, dict[str, Any]] = {}
    for trace in traces:
        prompt = prompt_id_for_trace(trace)
        rows_by_prompt[prompt] = read_trace(trace, roles, args.max_jobs)
        metrics_by_prompt[prompt] = read_metrics(trace)

    all_rows = [row for rows in rows_by_prompt.values() for row in rows]
    layout_names = ("expert_id", "frequency", "first_use", "greedy_pair")
    detail_rows: list[dict[str, Any]] = []
    summary: dict[str, dict[str, Any]] = {
        name: {"saved_reads": 0, "saved_ms_upper": 0.0, "missing": 0} for name in layout_names
    }
    total_decode_runs = 0
    for test_prompt, test_rows in sorted(rows_by_prompt.items()):
        train_prompts = [prompt for prompt in sorted(rows_by_prompt) if prompt != test_prompt]
        train_rows = [row for prompt in train_prompts for row in rows_by_prompt[prompt]]
        layouts = build_layouts(train_rows, all_rows)
        metrics = metrics_by_prompt[test_prompt]
        decode_ms = fnum(metrics.get("decode_ms"))
        decode_runs = inum(metrics.get("decode_runs"))
        iouring_wait_ms = fnum(metrics.get("iouring_wait_ms"))
        iouring_reads = inum(metrics.get("iouring_reads"))
        total_decode_runs += decode_runs
        per_read_wait_ms = iouring_wait_ms / iouring_reads if iouring_reads else 0.0
        for name in layout_names:
            row = eval_layout(test_rows, layouts[name], int(args.max_gap_mib * MIB))
            saved_ms = row["saved_reads"] * per_read_wait_ms
            bounded_decode_ms = max(1.0, decode_ms - saved_ms)
            out = {
                "train_prompts": train_prompts,
                "test_prompt": test_prompt,
                "layout": name,
                **row,
                "decode_ms": decode_ms,
                "decode_runs": decode_runs,
                "iouring_reads": iouring_reads,
                "iouring_wait_ms": iouring_wait_ms,
                "per_read_wait_ms": per_read_wait_ms,
                "saved_ms_upper": saved_ms,
                "bounded_decode_ms": bounded_decode_ms,
                "bounded_tok_s": decode_runs * 1000.0 / bounded_decode_ms if bounded_decode_ms > 0 else 0.0,
            }
            detail_rows.append(out)
            summary[name]["saved_reads"] += row["saved_reads"]
            summary[name]["saved_ms_upper"] += saved_ms
            summary[name]["missing"] += row["missing"]

    summary_rows: list[dict[str, Any]] = []
    for name in layout_names:
        saved_ms = summary[name]["saved_ms_upper"]
        bounded_decode_ms = max(1.0, args.baseline_decode_ms - saved_ms)
        summary_rows.append({
            "layout": name,
            "saved_reads": summary[name]["saved_reads"],
            "saved_ms_upper": saved_ms,
            "saved_ms_per_token": saved_ms / total_decode_runs if total_decode_runs else 0.0,
            "bounded_decode_ms": bounded_decode_ms,
            "bounded_tok_s": total_decode_runs * 1000.0 / bounded_decode_ms if bounded_decode_ms > 0 else 0.0,
            "missing": summary[name]["missing"],
        })
    summary_rows.sort(key=lambda row: row["saved_ms_upper"], reverse=True)

    best = summary_rows[0] if summary_rows else {"layout": "none", "saved_ms_per_token": 0.0, "bounded_tok_s": 0.0}
    interpretation = [
        (
            f"Best leave-one-prompt-out static layout is {best['layout']} with "
            f"{best['saved_ms_per_token']:.3f} ms/token saved and bounded {best['bounded_tok_s']:.3f} tok/s."
        ),
        (
            "This is still an optimistic per-read-wait upper bound: it assumes fewer extents translate linearly "
            "to less exposed iouring wait and ignores pack rebuild overhead."
        ),
        (
            "A runtime A/B is justified only if the LOO bound is comfortably above the target and a default-off "
            "pack layout can be built without prompt-specific held-out tuning."
        ),
    ]

    report = {
        "kind": "kimi_pack_layout_loo_bound",
        "traces": [str(trace) for trace in traces],
        "roles": sorted(roles),
        "max_jobs": args.max_jobs,
        "max_gap_mib": args.max_gap_mib,
        "baseline_decode_ms": args.baseline_decode_ms,
        "decode_runs": total_decode_runs,
        "summary_rows": summary_rows,
        "rows": detail_rows,
        "interpretation": interpretation,
        "reproduce_command": " ".join(__import__("sys").argv),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(args.out_md, report)


if __name__ == "__main__":
    main()
