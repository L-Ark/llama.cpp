#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
from dataclasses import dataclass, field
from typing import Any


GIB = 1024 ** 3
MIB = 1024 ** 2
TENSOR_RE = re.compile(r"blk\.(\d+)\.ffn_(up|gate|down)_exps\.weight")


@dataclass(frozen=True)
class Key:
    tensor: str
    expert_idx: int


@dataclass
class Batch:
    prompt: str
    seq: int
    op: str
    jobs: int
    wait_ms: float = 0.0
    rows: int = 0
    bytes: int = 0
    keys: set[Key] = field(default_factory=set)
    role_counts: dict[str, int] = field(default_factory=dict)
    layer_counts: dict[int, int] = field(default_factory=dict)


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


def role_layer(tensor: str) -> tuple[str, int]:
    match = TENSOR_RE.search(tensor or "")
    if not match:
        return "other", -1
    return match.group(2), int(match.group(1))


def prompt_id_for_trace(path: pathlib.Path) -> str:
    return path.parent.name if path.name == "io-read-trace.csv" else path.stem


def discover_traces(paths: list[pathlib.Path], exclude_re: re.Pattern[str]) -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    seen: set[pathlib.Path] = set()
    for path in paths:
        candidates: list[pathlib.Path]
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


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def load_waits(path: pathlib.Path) -> dict[int, float]:
    waits: dict[int, float] = {}
    for row in read_csv(path):
        if row.get("batch_seq") == "batch_seq":
            continue
        seq = inum(row.get("batch_seq"))
        waits[seq] = waits.get(seq, 0.0) + fnum(row.get("wait_ms"))
    return waits


def load_decode_runs(traces: list[pathlib.Path]) -> int:
    total = 0
    for trace in traces:
        metrics = trace.parent / "metrics.json"
        if not metrics.exists():
            continue
        try:
            data = json.loads(metrics.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError:
            continue
        total += inum(data.get("decode_runs"))
    return total


def load_batches(traces: list[pathlib.Path], max_jobs: int, roles: set[str]) -> tuple[list[Batch], dict[Key, int]]:
    batches: dict[tuple[str, int], Batch] = {}
    key_bytes: dict[Key, int] = {}
    for trace in traces:
        prompt = prompt_id_for_trace(trace)
        waits = load_waits(trace.parent / "io-wait-trace.csv")
        for row in read_csv(trace):
            if row.get("jobs") == "jobs" or not row.get("tensor"):
                continue
            jobs = inum(row.get("jobs"))
            if max_jobs > 0 and jobs > max_jobs:
                continue
            tensor = row["tensor"]
            role, layer = role_layer(tensor)
            if role not in roles:
                continue
            seq = inum(row.get("batch_seq"))
            key = Key(tensor=tensor, expert_idx=inum(row.get("expert_idx")))
            nbytes = inum(row.get("nbytes"))
            key_bytes[key] = max(key_bytes.get(key, 0), nbytes)
            batch_key = (prompt, seq)
            batch = batches.get(batch_key)
            if batch is None:
                batch = Batch(
                    prompt=prompt,
                    seq=seq,
                    op=row.get("op", ""),
                    jobs=jobs,
                    wait_ms=waits.get(seq, 0.0),
                )
                batches[batch_key] = batch
            batch.rows += 1
            batch.bytes += nbytes
            batch.jobs = max(batch.jobs, jobs)
            batch.keys.add(key)
            batch.role_counts[role] = batch.role_counts.get(role, 0) + 1
            batch.layer_counts[layer] = batch.layer_counts.get(layer, 0) + 1
    return list(batches.values()), key_bytes


def batch_resident_bytes(batch: Batch, key_bytes: dict[Key, int], selected: set[Key]) -> int:
    return sum(key_bytes.get(key, 0) for key in batch.keys if key not in selected)


def greedy_cover(batches: list[Batch], key_bytes: dict[Key, int], budget_mib: int) -> dict[str, Any]:
    budget = budget_mib * MIB
    selected: set[Key] = set()
    selected_batches: list[Batch] = []
    used = 0
    saved_wait = 0.0
    remaining = [batch for batch in batches if batch.wait_ms > 0 and batch.keys]
    while True:
        best_i = -1
        best_score = 0.0
        best_add = 0
        best_wait = 0.0
        for i, batch in enumerate(remaining):
            add = batch_resident_bytes(batch, key_bytes, selected)
            if add <= 0:
                score = batch.wait_ms / 1e-9
            elif used + add > budget:
                continue
            else:
                score = batch.wait_ms / (add / GIB)
            if score > best_score or (score == best_score and batch.wait_ms > best_wait):
                best_score = score
                best_i = i
                best_add = add
                best_wait = batch.wait_ms
        if best_i < 0:
            break
        batch = remaining.pop(best_i)
        add = batch_resident_bytes(batch, key_bytes, selected)
        if used + add > budget:
            continue
        selected.update(batch.keys)
        used += add
        saved_wait += batch.wait_ms
        selected_batches.append(batch)

    role_hist: dict[str, int] = {}
    op_hist: dict[str, int] = {}
    layer_hist: dict[str, int] = {}
    for batch in selected_batches:
        op_hist[batch.op] = op_hist.get(batch.op, 0) + 1
        for role, count in batch.role_counts.items():
            role_hist[role] = role_hist.get(role, 0) + count
        for layer, count in batch.layer_counts.items():
            layer_hist[str(layer)] = layer_hist.get(str(layer), 0) + count
    return {
        "budget_mib": budget_mib,
        "resident_bytes": used,
        "resident_gib": used / GIB,
        "selected_entries": len(selected),
        "covered_batches": len(selected_batches),
        "saved_wait_ms": saved_wait,
        "role_rows": dict(sorted(role_hist.items())),
        "op_batches": dict(sorted(op_hist.items())),
        "top_layers": sorted(layer_hist.items(), key=lambda item: item[1], reverse=True)[:20],
        "top_batches": [
            {
                "rank": i + 1,
                "prompt": batch.prompt,
                "seq": batch.seq,
                "op": batch.op,
                "jobs": batch.jobs,
                "rows": batch.rows,
                "wait_ms": batch.wait_ms,
                "resident_mib": sum(key_bytes.get(key, 0) for key in batch.keys) / MIB,
                "roles": dict(sorted(batch.role_counts.items())),
                "layers": dict(sorted(batch.layer_counts.items())),
            }
            for i, batch in enumerate(selected_batches[:50])
        ],
    }


def write_report(path: pathlib.Path, result: dict[str, Any]) -> None:
    lines = [
        "# Kimi batch-cover RAM oracle",
        "",
        "This is a dev-only upper bound. It selects arbitrary complete IO batches",
        "from profiled prompts and therefore can overfit the dev traces. It does",
        "not change runtime behavior or claim SOTA.",
        "",
        "## Inputs",
        "",
        f"- traces: `{result['trace_count']}`",
        f"- prompts: `{result['prompt_count']}`",
        f"- roles: `{','.join(result['roles'])}`",
        f"- max jobs: `{result['max_jobs']}`",
        f"- batches: `{result['batch_count']}`",
        f"- unique expert keys: `{result['unique_keys']}`",
        f"- decode runs: `{result['decode_runs']}`",
        f"- total wait: `{result['total_wait_ms']:.3f} ms`",
        f"- total wait/token: `{result['total_wait_ms_per_token']:.3f} ms/token`",
        "",
        "## Greedy Complete-Batch Cover Bound",
        "",
        "| budget MiB | resident GiB | entries | covered batches | saved wait ms | saved ms/token | remaining ms/token | bounded tok/s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    baseline_ms_per_token = result["baseline_decode_ms"] / max(result["decode_runs"], 1)
    for row in result["budget_rows"]:
        saved_per_token = row["saved_wait_ms"] / max(result["decode_runs"], 1)
        bounded_ms_per_token = max(1.0, baseline_ms_per_token - saved_per_token)
        lines.append(
            f"| {row['budget_mib']} | {row['resident_gib']:.3f} | {row['selected_entries']} | "
            f"{row['covered_batches']} | {row['saved_wait_ms']:.3f} | {saved_per_token:.3f} | "
            f"{bounded_ms_per_token:.3f} | {1000.0 / bounded_ms_per_token:.3f} |"
        )
    lines += [
        "",
        "## Decision Notes",
        "",
        "- If this oracle cannot reach the target, RAM batch-covering is not enough",
        "  even with dev-trace overfitting.",
        "- If it reaches the target only by selecting prompt-specific batches, the",
        "  next step must be a general predictor or storage-format change, not a",
        "  static dev hotset.",
        "- Runtime A/B needs a prompt-general policy that reproduces this coverage",
        "  without increasing TTFT by more than 20% or exceeding 16GB host RAM.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Dev-only complete-batch RAM cover oracle for Kimi IO traces.")
    ap.add_argument("--input-root", action="append", type=pathlib.Path, default=[])
    ap.add_argument("--io-read-trace", action="append", type=pathlib.Path, default=[])
    ap.add_argument("--out-json", required=True, type=pathlib.Path)
    ap.add_argument("--out-md", required=True, type=pathlib.Path)
    ap.add_argument("--roles", default="up,gate,down")
    ap.add_argument("--max-jobs", type=int, default=8)
    ap.add_argument("--budgets-mib", default="2048,4096,8192,10240")
    ap.add_argument("--baseline-decode-ms", type=float, required=True)
    ap.add_argument("--exclude-prompt-regex", default=r"heldout|test_")
    args = ap.parse_args()

    roles = {role.strip() for role in args.roles.split(",") if role.strip()}
    traces = discover_traces(args.input_root + args.io_read_trace, re.compile(args.exclude_prompt_regex))
    if not traces:
        raise SystemExit("No io-read-trace.csv inputs found")
    batches, key_bytes = load_batches(traces, args.max_jobs, roles)
    decode_runs = load_decode_runs(traces)
    prompt_count = len({prompt_id_for_trace(trace) for trace in traces})
    budgets = [int(item) for item in args.budgets_mib.split(",") if item.strip()]
    budget_rows = [greedy_cover(batches, key_bytes, budget) for budget in budgets]
    result = {
        "kind": "kimi_batch_cover_ram_oracle",
        "inputs": [str(trace) for trace in traces],
        "trace_count": len(traces),
        "prompt_count": prompt_count,
        "roles": sorted(roles),
        "max_jobs": args.max_jobs,
        "batch_count": len(batches),
        "unique_keys": len(key_bytes),
        "decode_runs": decode_runs,
        "baseline_decode_ms": args.baseline_decode_ms,
        "baseline_tok_s": decode_runs / (args.baseline_decode_ms / 1000.0) if args.baseline_decode_ms else 0.0,
        "total_wait_ms": sum(batch.wait_ms for batch in batches),
        "total_wait_ms_per_token": sum(batch.wait_ms for batch in batches) / max(decode_runs, 1),
        "budget_rows": budget_rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(args.out_md, result)
    print(args.out_md)


if __name__ == "__main__":
    main()
