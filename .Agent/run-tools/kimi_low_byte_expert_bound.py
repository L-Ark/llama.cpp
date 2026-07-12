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
TENSOR_RE = re.compile(r"blk\.(\d+)\.ffn_(up|gate|down)_exps\.weight")


@dataclass
class Batch:
    prompt: str
    seq: int
    op: str
    jobs: int
    wait_ms: float = 0.0
    bytes: int = 0
    role_bytes: dict[str, int] = field(default_factory=dict)
    layer_role_bytes: dict[str, int] = field(default_factory=dict)


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


def load_batches(traces: list[pathlib.Path], max_jobs: int, roles: set[str]) -> list[Batch]:
    batches: dict[tuple[str, int], Batch] = {}
    for trace in traces:
        prompt = prompt_id_for_trace(trace)
        waits = load_waits(trace.parent / "io-wait-trace.csv")
        for row in read_csv(trace):
            if not row.get("tensor"):
                continue
            jobs = inum(row.get("jobs"))
            if max_jobs > 0 and jobs > max_jobs:
                continue
            tensor = row["tensor"]
            role, layer = role_layer(tensor)
            if role not in roles:
                continue
            seq = inum(row.get("batch_seq"))
            nbytes = inum(row.get("nbytes"))
            key = (prompt, seq)
            batch = batches.get(key)
            if batch is None:
                batch = Batch(
                    prompt=prompt,
                    seq=seq,
                    op=row.get("op", ""),
                    jobs=jobs,
                    wait_ms=waits.get(seq, 0.0),
                )
                batches[key] = batch
            batch.bytes += nbytes
            batch.jobs = max(batch.jobs, jobs)
            batch.role_bytes[role] = batch.role_bytes.get(role, 0) + nbytes
            layer_key = f"blk.{layer}.{role}"
            batch.layer_role_bytes[layer_key] = batch.layer_role_bytes.get(layer_key, 0) + nbytes
    return list(batches.values())


def contribution_ms(batch: Batch, selected_roles: set[str] | None = None, selected_layer_roles: set[str] | None = None) -> float:
    if batch.bytes <= 0 or batch.wait_ms <= 0:
        return 0.0
    selected_bytes = 0
    if selected_roles is not None:
        selected_bytes += sum(n for role, n in batch.role_bytes.items() if role in selected_roles)
    if selected_layer_roles is not None:
        selected_bytes += sum(n for key, n in batch.layer_role_bytes.items() if key in selected_layer_roles)
    selected_bytes = min(selected_bytes, batch.bytes)
    return batch.wait_ms * selected_bytes / batch.bytes


def scenario_result(
    name: str,
    max_savable_ms: float,
    keep_ratio: float,
    baseline_decode_ms: float,
    decode_runs: int,
) -> dict[str, Any]:
    saved_ms = max_savable_ms * (1.0 - keep_ratio)
    bounded_decode_ms = max(0.0, baseline_decode_ms - saved_ms)
    return {
        "scenario": name,
        "keep_ratio": keep_ratio,
        "byte_reduction_pct": (1.0 - keep_ratio) * 100.0,
        "max_savable_ms": max_savable_ms,
        "saved_ms": saved_ms,
        "saved_ms_per_token": saved_ms / decode_runs if decode_runs else 0.0,
        "bounded_decode_ms": bounded_decode_ms,
        "bounded_ms_per_token": bounded_decode_ms / decode_runs if decode_runs else 0.0,
        "bounded_tok_s": decode_runs * 1000.0 / bounded_decode_ms if bounded_decode_ms > 0 else 0.0,
    }


def required_reduction(max_savable_ms: float, required_ms: float) -> dict[str, Any]:
    if required_ms <= 0:
        return {"possible": True, "required_reduction_pct": 0.0, "required_keep_ratio": 1.0}
    if max_savable_ms <= 0:
        return {"possible": False, "required_reduction_pct": None, "required_keep_ratio": None}
    fraction = required_ms / max_savable_ms
    if fraction > 1.0:
        return {
            "possible": False,
            "required_reduction_pct": fraction * 100.0,
            "required_keep_ratio": 1.0 - fraction,
        }
    return {
        "possible": True,
        "required_reduction_pct": fraction * 100.0,
        "required_keep_ratio": 1.0 - fraction,
    }


def fmt_ms(value: float) -> str:
    return f"{value:.3f}"


def fmt_tok(value: float) -> str:
    return f"{value:.3f}"


def write_markdown(path: pathlib.Path, report: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Kimi Low-Byte Expert Bound")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- traces: `{len(report['traces'])}`")
    for trace in report["traces"]:
        lines.append(f"  - `{trace}`")
    lines.append(f"- batches: `{report['batch_count']}`")
    lines.append(f"- decode runs: `{report['decode_runs']}`")
    lines.append(f"- baseline decode: `{fmt_ms(report['baseline_decode_ms'])} ms`")
    lines.append(f"- baseline token rate: `{fmt_tok(report['baseline_tok_s'])} tok/s`")
    lines.append(f"- total profiled IO wait: `{fmt_ms(report['total_wait_ms'])} ms`")
    lines.append(f"- total profiled payload: `{report['total_payload_gib']:.3f} GiB`")
    lines.append("")
    lines.append("## Role Payload")
    lines.append("")
    lines.append("| role | payload GiB | max linear wait ms | max linear wait ms/token | eliminate-role bound tok/s |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in report["role_rows"]:
        lines.append(
            "| {role} | {payload_gib:.3f} | {wait_ms:.3f} | {wait_ms_per_token:.3f} | {tok_s:.3f} |".format(**row)
        )
    lines.append("")
    lines.append("## Byte Reduction Scenarios")
    lines.append("")
    lines.append("| scenario | byte reduction | saved ms/token | bounded tok/s |")
    lines.append("|---|---:|---:|---:|")
    for row in report["scenario_rows"]:
        lines.append(
            "| {scenario} | {byte_reduction_pct:.0f}% | {saved_ms_per_token:.3f} | {bounded_tok_s:.3f} |".format(**row)
        )
    lines.append("")
    lines.append("## Target Feasibility")
    lines.append("")
    lines.append("| target tok/s | required saving ms/token | all-role required reduction | possible by IO-byte reduction only |")
    lines.append("|---:|---:|---:|---|")
    for row in report["target_rows"]:
        reduction = row["required_reduction_pct"]
        reduction_s = "n/a" if reduction is None else f"{reduction:.1f}%"
        lines.append(
            f"| {row['target_tok_s']:.1f} | {row['required_ms_per_token']:.3f} | {reduction_s} | {row['possible']} |"
        )
    lines.append("")
    lines.append("## Top Layer/Role Linear Contributions")
    lines.append("")
    lines.append("| rank | layer_role | payload GiB | max linear wait ms/token | eliminate bound tok/s |")
    lines.append("|---:|---|---:|---:|---:|")
    for i, row in enumerate(report["top_layer_role_rows"], 1):
        lines.append(
            f"| {i} | `{row['layer_role']}` | {row['payload_gib']:.3f} | {row['wait_ms_per_token']:.3f} | {row['tok_s']:.3f} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    for item in report["interpretation"]:
        lines.append(f"- {item}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", action="append", type=pathlib.Path, required=True)
    parser.add_argument("--out-json", type=pathlib.Path, required=True)
    parser.add_argument("--out-md", type=pathlib.Path, required=True)
    parser.add_argument("--roles", default="up,gate,down")
    parser.add_argument("--max-jobs", type=int, default=8)
    parser.add_argument("--keep-ratios", default="0.75,0.50,0.25,0.0")
    parser.add_argument("--targets", default="2.0,5.0")
    parser.add_argument("--baseline-decode-ms", type=float, required=True)
    parser.add_argument("--exclude-prompt-regex", default=r"(^|[_-])(test|holdout|heldout)([_-]|$)")
    args = parser.parse_args()

    roles = {role.strip() for role in args.roles.split(",") if role.strip()}
    keep_ratios = [float(item) for item in args.keep_ratios.split(",") if item.strip()]
    targets = [float(item) for item in args.targets.split(",") if item.strip()]
    traces = discover_traces(args.input_root, re.compile(args.exclude_prompt_regex, re.IGNORECASE))
    if not traces:
        raise SystemExit("no io-read-trace.csv files found")
    batches = load_batches(traces, args.max_jobs, roles)
    decode_runs = load_decode_runs(traces)
    if decode_runs <= 0:
        raise SystemExit("decode_runs is missing from metrics.json")

    total_wait_ms = sum(batch.wait_ms for batch in batches)
    total_payload = sum(batch.bytes for batch in batches)
    baseline_tok_s = decode_runs * 1000.0 / args.baseline_decode_ms

    role_payload: dict[str, int] = {}
    role_wait: dict[str, float] = {}
    layer_role_payload: dict[str, int] = {}
    layer_role_wait: dict[str, float] = {}
    for batch in batches:
        for role, nbytes in batch.role_bytes.items():
            role_payload[role] = role_payload.get(role, 0) + nbytes
        for layer_role, nbytes in batch.layer_role_bytes.items():
            layer_role_payload[layer_role] = layer_role_payload.get(layer_role, 0) + nbytes
        for role in roles:
            role_wait[role] = role_wait.get(role, 0.0) + contribution_ms(batch, selected_roles={role})
        for layer_role in batch.layer_role_bytes:
            layer_role_wait[layer_role] = layer_role_wait.get(layer_role, 0.0) + contribution_ms(
                batch, selected_layer_roles={layer_role}
            )

    all_wait = sum(contribution_ms(batch, selected_roles=roles) for batch in batches)
    role_rows = []
    for role in sorted(roles):
        wait_ms = role_wait.get(role, 0.0)
        bounded_decode_ms = args.baseline_decode_ms - wait_ms
        role_rows.append(
            {
                "role": role,
                "payload_gib": role_payload.get(role, 0) / GIB,
                "wait_ms": wait_ms,
                "wait_ms_per_token": wait_ms / decode_runs,
                "tok_s": decode_runs * 1000.0 / bounded_decode_ms if bounded_decode_ms > 0 else 0.0,
            }
        )

    scenario_defs = [
        ("gate", {"gate"}),
        ("up", {"up"}),
        ("down", {"down"}),
        ("up+gate", {"up", "gate"}),
        ("all", roles),
    ]
    scenario_rows = []
    for name, selected in scenario_defs:
        max_savable = sum(contribution_ms(batch, selected_roles=selected) for batch in batches)
        for keep_ratio in keep_ratios:
            scenario_rows.append(
                scenario_result(name, max_savable, keep_ratio, args.baseline_decode_ms, decode_runs)
            )

    target_rows = []
    for target in targets:
        target_decode_ms = decode_runs * 1000.0 / target
        required_ms = args.baseline_decode_ms - target_decode_ms
        req = required_reduction(all_wait, required_ms)
        target_rows.append(
            {
                "target_tok_s": target,
                "target_decode_ms": target_decode_ms,
                "required_ms": required_ms,
                "required_ms_per_token": required_ms / decode_runs,
                **req,
            }
        )

    top_layer_role_rows = []
    for key, wait_ms in sorted(layer_role_wait.items(), key=lambda item: item[1], reverse=True)[:40]:
        bounded_decode_ms = args.baseline_decode_ms - wait_ms
        top_layer_role_rows.append(
            {
                "layer_role": key,
                "payload_gib": layer_role_payload.get(key, 0) / GIB,
                "wait_ms": wait_ms,
                "wait_ms_per_token": wait_ms / decode_runs,
                "tok_s": decode_runs * 1000.0 / bounded_decode_ms if bounded_decode_ms > 0 else 0.0,
            }
        )

    all_50 = scenario_result("all", all_wait, 0.5, args.baseline_decode_ms, decode_runs)
    all_75 = scenario_result("all", all_wait, 0.25, args.baseline_decode_ms, decode_runs)
    all_100 = scenario_result("all", all_wait, 0.0, args.baseline_decode_ms, decode_runs)
    interpretation = [
        (
            f"All-role 50% byte reduction is bounded at {all_50['bounded_tok_s']:.3f} tok/s; "
            f"75% reduction is bounded at {all_75['bounded_tok_s']:.3f} tok/s."
        ),
        (
            f"Eliminating all profiled IO wait entirely is bounded at {all_100['bounded_tok_s']:.3f} tok/s, "
            "so reaching 5 tok/s cannot be achieved by IO-byte reduction alone on this baseline."
        ),
        (
            "This is a linear exposed-wait bound, not a quality result. Any lower-byte runtime change still "
            "must pass cold-start generalized-prompt quality, TTFT and 16GB host-RAM gates before it can be SOTA."
        ),
    ]

    report = {
        "traces": [str(trace) for trace in traces],
        "batch_count": len(batches),
        "decode_runs": decode_runs,
        "baseline_decode_ms": args.baseline_decode_ms,
        "baseline_tok_s": baseline_tok_s,
        "total_wait_ms": total_wait_ms,
        "total_payload_bytes": total_payload,
        "total_payload_gib": total_payload / GIB,
        "role_rows": role_rows,
        "scenario_rows": scenario_rows,
        "target_rows": target_rows,
        "top_layer_role_rows": top_layer_role_rows,
        "interpretation": interpretation,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(args.out_md, report)


if __name__ == "__main__":
    main()
