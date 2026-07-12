#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
from collections import defaultdict
from typing import Any


GIB = 1024 ** 3
LAYER_RE = re.compile(r"blk\.(\d+)\.")
KV_RE = re.compile(r"([A-Za-z0-9_]+)=([0-9.]+)")


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


def fmt(value: float) -> str:
    return f"{value:.3f}"


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_metrics(run_dir: pathlib.Path) -> dict[str, Any]:
    path = run_dir / "metrics.json"
    if path.exists():
        try:
            return json.loads(path.read_text(errors="replace"))
        except json.JSONDecodeError:
            return {}
    return {}


def parse_kv(line: str) -> dict[str, float]:
    return {key: fnum(value) for key, value in KV_RE.findall(line)}


def layer_from_tensor(name: str) -> int | None:
    match = LAYER_RE.search(name or "")
    return int(match.group(1)) if match else None


def role_from_tensor(name: str) -> str:
    if "ffn_up_exps" in name:
        return "up"
    if "ffn_gate_exps" in name:
        return "gate"
    if "ffn_down_exps" in name:
        return "down"
    return "other"


def read_git(run_dir: pathlib.Path) -> dict[str, str]:
    out: dict[str, str] = {}
    path = run_dir / "git.txt"
    if not path.exists():
        return out
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def iter_run_dirs(root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "metrics.json").exists()
    )


def parse_cpu_moe_profiles(stderr_path: pathlib.Path) -> dict[str, dict[str, float]]:
    profiles: dict[str, dict[str, float]] = {}
    if not stderr_path.exists():
        return profiles
    for line in stderr_path.read_text(errors="replace").splitlines():
        if "[kimi_cpu_moe_profile]" not in line:
            continue
        match = re.search(r"\[kimi_cpu_moe_profile\]\s+([A-Za-z0-9_]+)\s+", line)
        if not match:
            continue
        op = match.group(1)
        profiles[op] = parse_kv(line)
    return profiles


def parse_name_eligibility(stderr_path: pathlib.Path) -> dict[str, Any]:
    summary = {
        "entries": 0,
        "unsupported_entries": 0,
        "declined_entries": 0,
        "unsupported_decode": 0,
        "declined_calls": 0,
        "eligible_calls": 0,
        "accept_calls": 0,
    }
    if not stderr_path.exists():
        return summary
    current_name: str | None = None
    for line in stderr_path.read_text(errors="replace").splitlines():
        if "[kimi_cpu_moe_name_profile]" in line:
            name_match = re.search(r"name=([^ ]+)", line)
            current_name = name_match.group(1) if name_match else None
            kv = parse_kv(line)
            summary["entries"] += 1
            summary["declined_calls"] += int(kv.get("batch_decline", 0))
            summary["accept_calls"] += int(kv.get("batch_accept", 0))
            if kv.get("batch_decline", 0) > 0:
                summary["declined_entries"] += 1
        elif "[kimi_cpu_moe_eligibility_profile]" in line:
            kv = parse_kv(line)
            unsupported = int(kv.get("unsupported", 0))
            decode_unsupported = int(kv.get("decode_unsupported", 0))
            summary["eligible_calls"] += int(kv.get("eligible", 0))
            summary["unsupported_decode"] += decode_unsupported
            if unsupported > 0 or decode_unsupported > 0:
                summary["unsupported_entries"] += 1
            current_name = None
    return summary


def summarize_upgate(run_dir: pathlib.Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [row for row in read_csv(run_dir / "up-gate-profile.csv") if row.get("mode") == "decode"]
    total = defaultdict(float)
    by_layer: dict[int, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        layer = layer_from_tensor(row.get("up_tensor", ""))
        if layer is None:
            continue
        up_hits = inum(row.get("up_cache_hits"))
        up_miss = inum(row.get("up_cache_misses"))
        gate_hits = inum(row.get("gate_cache_hits"))
        gate_miss = inum(row.get("gate_cache_misses"))
        wait = max(fnum(row.get("up_wait_ms")), fnum(row.get("gate_wait_ms")))
        fields = {
            "calls": 1,
            "wall_ms": fnum(row.get("wall_ms")),
            "stage_ms": fnum(row.get("stage_ms")),
            "wait_ms": wait,
            "kernel_ms": fnum(row.get("kernel_ms")),
            "fuse_ms": fnum(row.get("fuse_ms")),
            "up_compute_ms": fnum(row.get("up_compute_ms")),
            "gate_compute_ms": fnum(row.get("gate_compute_ms")),
            "up_hits": up_hits,
            "up_misses": up_miss,
            "gate_hits": gate_hits,
            "gate_misses": gate_miss,
            "up_stage_jobs": inum(row.get("up_stage_jobs")),
            "gate_stage_jobs": inum(row.get("gate_stage_jobs")),
            "parallel_stage_calls": 1 if inum(row.get("parallel_stage")) else 0,
        }
        for key, value in fields.items():
            total[key] += value
            by_layer[layer][key] += value
    layer_rows: list[dict[str, Any]] = []
    for layer, acc in by_layer.items():
        calls = acc["calls"] or 1
        up_total = acc["up_hits"] + acc["up_misses"]
        gate_total = acc["gate_hits"] + acc["gate_misses"]
        layer_rows.append({
            "layer": layer,
            "calls": int(acc["calls"]),
            "wall_ms": acc["wall_ms"],
            "wait_ms": acc["wait_ms"],
            "stage_ms": acc["stage_ms"],
            "kernel_ms": acc["kernel_ms"],
            "up_hit_rate": acc["up_hits"] / up_total if up_total else 0.0,
            "gate_hit_rate": acc["gate_hits"] / gate_total if gate_total else 0.0,
            "up_miss_per_call": acc["up_misses"] / calls,
            "gate_miss_per_call": acc["gate_misses"] / calls,
            "avg_stage_jobs": (acc["up_stage_jobs"] + acc["gate_stage_jobs"]) / calls,
            "parallel_stage_rate": acc["parallel_stage_calls"] / calls,
        })
    layer_rows.sort(key=lambda row: row["wall_ms"], reverse=True)
    summary = dict(total)
    summary["rows"] = len(rows)
    return summary, layer_rows


def summarize_down(run_dir: pathlib.Path, max_decode_active: int) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    rows = read_csv(run_dir / "down-batch-profile.csv")
    decode_rows = [row for row in rows if inum(row.get("n_active")) <= max_decode_active]
    prompt_rows = [row for row in rows if inum(row.get("n_active")) > max_decode_active]
    total = defaultdict(float)
    prompt = defaultdict(float)
    by_layer: dict[int, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))

    def add_row(acc: defaultdict[str, float], row: dict[str, str]) -> None:
        acc["calls"] += 1
        acc["wall_ms"] += fnum(row.get("wall_ms"))
        acc["stage_ms"] += fnum(row.get("stage_ms"))
        acc["kernel_ms"] += fnum(row.get("kernel_ms"))
        acc["cache_hits"] += inum(row.get("cache_hits"))
        acc["cache_misses"] += inum(row.get("cache_misses"))
        acc["staged_jobs"] += inum(row.get("staged_jobs"))

    for row in decode_rows:
        layer = layer_from_tensor(row.get("tensor", ""))
        add_row(total, row)
        if layer is not None:
            add_row(by_layer[layer], row)
    for row in prompt_rows:
        add_row(prompt, row)

    layer_rows: list[dict[str, Any]] = []
    for layer, acc in by_layer.items():
        calls = acc["calls"] or 1
        total_events = acc["cache_hits"] + acc["cache_misses"]
        layer_rows.append({
            "layer": layer,
            "calls": int(acc["calls"]),
            "wall_ms": acc["wall_ms"],
            "stage_ms": acc["stage_ms"],
            "kernel_ms": acc["kernel_ms"],
            "hit_rate": acc["cache_hits"] / total_events if total_events else 0.0,
            "miss_per_call": acc["cache_misses"] / calls,
            "staged_jobs_per_call": acc["staged_jobs"] / calls,
        })
    layer_rows.sort(key=lambda row: row["wall_ms"], reverse=True)
    total["rows"] = len(decode_rows)
    prompt["rows"] = len(prompt_rows)
    return dict(total), layer_rows, dict(prompt)


def summarize_fallback(run_dir: pathlib.Path) -> dict[str, Any]:
    rows = read_csv(run_dir / "fallback-profile.csv")
    total_us = sum(inum(row.get("fallback_us")) for row in rows)
    by_phase_role: dict[str, int] = defaultdict(int)
    for row in rows:
        phase = row.get("phase", "")
        role = role_from_tensor(row.get("tensor", ""))
        by_phase_role[f"{phase}:{role}"] += inum(row.get("fallback_us"))
    return {
        "rows": len(rows),
        "fallback_ms": total_us / 1000.0,
        "by_phase_role_ms": {key: value / 1000.0 for key, value in sorted(by_phase_role.items())},
    }


def cpu_op_row(prompt_id: str, op: str, kv: dict[str, float]) -> dict[str, Any]:
    calls = kv.get("calls", 0.0)
    batch_accept = kv.get("batch_accept", 0.0)
    batch_decline = kv.get("batch_decline", 0.0)
    decode_calls = kv.get("decode_calls", 0.0)
    prompt_calls = kv.get("prompt_calls", 0.0)
    return {
        "prompt": prompt_id,
        "op": op,
        "calls": int(calls),
        "batch_accept": int(batch_accept),
        "batch_decline": int(batch_decline),
        "accept_rate": batch_accept / calls if calls else 0.0,
        "decode_calls": int(decode_calls),
        "prompt_calls": int(prompt_calls),
        "decode_total_ms_per_call": kv.get("decode_total", 0.0),
        "decode_cuda_ms_per_call": kv.get("decode_cuda", 0.0),
        "decode_fallback_ms_per_call": kv.get("decode_fallback", 0.0),
        "prompt_total_ms_per_call": kv.get("prompt_total", 0.0),
        "prompt_cuda_ms_per_call": kv.get("prompt_cuda", 0.0),
        "prompt_fallback_ms_per_call": kv.get("prompt_fallback", 0.0),
        "extension_miss": batch_decline > 0 or (calls > 0 and batch_accept < calls),
    }


def summarize_run(run_dir: pathlib.Path, max_decode_active: int) -> dict[str, Any]:
    metrics = read_metrics(run_dir)
    prompt_id = str(metrics.get("prompt_id") or run_dir.name)
    decode_runs = inum(metrics.get("decode_runs"))
    memory_peak = inum(metrics.get("memory.peak", metrics.get("memory_peak")))
    upgate, upgate_layers = summarize_upgate(run_dir)
    down, down_layers, prompt_down = summarize_down(run_dir, max_decode_active)
    fallback = summarize_fallback(run_dir)
    cpu_profiles = parse_cpu_moe_profiles(run_dir / "stderr.txt")
    cpu_rows = [cpu_op_row(prompt_id, op, kv) for op, kv in sorted(cpu_profiles.items())]
    name_summary = parse_name_eligibility(run_dir / "stderr.txt")
    extension_misses = [row for row in cpu_rows if row["extension_miss"]]
    true_fallback = fallback["rows"] > 0 or fallback["fallback_ms"] > 0.0
    return {
        "prompt_id": prompt_id,
        "run_dir": str(run_dir),
        "quality": metrics.get("quality", "unknown"),
        "token_rate": fnum(metrics.get("token_rate")),
        "ttft_ms": fnum(metrics.get("ttft_ms")),
        "decode_ms": fnum(metrics.get("decode_ms")),
        "decode_runs": decode_runs,
        "decode_ms_per_token": fnum(metrics.get("decode_ms")) / decode_runs if decode_runs else 0.0,
        "memory_peak": memory_peak,
        "active_file": inum(metrics.get("active_file")),
        "inactive_file": inum(metrics.get("inactive_file")),
        "upgate": upgate,
        "upgate_layers": upgate_layers,
        "down": down,
        "down_layers": down_layers,
        "prompt_down": prompt_down,
        "fallback": fallback,
        "cpu_rows": cpu_rows,
        "name_summary": name_summary,
        "true_fallback": true_fallback,
        "extension_miss": bool(extension_misses),
        "git": read_git(run_dir),
    }


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    total_decode_runs = sum(run["decode_runs"] for run in runs)
    total_decode_ms = sum(run["decode_ms"] for run in runs)
    agg = {
        "runs": len(runs),
        "total_decode_runs": total_decode_runs,
        "total_decode_ms": total_decode_ms,
        "weighted_ms_per_token": total_decode_ms / total_decode_runs if total_decode_runs else 0.0,
        "weighted_token_rate": 1000.0 / (total_decode_ms / total_decode_runs) if total_decode_runs and total_decode_ms else 0.0,
        "true_fallback_runs": sum(1 for run in runs if run["true_fallback"]),
        "extension_miss_runs": sum(1 for run in runs if run["extension_miss"]),
        "fallback_rows": sum(run["fallback"]["rows"] for run in runs),
        "fallback_ms": sum(run["fallback"]["fallback_ms"] for run in runs),
        "upgate_wall_ms": sum(run["upgate"].get("wall_ms", 0.0) for run in runs),
        "upgate_wait_ms": sum(run["upgate"].get("wait_ms", 0.0) for run in runs),
        "upgate_stage_ms": sum(run["upgate"].get("stage_ms", 0.0) for run in runs),
        "down_wall_ms": sum(run["down"].get("wall_ms", 0.0) for run in runs),
        "down_stage_ms": sum(run["down"].get("stage_ms", 0.0) for run in runs),
        "prompt_down_wall_ms": sum(run["prompt_down"].get("wall_ms", 0.0) for run in runs),
    }
    return agg


def make_rows(runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    per_prompt: list[dict[str, Any]] = []
    cpu_rows: list[dict[str, Any]] = []
    up_layers_acc: dict[int, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    down_layers_acc: dict[int, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    for run in runs:
        decode_runs = run["decode_runs"] or 1
        per_prompt.append({
            "prompt": run["prompt_id"],
            "quality": run["quality"],
            "token_rate": fmt(run["token_rate"]),
            "decode_ms_per_token": fmt(run["decode_ms_per_token"]),
            "ttft_ms": fmt(run["ttft_ms"]),
            "ram_peak_gib": fmt(run["memory_peak"] / GIB),
            "fallback_rows": run["fallback"]["rows"],
            "extension_miss": int(run["extension_miss"]),
            "upgate_ms_per_token": fmt(run["upgate"].get("wall_ms", 0.0) / decode_runs),
            "upgate_wait_ms_per_token": fmt(run["upgate"].get("wait_ms", 0.0) / decode_runs),
            "down_ms_per_token": fmt(run["down"].get("wall_ms", 0.0) / decode_runs),
            "down_stage_ms_per_token": fmt(run["down"].get("stage_ms", 0.0) / decode_runs),
        })
        cpu_rows.extend(run["cpu_rows"])
        for row in run["upgate_layers"]:
            acc = up_layers_acc[int(row["layer"])]
            for key in ["calls", "wall_ms", "wait_ms", "stage_ms", "kernel_ms", "up_miss_per_call", "gate_miss_per_call", "avg_stage_jobs"]:
                acc[key] += float(row[key])
            acc["row_count"] += 1
        for row in run["down_layers"]:
            acc = down_layers_acc[int(row["layer"])]
            for key in ["calls", "wall_ms", "stage_ms", "kernel_ms", "miss_per_call", "staged_jobs_per_call"]:
                acc[key] += float(row[key])
            acc["row_count"] += 1

    up_rows: list[dict[str, Any]] = []
    for layer, acc in up_layers_acc.items():
        denom = acc["row_count"] or 1.0
        calls = acc["calls"] or 1.0
        up_rows.append({
            "layer": layer,
            "calls": int(acc["calls"]),
            "wall_ms": fmt(acc["wall_ms"]),
            "wait_ms": fmt(acc["wait_ms"]),
            "stage_ms": fmt(acc["stage_ms"]),
            "kernel_ms": fmt(acc["kernel_ms"]),
            "up_miss_per_call": fmt(acc["up_miss_per_call"] / denom),
            "gate_miss_per_call": fmt(acc["gate_miss_per_call"] / denom),
            "avg_stage_jobs": fmt(acc["avg_stage_jobs"] / denom),
        })
    up_rows.sort(key=lambda row: float(row["wall_ms"]), reverse=True)

    down_rows: list[dict[str, Any]] = []
    for layer, acc in down_layers_acc.items():
        denom = acc["row_count"] or 1.0
        down_rows.append({
            "layer": layer,
            "calls": int(acc["calls"]),
            "wall_ms": fmt(acc["wall_ms"]),
            "stage_ms": fmt(acc["stage_ms"]),
            "kernel_ms": fmt(acc["kernel_ms"]),
            "miss_per_call": fmt(acc["miss_per_call"] / denom),
            "staged_jobs_per_call": fmt(acc["staged_jobs_per_call"] / denom),
        })
    down_rows.sort(key=lambda row: float(row["wall_ms"]), reverse=True)
    return per_prompt, cpu_rows, up_rows, down_rows


def write_report(out: pathlib.Path, input_root: pathlib.Path, runs: list[dict[str, Any]], agg: dict[str, Any], per_prompt: list[dict[str, Any]], cpu_rows: list[dict[str, Any]], up_rows: list[dict[str, Any]], down_rows: list[dict[str, Any]], top_n: int) -> None:
    first_git = runs[0]["git"] if runs else {}
    branch = first_git.get("branch", "unknown")
    head = first_git.get("short", first_git.get("head", "unknown"))
    copy_profile_runs = sum(1 for run in runs if (pathlib.Path(run["run_dir"]) / "copy-profile.csv").exists())
    decision = "fallback_hook_not_next"
    if agg["true_fallback_runs"] or agg["extension_miss_runs"]:
        decision = "investigate_extension_miss_before_scheduler_work"
    lines = [
        "# Kimi CPU/defer GPU-extension audit",
        "",
        f"Input root: `{input_root}`",
        f"Branch at source run: `{branch}`",
        f"Commit at source run: `{head}`",
        f"Runs: `{agg['runs']}`",
        "",
        "## Verdict",
        "",
        f"- Decision: `{decision}`.",
        f"- True CPU fallback rows: `{agg['fallback_rows']}`; fallback ms: `{fmt(agg['fallback_ms'])}`.",
        f"- Runs with CPU/defer extension miss: `{agg['extension_miss_runs']}`.",
        f"- Weighted decode: `{fmt(agg['weighted_ms_per_token'])} ms/token`, `{fmt(agg['weighted_token_rate'])} tok/s`.",
        f"- Up/gate GPU-extension wall: `{fmt(agg['upgate_wall_ms'] / max(agg['total_decode_runs'], 1))} ms/token`.",
        f"- Up/gate exposed wait proxy: `{fmt(agg['upgate_wait_ms'] / max(agg['total_decode_runs'], 1))} ms/token`.",
        f"- Decode down GPU-extension wall: `{fmt(agg['down_wall_ms'] / max(agg['total_decode_runs'], 1))} ms/token`.",
        "",
    ]
    if decision == "fallback_hook_not_next":
        lines += [
            "Interpretation:",
            "",
            "- The DeepSeek CPU/defer GPU-extension pattern is already active for Kimi in this profiled SOTA path.",
            "- `up_gate` and `down` CPU/defer ops are accepted by the CUDA batch extension; the fallback CSV is empty.",
            "- The next implementation should not be another broad CPU fallback rewrite.",
            "- The next bottleneck to attack is exposed up/gate staging and io_uring demand-read starvation, followed by down staging.",
            "",
        ]
    else:
        lines += [
            "Interpretation:",
            "",
            "- At least one run still has fallback rows or CPU/defer extension declines.",
            "- The next implementation should first isolate those role/layer/dtype predicates before scheduler work.",
            "",
        ]

    lines += [
        "## Per Prompt",
        "",
        "| Prompt | quality | tok/s | decode ms/token | TTFT ms | RAM peak GiB | fallback rows | extension miss | up/gate ms/token | up/gate wait ms/token | down ms/token | down stage ms/token |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in per_prompt:
        lines.append(
            f"| `{row['prompt']}` | {row['quality']} | {row['token_rate']} | {row['decode_ms_per_token']} | "
            f"{row['ttft_ms']} | {row['ram_peak_gib']} | {row['fallback_rows']} | {row['extension_miss']} | "
            f"{row['upgate_ms_per_token']} | {row['upgate_wait_ms_per_token']} | {row['down_ms_per_token']} | {row['down_stage_ms_per_token']} |"
        )

    lines += [
        "",
        "## CPU/defer Op Acceptance",
        "",
        "| Prompt | op | calls | batch accept | batch decline | accept rate | decode calls | prompt calls | decode cuda ms/call | decode fallback ms/call | prompt cuda ms/call | prompt fallback ms/call |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in cpu_rows:
        lines.append(
            f"| `{row['prompt']}` | `{row['op']}` | {row['calls']} | {row['batch_accept']} | "
            f"{row['batch_decline']} | {fmt(row['accept_rate'])} | {row['decode_calls']} | {row['prompt_calls']} | "
            f"{fmt(row['decode_cuda_ms_per_call'])} | {fmt(row['decode_fallback_ms_per_call'])} | "
            f"{fmt(row['prompt_cuda_ms_per_call'])} | {fmt(row['prompt_fallback_ms_per_call'])} |"
        )

    lines += [
        "",
        f"## Top {top_n} Up/Gate Layers By Wall",
        "",
        "| layer | calls | wall ms | wait ms | stage ms | kernel ms | up miss/call | gate miss/call | avg stage jobs |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in up_rows[:top_n]:
        lines.append(
            f"| {row['layer']} | {row['calls']} | {row['wall_ms']} | {row['wait_ms']} | {row['stage_ms']} | "
            f"{row['kernel_ms']} | {row['up_miss_per_call']} | {row['gate_miss_per_call']} | {row['avg_stage_jobs']} |"
        )

    lines += [
        "",
        f"## Top {top_n} Decode Down Layers By Wall",
        "",
        "| layer | calls | wall ms | stage ms | kernel ms | miss/call | staged jobs/call |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in down_rows[:top_n]:
        lines.append(
            f"| {row['layer']} | {row['calls']} | {row['wall_ms']} | {row['stage_ms']} | "
            f"{row['kernel_ms']} | {row['miss_per_call']} | {row['staged_jobs_per_call']} |"
        )

    lines += [
        "",
        "## Evidence Scope",
        "",
        "- This audit uses existing `metrics.json`, `stderr.txt`, `fallback-profile.csv`, `up-gate-profile.csv`, and `down-batch-profile.csv` artifacts.",
        "- It proves CPU fallback and CUDA batch-extension acceptance for the profiled runs.",
        f"- Runs with `copy-profile.csv`: `{copy_profile_runs}/{len(runs)}`.",
    ]
    if copy_profile_runs:
        lines += [
            "- For runs collected with `COPY_PROFILE=1`, use `kimi_copy_profile_breakdown.py` to split expert-pack/io_uring wait from H2D enqueue/copy.",
        ]
    else:
        lines += [
            "- It does not split H2D from host staging because no input run was collected with `COPY_PROFILE=1`.",
        ]
    lines += [
        "- A later source-change A/B must still run fresh cold-start profiles with copy/io traces before claiming SOTA.",
        "",
        "## Next Action",
        "",
        "1. Skip broad Kimi fallback hook work unless a fresh fallback-reason profile contradicts this audit.",
        "2. Profile with `COPY_PROFILE=1` and IO batch traces on France plus a held-out prompt.",
        "3. Design the next default-off A/B around reducing up/gate demand-read wait and preserving down overlap.",
        "4. Reject any cache/RAM policy that improves hit rate but does not reduce endpoint decode time under the 16GB RAM gate.",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit Kimi CPU/defer MoE GPU-extension coverage from profiled runs.")
    ap.add_argument("--input", required=True, type=pathlib.Path, help="Run root containing prompt subdirectories.")
    ap.add_argument("--out", required=True, type=pathlib.Path, help="Output directory.")
    ap.add_argument("--decode-down-max-active", type=int, default=8)
    ap.add_argument("--top-n", type=int, default=20)
    args = ap.parse_args()

    run_dirs = iter_run_dirs(args.input)
    if not run_dirs:
        raise SystemExit(f"No prompt run directories with metrics.json found under {args.input}")
    runs = [summarize_run(run_dir, args.decode_down_max_active) for run_dir in run_dirs]
    agg = aggregate(runs)
    per_prompt, cpu_rows, up_rows, down_rows = make_rows(runs)

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "per_prompt.csv", per_prompt, [
        "prompt", "quality", "token_rate", "decode_ms_per_token", "ttft_ms", "ram_peak_gib",
        "fallback_rows", "extension_miss", "upgate_ms_per_token", "upgate_wait_ms_per_token",
        "down_ms_per_token", "down_stage_ms_per_token",
    ])
    write_csv(args.out / "cpu_defer_ops.csv", cpu_rows, [
        "prompt", "op", "calls", "batch_accept", "batch_decline", "accept_rate",
        "decode_calls", "prompt_calls", "decode_total_ms_per_call", "decode_cuda_ms_per_call",
        "decode_fallback_ms_per_call", "prompt_total_ms_per_call", "prompt_cuda_ms_per_call",
        "prompt_fallback_ms_per_call", "extension_miss",
    ])
    write_csv(args.out / "top_upgate_layers.csv", up_rows, [
        "layer", "calls", "wall_ms", "wait_ms", "stage_ms", "kernel_ms",
        "up_miss_per_call", "gate_miss_per_call", "avg_stage_jobs",
    ])
    write_csv(args.out / "top_down_layers.csv", down_rows, [
        "layer", "calls", "wall_ms", "stage_ms", "kernel_ms", "miss_per_call", "staged_jobs_per_call",
    ])
    summary = {
        "input": str(args.input),
        "aggregate": agg,
        "runs": runs,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_report(args.out, args.input, runs, agg, per_prompt, cpu_rows, up_rows, down_rows, args.top_n)


if __name__ == "__main__":
    main()
