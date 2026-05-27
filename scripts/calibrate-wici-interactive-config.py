#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import posixpath
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_REMOTE_DIR = "/home/wici/venti/ik_llama"
DEFAULT_TRACE_DIR = "bench/wici-glm51-interactive-latency/traces"
DEFAULT_TTFT_TRACE_DIR = "bench/wici-glm51-interactive-latency/ttft"
DEFAULT_PROMPT = "Write a concise explanation of why NVMe latency matters for MoE inference."
EXPECTED_N36_SHA = "6d9e7cba8a78e6ea35d4ee4c56f4730642cf7cb78393779eb1d70b50508ed8b9"
HISTORICAL_TTFT_S = 15.07
DEFAULT_MAX_TTFT_S = round(HISTORICAL_TTFT_S * 0.90, 2)
PRESET_RE = re.compile(r"^preset=(?P<path>.+)$", re.MULTILINE)
PRESET_JSON_RE = re.compile(r"^preset_json=(?P<json>.+)$", re.MULTILINE)

BASE_ENV = {
    "MLA": "3",
    "CPU_MOE": "1",
    "IGNORE_EOS": "1",
    "GGML_CUDA_NO_PINNED": "1",
    "GGML_MOE_STREAM": "1",
    "GGML_MOE_STREAM_BATCH_ONLY": "1",
    "GGML_MOE_STREAM_DEFER": "1",
    "GGML_MOE_STREAM_CPU_OPS": "1",
    "GGML_MOE_PARALLEL_EXPERTS": "1",
    "GGML_MOE_STREAM_FUSED_UP_GATE": "1",
    "GGML_MOE_VRAM_CACHE_SPLIT": "1",
    "GGML_MOE_STREAM_ONE_CACHE_MIB": "0",
    "GGML_MOE_VRAM_CACHE_POLICY": "lfu_lru",
    "GGML_MOE_STAGE_PINNED_SLOTS": "16",
    "GGML_MOE_PREFETCH_DOWN": "0",
    "GGML_MOE_PREFETCH_DOWN_DEPTH": "8",
}


def run_cmd(cmd: list[str], *, timeout: int | None = None, dry_run: bool = False) -> subprocess.CompletedProcess[str]:
    print(" ".join(shlex.quote(part) for part in cmd), file=sys.stderr)
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)


def remote_path(args: argparse.Namespace, path: str) -> str:
    if posixpath.isabs(path):
        return path
    remote_dir = getattr(args, "remote_dir", ".")
    return posixpath.join(remote_dir, path)


def remote_cat(args: argparse.Namespace, path: str, timeout: int = 30) -> str:
    resolved = remote_path(args, path)
    result = run_cmd(["ssh", args.ssh_host, f"cat -- {shlex.quote(resolved)}"], timeout=timeout, dry_run=args.dry_run)
    if result.returncode != 0:
        raise SystemExit(result.stderr + result.stdout)
    return result.stdout


def remote_file_exists(args: argparse.Namespace, path: str, timeout: int = 30) -> bool:
    resolved = remote_path(args, path)
    result = run_cmd(["ssh", args.ssh_host, f"test -f {shlex.quote(resolved)}"], timeout=timeout, dry_run=args.dry_run)
    return args.dry_run or result.returncode == 0


def remote_file_contains(args: argparse.Namespace, path: str, needle: str, timeout: int = 30) -> bool:
    resolved = remote_path(args, path)
    result = run_cmd(
        ["ssh", args.ssh_host, f"test -f {shlex.quote(resolved)} && grep -F -- {shlex.quote(needle)} {shlex.quote(resolved)} >/dev/null"],
        timeout=timeout,
        dry_run=args.dry_run,
    )
    return args.dry_run or result.returncode == 0


def remote_cat_when_ready(args: argparse.Namespace, path: str, wait_s: float = 180.0) -> str | None:
    deadline = time.monotonic() + wait_s
    while True:
        if remote_file_exists(args, path):
            return remote_cat(args, path)
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.25)


def remote_file_contains_when_ready(
    args: argparse.Namespace,
    path: str,
    needle: str,
    wait_s: float = 180.0,
) -> bool:
    deadline = time.monotonic() + wait_s
    while True:
        if remote_file_contains(args, path, needle):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.25)


def remote_write_text(args: argparse.Namespace, path: str, text: str, timeout: int = 30) -> None:
    remote_dir = posixpath.dirname(path.rstrip("/"))
    remote_cmd = f"mkdir -p {shlex.quote(remote_dir)} && cat > {shlex.quote(path)}"
    print(f"ssh {shlex.quote(args.ssh_host)} {shlex.quote(remote_cmd)}", file=sys.stderr)
    if args.dry_run:
        return
    result = subprocess.run(
        ["ssh", args.ssh_host, remote_cmd],
        input=text,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(result.stderr + result.stdout)


def wait_for_gpu_idle(args: argparse.Namespace, label: str) -> None:
    if args.dry_run or args.gpu_idle_timeout_s <= 0:
        return
    deadline = time.monotonic() + args.gpu_idle_timeout_s
    cmd = [
        "ssh",
        args.ssh_host,
        "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits",
    ]
    last = ""
    while time.monotonic() < deadline:
        result = run_cmd(cmd, timeout=15)
        last = (result.stdout or result.stderr).strip()
        if result.returncode == 0:
            line = result.stdout.strip().splitlines()[0]
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 2:
                try:
                    used_mib = int(parts[0])
                    util_pct = int(parts[1])
                except ValueError:
                    used_mib = args.gpu_idle_max_used_mib + 1
                    util_pct = args.gpu_idle_max_util_pct + 1
                if used_mib <= args.gpu_idle_max_used_mib and util_pct <= args.gpu_idle_max_util_pct:
                    return
        time.sleep(args.gpu_idle_poll_s)
    raise SystemExit(f"GPU did not become idle before {label}: last nvidia-smi={last}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"missing jsonl output: {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def one_ok_record(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    ok = [row for row in rows if row.get("ok")]
    if not ok:
        raise SystemExit(f"no successful runs in {path}")
    return ok[-1]


def bench_summary_path(output: Path) -> Path:
    return output.with_suffix(".summary.json")


def bench_command(
    args: argparse.Namespace,
    output: Path,
    predict_tokens: int,
    extra: list[str],
    remote_env: list[str] | None = None,
    runs: int | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "bench-wici-glm51-interactive-latency.py"),
        "--ssh-host", args.ssh_host,
        "--remote-dir", args.remote_dir,
        "--runs", str(runs if runs is not None else args.runs),
        "--predict-tokens", str(predict_tokens),
        "--prompt", args.prompt,
        "--output", str(output),
        "--summary", str(bench_summary_path(output)),
    ]
    if args.post_ready_delay_s > 0:
        cmd.extend(["--post-ready-delay-s", str(args.post_ready_delay_s)])
    if args.chat_load_mode != "fast-prompt":
        cmd.extend(["--chat-load-mode", args.chat_load_mode])
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.chat_active_prewarm:
        cmd.append("--chat-active-prewarm")
    if args.ignore_eos:
        cmd.append("--ignore-eos")
    else:
        cmd.append("--no-ignore-eos")
    for env in [*args.remote_env, *(remote_env or [])]:
        cmd.extend(["--remote-env", env])
    cmd.extend(extra)
    return cmd


def capture_trace(args: argparse.Namespace, stamp: str) -> tuple[Path, str, dict[str, Any]]:
    output = args.output_dir / f"{stamp}-trace-capture-n{args.capture_tokens}.jsonl"
    cmd = bench_command(
        args,
        output,
        args.capture_tokens,
        ["--route-trace-dir", args.route_trace_dir, "--ttft-trace-dir", args.ttft_trace_dir],
    )
    wait_for_gpu_idle(args, "trace capture")
    result = run_cmd(cmd, timeout=args.runtime_timeout_s, dry_run=args.dry_run)
    if result.returncode != 0:
        raise SystemExit(result.stderr + result.stdout)
    if args.dry_run:
        return output, "<dry-run-route-trace>", {}
    record = one_ok_record(output)
    trace = record.get("route_trace_out")
    if not trace:
        raise SystemExit(f"capture run did not report route_trace_out: {output}")
    return output, str(trace), record


def recommend_preset(args: argparse.Namespace, trace: str) -> tuple[str, dict[str, Any]]:
    moe_args = "python3 scripts/moe-run.py --chat --dry-run --print-preset-json"
    if args.force_trace_preset:
        moe_args += f" --force --route-trace {shlex.quote(trace)}"
    remote_cmd = f"cd {shlex.quote(args.remote_dir)} && {moe_args}"
    result = run_cmd(["ssh", args.ssh_host, remote_cmd], timeout=args.runtime_timeout_s, dry_run=args.dry_run)
    if result.returncode != 0:
        raise SystemExit(result.stderr + result.stdout)
    if args.dry_run:
        return "<dry-run-preset>", {}
    match = PRESET_RE.search(result.stdout)
    if not match:
        raise SystemExit(f"moe-run dry-run did not print preset path:\n{result.stdout}")
    preset = match.group("path").strip()
    json_match = PRESET_JSON_RE.search(result.stdout)
    if json_match:
        data = json.loads(json_match.group("json"))
        if not remote_file_exists(args, preset):
            remote_write_text(args, preset, json.dumps(data, indent=2, sort_keys=True) + "\n")
        return preset, data
    data = json.loads(remote_cat(args, preset))
    return preset, data


def validate_preset(
    args: argparse.Namespace,
    stamp: str,
    trace: str,
    preset_path: str,
    *,
    label: str = "validate",
    remote_env: list[str] | None = None,
    runs: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    output = args.output_dir / f"{stamp}-{label}-n{args.validate_tokens}.jsonl"
    extra = ["--moe-preset", preset_path]
    if args.force_validation_preset:
        extra = ["--force-preset", "--moe-route-trace", trace]
    cmd = bench_command(
        args,
        output,
        args.validate_tokens,
        extra,
        remote_env=remote_env,
        runs=runs,
    )
    wait_for_gpu_idle(args, "validation")
    result = run_cmd(cmd, timeout=args.runtime_timeout_s, dry_run=args.dry_run)
    if result.returncode != 0:
        raise SystemExit(result.stderr + result.stdout)
    if args.dry_run:
        return output, {}
    return output, one_ok_record(output)


def validation_records_pass_gate(
    args: argparse.Namespace,
    output: Path,
    baseline_record: dict[str, Any] | None = None,
) -> tuple[bool, list[str], dict[str, Any]]:
    rows = read_jsonl(output)
    failures: list[str] = []
    ok_rows = [row for row in rows if row.get("ok")]
    if len(ok_rows) != len(rows) or not rows:
        failures.append(f"{output} has {len(ok_rows)}/{len(rows)} successful runs")
    for index, row in enumerate(ok_rows, start=1):
        if not validation_passes_latency_gate(args, row, baseline_record):
            failures.append(
                f"{output} run {index} failed validation gate: "
                f"ttft={row.get('interactive_ttft_s')} eval={row.get('eval_tokens_per_s')} "
                f"cache_failed={row.get('moe_vram_cache_failed')}"
            )
    if not ok_rows:
        return False, failures, {}
    worst = max(ok_rows, key=lambda row: float(row.get("interactive_ttft_s") or 1e9))
    return not failures, failures, worst


def run_canary(
    args: argparse.Namespace,
    stamp: str,
    trace: str,
    preset_path: str,
    remote_env: list[str] | None = None,
    label: str = "canary",
) -> tuple[Path, dict[str, Any], str]:
    output = args.output_dir / f"{stamp}-{label}-n36.jsonl"
    extra = ["--moe-preset", preset_path]
    if args.force_validation_preset:
        extra = ["--force-preset", "--moe-route-trace", trace]
    cmd = bench_command(
        args,
        output,
        36,
        extra,
        remote_env=remote_env,
    )
    wait_for_gpu_idle(args, "canary")
    result = run_cmd(cmd, timeout=args.runtime_timeout_s, dry_run=args.dry_run)
    if result.returncode != 0:
        raise SystemExit(result.stderr + result.stdout)
    if args.dry_run:
        return output, {}, "<dry-run-sha>"
    record = one_ok_record(output)
    digest = hashlib.sha256(str(record.get("assistant_text", "")).encode("utf-8")).hexdigest()
    return output, record, digest


def metric(record: dict[str, Any], key: str) -> float | None:
    value = record.get(key)
    return float(value) if value is not None else None


def evaluate(args: argparse.Namespace, validate_record: dict[str, Any], canary_sha: str | None) -> list[str]:
    failures = []
    if args.require_validation and not validate_record:
        failures.append("validation was required but did not run")
    if validate_record:
        if validate_record.get("moe_vram_cache_failed"):
            failures.append("validation reported moe_vram_cache_failed")
        eval_tps = metric(validate_record, "eval_tokens_per_s")
        if eval_tps is None or eval_tps < args.min_eval_tps:
            failures.append(f"eval_tokens_per_s {eval_tps} < {args.min_eval_tps}")
        ttt = metric(validate_record, "time_to_type_s")
        if args.max_time_to_type_s is not None and (ttt is None or ttt > args.max_time_to_type_s):
            failures.append(f"time_to_type_s {ttt} > {args.max_time_to_type_s}")
        ttft = metric(validate_record, "interactive_ttft_s")
        if args.max_ttft_s is not None and (ttft is None or ttft > args.max_ttft_s):
            failures.append(f"interactive_ttft_s {ttft} > {args.max_ttft_s}")
    if args.require_canary:
        if canary_sha is None:
            failures.append("canary was required but did not run")
        elif canary_sha != EXPECTED_N36_SHA:
            failures.append(f"canary sha {canary_sha} != {EXPECTED_N36_SHA}")
    return failures


def parse_int_list(value: str) -> list[int]:
    items: list[int] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            item = int(raw)
        except ValueError as exc:
            raise SystemExit(f"invalid integer in candidate list: {raw}") from exc
        if item < 0:
            raise SystemExit("startup preload candidates must be non-negative")
        items.append(item)
    if not items:
        raise SystemExit("candidate list must contain at least one integer")
    return items


def startup_preload_candidates(args: argparse.Namespace, preset: dict[str, Any]) -> list[int]:
    if args.startup_preload_candidates:
        raw_candidates = parse_int_list(args.startup_preload_candidates)
    else:
        env = {str(k): str(v) for k, v in preset.get("env", {}).items()}
        current = int(env.get("LLAMA_CHAT_STARTUP_PROFILE_PRELOAD_TENSORS", "3"))
        raw_candidates = [0, max(1, current - 1), current, current + 1]
    capped = [min(candidate, args.max_startup_preload_tensors) for candidate in raw_candidates]
    return sorted(set(capped))


def cache_budget_candidates(args: argparse.Namespace, preset: dict[str, Any]) -> list[int]:
    if args.cache_budget_candidates:
        raw_candidates = parse_int_list(args.cache_budget_candidates)
    else:
        current = preset_vram_cache_mib(preset) or 1536
        raw_candidates = [max(128, current - 128), current, current + 128]
    capped = [min(candidate, args.max_cache_budget_mib) for candidate in raw_candidates]
    return sorted(set(capped))


def profile_reserve_candidates(args: argparse.Namespace, preset: dict[str, Any]) -> list[int]:
    if args.profile_reserve_candidates:
        raw_candidates = parse_int_list(args.profile_reserve_candidates)
    else:
        current = preset_profile_reserve_pct(preset) or 10
        raw_candidates = [current, current + 10, current + 20]
    capped = [min(candidate, args.max_profile_reserve_pct) for candidate in raw_candidates]
    return sorted(set(capped))


def preset_startup_preload(preset: dict[str, Any]) -> int | None:
    env = {str(k): str(v) for k, v in preset.get("env", {}).items()}
    value = env.get("LLAMA_CHAT_STARTUP_PROFILE_PRELOAD_TENSORS")
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def preset_vram_cache_mib(preset: dict[str, Any]) -> int | None:
    env = {str(k): str(v) for k, v in preset.get("env", {}).items()}
    value = env.get("GGML_MOE_VRAM_CACHE_MIB")
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def preset_profile_reserve_pct(preset: dict[str, Any]) -> int | None:
    env = {str(k): str(v) for k, v in preset.get("env", {}).items()}
    value = env.get("GGML_MOE_VRAM_PROFILE_RESERVE_PCT")
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def ttft_improvement_pct(baseline_record: dict[str, Any] | None, record: dict[str, Any]) -> float | None:
    if not baseline_record:
        return None
    baseline_ttft = metric(baseline_record, "interactive_ttft_s")
    candidate_ttft = metric(record, "interactive_ttft_s")
    if baseline_ttft is None or baseline_ttft <= 0 or candidate_ttft is None:
        return None
    return 100.0 * (baseline_ttft - candidate_ttft) / baseline_ttft


def validation_passes_latency_gate(
    args: argparse.Namespace,
    record: dict[str, Any],
    baseline_record: dict[str, Any] | None = None,
) -> bool:
    if not record or not record.get("ok"):
        return False
    if record.get("moe_vram_cache_failed"):
        return False
    eval_tps = metric(record, "eval_tokens_per_s")
    if eval_tps is None or eval_tps < args.min_eval_tps:
        return False
    ttft = metric(record, "interactive_ttft_s")
    if ttft is None:
        return False
    if args.max_ttft_s is not None and ttft > args.max_ttft_s:
        return False
    ttt = metric(record, "time_to_type_s")
    if args.max_time_to_type_s is not None and (ttt is None or ttt > args.max_time_to_type_s):
        return False
    improvement = ttft_improvement_pct(baseline_record, record)
    if improvement is not None and improvement < args.min_ttft_improvement_pct:
        return False
    return True


def latency_candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, float, float, int]:
    record = candidate["record"]
    tie_breaker = int(candidate.get("preload_tensors", candidate.get("cache_mib", candidate.get("profile_reserve_pct", 0))))
    return (
        float(record["interactive_ttft_s"]),
        -float(record.get("eval_tokens_per_s") or 0.0),
        float(record.get("time_to_type_s") or 1e9),
        tie_breaker,
    )


def ordered_latency_candidates(
    args: argparse.Namespace,
    candidates: list[dict[str, Any]],
    baseline_record: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    passing = [
        candidate
        for candidate in candidates
        if "record" in candidate and validation_passes_latency_gate(args, candidate["record"], baseline_record)
    ]
    return sorted(passing, key=latency_candidate_sort_key)


def select_latency_candidate(
    args: argparse.Namespace,
    candidates: list[dict[str, Any]],
    baseline_record: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    ordered = ordered_latency_candidates(args, candidates, baseline_record)
    return ordered[0] if ordered else None


def startup_preload_remote_env(candidate: dict[str, Any] | None) -> list[str]:
    if candidate is None:
        return []
    return [f"LLAMA_CHAT_STARTUP_PROFILE_PRELOAD_TENSORS={candidate['preload_tensors']}"]


def selected_candidate_remote_env(selected_startup_preload: dict[str, Any] | None) -> list[str]:
    return startup_preload_remote_env(selected_startup_preload)


def cache_budget_remote_env(candidate: dict[str, Any] | None) -> list[str]:
    if candidate is None:
        return []
    return [f"GGML_MOE_VRAM_CACHE_MIB={candidate['cache_mib']}"]


def profile_reserve_remote_env(candidate: dict[str, Any] | None) -> list[str]:
    if candidate is None:
        return []
    return [f"GGML_MOE_VRAM_PROFILE_RESERVE_PCT={candidate['profile_reserve_pct']}"]


def tune_cache_budget(
    args: argparse.Namespace,
    stamp: str,
    trace: str,
    preset_path: str,
    preset: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if not args.tune_cache_budget or args.skip_validation:
        return None, []

    results: list[dict[str, Any]] = []
    for value in cache_budget_candidates(args, preset):
        label = f"candidate-cache{value}"
        output, record = validate_preset(
            args,
            stamp,
            trace,
            preset_path,
            label=label,
            remote_env=[f"GGML_MOE_VRAM_CACHE_MIB={value}"],
        )
        results.append(
            {
                "cache_mib": value,
                "output": str(output),
                "summary": str(bench_summary_path(output)),
                "record": record,
                "passes_latency_gate": validation_passes_latency_gate(args, record),
            }
        )
        if args.phase_pause_s > 0:
            time.sleep(args.phase_pause_s)

    selected = select_latency_candidate(args, results)
    if selected is None:
        return None, results

    preset_env = preset.setdefault("env", {})
    preset_env["GGML_MOE_VRAM_CACHE_MIB"] = str(selected["cache_mib"])
    return selected, results


def tune_profile_reserve(
    args: argparse.Namespace,
    stamp: str,
    trace: str,
    preset_path: str,
    preset: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if not args.tune_profile_reserve or args.skip_validation:
        return None, []

    results: list[dict[str, Any]] = []
    for value in profile_reserve_candidates(args, preset):
        label = f"candidate-reserve{value}"
        output, record = validate_preset(
            args,
            stamp,
            trace,
            preset_path,
            label=label,
            remote_env=[f"GGML_MOE_VRAM_PROFILE_RESERVE_PCT={value}"],
        )
        results.append(
            {
                "profile_reserve_pct": value,
                "output": str(output),
                "summary": str(bench_summary_path(output)),
                "record": record,
                "passes_latency_gate": validation_passes_latency_gate(args, record),
            }
        )
        if args.phase_pause_s > 0:
            time.sleep(args.phase_pause_s)

    selected = select_latency_candidate(args, results)
    if selected is None:
        return None, results

    preset_env = preset.setdefault("env", {})
    preset_env["GGML_MOE_VRAM_PROFILE_RESERVE_PCT"] = str(selected["profile_reserve_pct"])
    return selected, results


def tune_startup_preload(
    args: argparse.Namespace,
    stamp: str,
    trace: str,
    preset_path: str,
    preset: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any]]:
    if not args.tune_startup_preload or args.skip_validation:
        return None, [], {}

    baseline: dict[str, Any] = {}
    baseline_record: dict[str, Any] = {}
    if args.measure_session_baseline:
        baseline_output, baseline_record = validate_preset(
            args,
            stamp,
            trace,
            preset_path,
            label="session-baseline",
        )
        baseline = {
            "output": str(baseline_output),
            "summary": str(bench_summary_path(baseline_output)),
            "record": baseline_record,
        }
        if args.phase_pause_s > 0:
            time.sleep(args.phase_pause_s)

    results: list[dict[str, Any]] = []
    baseline_preload = preset_startup_preload(preset) if args.measure_session_baseline else None
    for value in startup_preload_candidates(args, preset):
        if baseline_preload is not None and value == baseline_preload:
            results.append(
                {
                    "preload_tensors": value,
                    "skipped": True,
                    "skip_reason": "matches_session_baseline",
                }
            )
            continue
        label = f"candidate-preload{value}"
        output, record = validate_preset(
            args,
            stamp,
            trace,
            preset_path,
            label=label,
            remote_env=[f"LLAMA_CHAT_STARTUP_PROFILE_PRELOAD_TENSORS={value}"],
        )
        results.append(
            {
                "preload_tensors": value,
                "output": str(output),
                "summary": str(bench_summary_path(output)),
                "record": record,
                "ttft_improvement_pct": ttft_improvement_pct(baseline_record, record),
                "passes_latency_gate": validation_passes_latency_gate(args, record, baseline_record),
            }
        )
        if args.phase_pause_s > 0:
            time.sleep(args.phase_pause_s)

    selected = select_latency_candidate(args, results, baseline_record)
    if selected is None:
        return None, results, baseline

    preset_env = preset.setdefault("env", {})
    preset_env["LLAMA_CHAT_STARTUP_PROFILE_PRELOAD_TENSORS"] = str(selected["preload_tensors"])
    return selected, results, baseline


def selected_validation_result(
    selected_startup_preload: dict[str, Any] | None,
    session_baseline: dict[str, Any],
) -> tuple[Path | None, dict[str, Any]]:
    if selected_startup_preload is not None:
        return Path(str(selected_startup_preload["output"])), selected_startup_preload["record"]
    if session_baseline:
        return Path(str(session_baseline["output"])), session_baseline["record"]
    return None, {}


def summarize_ttft_trace_csv(trace_text: str) -> dict[str, Any]:
    rows = list(csv.DictReader(io.StringIO(trace_text)))
    if not rows:
        return {"events": 0, "by_op": {}, "submit_to_first_token_ms": None}

    start_index = 0
    end_index = len(rows)
    for index, row in enumerate(rows):
        if row.get("op") == "mark_submit":
            start_index = index
            break
    for index, row in enumerate(rows[start_index:], start=start_index):
        if row.get("op") == "mark_first_token":
            end_index = index
            break

    by_op: dict[str, dict[str, float | int]] = {}
    window = rows[start_index:end_index]
    for row in window:
        op = row.get("op") or ""
        if not op or op.startswith("mark_"):
            continue
        item = by_op.setdefault(op, {"count": 0, "elapsed_ms": 0.0})
        item["count"] = int(item["count"]) + 1
        item["elapsed_ms"] = float(item["elapsed_ms"]) + float(row.get("copy_ms") or 0.0)

    submit_to_first_token_ms = None
    if 0 <= start_index < len(rows) and 0 <= end_index < len(rows):
        submit_to_first_token_ms = float(rows[end_index].get("t_ms") or 0.0) - float(rows[start_index].get("t_ms") or 0.0)

    elapsed = {op: float(data.get("elapsed_ms") or 0.0) for op, data in by_op.items()}
    def elapsed_any(*ops: str) -> float:
        return sum(elapsed.get(op, 0.0) for op in ops)
    prompt_moe_subspans = {
        "group": elapsed_any("cpu_up_group", "cpu_prompt_upgate_group") + elapsed_any("cpu_down_group", "cpu_prompt_down_group"),
        "prefetch_preload": elapsed_any(
            "cpu_up_prep",
            "cpu_prompt_upgate_prefetch_preload",
            "cpu_prompt_upgate_prefe",
        ) + elapsed_any(
            "cpu_down_prep",
            "cpu_prompt_down_prefetch_preload",
            "cpu_prompt_down_prefetc",
        ),
        "cuda_probe": elapsed_any("cpu_up_cuda_probe", "cpu_prompt_upgate_cuda_probe"),
        "compute": elapsed_any(
            "cpu_up_compute",
            "cpu_up_dyn_compute",
            "cpu_up_dynamic_compute",
            "cpu_up_many_compute",
            "cpu_prompt_upgate_compute",
            "cpu_prompt_upgate_compu",
        ) + elapsed_any("cpu_down_compute", "cpu_down_dyn_compute", "cpu_down_dynamic_compute", "cpu_down_dynamic_comput", "cpu_down_many_compute", "cpu_prompt_down_compute"),
    }
    prompt_moe_faults = {
        "minor": elapsed_any("cpu_up_minflt") + elapsed_any("cpu_down_minflt"),
        "major": elapsed_any("cpu_up_majflt") + elapsed_any("cpu_down_majflt"),
    }
    route_keys: set[tuple[str, str]] = set()
    route_unique_bytes = 0
    route_events = 0
    for row in window:
        if row.get("op") not in ("cpu_up_route", "cpu_gate_route", "cpu_down_route"):
            continue
        route_events += 1
        key = (row.get("tensor") or "", row.get("expert_idx") or "")
        if key in route_keys:
            continue
        route_keys.add(key)
        route_unique_bytes += int(row.get("expert_bytes") or 0)

    return {
        "events": len(window),
        "by_op": by_op,
        "prompt_moe_subspans": prompt_moe_subspans,
        "prompt_moe_faults": prompt_moe_faults,
        "prompt_moe_route_events": route_events,
        "prompt_moe_route_unique_experts": len(route_keys),
        "prompt_moe_route_unique_gib": route_unique_bytes / float(1024 ** 3),
        "submit_to_first_token_ms": submit_to_first_token_ms,
    }


def build_mechanism_plan(ttft_summary: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    by_op = ttft_summary.get("by_op", {})
    elapsed = {str(op): float(data.get("elapsed_ms") or 0.0) for op, data in by_op.items()}
    total_ms = float(ttft_summary.get("submit_to_first_token_ms") or 0.0)
    if total_ms <= 0:
        total_ms = float(record.get("interactive_ttft_s") or 0.0) * 1000.0
    prompt_moe_ms = elapsed.get("cpu_prompt_upgate", 0.0) + elapsed.get("cpu_prompt_down", 0.0)
    cuda_load_ms = elapsed.get("runtime_load", 0.0) + elapsed.get("preload_load", 0.0)
    cuda_call_ms = elapsed.get("call_upgate", 0.0) + elapsed.get("call_down", 0.0)
    subspans = {str(k): float(v or 0.0) for k, v in ttft_summary.get("prompt_moe_subspans", {}).items()}
    faults = {str(k): float(v or 0.0) for k, v in ttft_summary.get("prompt_moe_faults", {}).items()}
    route_unique_gib = float(ttft_summary.get("prompt_moe_route_unique_gib") or 0.0)
    subspan_total_ms = sum(subspans.values())

    plan: dict[str, Any] = {
        "schema": "wici_ttft_mechanism_plan_v1",
        "total_ms": total_ms,
        "bottleneck_ms": {
            "prompt_moe_cpu": prompt_moe_ms,
            "cuda_expert_load": cuda_load_ms,
            "cuda_decode_call": cuda_call_ms,
        },
        "prompt_moe_subspans_ms": subspans,
        "prompt_moe_faults": faults,
        "prompt_moe_route_unique_gib": route_unique_gib,
        "recent_breakthrough_scope": "Use 2024-05-28 or newer mechanisms only; reject older FlexGen/MoE-Infinity-era offload heuristics unless revalidated by a recent system.",
        "recent_mechanism_basis": [
            "Tutti-style critical-path removal and slack-aware bulk movement",
            "FineMoE-style route-aware expert cache/offload planning",
            "Speculative expert prefetch only when an observed first-prompt trace exposes prefetchable expert movement",
            "SpecMD-style speculation only for token-rate work after the deterministic canary path is protected",
            "Fast-MoE-style predictive prefetch and expert replication only after the trace proves expert waits or load imbalance are critical",
            "HarMoEny-style load-balance scheduling only if a multi-device or intra-host imbalance trace exists",
            "Recent serving-runtime practice from vLLM/SGLang/DeepEP-style MoE scheduling, adapted only where single-host interactive TTFT benefits",
        ],
        "evidence_gate": (
            "Before accepting a recent mechanism, prove it on the traced critical path "
            "with a one-run canary. For exact CUDA prompt MoE, first prove the max-error "
            "routed row from validation against the CPU IQK dot path."
        ),
        "candidate_env": {},
        "implementation_target": "",
        "avoid": [],
        "next_experiment": "",
        "rationale": "",
    }

    if total_ms > 0 and prompt_moe_ms / total_ms >= 0.50:
        plan["bottleneck"] = "prompt_moe_compute"
        if subspan_total_ms > 0 and prompt_moe_ms > 0 and subspans.get("prefetch_preload", 0.0) / prompt_moe_ms >= 0.35:
            plan["next_experiment"] = "prompt_critical_residency_or_ttft_pack"
            plan["rationale"] = (
                "Most submit-to-first-token time is CPU prompt MoE, and the "
                "subtrace points at prefetch/preload rather than arithmetic. Build a "
                "prompt-critical resident CPU pack before testing thread or cache knobs."
            )
        elif subspan_total_ms > 0 and prompt_moe_ms > 0 and subspans.get("compute", 0.0) / prompt_moe_ms >= 0.45:
            if faults.get("major", 0.0) >= 16.0 and route_unique_gib >= 12.0:
                plan["next_experiment"] = "speculative_expert_prefetch_accuracy_probe"
                plan["implementation_target"] = (
                    "Build a route-prediction accuracy report before adding another "
                    "runtime prefetcher. Score whether recent Speculating-Experts- "
                    "style lookahead can predict enough next-layer expert bytes to "
                    "hide the observed CPU mmap faults. Only if predicted hit bytes "
                    "before first use exceed the faulting working set should the next "
                    "patch add cross-layer page-cache or VRAM prefetch."
                )
                plan["rationale"] = (
                    "Most submit-to-first-token time is CPU prompt MoE with major "
                    "page faults, and the first-prompt route set is far too large for "
                    "a bounded CPU/pack residency fix. The corrected mmap TTFT pack "
                    "preserved the canary but did not reduce traced prompt compute, "
                    "same-layer prefetch regressed immediate TTFT, and exact Q8_K "
                    "prompt staging changed deterministic output while regressing "
                    "TTFT. The next mechanism is therefore a predictor-gated "
                    "cross-layer prefetch design, not more cache/thread sweeps or "
                    "another exact prompt CUDA latency probe."
                )
            elif faults.get("major", 0.0) >= 16.0:
                plan["next_experiment"] = "prompt_critical_residency_or_ttft_pack"
                plan["rationale"] = (
                    "Most submit-to-first-token time is CPU prompt MoE, and the "
                    "compute span incurred major page faults. Build a prompt-critical "
                    "resident CPU pack before changing thread/cache knobs."
                )
            else:
                plan["next_experiment"] = "exact_row_batched_prompt_moe"
                plan["rationale"] = (
                    "Most submit-to-first-token time is CPU prompt MoE, and the "
                    "subtrace points at IQK compute without enough major faults to "
                    "blame storage. Work on an exact row-batched prompt MoE kernel "
                    "that matches the IQK prompt dot path."
                )
        elif subspan_total_ms > 0:
            plan["next_experiment"] = "cpu_prompt_moe_subtrace"
            plan["rationale"] = (
                "Most submit-to-first-token time is CPU prompt MoE. Capture the "
                "new subtrace events to choose between exact row-batched prompt MoE "
                "and prompt-critical CPU residency."
            )
        else:
            plan["next_experiment"] = "exact_row_batched_prompt_moe"
            plan["rationale"] = (
                "Most submit-to-first-token time is CPU prompt MoE. Work on an exact "
                "row-batched prompt MoE kernel that matches the IQK prompt dot path before "
                "testing cache or thread parameters."
            )
        plan["avoid"] = [
            "Do not treat GGML_MOE_STREAM_PROMPT_UP_GATE=1 as prompt CUDA; it is accepted only as the decode-control path.",
            "Do not enable GGML_MOE_STREAM_PROMPT_UP_GATE=unsafe-q8-1 or exact-q8-k* for IQ2_S; chunked exact staging still fails the n36 canary and regresses TTFT.",
            "Do not promote LLAMA_CHAT_PREFIX_PREFILL for this model; it reduced TTFT but changed the n36 canary.",
            "Do not sweep cache/thread/GPU-layer knobs until prompt MoE compute is no longer dominant.",
        ]
    elif total_ms > 0 and cuda_load_ms / total_ms >= 0.25:
        plan["bottleneck"] = "expert_movement"
        plan["next_experiment"] = "slack_aware_expert_scheduler"
        plan["rationale"] = (
            "Expert movement is a material part of TTFT. Build a layer-aware scheduler "
            "that batches and prefetches exact routed experts under compute slack."
        )
        plan["candidate_env"] = {"GGML_MOE_VRAM_PROFILE_PRELOAD_EVICT": "1"}
    else:
        plan["bottleneck"] = "mixed_or_unclassified"
        plan["next_experiment"] = "expand_critical_path_trace"
        plan["rationale"] = "No single traced component dominates; add graph/setup and scheduler wait events before tuning."
    return plan


def choose_canary_preserving_candidate(
    args: argparse.Namespace,
    stamp: str,
    trace: str,
    preset_path: str,
    candidates: list[dict[str, Any]],
    baseline_record: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, Path | None, dict[str, Any], str | None, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    last_output: Path | None = None
    last_record: dict[str, Any] = {}
    last_digest: str | None = None
    for candidate in ordered_latency_candidates(args, candidates, baseline_record):
        if attempts and args.phase_pause_s > 0:
            time.sleep(args.phase_pause_s)
        output, record, digest = run_canary(
            args,
            stamp,
            trace,
            preset_path,
            startup_preload_remote_env(candidate),
            label=f"canary-preload{candidate['preload_tensors']}",
        )
        last_output = output
        last_record = record
        last_digest = digest
        attempt = {
            "preload_tensors": candidate["preload_tensors"],
            "output": str(output),
            "summary": str(bench_summary_path(output)),
            "sha256": digest,
            "passes_canary": digest == EXPECTED_N36_SHA,
        }
        attempts.append(attempt)
        candidate["canary"] = attempt
        if digest == EXPECTED_N36_SHA:
            return candidate, output, record, digest, attempts
    return None, last_output, last_record, last_digest, attempts


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    if not slug:
        raise SystemExit("--promote-name must contain at least one safe filename character")
    return slug


def trace_to_profile_csv(trace_text: str) -> str:
    rows = csv.DictReader(io.StringIO(trace_text))
    counts: dict[tuple[str, int], dict[str, int | str]] = {}
    for row in rows:
        tensor = row["tensor"]
        expert_idx = int(row["expert_idx"])
        key = (tensor, expert_idx)
        item = counts.setdefault(key, {"tensor": tensor, "expert_idx": expert_idx, "count": 0, "expert_bytes": 0})
        item["count"] = int(item["count"]) + 1
        item["expert_bytes"] = max(int(item["expert_bytes"]), int(row["expert_bytes"]))
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["tensor", "expert_idx", "count", "expert_bytes"], lineterminator="\n")
    writer.writeheader()
    for item in sorted(counts.values(), key=lambda r: (int(r["count"]), int(r["expert_bytes"]), str(r["tensor"])), reverse=True):
        writer.writerow(item)
    return output.getvalue()


def write_env_file(path: Path, env: dict[str, str]) -> None:
    merged = dict(BASE_ENV)
    merged.update(env)
    lines = [f"# Generated from {path.with_suffix('.json').as_posix()}."]
    for key in merged:
        lines.append(f"export {key}={shlex.quote(str(merged[key]))}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def promote_groundtruth(
    args: argparse.Namespace,
    stamp: str,
    trace: str,
    preset: dict[str, Any],
    validate_record: dict[str, Any],
    canary_record: dict[str, Any],
    canary_sha: str | None,
    failures: list[str],
) -> dict[str, str] | None:
    if not args.promote_name:
        return None
    if args.dry_run:
        return {"dry_run": "1"}
    if failures:
        raise SystemExit("--promote-name was set, but calibration gates failed; refusing to promote")

    name = slugify(args.promote_name)
    target_dir = REPO / "presets" / "moe" / "groundtruth"
    target_dir.mkdir(parents=True, exist_ok=True)
    json_path = target_dir / f"{name}.json"
    env_path = target_dir / f"{name}.env"
    route_path = target_dir / f"{name}.route.csv"
    existing = [path for path in (json_path, env_path, route_path) if path.exists()]
    if existing and not args.replace_existing_promotion:
        paths = ", ".join(str(path) for path in existing)
        raise SystemExit(f"promotion target already exists; pass --replace-existing-promotion to overwrite: {paths}")

    data = json.loads(json.dumps(preset))
    data["name"] = name
    data["created_by"] = "scripts/calibrate-wici-interactive-config.py"
    data["accepted"] = {
        "timestamp": stamp,
        "route_trace": trace,
        "validate": {
            "time_to_type_s": validate_record.get("time_to_type_s"),
            "interactive_ttft_s": validate_record.get("interactive_ttft_s"),
            "eval_tokens_per_s": validate_record.get("eval_tokens_per_s"),
            "prompt_eval_tokens_per_s": validate_record.get("prompt_eval_tokens_per_s"),
        },
        "canary": {
            "sha256": canary_sha,
            "eval_tokens_per_s": canary_record.get("eval_tokens_per_s"),
        },
        "gates": {
            "min_eval_tps": args.min_eval_tps,
            "max_time_to_type_s": args.max_time_to_type_s,
            "max_ttft_s": args.max_ttft_s,
            "measure_session_baseline": args.measure_session_baseline,
            "min_ttft_improvement_pct": args.min_ttft_improvement_pct,
            "require_validation": args.require_validation,
            "require_canary": args.require_canary,
        },
    }
    data["notes"] = (
        "Promoted by the short interactive calibration harness. Re-run "
        "scripts/calibrate-wici-interactive-config.py after changing model, pack, GPU, storage, or prompt length."
    )

    env = {str(k): str(v) for k, v in data.get("env", {}).items()}
    profile = env.get("GGML_MOE_VRAM_PROFILE", "")
    profile_path = (REPO / profile) if profile and not Path(profile).is_absolute() else Path(profile)
    if not profile or not profile_path.is_file() or args.promote_trace_profile:
        trace_text = remote_cat(args, trace)
        route_path.write_text(trace_to_profile_csv(trace_text), encoding="utf-8")
        profile_repo_path = route_path.relative_to(REPO).as_posix()
        env["GGML_MOE_VRAM_PROFILE"] = profile_repo_path
        match = data.setdefault("match", {})
        match["profile_basename"] = route_path.name
    data["env"] = env

    json_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_env_file(env_path, env)
    return {
        "json": str(json_path),
        "env": str(env_path),
        "route_profile": str(route_path) if route_path.is_file() else "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Short trace-driven calibration for wici interactive GLM-5.1 config."
    )
    parser.add_argument("--ssh-host", default="wici")
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    parser.add_argument("--route-trace-dir", default=DEFAULT_TRACE_DIR)
    parser.add_argument("--ttft-trace-dir", default=DEFAULT_TTFT_TRACE_DIR)
    parser.add_argument("--output-dir", type=Path, default=REPO / "bench" / "wici-glm51-interactive-calibration")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=1, help="Runs per phase. Default is intentionally short.")
    parser.add_argument("--capture-tokens", type=int, default=36)
    parser.add_argument("--validate-tokens", type=int, default=84)
    parser.add_argument("--chat-load-mode", choices=("fast-prompt", "eager"), default="fast-prompt")
    parser.add_argument("--chat-active-prewarm", action="store_true")
    parser.add_argument("--post-ready-delay-s", type=float, default=0.0, help="Wait after the first ready marker before submitting the prompt. Use only when calibrating human typing latency.")
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--remote-env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--route-trace", help="Reuse an existing remote route trace instead of capturing one.")
    parser.add_argument("--force-trace-preset", action="store_true", help="Recompute a trace-aware preset instead of tuning the accepted groundtruth base preset. This can improve latency but must pass canary before promotion.")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-canary", action="store_true")
    parser.add_argument("--force-validation-preset", action="store_true", help="Recompute the trace preset during validation; off by default so time-to-type matches the cached user path.")
    parser.add_argument("--min-eval-tps", type=float, default=0.93)
    parser.add_argument("--max-time-to-type-s", type=float)
    parser.add_argument("--max-ttft-s", type=float, default=DEFAULT_MAX_TTFT_S)
    parser.add_argument("--require-validation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-canary", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--canary-fallback", action=argparse.BooleanOptionalAction, default=True, help="If the fastest validated latency candidate fails canary, try the next validated candidates before rejecting the calibration.")
    parser.add_argument("--tune-cache-budget", action=argparse.BooleanOptionalAction, default=False, help="Try a short GGML_MOE_VRAM_CACHE_MIB candidate set before startup-preload tuning. Disabled by default because it adds remote runs.")
    parser.add_argument("--cache-budget-candidates", help="Comma-separated GGML_MOE_VRAM_CACHE_MIB candidates. Default is current cache budget plus immediate +/-128 MiB neighbors.")
    parser.add_argument("--max-cache-budget-mib", type=int, default=2048)
    parser.add_argument("--tune-profile-reserve", action=argparse.BooleanOptionalAction, default=False, help="Try a short GGML_MOE_VRAM_PROFILE_RESERVE_PCT candidate set. Disabled by default because it adds remote runs.")
    parser.add_argument("--profile-reserve-candidates", help="Comma-separated GGML_MOE_VRAM_PROFILE_RESERVE_PCT candidates. Default is current reserve plus +10 and +20.")
    parser.add_argument("--max-profile-reserve-pct", type=int, default=50)
    parser.add_argument("--tune-startup-preload", action=argparse.BooleanOptionalAction, default=False, help="Try a short derived LLAMA_CHAT_STARTUP_PROFILE_PRELOAD_TENSORS candidate set. Disabled by default; mechanism traces, not parameter sweeps, choose the next optimization.")
    parser.add_argument("--measure-session-baseline", action=argparse.BooleanOptionalAction, default=True, help="Measure the untuned preset in the same calibration session and require latency candidates to improve against it.")
    parser.add_argument("--min-ttft-improvement-pct", type=float, default=10.0, help="Minimum interactive TTFT improvement over the session baseline required for a tuned latency candidate.")
    parser.add_argument("--startup-preload-candidates", help="Comma-separated startup preload candidate values. Default is 0 plus the generated value and its immediate neighbors.")
    parser.add_argument("--max-startup-preload-tensors", type=int, default=6)
    parser.add_argument("--promote-name", help="After all gates pass, write the accepted preset to presets/moe/groundtruth/<name>.* for future reuse.")
    parser.add_argument("--promotion-confirm-runs", type=int, default=2, help="When promoting, rerun the selected validation path this many times and require every run to pass. Use 1 to disable.")
    parser.add_argument("--promote-trace-profile", action=argparse.BooleanOptionalAction, default=True, help="When promoting, convert the captured route trace into the promoted groundtruth route profile.")
    parser.add_argument("--replace-existing-promotion", action="store_true", help="Allow --promote-name to overwrite an existing groundtruth preset.")
    parser.add_argument("--phase-pause-s", type=float, default=5.0, help="Pause between validation phases to let GPU memory settle.")
    parser.add_argument("--gpu-idle-timeout-s", type=float, default=90.0)
    parser.add_argument("--gpu-idle-poll-s", type=float, default=2.0)
    parser.add_argument("--gpu-idle-max-used-mib", type=int, default=1536)
    parser.add_argument("--gpu-idle-max-util-pct", type=int, default=5)
    parser.add_argument("--runtime-timeout-s", type=int, default=360)
    parser.add_argument("--ttft-trace-wait-s", type=float, default=240.0, help="Bounded wait for the remote TTFT CSV after a trace-capture run.")
    parser.add_argument("--plan-only", action="store_true", help="Capture and classify the first-prompt trace, then stop before validation/canary runs.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.runs <= 0:
        raise SystemExit("--runs must be positive")
    if args.capture_tokens <= 0 or args.validate_tokens <= 0:
        raise SystemExit("--capture-tokens and --validate-tokens must be positive")
    if args.max_startup_preload_tensors < 0:
        raise SystemExit("--max-startup-preload-tensors must be non-negative")
    if args.max_cache_budget_mib <= 0:
        raise SystemExit("--max-cache-budget-mib must be positive")
    if args.max_profile_reserve_pct <= 0 or args.max_profile_reserve_pct > 95:
        raise SystemExit("--max-profile-reserve-pct must be in 1..95")
    if args.min_ttft_improvement_pct < 0:
        raise SystemExit("--min-ttft-improvement-pct must be non-negative")
    if args.promotion_confirm_runs <= 0:
        raise SystemExit("--promotion-confirm-runs must be positive")
    if args.post_ready_delay_s < 0:
        raise SystemExit("--post-ready-delay-s must be non-negative")
    for value in args.remote_env:
        if "=" not in value or value.startswith("="):
            raise SystemExit("--remote-env values must be KEY=VALUE")
    return args


def main() -> int:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    capture_output = None
    capture_record: dict[str, Any] = {}
    if args.route_trace:
        trace = args.route_trace
    else:
        capture_output, trace, capture_record = capture_trace(args, stamp)

    ttft_trace_summary: dict[str, Any] = {}
    mechanism_plan: dict[str, Any] = {}
    ttft_trace = capture_record.get("ttft_trace_out")
    if ttft_trace and not args.dry_run:
        stderr_log = capture_record.get("stderr_log")
        if stderr_log:
            remote_file_contains_when_ready(
                args,
                str(stderr_log),
                "TTFT trace written",
                wait_s=args.ttft_trace_wait_s,
            )
        ttft_trace_text = remote_cat_when_ready(args, str(ttft_trace), wait_s=args.ttft_trace_wait_s)
        if ttft_trace_text is None:
            mechanism_plan = {
                "schema": "wici_ttft_mechanism_plan_v1",
                "bottleneck": "trace_missing",
                "next_experiment": "fix_ttft_trace_capture",
                "rationale": "The interactive run succeeded, but the TTFT trace was not readable before the bounded wait expired.",
            }
        else:
            ttft_trace_summary = summarize_ttft_trace_csv(ttft_trace_text)
            mechanism_plan = build_mechanism_plan(ttft_trace_summary, capture_record)

    if args.plan_only:
        summary = {
            "schema": "wici_interactive_config_calibration_v1",
            "timestamp": stamp,
            "ok": bool(mechanism_plan),
            "failures": [] if mechanism_plan else ["mechanism_plan_missing"],
            "route_trace": trace,
            "ttft_trace": ttft_trace,
            "ttft_trace_summary": ttft_trace_summary,
            "mechanism_plan": mechanism_plan,
            "capture_output": str(capture_output) if capture_output else None,
            "capture_summary": str(bench_summary_path(capture_output)) if capture_output else None,
            "capture_metrics": capture_record,
            "plan_only": True,
        }
        summary_path = args.output_dir / f"{stamp}-summary.json"
        if not args.dry_run:
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["ok"] else 1

    preset_path, preset = recommend_preset(args, trace)

    validate_output = None
    validate_record: dict[str, Any] = {}
    selected_cache_budget: dict[str, Any] | None = None
    cache_budget_candidates_ran: list[dict[str, Any]] = []
    if not args.skip_validation:
        selected_cache_budget, cache_budget_candidates_ran = tune_cache_budget(
            args,
            stamp,
            trace,
            preset_path,
            preset,
        )
        if selected_cache_budget is not None:
            args.remote_env.append(f"GGML_MOE_VRAM_CACHE_MIB={selected_cache_budget['cache_mib']}")
            validate_output = Path(str(selected_cache_budget["output"]))
            validate_record = selected_cache_budget["record"]

    selected_profile_reserve: dict[str, Any] | None = None
    profile_reserve_candidates_ran: list[dict[str, Any]] = []
    if not args.skip_validation:
        selected_profile_reserve, profile_reserve_candidates_ran = tune_profile_reserve(
            args,
            stamp,
            trace,
            preset_path,
            preset,
        )
        if selected_profile_reserve is not None:
            args.remote_env.append(f"GGML_MOE_VRAM_PROFILE_RESERVE_PCT={selected_profile_reserve['profile_reserve_pct']}")
            validate_output = Path(str(selected_profile_reserve["output"]))
            validate_record = selected_profile_reserve["record"]

    selected_startup_preload: dict[str, Any] | None = None
    startup_preload_candidates_ran: list[dict[str, Any]] = []
    session_baseline: dict[str, Any] = {}
    if not args.skip_validation:
        selected_startup_preload, startup_preload_candidates_ran, session_baseline = tune_startup_preload(
            args,
            stamp,
            trace,
            preset_path,
            preset,
        )
        startup_validate_output, startup_validate_record = selected_validation_result(selected_startup_preload, session_baseline)
        if startup_validate_output is not None:
            validate_output, validate_record = startup_validate_output, startup_validate_record
        elif validate_output is None:
            validate_output, validate_record = validate_preset(args, stamp, trace, preset_path)

    canary_output = None
    canary_record: dict[str, Any] = {}
    canary_sha = None
    canary_attempts: list[dict[str, Any]] = []
    if not args.skip_canary:
        if args.phase_pause_s > 0 and validate_output is not None:
            time.sleep(args.phase_pause_s)
        if args.canary_fallback and selected_startup_preload is not None:
            selected_with_canary, canary_output, canary_record, canary_sha, canary_attempts = choose_canary_preserving_candidate(
                args,
                stamp,
                trace,
                preset_path,
                startup_preload_candidates_ran,
                session_baseline.get("record", {}),
            )
            if selected_with_canary is not None:
                selected_startup_preload = selected_with_canary
                validate_output = Path(str(selected_startup_preload["output"]))
                validate_record = selected_startup_preload["record"]
                preset.setdefault("env", {})["LLAMA_CHAT_STARTUP_PROFILE_PRELOAD_TENSORS"] = str(selected_startup_preload["preload_tensors"])
        else:
            canary_output, canary_record, canary_sha = run_canary(
                args,
                stamp,
                trace,
                preset_path,
                startup_preload_remote_env(selected_startup_preload),
            )

    failures = [] if args.dry_run else evaluate(args, validate_record, canary_sha)
    promotion_confirm: dict[str, Any] = {}
    if (
        args.promote_name
        and not failures
        and not args.dry_run
        and args.promotion_confirm_runs > 1
    ):
        if args.phase_pause_s > 0:
            time.sleep(args.phase_pause_s)
        confirm_output, _ = validate_preset(
            args,
            stamp,
            trace,
            preset_path,
            label=f"promotion-confirm-r{args.promotion_confirm_runs}",
            remote_env=selected_candidate_remote_env(selected_startup_preload),
            runs=args.promotion_confirm_runs,
        )
        confirm_ok, confirm_failures, confirm_worst = validation_records_pass_gate(
            args,
            confirm_output,
            session_baseline.get("record", {}),
        )
        promotion_confirm = {
            "output": str(confirm_output),
            "summary": str(bench_summary_path(confirm_output)),
            "runs": args.promotion_confirm_runs,
            "ok": confirm_ok,
            "worst_metrics": confirm_worst,
            "failures": confirm_failures,
        }
        if not confirm_ok:
            failures.extend(confirm_failures)
        else:
            validate_output = confirm_output
            validate_record = confirm_worst
    promoted = promote_groundtruth(
        args,
        stamp,
        trace,
        preset,
        validate_record,
        canary_record,
        canary_sha,
        failures,
    )
    summary = {
        "schema": "wici_interactive_config_calibration_v1",
        "timestamp": stamp,
        "ok": not failures,
        "failures": failures,
        "route_trace": trace,
        "ttft_trace": ttft_trace,
        "ttft_trace_summary": ttft_trace_summary,
        "mechanism_plan": mechanism_plan,
        "preset": preset_path,
        "preset_env": preset.get("env", {}) if preset else {},
        "selected_cache_budget": selected_cache_budget,
        "cache_budget_candidates": cache_budget_candidates_ran,
        "selected_profile_reserve": selected_profile_reserve,
        "profile_reserve_candidates": profile_reserve_candidates_ran,
        "session_baseline_output": session_baseline.get("output"),
        "session_baseline_metrics": session_baseline.get("record", {}),
        "selected_startup_preload": selected_startup_preload,
        "startup_preload_candidates": startup_preload_candidates_ran,
        "capture_output": str(capture_output) if capture_output else None,
        "capture_summary": str(bench_summary_path(capture_output)) if capture_output else None,
        "validate_output": str(validate_output) if validate_output else None,
        "validate_summary": str(bench_summary_path(validate_output)) if validate_output else None,
        "canary_output": str(canary_output) if canary_output else None,
        "canary_summary": str(bench_summary_path(canary_output)) if canary_output else None,
        "canary_attempts": canary_attempts,
        "promotion_confirm": promotion_confirm,
        "promoted": promoted,
        "capture_metrics": capture_record,
        "validate_metrics": validate_record,
        "canary_sha": canary_sha,
        "canary_metrics": canary_record,
        "gates": {
            "min_eval_tps": args.min_eval_tps,
            "max_time_to_type_s": args.max_time_to_type_s,
            "max_ttft_s": args.max_ttft_s,
            "measure_session_baseline": args.measure_session_baseline,
            "post_ready_delay_s": args.post_ready_delay_s,
            "min_ttft_improvement_pct": args.min_ttft_improvement_pct,
            "canary_fallback": args.canary_fallback,
            "tune_profile_reserve": args.tune_profile_reserve,
            "promotion_confirm_runs": args.promotion_confirm_runs,
            "require_validation": args.require_validation,
            "require_canary": args.require_canary,
        },
    }
    summary_path = args.output_dir / f"{stamp}-summary.json"
    if not args.dry_run:
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
