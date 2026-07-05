#!/usr/bin/env python3
"""Strict 16 GB oracle verifier window probe for DeepSeek V4 Flash."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path("/root/lfz/vendor/llama.cpp-deepseek-v4")
DEFAULT_BIN = REPO / "build-ds4-moe-stream/bin/llama-results"
DEFAULT_MODEL = Path("/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf")
DEFAULT_TEXT = Path("/root/lfz/runs/vendor-ds4-16gb/20260704T171004Z-results-top1-current-selfcheck/selfcheck/fixed-france-text.txt")
DEFAULT_RUN_ROOT = Path("/root/lfz/runs/vendor-ds4-16gb")
MEMORY_MAX_BYTES = 16_000_000_000


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_text(cmd: list[str], cwd: Path | None = None, check: bool = False) -> str:
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=check)
    return proc.stdout.strip()


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_time_v(text: str) -> dict[str, float | int | None]:
    out: dict[str, float | int | None] = {
        "max_rss_kb": None,
        "elapsed_seconds": None,
    }
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    if match:
        out["max_rss_kb"] = int(match.group(1))
    match = re.search(r"Elapsed \(wall clock\) time[^\n]*\):\s*([0-9:.]+)", text)
    if match:
        parts = match.group(1).split(":")
        try:
            if len(parts) == 3:
                out["elapsed_seconds"] = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:
                out["elapsed_seconds"] = int(parts[0]) * 60 + float(parts[1])
            else:
                out["elapsed_seconds"] = float(parts[0])
        except ValueError:
            pass
    return out


def parse_memory_kv(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            out[parts[0]] = int(parts[1])
        except ValueError:
            pass
    return out


def read_int(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8", errors="replace").strip())
    except ValueError:
        return None


def top1_ids(path: Path) -> list[int]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    ids: list[int] = []
    for item in data.get("positions", []):
        top1 = item.get("top1") or {}
        if "id" in top1:
            ids.append(int(top1["id"]))
    return ids


def top1_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "mode": data.get("mode"),
        "n_tokens": data.get("n_tokens"),
        "n_vocab": data.get("n_vocab"),
        "next_token_positions": data.get("next_token_positions"),
        "top1_matches_next_token": data.get("top1_matches_next_token"),
    }


def env_exports(env: dict[str, str]) -> str:
    return "\n".join(f"export {key}={shlex.quote(value)}" for key, value in sorted(env.items()))


def command_for_window(binary: Path, model: Path, text: Path, window: int) -> list[str]:
    cmd = [
        str(binary),
        "-m", str(model),
        "-f", str(text),
        "-c", "256",
        "-t", "20",
        "-tb", "20",
        "-ngl", "all",
        "--fit", "on",
        "-fa", "auto",
        "--n-cpu-moe", "40",
        "--defer-experts",
        "-b", str(window),
        "-ub", str(window),
        "--top1-report", "top1.json",
    ]
    if window == 1:
        cmd.append("--sequential-logits")
    return cmd


def write_case_script(path: Path, case_dir: Path, env: dict[str, str], cmd: list[str]) -> None:
    exact_command = " ".join(shlex.quote(part) for part in cmd)
    script = f"""#!/usr/bin/env bash
set -euo pipefail
RUN_DIR={shlex.quote(str(case_dir))}
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"
{env_exports(env)}
printf '%s\\n' {shlex.quote(exact_command)} > exact_command.txt
env | sort > environment.txt
date -Is > start_time.txt
start_ns=$(date +%s%N)
echo "$start_ns" > start_ns.txt
CG_REL=$(awk -F: '$2 == "" {{ print $3 }}' /proc/self/cgroup | tail -n 1)
CG_DIR="/sys/fs/cgroup${{CG_REL}}"
printf '%s\\n' "$CG_DIR" > cgroup_path.txt
set +e
/usr/bin/time -v {exact_command} > stdout.txt 2> stderr.txt &
CMD_PID=$!
(
  printf 'time_epoch\\tmemory_current\\tmemory_peak\\tgpu_mem_used_mib\\tgpu_mem_free_mib\\tgpu_util_pct\\n'
  while kill -0 "$CMD_PID" 2>/dev/null; do
    now=$(date +%s)
    mem_cur=$(cat "$CG_DIR/memory.current" 2>/dev/null || true)
    mem_peak=$(cat "$CG_DIR/memory.peak" 2>/dev/null || true)
    gpu=$(nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ')
    if [ -n "$gpu" ]; then
      printf '%s\\t%s\\t%s\\t%s\\n' "$now" "$mem_cur" "$mem_peak" "$(printf '%s' "$gpu" | tr ',' '\\t')"
    else
      printf '%s\\t%s\\t%s\\t\\t\\t\\n' "$now" "$mem_cur" "$mem_peak"
    fi
    if [ -n "$mem_cur" ] && [ "$mem_cur" -gt {MEMORY_MAX_BYTES} ]; then
      printf 'ram_limit_exceeded current=%s threshold=%s\\n' "$mem_cur" "{MEMORY_MAX_BYTES}" > ram_limit_exceeded.txt
      kill "$CMD_PID" 2>/dev/null || true
      exit 0
    fi
    sleep 1
  done
) > resource_samples.tsv &
MONITOR_PID=$!
wait "$CMD_PID"
STATUS=$?
set -e
kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
date -Is > end_time.txt
echo "$STATUS" > exit_status.txt
cp "$CG_DIR/memory.current" memory.current 2>/dev/null || true
cp "$CG_DIR/memory.peak" memory.peak 2>/dev/null || true
cp "$CG_DIR/memory.events" memory.events 2>/dev/null || true
cp "$CG_DIR/memory.stat" memory.stat 2>/dev/null || true
cat stdout.txt stderr.txt > combined.txt
exit "$STATUS"
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def summarize_case(case_dir: Path, window: int, unit: str, systemd_status: int) -> dict[str, Any]:
    stderr_text = (case_dir / "stderr.txt").read_text(encoding="utf-8", errors="replace") if (case_dir / "stderr.txt").exists() else ""
    time_v = parse_time_v(stderr_text)
    memory_events = parse_memory_kv(case_dir / "memory.events")
    memory_stat = parse_memory_kv(case_dir / "memory.stat")
    memory_peak = read_int(case_dir / "memory.peak")
    exit_status = read_int(case_dir / "exit_status.txt")
    top1 = top1_summary(case_dir / "top1.json")
    oom_seen = memory_events.get("oom", 0) > 0 or memory_events.get("oom_kill", 0) > 0
    summary: dict[str, Any] = {
        "case_dir": str(case_dir),
        "window": window,
        "sequential_logits": window == 1,
        "unit": unit,
        "systemd_status": systemd_status,
        "exit_status": exit_status,
        "elapsed_seconds": time_v["elapsed_seconds"],
        "max_rss_kb": time_v["max_rss_kb"],
        "memory_peak_bytes": memory_peak,
        "memory_file_bytes": memory_stat.get("file"),
        "memory_current_bytes": read_int(case_dir / "memory.current"),
        "memory_events": memory_events,
        "memory_stat_relevant": {
            key: memory_stat.get(key)
            for key in [
                "anon", "file", "kernel", "slab", "inactive_file", "active_file",
                "file_mapped", "pgfault", "pgmajfault", "workingset_refault_file",
                "workingset_activate_file", "workingset_restore_file",
            ]
            if key in memory_stat
        },
        "oom_seen": oom_seen,
        "ram_limit_killed": (case_dir / "ram_limit_exceeded.txt").exists(),
        "ram_ok": memory_peak is not None and memory_peak <= MEMORY_MAX_BYTES and not oom_seen and not (case_dir / "ram_limit_exceeded.txt").exists(),
        "top1_report": top1,
        "top1_ids_count": len(top1_ids(case_dir / "top1.json")),
    }
    write_json(case_dir / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BIN)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--text", type=Path, default=DEFAULT_TEXT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-name", default="dflash-oracle-verifier-window-probe")
    parser.add_argument("--windows", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--no-drop-caches", action="store_true")
    args = parser.parse_args()

    if not args.binary.exists():
        raise SystemExit(f"missing binary: {args.binary}")
    if not args.model.exists():
        raise SystemExit(f"missing model: {args.model}")
    if not args.text.exists():
        raise SystemExit(f"missing fixed text: {args.text}")
    if any(window < 1 for window in args.windows):
        raise SystemExit("--windows must be positive")

    run_dir = args.run_root / f"{utc_stamp()}-{args.run_name}"
    run_dir.mkdir(parents=True, exist_ok=False)

    env = {
        "CUDA_VISIBLE_DEVICES": "0",
        "GGML_CUDA_DISABLE_GRAPHS": "1",
        "GGML_MOE_KEEP_TOPK_LAYER_RANGE": "10-39",
        "GGML_MOE_KEEP_TOPK_LAYER_VALUE": "3",
        "GGML_MOE_KEEP_TOPK_UPDOWN": "4",
        "GGML_MOE_STREAM": "1",
        "GGML_MOE_STREAM_CACHE_ADMIT_PROFILE": str(REPO / ".Agent/profiles/vendor-ds4/current_sota_gate_freq_ge2.tsv"),
        "GGML_MOE_STREAM_DONTNEED": "1",
        "GGML_MOE_STREAM_ONE_CACHE_MIB": "13568",
        "GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4": "1",
        "GGML_MOE_STREAM_ONE_EXPERT_PACK": "/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-gate-miss-firstorder-20260702.pack",
        "GGML_MOE_STREAM_ONE_EXPERT_PACK_IO": "direct",
        "GGML_MOE_STREAM_ONE_NAME_FILTER": "ffn_gate_exps",
        "GGML_MOE_STREAM_ONE_PREFILL_LIMIT": "3000",
        "GGML_MOE_STREAM_ONE_PREFILL_PROFILE": str(REPO / ".Agent/profiles/vendor-ds4/current_sota_gate_freq_ge2.tsv"),
        "GGML_MOE_VRAM_CACHE_GB": "0",
    }
    metadata = {
        "created_utc": utc_stamp(),
        "repo": str(REPO),
        "branch": run_text(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO),
        "commit": run_text(["git", "rev-parse", "HEAD"], cwd=REPO),
        "status": run_text(["git", "status", "--short"], cwd=REPO),
        "binary": str(args.binary),
        "binary_sha256": run_text(["sha256sum", str(args.binary)]).split()[0],
        "model": str(args.model),
        "model_size_bytes": args.model.stat().st_size,
        "fixed_text": str(args.text),
        "fixed_text_size_bytes": args.text.stat().st_size,
        "memory_max_bytes": MEMORY_MAX_BYTES,
        "drop_caches_before_each_case": not args.no_drop_caches,
        "windows": args.windows,
        "env": env,
        "purpose": "diagnostic-only oracle verifier window cost probe; not a generated-output SOTA benchmark",
    }
    write_json(run_dir / "metadata.json", metadata)

    summaries: list[dict[str, Any]] = []
    for window in args.windows:
        case_dir = run_dir / f"w{window}{'-seq' if window == 1 else ''}"
        case_dir.mkdir()
        cmd = command_for_window(args.binary, args.model, args.text, window)
        write_case_script(case_dir / "run_case.sh", case_dir, env, cmd)
        unit = f"vendor-ds4-oracle-w{window}-{utc_stamp()}.service"
        if not args.no_drop_caches:
            (case_dir / "cold_start_procedure.txt").write_text("sync; echo 3 > /proc/sys/vm/drop_caches\n", encoding="utf-8")
            subprocess.run(["sync"], check=True)
            Path("/proc/sys/vm/drop_caches").write_text("3\n", encoding="utf-8")
        systemd_cmd = [
            "systemd-run",
            f"--unit={unit}",
            "--collect",
            "--wait",
            f"--property=MemoryMax={MEMORY_MAX_BYTES}",
            "--property=MemorySwapMax=0",
            str(case_dir / "run_case.sh"),
        ]
        (case_dir / "systemd_command.txt").write_text(" ".join(shlex.quote(part) for part in systemd_cmd) + "\n", encoding="utf-8")
        start = time.time()
        proc = subprocess.run(systemd_cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (case_dir / "systemd-run.out").write_text(proc.stdout, encoding="utf-8", errors="replace")
        (case_dir / "systemd_run_status.txt").write_text(f"{proc.returncode}\n", encoding="utf-8")
        (case_dir / "runner_elapsed_seconds.txt").write_text(f"{time.time() - start:.3f}\n", encoding="utf-8")
        with (case_dir / "unit.properties").open("w", encoding="utf-8") as out:
            subprocess.run(["systemctl", "show", unit], text=True, stdout=out, stderr=subprocess.DEVNULL)
        with (case_dir / "journal.log").open("w", encoding="utf-8") as out:
            subprocess.run(["journalctl", "-u", unit, "--no-pager", "-n", "200"], text=True, stdout=out, stderr=subprocess.DEVNULL)
        summaries.append(summarize_case(case_dir, window, unit, proc.returncode))

    baseline_ids = top1_ids(Path(summaries[0]["case_dir"]) / "top1.json") if summaries else []
    for item in summaries:
        ids = top1_ids(Path(item["case_dir"]) / "top1.json")
        item["same_top1_as_w1"] = bool(baseline_ids) and ids == baseline_ids
        item["same_top1_count_vs_w1"] = sum(1 for a, b in zip(ids, baseline_ids) if a == b)
        item["same_top1_total_vs_w1"] = min(len(ids), len(baseline_ids))
        write_json(Path(item["case_dir"]) / "summary.json", item)

    elapsed_by_w = {int(item["window"]): item.get("elapsed_seconds") for item in summaries}
    baseline_elapsed = elapsed_by_w.get(1)
    comparisons: dict[str, Any] = {}
    if isinstance(baseline_elapsed, (int, float)) and baseline_elapsed > 0:
        for item in summaries:
            elapsed = item.get("elapsed_seconds")
            if isinstance(elapsed, (int, float)) and elapsed > 0:
                window = int(item["window"])
                comparisons[f"w{window}"] = {
                    "elapsed_seconds": elapsed,
                    "speedup_vs_w1_elapsed": baseline_elapsed / elapsed,
                    "cost_x_w1_elapsed": elapsed / baseline_elapsed,
                }

    result = {
        "metadata": metadata,
        "run_dir": str(run_dir),
        "summaries": summaries,
        "comparisons": comparisons,
        "decision_hint": "not_evaluated",
    }
    w4 = comparisons.get("w4", {})
    w8 = comparisons.get("w8", {})
    best_speedup = max(
        [v.get("speedup_vs_w1_elapsed", 0.0) for v in comparisons.values() if isinstance(v, dict)],
        default=0.0,
    )
    all_ram_ok = all(bool(item.get("ram_ok")) for item in summaries)
    all_top1_same = all(bool(item.get("same_top1_as_w1")) for item in summaries)
    if not all_ram_ok:
        result["decision_hint"] = "reject_ram_or_oom"
    elif not all_top1_same:
        result["decision_hint"] = "reject_top1_differs_from_w1"
    elif best_speedup < 1.128429:
        result["decision_hint"] = "close_dflash_current_target_verifier_no_sublinear_gain"
    elif (w4.get("cost_x_w1_elapsed", 999.0) <= 1.719568) or (w8.get("cost_x_w1_elapsed", 999.0) <= 1.719568):
        result["decision_hint"] = "potentially_reopen_dflash_requires_more_precise_verifier_timing"
    else:
        result["decision_hint"] = "insufficient_for_dflash_threshold"

    write_json(run_dir / "summary.json", result)
    with (run_dir / "results.tsv").open("w", encoding="utf-8") as out:
        cols = [
            "window", "elapsed_seconds", "same_top1_as_w1", "same_top1_count_vs_w1",
            "same_top1_total_vs_w1", "memory_peak_bytes", "memory_file_bytes",
            "ram_ok", "oom_seen", "top1_matches_next_token", "case_dir",
        ]
        out.write("\t".join(cols) + "\n")
        for item in summaries:
            top1 = item.get("top1_report", {})
            row = {
                **item,
                "top1_matches_next_token": top1.get("top1_matches_next_token") if isinstance(top1, dict) else None,
            }
            out.write("\t".join(str(row.get(col, "")) for col in cols) + "\n")

    print(run_dir)
    print((run_dir / "results.tsv").read_text(encoding="utf-8"))
    print(json.dumps({"decision_hint": result["decision_hint"], "comparisons": comparisons}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
