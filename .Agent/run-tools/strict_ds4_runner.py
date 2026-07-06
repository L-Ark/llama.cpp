#!/usr/bin/env python3
"""Strict 16 GB DeepSeek V4 runner for vendor llama.cpp experiments."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BIN = REPO_ROOT / "build-cuda-batch/bin/llama-cli"
DEFAULT_MODEL = Path("/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf")
DEFAULT_OUT_ROOT = Path("/root/lfz/runs/vendor-ds4-16gb")
DEFAULT_PROMPT = "Please introduce France in a short paragraph."
MEMORY_MAX_BYTES = 16_000_000_000


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_text(cmd: list[str], cwd: Path | None = None, check: bool = False) -> str:
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=check)
    return proc.stdout.strip()


def parse_rate(text: str, label: str) -> float | None:
    patterns = [
        rf"{label}\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*t/s",
        rf"{label.lower()}.*?([0-9]+(?:\.[0-9]+)?)\s*tokens per second",
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if match:
            return float(match.group(1))
    return None


def parse_timing_ms(text: str, name: str) -> float | None:
    match = re.search(rf"{re.escape(name)}\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ms", text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


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


def parse_memory_events(text: str) -> dict[str, int]:
    events: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            events[parts[0]] = int(parts[1])
        except ValueError:
            pass
    return events


def parse_memory_stat(text: str) -> dict[str, int]:
    keys = {
        "anon",
        "file",
        "kernel",
        "slab",
        "inactive_file",
        "active_file",
        "file_mapped",
        "pgfault",
        "pgmajfault",
        "workingset_refault_file",
        "workingset_activate_file",
        "workingset_restore_file",
    }
    stats: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 2 or parts[0] not in keys:
            continue
        try:
            stats[parts[0]] = int(parts[1])
        except ValueError:
            pass
    return stats


def extract_answer(stdout_text: str, prompt: str) -> str:
    text = stdout_text.replace("\b", "").replace("\r", "\n").strip()
    if prompt in text:
        text = text.rsplit(prompt, 1)[-1]
    split = re.split(r"\[\s*Prompt\s*:", text, maxsplit=1)
    if split:
        text = split[0].strip()
    text = re.sub(r"^Question:.*?Answer:\s*", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    text = re.sub(r"^Answer:\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^[>\s|/\\-]+", "", text).strip()
    return text


def check_correctness(answer: str, prompt: str) -> tuple[bool, str]:
    lowered = answer.lower()
    prompt_lowered = prompt.lower()
    failures: list[str] = []
    min_len = 160 if "france" in prompt_lowered else 80
    if len(answer) < min_len:
        failures.append("too_short")
    if "france" in prompt_lowered:
        if "france" not in lowered:
            failures.append("missing_france")
        if "europe" not in lowered and "european" not in lowered:
            failures.append("missing_europe")
        if not any(term in lowered for term in ["paris", "culture", "history", "cuisine", "landmark", "art", "wine"]):
            failures.append("missing_expected_context")
    if re.search(r"(?:!{3,}|#{3,}|\ufffd|\b(\w+)\s+\1\s+\1\b)", lowered):
        failures.append("degenerate_text")
    if re.search(r"\b(the the|of of|in the the)\b", lowered):
        failures.append("repetition")
    if re.search(r"\b(wait,?\s+i\s+already\s+said|let\s+me\s+try\s+again|start\s+over)\b", lowered):
        failures.append("self_correction")
    if re.search(r"\b(eiff tower|eiff\s*$)\b", lowered):
        failures.append("truncated_or_corrupt_landmark")
    if re.search(r"\b(a|an|the|in|of|for|with|and|or|to|from|as|is|are)\s*[.。!！?？]?$", lowered):
        failures.append("unfinished_sentence")
    if failures:
        return False, ",".join(failures)
    return True, "heuristic_pass_manual_review_required"


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def shell_quote_env(env: dict[str, str]) -> str:
    return "\n".join(f"export {k}={shlex.quote(v)}" for k, v in sorted(env.items()))


def build_case_script(
    path: Path,
    *,
    case_dir: Path,
    binary: Path,
    model: Path,
    cpu_moe: int,
    env: dict[str, str],
    extra_args: list[str],
    ram_kill_threshold_bytes: int,
    prompt: str,
) -> None:
    args = [
        str(binary),
        "-m", str(model),
        "-p", prompt,
        "-n", "192",
        "-c", "512",
        "-b", "64",
        "-ub", "64",
        "-t", "20",
        "-tb", "20",
        "-ngl", "all",
        "--fit", "on",
        "-fa", "auto",
        "--temp", "0",
        "--top-p", "1",
        "--top-k", "1",
        "--seed", "1",
        "--no-display-prompt",
        "--n-cpu-moe", str(cpu_moe),
        "--defer-experts",
    ] + extra_args
    exact_command = " ".join(shlex.quote(x) for x in args)
    script = f"""#!/usr/bin/env bash
set -euo pipefail
RUN_DIR={shlex.quote(str(case_dir))}
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"
cat > prompt.txt <<'EOF_PROMPT'
{prompt}
EOF_PROMPT
printf "/exit\\n" > stdin.txt
cat > detect_answer_started.py <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
prompt = sys.argv[2]
if not path.exists():
    raise SystemExit(1)

text = path.read_text(encoding="utf-8", errors="ignore")
text = text.replace("\\b", "").replace("\\r", "\\n")
if prompt not in text:
    raise SystemExit(1)

tail = text.rsplit(prompt, 1)[-1]
tail = re.split(r"\\[\\s*Prompt\\s*:", tail, maxsplit=1)[0]
tail = re.sub(r"^[>\\s|/\\\\-]+", "", tail).strip()
raise SystemExit(0 if re.search(r"[A-Za-z]{{2,}}", tail) else 1)
PY
{shell_quote_env(env)}
printf '%s\\n' {shlex.quote(exact_command)} > exact_command.txt
env | sort > environment.txt
date -Is > start_time.txt
start_ns=$(date +%s%N)
echo "$start_ns" > start_ns.txt
CG_REL=$(awk -F: '$2 == "" {{ print $3 }}' /proc/self/cgroup | tail -n 1)
CG_DIR="/sys/fs/cgroup${{CG_REL}}"
printf '%s\\n' "$CG_DIR" > cgroup_path.txt
set +e
/usr/bin/time -v {exact_command} < stdin.txt > stdout.txt 2> stderr.txt &
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
    if [ -n "$mem_cur" ] && [ "$mem_cur" -gt {ram_kill_threshold_bytes} ]; then
      printf 'ram_limit_exceeded current=%s threshold=%s\\n' "$mem_cur" "{ram_kill_threshold_bytes}" > ram_limit_exceeded.txt
      kill "$CMD_PID" 2>/dev/null || true
      exit 0
    fi
    sleep 1
  done
) > resource_samples.tsv &
MONITOR_PID=$!
(
  while kill -0 "$CMD_PID" 2>/dev/null; do
    if python3 detect_answer_started.py stdout.txt {shlex.quote(prompt)}; then
      first_ns=$(date +%s%N)
      python3 - "$start_ns" "$first_ns" > first_answer_ms.txt <<'PY'
import sys
print((int(sys.argv[2]) - int(sys.argv[1])) / 1000000.0)
PY
      exit 0
    fi
    sleep 0.05
  done
) &
WATCHER_PID=$!
wait "$CMD_PID"
STATUS=$?
set -e
kill "$WATCHER_PID" 2>/dev/null || true
wait "$WATCHER_PID" 2>/dev/null || true
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


def summarize_case(
    case_dir: Path,
    cpu_moe: int,
    unit: str,
    systemd_status: int,
    memory_max_bytes: int,
    ram_kill_threshold_bytes: int,
    prompt: str,
) -> dict[str, object]:
    stdout_text = (case_dir / "stdout.txt").read_text(encoding="utf-8", errors="replace") if (case_dir / "stdout.txt").exists() else ""
    stderr_text = (case_dir / "stderr.txt").read_text(encoding="utf-8", errors="replace") if (case_dir / "stderr.txt").exists() else ""
    combined = stdout_text + "\n" + stderr_text
    answer = extract_answer(stdout_text, prompt)
    correct, reason = check_correctness(answer, prompt)
    load_ms = parse_timing_ms(combined, "load time")
    prompt_ms = parse_timing_ms(combined, "prompt eval time")
    eval_ms = parse_timing_ms(combined, "eval time")
    ttft_estimate_ms = load_ms + prompt_ms if load_ms is not None and prompt_ms is not None else None
    time_v = parse_time_v(stderr_text)
    memory_peak_bytes = None
    if (case_dir / "memory.peak").exists():
        try:
            memory_peak_bytes = int((case_dir / "memory.peak").read_text().strip())
        except ValueError:
            pass
    memory_events = (case_dir / "memory.events").read_text(encoding="utf-8", errors="replace") if (case_dir / "memory.events").exists() else ""
    memory_event_counts = parse_memory_events(memory_events)
    memory_stat = (case_dir / "memory.stat").read_text(encoding="utf-8", errors="replace") if (case_dir / "memory.stat").exists() else ""
    memory_stat_relevant = parse_memory_stat(memory_stat)
    oom_seen = memory_event_counts.get("oom", 0) > 0 or memory_event_counts.get("oom_kill", 0) > 0
    max_events = memory_event_counts.get("max", 0)
    ram_limit_killed = (case_dir / "ram_limit_exceeded.txt").exists()
    first_answer_ms = None
    if (case_dir / "first_answer_ms.txt").exists():
        try:
            first_answer_ms = float((case_dir / "first_answer_ms.txt").read_text().strip())
        except ValueError:
            pass
    summary: dict[str, object] = {
        "case_dir": str(case_dir),
        "unit": unit,
        "cpu_moe": cpu_moe,
        "systemd_status": systemd_status,
        "exit_status": int((case_dir / "exit_status.txt").read_text().strip()) if (case_dir / "exit_status.txt").exists() else None,
        "prompt_tok_s": parse_rate(combined, "Prompt"),
        "eval_tok_s": parse_rate(combined, "Generation"),
        "load_ms": load_ms,
        "prompt_eval_ms": prompt_ms,
        "eval_ms": eval_ms,
        "ttft_estimate_ms": first_answer_ms if first_answer_ms is not None else ttft_estimate_ms,
        "first_answer_ms": first_answer_ms,
        "memory_peak_bytes": memory_peak_bytes,
        "memory_max_bytes": memory_max_bytes,
        "ram_kill_threshold_bytes": ram_kill_threshold_bytes,
        "memory_max_events": max_events,
        "memory_stat_relevant": memory_stat_relevant,
        "memory_file_bytes": memory_stat_relevant.get("file"),
        "pgmajfault": memory_stat_relevant.get("pgmajfault"),
        "workingset_refault_file": memory_stat_relevant.get("workingset_refault_file"),
        "max_rss_kb": time_v["max_rss_kb"],
        "elapsed_seconds": time_v["elapsed_seconds"],
        "oom_seen": oom_seen,
        "ram_limit_killed": ram_limit_killed,
        "ram_ok": memory_peak_bytes is not None and memory_peak_bytes <= ram_kill_threshold_bytes and not ram_limit_killed and not oom_seen,
        "correctness_ok": correct,
        "correctness_reason": reason,
        "answer": answer,
    }
    write_json(case_dir / "summary.json", summary)
    with (case_dir / "summary.env").open("w", encoding="utf-8") as f:
        for key, value in summary.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, sort_keys=True)
            f.write(f"{key}={json.dumps(value) if isinstance(value, str) else value}\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BIN)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-name", default="baseline")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--case-name", default=None, help="Case directory prefix. Defaults to france.")
    parser.add_argument("--cpu-moe", type=int, action="append", required=True)
    parser.add_argument("--vram-cache-gb", type=int, default=2)
    parser.add_argument("--memory-max-bytes", type=int, default=MEMORY_MAX_BYTES)
    parser.add_argument("--ram-kill-threshold-bytes", type=int, default=MEMORY_MAX_BYTES)
    parser.add_argument("--drop-caches-before-case", action="store_true")
    parser.add_argument("--env", action="append", default=[], help="Additional environment variable as KEY=VALUE. VALUE may contain {case_dir}.")
    parser.add_argument("--extra-arg", action="append", default=[])
    args = parser.parse_args()

    if args.memory_max_bytes > MEMORY_MAX_BYTES:
        print(f"MemoryMax must not exceed {MEMORY_MAX_BYTES} bytes for strict 16GB runs.", file=sys.stderr)
        return 2
    if args.ram_kill_threshold_bytes > args.memory_max_bytes:
        print("RAM kill threshold must not exceed MemoryMax.", file=sys.stderr)
        return 2
    if not args.binary.exists():
        print(f"missing binary: {args.binary}", file=sys.stderr)
        return 2
    if not args.model.exists():
        print(f"missing model: {args.model}", file=sys.stderr)
        return 2

    repo = Path.cwd()
    run_dir = args.out_root / f"{utc_stamp()}-{args.run_name}"
    run_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        "created_utc": utc_stamp(),
        "repo": str(repo),
        "branch": run_text(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo),
        "commit": run_text(["git", "rev-parse", "HEAD"], cwd=repo),
        "status": run_text(["git", "status", "--short"], cwd=repo),
        "binary": str(args.binary),
        "model": str(args.model),
        "memory_max_bytes": args.memory_max_bytes,
        "ram_kill_threshold_bytes": args.ram_kill_threshold_bytes,
        "drop_caches_before_case": args.drop_caches_before_case,
        "prompt": args.prompt,
    }
    write_json(run_dir / "metadata.json", metadata)
    (run_dir / "prompt.txt").write_text(args.prompt + "\n", encoding="utf-8")

    env = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        "GGML_MOE_STREAM": "1",
        "GGML_MOE_VRAM_CACHE_GB": str(args.vram_cache_gb),
        "GGML_CUDA_DISABLE_GRAPHS": "1",
        "GGML_MOE_STREAM_DONTNEED": "1",
    }
    extra_env: dict[str, str] = {}
    for item in args.env:
        if "=" not in item:
            print(f"--env must be KEY=VALUE, got: {item}", file=sys.stderr)
            return 2
        key, value = item.split("=", 1)
        if not key:
            print(f"--env key must not be empty: {item}", file=sys.stderr)
            return 2
        extra_env[key] = value

    summaries: list[dict[str, object]] = []
    extra_args = []
    for item in args.extra_arg:
        extra_args.extend(shlex.split(item))

    for cpu_moe in args.cpu_moe:
        case_name = args.case_name or "france"
        case_dir = run_dir / f"{case_name}-cpu{cpu_moe}-vram{args.vram_cache_gb}gb"
        case_dir.mkdir()
        script = case_dir / "run_case.sh"
        build_case_script(
            script,
            case_dir=case_dir,
            binary=args.binary,
            model=args.model,
            cpu_moe=cpu_moe,
            env={**env, **{key: value.format(case_dir=str(case_dir)) for key, value in extra_env.items()}},
            extra_args=extra_args,
            ram_kill_threshold_bytes=args.ram_kill_threshold_bytes,
            prompt=args.prompt,
        )
        unit = f"vendor-ds4-16gb-{utc_stamp()}-cpu{cpu_moe}.service"
        if args.drop_caches_before_case:
            (case_dir / "cold_start_procedure.txt").write_text("sync; echo 3 > /proc/sys/vm/drop_caches\n", encoding="utf-8")
            subprocess.run(["sync"], check=True)
            Path("/proc/sys/vm/drop_caches").write_text("3\n", encoding="utf-8")
        systemd_cmd = [
            "systemd-run",
            f"--unit={unit}",
            "--collect",
            "--wait",
            f"--property=MemoryMax={args.memory_max_bytes}",
            "--property=MemorySwapMax=0",
            str(script),
        ]
        (case_dir / "systemd_command.txt").write_text(" ".join(shlex.quote(x) for x in systemd_cmd) + "\n", encoding="utf-8")
        start = time.time()
        proc = subprocess.run(systemd_cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (case_dir / "systemd-run.out").write_text(proc.stdout, encoding="utf-8", errors="replace")
        (case_dir / "systemd_run_status.txt").write_text(f"{proc.returncode}\n", encoding="utf-8")
        (case_dir / "runner_elapsed_seconds.txt").write_text(f"{time.time() - start:.3f}\n", encoding="utf-8")
        subprocess.run(["systemctl", "show", unit], text=True, stdout=(case_dir / "unit.properties").open("w"), stderr=subprocess.DEVNULL)
        subprocess.run(["journalctl", "-u", unit, "--no-pager", "-n", "200"], text=True, stdout=(case_dir / "journal.log").open("w"), stderr=subprocess.DEVNULL)
        summaries.append(summarize_case(case_dir, cpu_moe, unit, proc.returncode, args.memory_max_bytes, args.ram_kill_threshold_bytes, args.prompt))

    write_json(run_dir / "summaries.json", summaries)
    with (run_dir / "results.tsv").open("w", encoding="utf-8") as f:
        cols = [
            "cpu_moe",
            "eval_tok_s",
            "prompt_tok_s",
            "ttft_estimate_ms",
            "memory_peak_bytes",
            "memory_max_events",
            "memory_file_bytes",
            "pgmajfault",
            "workingset_refault_file",
            "ram_ok",
            "ram_limit_killed",
            "correctness_ok",
            "correctness_reason",
            "case_dir",
        ]
        f.write("\t".join(cols) + "\n")
        for item in summaries:
            f.write("\t".join(str(item.get(col, "")) for col in cols) + "\n")

    print(run_dir)
    print((run_dir / "results.tsv").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
