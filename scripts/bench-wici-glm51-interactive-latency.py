#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import posixpath
import queue
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_REMOTE_DIR = "/home/wici/venti/ik_llama"
DEFAULT_PROMPT = "Write a concise explanation of why NVMe latency matters for MoE inference."
EXPECTED_N36_SHA = "6d9e7cba8a78e6ea35d4ee4c56f4730642cf7cb78393779eb1d70b50508ed8b9"
START_MARKER = "[[WICI_INTERACTIVE_BENCH_START]]"
READY_RE = re.compile(r"\[\[LLAMA_CHAT_READY:(\d+)]]")
ANSI_RE = re.compile(
    r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1B\\))"
)
VISIBLE_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
PROMPT_MARKER_RE = re.compile(r"\n?> \n?$")
STDERR_LOG_RE = re.compile(r"stderr_log=([^\r\n]+)")
TIMING_RE = re.compile(
    r"llama_print_timings:\s+(?P<name>prompt eval|eval|total) time =\s+"
    r"(?P<ms>[0-9.]+) ms /(?:\s+(?P<tokens>[0-9]+) (?:tokens|runs))?"
    r".*?(?P<tps>[0-9.]+|-nan|inf) tokens per second"
)
CACHE_BUDGET_RE = re.compile(
    r"\[moe_stream_batch\] VRAM cache budget: "
    r"requested=(?P<requested>[0-9]+) MiB actual=(?P<actual>[0-9]+) MiB "
    r"free=(?P<free>[0-9]+) MiB total=(?P<total>[0-9]+) MiB "
    r"graph_reserve=(?P<graph_reserve>[0-9]+) MiB safety=(?P<safety>[0-9]+) MiB "
    r"clamp=(?P<clamp>[01])"
)
CACHE_ALLOC_RE = re.compile(
    r"\[moe_stream_batch\] VRAM cache: (?P<gib>[0-9.]+) GiB, "
    r"(?P<slots>[0-9]+) slots \((?P<slot_mib>[0-9.]+) MiB each\)"
)


@dataclass
class TimedChunk:
    t: float
    data: bytes


class ReaderThread(threading.Thread):
    def __init__(self, stream: Any, chunks: "queue.Queue[TimedChunk | None]") -> None:
        super().__init__(daemon=True)
        self.stream = stream
        self.chunks = chunks

    def run(self) -> None:
        try:
            while True:
                data = os.read(self.stream.fileno(), 4096)
                if not data:
                    break
                self.chunks.put(TimedChunk(time.perf_counter(), data))
        finally:
            self.chunks.put(None)


def strip_ansi(text: str) -> str:
    text = ANSI_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 32)


def has_visible_text(text: str) -> bool:
    return any(not ch.isspace() for ch in text)


def visible_token_estimate(text: str) -> int:
    return len(VISIBLE_TOKEN_RE.findall(text))


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = math.ceil((pct / 100.0) * len(ordered)) - 1
    return ordered[max(0, min(idx, len(ordered) - 1))]


def summary_stats(values: list[float]) -> dict[str, float | None]:
    return {
        "min": min(values) if values else None,
        "p50": percentile(values, 50.0),
        "p95": percentile(values, 95.0),
        "max": max(values) if values else None,
    }


def shell_join(args: list[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in args)


def remote_route_trace_path(args: argparse.Namespace, run_index: int) -> str | None:
    if not args.route_trace_dir:
        return None
    prefix = getattr(args, "route_trace_prefix", "interactive-latency")
    name = f"{prefix}-run{run_index:02d}.trace.csv"
    return posixpath.join(args.route_trace_dir.rstrip("/"), name)


def remote_ttft_trace_path(args: argparse.Namespace, run_index: int) -> str | None:
    if not args.ttft_trace_dir:
        return None
    prefix = getattr(args, "route_trace_prefix", "interactive-latency")
    name = f"{prefix}-run{run_index:02d}.ttft.csv"
    return posixpath.join(args.ttft_trace_dir.rstrip("/"), name)


def build_remote_command(args: argparse.Namespace, run_index: int) -> str:
    moe_args = ["python3", "scripts/moe-run.py", "--chat", "--chat-load-mode", args.chat_load_mode]
    if args.force_preset:
        moe_args.append("--force")
    if args.moe_preset:
        moe_args.extend(["--preset", args.moe_preset])
    if args.moe_route_trace:
        moe_args.extend(["--route-trace", args.moe_route_trace])
    if args.chat_active_prewarm:
        moe_args.append("--chat-active-prewarm")
    if args.threads is not None:
        moe_args.extend(["--threads", str(args.threads)])
    if args.chat_threads_batch is not None:
        moe_args.extend(["--chat-threads-batch", str(args.chat_threads_batch)])
    if args.chat_gpu_layers is not None:
        moe_args.extend(["--chat-gpu-layers", str(args.chat_gpu_layers)])

    cli_extra = ["--seed", str(args.seed), "-n", str(args.predict_tokens)]
    if args.ignore_eos:
        cli_extra.append("--ignore-eos")
    cli_extra.extend(args.extra)

    trace_out = remote_route_trace_path(args, run_index)
    ttft_trace_out = remote_ttft_trace_path(args, run_index)
    generated_env = []
    if trace_out:
        generated_env.append(f"GGML_MOE_ROUTE_TRACE_OUT={trace_out}")
    if ttft_trace_out:
        generated_env.append(f"GGML_MOE_TTFT_TRACE_OUT={ttft_trace_out}")

    setup = ""
    setup_dirs = []
    if trace_out:
        setup_dirs.append(posixpath.dirname(trace_out))
    if ttft_trace_out:
        setup_dirs.append(posixpath.dirname(ttft_trace_out))
    if setup_dirs:
        quoted_dirs = " ".join(shlex.quote(path) for path in sorted(set(setup_dirs)))
        setup = f"mkdir -p {quoted_dirs} && "

    env_args = [
        "env",
        "LLAMA_CHAT_READY_MARKER=1",
        "LLAMA_CHAT_EXIT_COMMAND=1",
        "PYTHONUNBUFFERED=1",
        *generated_env,
        *args.remote_env,
    ]
    return (
        f"cd {shlex.quote(args.remote_dir)} && "
        f"{setup}"
        f"printf '%s\\n' {shlex.quote(START_MARKER)} && "
        f"exec {shell_join(env_args + moe_args + ['--'] + cli_extra)}"
    )


def build_ssh_command(args: argparse.Namespace, run_index: int) -> list[str]:
    return ["ssh", "-tt", args.ssh_host, build_remote_command(args, run_index)]


def terminate_process(proc: subprocess.Popen[bytes], graceful: bool = False) -> None:
    if proc.poll() is not None:
        return
    try:
        if proc.stdin:
            proc.stdin.write(b"/exit\n" if graceful else b"\x03")
            proc.stdin.flush()
    except OSError:
        pass

    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if graceful:
            terminate_process(proc, graceful=False)
        else:
            proc.kill()
            proc.wait(timeout=10)


def consume_ready_marker(
    proc: subprocess.Popen[bytes],
    chunks: "queue.Queue[TimedChunk | None]",
    start: float,
    timeout: float,
) -> tuple[float, float | None, str, str | None]:
    deadline = time.perf_counter() + timeout
    clean_seen = ""
    raw_seen = bytearray()
    remote_start_t: float | None = None

    while time.perf_counter() < deadline:
        remaining = max(0.1, deadline - time.perf_counter())
        try:
            item = chunks.get(timeout=min(0.5, remaining))
        except queue.Empty:
            if proc.poll() is not None:
                break
            continue
        if item is None:
            break

        raw_seen.extend(item.data)
        clean_seen += strip_ansi(item.data.decode("utf-8", errors="replace"))
        if remote_start_t is None and START_MARKER in clean_seen:
            remote_start_t = item.t
        match = READY_RE.search(clean_seen)
        if match:
            start_t = remote_start_t if remote_start_t is not None else start
            return item.t - start_t, remote_start_t, raw_seen.decode("utf-8", errors="replace"), None

    return 0.0, remote_start_t, raw_seen.decode("utf-8", errors="replace"), "ready marker timeout"


def split_ready(text: str) -> tuple[str, str | None]:
    match = READY_RE.search(text)
    if not match:
        return text, None
    return text[: match.start()], match.group(1)


def strip_trailing_prompt_marker(text: str) -> str:
    return PROMPT_MARKER_RE.sub("", text)


def parse_stderr_log_path(text: str) -> str | None:
    match = STDERR_LOG_RE.search(text)
    return match.group(1) if match else None


def fetch_remote_text(args: argparse.Namespace, path: str) -> str | None:
    try:
        return subprocess.check_output(
            ["ssh", args.ssh_host, f"cat -- {shlex.quote(path)}"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def add_stderr_metrics(record: dict[str, Any], text: str) -> None:
    record["moe_vram_cache_failed"] = "VRAM cache: cudaMalloc" in text or "CUDA error" in text
    budget = CACHE_BUDGET_RE.search(text)
    if budget:
        record["moe_vram_cache_requested_mib"] = int(budget.group("requested"))
        record["moe_vram_cache_actual_mib"] = int(budget.group("actual"))
        record["moe_vram_cache_free_mib"] = int(budget.group("free"))
        record["moe_vram_cache_total_mib"] = int(budget.group("total"))
        record["moe_vram_cache_graph_reserve_mib"] = int(budget.group("graph_reserve"))
        record["moe_vram_cache_safety_mib"] = int(budget.group("safety"))
        record["moe_vram_cache_clamp"] = budget.group("clamp") == "1"

    allocs = list(CACHE_ALLOC_RE.finditer(text))
    if allocs:
        record["moe_vram_cache_allocations"] = [
            {
                "gib": float(match.group("gib")),
                "slots": int(match.group("slots")),
                "slot_mib": float(match.group("slot_mib")),
            }
            for match in allocs
        ]
        record["moe_vram_cache_allocated_gib"] = sum(float(match.group("gib")) for match in allocs)
        record["moe_vram_cache_slots"] = sum(int(match.group("slots")) for match in allocs)

    for match in TIMING_RE.finditer(text):
        prefix = match.group("name").replace(" ", "_")
        record[f"{prefix}_time_ms"] = float(match.group("ms"))
        if match.group("tokens") is not None:
            record[f"{prefix}_tokens"] = int(match.group("tokens"))
        tps = match.group("tps")
        if tps not in ("-nan", "inf"):
            record[f"{prefix}_tokens_per_s"] = float(tps)


def add_remote_timings(record: dict[str, Any], args: argparse.Namespace) -> None:
    path = parse_stderr_log_path(record.get("startup_raw_tail", ""))
    if not path:
        return
    record["stderr_log"] = path
    text = fetch_remote_text(args, path)
    if text is None:
        record["stderr_log_fetch_error"] = True
        return
    add_stderr_metrics(record, text)


def run_one(args: argparse.Namespace, run_index: int) -> dict[str, Any]:
    cmd = build_ssh_command(args, run_index)
    route_trace_out = remote_route_trace_path(args, run_index)
    ttft_trace_out = remote_ttft_trace_path(args, run_index)
    chunks: "queue.Queue[TimedChunk | None]" = queue.Queue()
    start = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert proc.stdout is not None
    ReaderThread(proc.stdout, chunks).start()

    record: dict[str, Any] = {
        "schema": "wici_glm51_interactive_latency_v1",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run": run_index,
        "ssh_host": args.ssh_host,
        "remote_dir": args.remote_dir,
        "chat_load_mode": args.chat_load_mode,
        "chat_active_prewarm": args.chat_active_prewarm,
        "threads": args.threads,
        "chat_threads_batch": args.chat_threads_batch,
        "chat_gpu_layers": args.chat_gpu_layers,
        "force_preset": args.force_preset,
        "moe_preset": args.moe_preset,
        "moe_route_trace": args.moe_route_trace,
        "route_trace_out": route_trace_out,
        "ttft_trace_out": ttft_trace_out,
        "prompt": args.prompt,
        "seed": args.seed,
        "predict_tokens": args.predict_tokens,
        "ignore_eos": args.ignore_eos,
        "command": cmd,
    }

    try:
        time_to_type, remote_start_t, startup_raw, error = consume_ready_marker(
            proc, chunks, start, args.startup_timeout
        )
        record["startup_raw_tail"] = startup_raw[-4000:]
        if remote_start_t is not None:
            record["ssh_to_remote_start_s"] = remote_start_t - start
        if error:
            record["ok"] = False
            record["error"] = error
            record["returncode"] = proc.poll()
            return record

        record["time_to_type_s"] = time_to_type
        if proc.stdin is None:
            raise RuntimeError("ssh stdin is not available")

        if args.post_ready_delay_s > 0:
            time.sleep(args.post_ready_delay_s)
        record["post_ready_delay_s"] = args.post_ready_delay_s
        submit_t = time.perf_counter()
        proc.stdin.write(args.prompt.encode("utf-8") + b"\n")
        proc.stdin.flush()

        generation_deadline = time.perf_counter() + args.generation_timeout
        clean_buffer = ""
        assistant_started = False
        echo_buffer = ""
        assistant_parts: list[str] = []
        visible_times: list[float] = []
        second_ready_t: float | None = None
        second_ready_seq: str | None = None
        raw_tail = bytearray()

        while time.perf_counter() < generation_deadline:
            remaining = max(0.1, generation_deadline - time.perf_counter())
            try:
                item = chunks.get(timeout=min(0.5, remaining))
            except queue.Empty:
                if proc.poll() is not None:
                    break
                continue
            if item is None:
                break

            raw_tail.extend(item.data)
            if len(raw_tail) > 8000:
                del raw_tail[:-8000]

            clean_buffer += strip_ansi(item.data.decode("utf-8", errors="replace"))
            before_marker, marker_seq = split_ready(clean_buffer)
            if marker_seq is not None:
                clean_buffer = ""
                second_ready_t = item.t
                second_ready_seq = marker_seq
                part = strip_trailing_prompt_marker(before_marker)
                done = True
            else:
                part = clean_buffer
                clean_buffer = ""
                done = False

            if not assistant_started:
                echo_buffer += part
                compact_echo = echo_buffer.replace("\n", "")
                prompt_compact = args.prompt.replace("\n", "")
                if prompt_compact.startswith(compact_echo) and compact_echo != prompt_compact:
                    if done:
                        break
                    continue
                if compact_echo.startswith(prompt_compact):
                    skipped = len(prompt_compact)
                    kept_chars = []
                    seen = 0
                    for ch in echo_buffer:
                        if ch != "\n" and seen < skipped:
                            seen += 1
                            continue
                        kept_chars.append(ch)
                    part = "".join(kept_chars).lstrip()
                else:
                    part = echo_buffer.lstrip()
                echo_buffer = ""
                if has_visible_text(part):
                    assistant_started = True
                else:
                    if done:
                        break
                    continue

            if has_visible_text(part):
                assistant_parts.append(part)
                visible_times.append(item.t)

            if done:
                break

        if second_ready_t is None:
            record["ok"] = False
            record["error"] = "second ready marker timeout"
        elif not visible_times:
            record["ok"] = False
            record["error"] = "no visible assistant output before second ready marker"
        else:
            assistant_text = "".join(assistant_parts)
            gaps = [b - a for a, b in zip(visible_times, visible_times[1:])]
            duration = max(0.0, second_ready_t - visible_times[0])
            token_estimate = visible_token_estimate(assistant_text)
            record.update(
                {
                    "ok": True,
                    "ready_seq_after_generation": second_ready_seq,
                    "interactive_ttft_s": visible_times[0] - submit_t,
                    "visible_chunk_count": len(visible_times),
                    "visible_chunk_gap_p95_s": percentile(gaps, 95.0),
                    "visible_chunk_gap_max_s": max(gaps) if gaps else None,
                    "sustained_duration_s": duration,
                    "visible_token_estimate": token_estimate,
                    "sustained_visible_token_estimate_per_s": (
                        token_estimate / duration if duration > 0 else None
                    ),
                    "sustained_requested_token_per_s": (
                        args.predict_tokens / duration if duration > 0 else None
                    ),
                    "assistant_text": assistant_text,
                    "assistant_sha256": hashlib.sha256(assistant_text.encode("utf-8")).hexdigest(),
                }
            )
            if args.seed == 42 and args.predict_tokens == 36 and args.prompt == DEFAULT_PROMPT:
                record["n36_canary_ok"] = record["assistant_sha256"] == EXPECTED_N36_SHA

        record["raw_tail"] = raw_tail.decode("utf-8", errors="replace")
        return record
    finally:
        terminate_process(proc, graceful=True)
        add_remote_timings(record, args)
        record["returncode"] = proc.returncode


SUMMARY_KEYS = (
    "time_to_type_s",
    "interactive_ttft_s",
    "visible_chunk_gap_p95_s",
    "visible_chunk_gap_max_s",
    "eval_tokens_per_s",
    "sustained_visible_token_estimate_per_s",
    "sustained_requested_token_per_s",
    "moe_vram_cache_requested_mib",
    "moe_vram_cache_actual_mib",
    "moe_vram_cache_free_mib",
    "moe_vram_cache_safety_mib",
    "moe_vram_cache_slots",
    "moe_vram_cache_allocated_gib",
)


def build_summary(records: list[dict[str, Any]], output: Path | None = None) -> dict[str, Any]:
    ok = [record for record in records if record.get("ok")]
    summary: dict[str, Any] = {
        "schema": "wici_glm51_interactive_latency_summary_v1",
        "jsonl": str(output) if output is not None else None,
        "runs": len(records),
        "ok": len(ok),
        "failed": len(records) - len(ok),
        "moe_vram_cache_failed_runs": sum(1 for record in records if record.get("moe_vram_cache_failed")),
        "stats": {},
    }
    canary_values = [record.get("n36_canary_ok") for record in ok if record.get("n36_canary_ok") is not None]
    if canary_values:
        summary["n36_canary_ok_runs"] = sum(1 for value in canary_values if value)
        summary["n36_canary_failed_runs"] = sum(1 for value in canary_values if not value)
    for key in SUMMARY_KEYS:
        values = [float(record[key]) for record in ok if record.get(key) is not None]
        summary["stats"][key] = summary_stats(values)
    clamp_values = [record.get("moe_vram_cache_clamp") for record in ok if record.get("moe_vram_cache_clamp") is not None]
    if clamp_values:
        summary["moe_vram_cache_clamp_runs"] = sum(1 for value in clamp_values if value)
    return summary


def print_summary(records: list[dict[str, Any]], output: Path | None = None) -> None:
    summary = build_summary(records, output)
    print(f"runs={summary['runs']} ok={summary['ok']} failed={summary['failed']}")
    for key in SUMMARY_KEYS:
        stats = summary["stats"][key]
        if stats["min"] is None:
            continue
        print(
            f"{key}: "
            f"min={fmt(stats['min'])} p50={fmt(stats['p50'])} "
            f"p95={fmt(stats['p95'])} max={fmt(stats['max'])}"
        )


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure GLM-5.1 interactive chat latency through ssh -tt on wici."
    )
    parser.add_argument("--ssh-host", default="wici")
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--predict-tokens", type=int, default=84, help="llama-cli -n value. The default yields 64 generated tokens for the fixed GLM-5.1 chat prompt.")
    parser.add_argument("--chat-load-mode", choices=("fast-prompt", "eager"), default="fast-prompt")
    parser.add_argument("--chat-active-prewarm", action="store_true")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--chat-threads-batch", type=int)
    parser.add_argument("--chat-gpu-layers", type=int)
    parser.add_argument("--force-preset", action="store_true", help="Pass --force to moe-run.py so the preset is recomputed for this run.")
    parser.add_argument("--moe-preset", help="Remote preset JSON passed to moe-run.py --preset for direct preset validation.")
    parser.add_argument("--moe-route-trace", help="Remote route trace CSV passed to moe-run.py --route-trace for trace-aware preset selection.")
    parser.add_argument("--route-trace-dir", help="Remote directory for GGML_MOE_ROUTE_TRACE_OUT files captured during each run.")
    parser.add_argument("--ttft-trace-dir", help="Remote directory for GGML_MOE_TTFT_TRACE_OUT per-expert load traces.")
    parser.add_argument("--remote-env", action="append", default=[], metavar="KEY=VALUE", help="Extra environment variable for the remote moe-run.py process. May be repeated.")
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--generation-timeout", type=float, default=300.0)
    parser.add_argument("--post-ready-delay-s", type=float, default=0.0, help="Wait after the first ready marker before submitting the prompt.")
    parser.add_argument("--run-pause-s", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", type=Path, help="Write aggregate p50/p95 summary JSON.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="Arguments after -- are passed to llama-cli.")
    args = parser.parse_args()

    if args.extra and args.extra[0] == "--":
        args.extra = args.extra[1:]
    if args.runs <= 0:
        raise SystemExit("--runs must be positive")
    if args.predict_tokens <= 0:
        raise SystemExit("--predict-tokens must be positive")
    if args.post_ready_delay_s < 0:
        raise SystemExit("--post-ready-delay-s must be non-negative")
    for value in args.remote_env:
        if "=" not in value or value.startswith("="):
            raise SystemExit("--remote-env values must be KEY=VALUE")
    return args


def main() -> int:
    args = parse_args()
    output = args.output
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output = REPO / "bench" / "wici-glm51-interactive-latency" / f"{stamp}.jsonl"
    args.route_trace_prefix = output.stem

    if args.dry_run:
        print(shell_join(build_ssh_command(args, 1)))
        print(f"output={output}")
        if args.summary:
            print(f"summary={args.summary}")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with output.open("a", encoding="utf-8") as out:
        for run_index in range(1, args.runs + 1):
            print(f"run {run_index}/{args.runs}", file=sys.stderr)
            record = run_one(args, run_index)
            records.append(record)
            out.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            out.flush()
            if not record.get("ok"):
                print(f"run {run_index} failed: {record.get('error')}", file=sys.stderr)
            elif args.run_pause_s > 0:
                time.sleep(args.run_pause_s)

    print(f"jsonl={output}")
    print_summary(records, output)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(build_summary(records, output), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"summary={args.summary}")
    return 0 if all(record.get("ok") for record in records) else 1


if __name__ == "__main__":
    if os.name == "nt":
        signal.signal(signal.SIGINT, signal.SIG_DFL)
    raise SystemExit(main())
