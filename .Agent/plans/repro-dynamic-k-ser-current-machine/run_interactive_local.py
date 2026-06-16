#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pty
import re
import select
import shlex
import signal
import subprocess
import time
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
READY_RE = re.compile(r"\[\[LLAMA_CHAT_READY:(\d+)]]")
STDERR_LOG_RE = re.compile(r"stderr_log=([^\r\n]+)")
ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1B\\))")
PROMPT_MARKER_RE = re.compile(r"\n?> \n?$")
VISIBLE_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
TIMING_RE = re.compile(
    r"llama_print_timings:\s+(?P<name>prompt eval|eval|total) time =\s+"
    r"(?P<ms>[0-9.]+) ms /(?:\s+(?P<tokens>[0-9]+) (?:tokens|runs))?"
    r".*?(?P<tps>[0-9.]+|-nan|inf) tokens per second"
)
PACK_RE = re.compile(
    r"\[moe_stream_batch\] expert pack: hits=(?P<hits>\d+) misses=(?P<misses>\d+) "
    r"read_failures=(?P<read_failures>\d+) direct_reads=(?P<direct_reads>\d+) "
    r"direct_fallbacks=(?P<direct_fallbacks>\d+)"
)
VCACHE_RE = re.compile(
    r"\[moe_stream_batch\] VRAM cache: hits=(?P<hits>\d+) misses=(?P<misses>\d+) "
    r"preloads=(?P<preloads>\d+) pinned=(?P<pinned>\d+) hit_rate=(?P<hit_rate>[0-9.]+)%"
)


def strip_ansi(text: str) -> str:
    text = ANSI_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 32)


def visible_token_estimate(text: str) -> int:
    return len(VISIBLE_RE.findall(text))


def shell_join(args: list[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in args)


def parse_stderr_log_path(text: str) -> str | None:
    match = STDERR_LOG_RE.search(text)
    return match.group(1) if match else None


def read_pty(master_fd: int, proc: subprocess.Popen[bytes], timeout: float) -> bytes:
    deadline = time.perf_counter() + timeout
    raw = bytearray()
    while time.perf_counter() < deadline:
        if proc.poll() is not None:
            drain_timeout = 0.0
        else:
            drain_timeout = min(0.2, max(0.0, deadline - time.perf_counter()))
        r, _, _ = select.select([master_fd], [], [], drain_timeout)
        if not r:
            if proc.poll() is not None:
                break
            continue
        try:
            data = os.read(master_fd, 4096)
        except OSError:
            break
        if not data:
            break
        raw.extend(data)
    return bytes(raw)


def consume_until_ready(master_fd: int, proc: subprocess.Popen[bytes], timeout: float) -> tuple[float | None, bytes, str | None]:
    start = time.perf_counter()
    clean = ""
    raw = bytearray()
    while time.perf_counter() - start < timeout:
        r, _, _ = select.select([master_fd], [], [], 0.2)
        if not r:
            if proc.poll() is not None:
                return None, bytes(raw), f"process exited before ready: {proc.returncode}"
            continue
        now = time.perf_counter()
        try:
            data = os.read(master_fd, 4096)
        except OSError as exc:
            return None, bytes(raw), f"pty read failed before ready: {exc}"
        if not data:
            return None, bytes(raw), "pty closed before ready"
        raw.extend(data)
        clean += strip_ansi(data.decode("utf-8", errors="replace"))
        if READY_RE.search(clean):
            return now - start, bytes(raw), None
    return None, bytes(raw), "ready marker timeout"


def add_stderr_metrics(result: dict[str, Any], stderr_log: str | None) -> None:
    if not stderr_log:
        return
    path = Path(stderr_log)
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    for match in TIMING_RE.finditer(text):
        key = match.group("name").replace(" ", "_")
        result[f"{key}_ms"] = float(match.group("ms"))
        if match.group("tokens"):
            result[f"{key}_tokens"] = int(match.group("tokens"))
        tps = match.group("tps")
        if tps and tps not in {"-nan", "inf"}:
            result[f"{key}_tokens_per_s"] = float(tps)
    pack_matches = list(PACK_RE.finditer(text))
    if pack_matches:
        for key, value in pack_matches[-1].groupdict().items():
            result[f"expert_pack_{key}"] = int(value)
    cache_matches = list(VCACHE_RE.finditer(text))
    if cache_matches:
        cache = cache_matches[-1]
        result["vram_cache_hits"] = int(cache.group("hits"))
        result["vram_cache_misses"] = int(cache.group("misses"))
        result["vram_cache_preloads"] = int(cache.group("preloads"))
        result["vram_cache_pinned"] = int(cache.group("pinned"))
        result["vram_cache_hit_rate_pct"] = float(cache.group("hit_rate"))
    result["ram_hit_rate_pct"] = 0.0


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.txt"
    startup_path = run_dir / "startup.txt"
    result_path = run_dir / "result.json"
    status_path = run_dir / "status.txt"
    env_path = run_dir / "env.txt"
    trace_path = run_dir / "route.trace.csv"
    ttft_path = run_dir / "ttft.trace.csv"

    env_pairs = [
        "LLAMA_CHAT_READY_MARKER=1",
        "LLAMA_CHAT_EXIT_COMMAND=1",
        "PYTHONUNBUFFERED=1",
        f"LLAMA_CACHE={run_dir / 'llama-cache'}",
        f"MODEL={REPO / 'models/GLM-5.1-UD-IQ3_XXS/GLM-5.1-UD-IQ3_XXS-00001-of-00007.gguf'}",
        f"GGML_MOE_EXPERT_PACK={REPO / 'models/GLM-5.1-UD-IQ3_XXS/glm51-iq3xxs.expert-pack'}",
        f"GGML_MOE_ROUTE_TRACE_OUT={trace_path}",
        f"GGML_MOE_TTFT_TRACE_OUT={ttft_path}",
        "GGML_MOE_RAM_TIER_MIB=0",
        "GGML_MOE_RAM_TIER_SKIP=0",
        "GGML_MOE_IO_BACKEND=direct",
        "GGML_MOE_STAGE_PINNED=0",
        "GGML_MOE_VRAM_CACHE_MIB=12288",
        "GGML_MOE_VRAM_PROFILE=presets/moe/groundtruth/wici-glm51-interactive-n84-expert-top8.runtime.csv",
        "GGML_MOE_VRAM_PROFILE_PROTECT=1",
        "GGML_MOE_VRAM_PROFILE_RESERVE_PCT=10",
        "GGML_MOE_VRAM_CACHE_UPGATE_PCT=60",
        "GGML_MOE_VRAM_CACHE_POLICY=lfu_lru",
        "GGML_MOE_VRAM_CACHE_AUTO_CLAMP=0",
        "LLAMA_CHAT_STARTUP_PROFILE_PRELOAD_TENSORS=3",
    ]
    if args.extra_env:
        env_pairs.extend(args.extra_env)

    moe_args = [
        "python3", "scripts/moe-run.py",
        "--chat",
        "--chat-load-mode", "fast-prompt",
        "--preset", "presets/moe/glm51/rtx5090-interactive-n84-repro.json",
        "--chat-gpu-layers", "79",
        "--threads", "8",
        "--chat-threads-batch", "24",
        "--",
        "-n", str(args.predict_tokens),
        "-b", "2048",
        "-tb", "24",
        "-t", "8",
        "--simple-io",
        "--seed", "42",
        "--ignore-eos",
    ]
    if args.ser:
        moe_args.extend(["--smart-expert-reduction", args.ser])

    inner = shell_join(["env", *env_pairs, *moe_args])
    cmd = [
        "systemd-run", "--user", "--pty", "--wait", "--collect",
        "-p", f"WorkingDirectory={REPO}",
        "-p", "MemoryMax=2G",
        "-p", "MemorySwapMax=0",
        "bash", "-lc", inner,
    ]
    env_path.write_text("\n".join(env_pairs) + "\ncommand=" + shell_join(cmd) + "\n", encoding="utf-8")

    master_fd, slave_fd = pty.openpty()
    start = time.perf_counter()
    proc = subprocess.Popen(cmd, stdin=slave_fd, stdout=slave_fd, stderr=slave_fd, close_fds=True)
    os.close(slave_fd)

    time_to_type: float | None = None
    startup_raw = b""
    visible_times: list[float] = []
    assistant_parts: list[str] = []
    second_ready_t: float | None = None
    second_ready_seq: str | None = None
    submit_t: float | None = None

    try:
        time_to_type, startup_raw, ready_error = consume_until_ready(master_fd, proc, args.ready_timeout_s)
        startup_path.write_bytes(startup_raw)
        if ready_error:
            raw_rest = read_pty(master_fd, proc, 5.0)
            stdout_path.write_bytes(startup_raw + raw_rest)
            result = {
                "config": args.config_name,
                "ser": args.ser,
                "returncode": proc.poll(),
                "error": ready_error,
                "time_to_type_s": time_to_type,
                "stderr_log": parse_stderr_log_path(startup_raw.decode("utf-8", errors="replace")),
            }
            result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            status_path.write_text("failed\n", encoding="utf-8")
            return result

        submit_t = time.perf_counter()
        os.write(master_fd, (args.prompt + "\n").encode("utf-8"))

        clean = ""
        raw = bytearray(startup_raw)
        echo_buffer = ""
        assistant_started = False
        deadline = time.perf_counter() + args.generation_timeout_s
        while time.perf_counter() < deadline:
            if proc.poll() is not None:
                break
            r, _, _ = select.select([master_fd], [], [], 0.2)
            if not r:
                continue
            now = time.perf_counter()
            try:
                data = os.read(master_fd, 4096)
            except OSError:
                break
            if not data:
                break
            raw.extend(data)
            clean += strip_ansi(data.decode("utf-8", errors="replace"))
            match = READY_RE.search(clean)
            if match:
                before_marker = clean[:match.start()]
                clean = ""
                done = True
            else:
                before_marker = clean
                clean = ""
                done = False

            part = before_marker
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
                    kept_chars: list[str] = []
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
                if part.strip():
                    assistant_started = True
                else:
                    if done:
                        break
                    continue

            part = PROMPT_MARKER_RE.sub("", part)
            if part.strip():
                assistant_parts.append(part)
                visible_times.append(now)
            if match:
                second_ready_t = now
                second_ready_seq = match.group(1)
                break

        try:
            os.write(master_fd, b"/exit\n")
        except OSError:
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)

        raw.extend(read_pty(master_fd, proc, 2.0))
    finally:
        try:
            os.close(master_fd)
        except OSError:
            pass

    stdout_path.write_bytes(bytes(raw))
    clean_all = strip_ansi(bytes(raw).decode("utf-8", errors="replace"))
    stderr_log = parse_stderr_log_path(clean_all)
    assistant_text = "".join(assistant_parts).strip()
    result: dict[str, Any] = {
        "config": args.config_name,
        "ser": args.ser,
        "returncode": proc.returncode,
        "time_to_type_s": time_to_type,
        "interactive_ttft_s": (visible_times[0] - submit_t) if visible_times and submit_t else None,
        "first_visible_s": (visible_times[0] - start) if visible_times else None,
        "generation_visible_duration_s": (second_ready_t - visible_times[0]) if second_ready_t and visible_times else None,
        "ready_seq_after_generation": second_ready_seq,
        "visible_token_estimate": visible_token_estimate(assistant_text),
        "assistant_sha256": hashlib.sha256(assistant_text.encode("utf-8")).hexdigest() if assistant_text else None,
        "stderr_log": stderr_log,
        "route_trace": str(trace_path) if trace_path.exists() else None,
        "ttft_trace": str(ttft_path) if ttft_path.exists() else None,
        "stdout": str(stdout_path),
        "env": str(env_path),
        "elapsed_process_s": time.perf_counter() - start,
    }
    add_stderr_metrics(result, stderr_log)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    status_path.write_text(("ok" if proc.returncode == 0 and visible_times else "failed") + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Task-local interactive SER reproduction runner.")
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ser")
    parser.add_argument("--predict-tokens", type=int, default=36)
    parser.add_argument("--prompt", default="Write a concise explanation of why NVMe latency matters for MoE inference.")
    parser.add_argument("--ready-timeout-s", type=float, default=240.0)
    parser.add_argument("--generation-timeout-s", type=float, default=360.0)
    parser.add_argument("--extra-env", action="append", default=[])
    args = parser.parse_args()

    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("returncode") == 0 and not result.get("error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
