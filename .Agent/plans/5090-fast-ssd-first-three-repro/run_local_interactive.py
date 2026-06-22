#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pty
import re
import select
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
READY_RE = re.compile(r"\[\[LLAMA_CHAT_READY:(\d+)]]")
STDERR_LOG_RE = re.compile(r"stderr_log=([^\r\n]+)")
ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1B\\))")
PROMPT_MARKER_RE = re.compile(r"\n?> \n?$")
VISIBLE_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


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


def read_pty(master_fd: int, proc: subprocess.Popen[bytes], timeout: float) -> tuple[bytes, list[tuple[float, bytes]]]:
    deadline = time.perf_counter() + timeout
    chunks: list[tuple[float, bytes]] = []
    raw = bytearray()
    while time.perf_counter() < deadline:
        if proc.poll() is not None:
            while True:
                r, _, _ = select.select([master_fd], [], [], 0)
                if not r:
                    break
                try:
                    data = os.read(master_fd, 4096)
                except OSError:
                    break
                if not data:
                    break
                raw.extend(data)
                chunks.append((time.perf_counter(), data))
            break
        r, _, _ = select.select([master_fd], [], [], min(0.2, max(0.0, deadline - time.perf_counter())))
        if not r:
            continue
        try:
            data = os.read(master_fd, 4096)
        except OSError:
            break
        if not data:
            break
        raw.extend(data)
        chunks.append((time.perf_counter(), data))
    return bytes(raw), chunks


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
        f"GGML_MOE_VRAM_CACHE_MIB={args.vram_mib}",
        f"GGML_MOE_VRAM_PROFILE={args.profile}",
        "GGML_MOE_VRAM_PROFILE_PROTECT=1",
        f"GGML_MOE_VRAM_PROFILE_RESERVE_PCT={args.profile_reserve_pct}",
        f"GGML_MOE_VRAM_CACHE_UPGATE_PCT={args.upgate_pct}",
        "GGML_MOE_VRAM_CACHE_POLICY=lfu_lru",
        "GGML_MOE_VRAM_CACHE_AUTO_CLAMP=0",
        f"LLAMA_CHAT_STARTUP_PROFILE_PRELOAD_TENSORS={args.startup_preload_tensors}",
    ]
    if args.extra_env:
        env_pairs.extend(args.extra_env)

    moe_args = [
        "python3", "scripts/moe-run.py",
        "--chat",
        "--chat-load-mode", "fast-prompt",
        "--preset", args.preset,
        "--chat-gpu-layers", str(args.gpu_layers),
        "--threads", str(args.threads),
        "--chat-threads-batch", str(args.threads_batch),
        "--",
        "-n", str(args.predict_tokens),
        "-b", str(args.batch),
        "-tb", str(args.threads_batch),
        "-t", str(args.threads),
        "--simple-io",
        "--seed", str(args.seed),
        "--ignore-eos",
    ]

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

    try:
        time_to_type, startup_raw, ready_error = consume_until_ready(master_fd, proc, args.ready_timeout_s)
        startup_path.write_bytes(startup_raw)
        if ready_error:
            raw_rest, _ = read_pty(master_fd, proc, 5.0)
            stdout_path.write_bytes(startup_raw + raw_rest)
            result = {
                "config": args.config_name,
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
        visible_times: list[float] = []
        assistant_parts: list[str] = []
        second_ready_t: float | None = None
        second_ready_seq: str | None = None
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

        try:
            tail, _ = read_pty(master_fd, proc, 2.0)
            raw.extend(tail)
        except Exception:
            pass

    finally:
        try:
            os.close(master_fd)
        except OSError:
            pass

    stdout_path.write_bytes(bytes(raw))
    clean_all = strip_ansi(bytes(raw).decode("utf-8", errors="replace"))
    stderr_log = parse_stderr_log_path(clean_all)
    assistant_text = "".join(assistant_parts).strip()
    result = {
        "config": args.config_name,
        "returncode": proc.returncode,
        "time_to_type_s": time_to_type,
        "interactive_ttft_s": (visible_times[0] - submit_t) if visible_times else None,
        "first_visible_s": (visible_times[0] - start) if visible_times else None,
        "generation_visible_duration_s": (second_ready_t - visible_times[0]) if second_ready_t and visible_times else None,
        "ready_seq_after_generation": second_ready_seq,
        "visible_token_estimate": visible_token_estimate(assistant_text),
        "assistant_sha256": __import__("hashlib").sha256(assistant_text.encode("utf-8")).hexdigest() if assistant_text else None,
        "stderr_log": stderr_log,
        "route_trace": str(trace_path) if trace_path.exists() else None,
        "ttft_trace": str(ttft_path) if ttft_path.exists() else None,
        "stdout": str(stdout_path),
        "env": str(env_path),
        "elapsed_process_s": time.perf_counter() - start,
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    status_path.write_text(("ok" if proc.returncode == 0 and visible_times else "failed") + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Local systemd-run PTY GLM interactive benchmark harness.")
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preset", default="presets/moe/glm51/rtx5090-interactive-n84-repro.json")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--vram-mib", type=int, required=True)
    parser.add_argument("--upgate-pct", type=int, default=60)
    parser.add_argument("--profile-reserve-pct", type=int, default=10)
    parser.add_argument("--startup-preload-tensors", type=int, default=3)
    parser.add_argument("--gpu-layers", type=int, default=79)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--threads-batch", type=int, default=24)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--predict-tokens", type=int, default=84)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="Write a concise explanation of why NVMe latency matters for MoE inference.")
    parser.add_argument("--ready-timeout-s", type=float, default=180.0)
    parser.add_argument("--generation-timeout-s", type=float, default=240.0)
    parser.add_argument("--extra-env", action="append", default=[])
    args = parser.parse_args()

    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("returncode") == 0 and not result.get("error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
