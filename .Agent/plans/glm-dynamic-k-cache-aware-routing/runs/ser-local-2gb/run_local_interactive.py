#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import pty
import re
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path


REPO = Path("/root/lfz/ik_llama")
PROMPT = "Write a concise explanation of why NVMe latency matters for MoE inference."
START_MARKER = "[[LOCAL_INTERACTIVE_BENCH_START]]"
READY_RE = re.compile(r"\[\[LLAMA_CHAT_READY:(\d+)]]")
STDERR_LOG_RE = re.compile(r"stderr_log=([^\r\n]+)")
VISIBLE_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1B\\))")
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


def visible_tokens(text: str) -> int:
    return len(VISIBLE_RE.findall(text))


def add_stderr_metrics(record: dict[str, object]) -> None:
    stderr_log = record.get("stderr_log")
    if not stderr_log:
        return
    path = Path(str(stderr_log))
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    for match in TIMING_RE.finditer(text):
        prefix = match.group("name").replace(" ", "_")
        record[f"{prefix}_ms"] = float(match.group("ms"))
        if match.group("tokens"):
            record[f"{prefix}_tokens"] = int(match.group("tokens"))
        tps = match.group("tps")
        if tps not in {"-nan", "inf"}:
            record[f"{prefix}_tokens_per_s"] = float(tps)
    pack_matches = list(PACK_RE.finditer(text))
    if pack_matches:
        pack = pack_matches[-1]
        for key, value in pack.groupdict().items():
            record[f"expert_pack_{key}"] = int(value)
    vcache_matches = list(VCACHE_RE.finditer(text))
    if vcache_matches:
        vcache = vcache_matches[-1]
        record["vram_cache_hits"] = int(vcache.group("hits"))
        record["vram_cache_misses"] = int(vcache.group("misses"))
        record["vram_cache_preloads"] = int(vcache.group("preloads"))
        record["vram_cache_pinned"] = int(vcache.group("pinned"))
        record["vram_cache_hit_rate_pct"] = float(vcache.group("hit_rate"))
    record["ram_hit_rate_pct"] = 0.0


def build_inner_command(args: argparse.Namespace) -> list[str]:
    env = [
        "env",
        "LLAMA_CHAT_READY_MARKER=1",
        "LLAMA_CHAT_EXIT_COMMAND=1",
        "PYTHONUNBUFFERED=1",
        "GGML_MOE_RAM_TIER_MIB=0",
        "GGML_MOE_RAM_TIER_SKIP=0",
        "GGML_MOE_VRAM_CACHE_MIB=12288",
        "GGML_MOE_VRAM_PROFILE=presets/moe/groundtruth/wici-glm51-interactive-n84-expert-top8.runtime.csv",
        "GGML_MOE_IO_BACKEND=direct",
        "GGML_MOE_STAGE_PINNED=0",
    ]
    moe = [
        "python3",
        "scripts/moe-run.py",
        "--chat",
        "--chat-load-mode",
        "fast-prompt",
        "--low-host-ram-profile",
        "glm51-2gb-vram12",
    ]
    cli = ["--", "--seed", str(args.seed), "-n", str(args.predict_tokens), "--ignore-eos"]
    if args.ser:
        cli += ["--smart-expert-reduction", args.ser]
    return env + moe + cli


def build_command(args: argparse.Namespace) -> list[str]:
    inner = build_inner_command(args)
    mode_flag = "--user" if args.user_scope else "--same-dir"
    return [
        "systemd-run",
        mode_flag,
        "--pty",
        "--wait",
        "--collect",
        "-p",
        "MemoryMax=2G",
        "-p",
        "MemorySwapMax=0",
        *inner,
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--raw-output", required=True, type=Path)
    parser.add_argument("--ser", help="smart expert reduction setting, e.g. 4,0.05")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--predict-tokens", type=int, default=36)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--generation-timeout", type=float, default=300.0)
    parser.add_argument("--user-scope", action="store_true")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)

    master_fd, slave_fd = pty.openpty()
    flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
    fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    cmd = build_command(args)
    proc = subprocess.Popen(
        cmd,
        cwd=REPO,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
        start_new_session=True,
    )
    os.close(slave_fd)

    sel = selectors.DefaultSelector()
    sel.register(master_fd, selectors.EVENT_READ)
    start = time.perf_counter()
    raw = bytearray()
    clean = ""
    ready_t = None
    submit_t = None
    first_visible_t = None
    second_ready_t = None
    assistant_parts: list[str] = []
    submitted = False
    prompt_echo_seen = False
    post_submit_text = ""

    print(START_MARKER)
    print("command=" + " ".join(cmd), flush=True)

    try:
        deadline = start + args.startup_timeout
        gen_deadline = None
        while True:
            now = time.perf_counter()
            if ready_t is None and now > deadline:
                raise TimeoutError("ready marker timeout")
            if gen_deadline is not None and now > gen_deadline:
                raise TimeoutError("generation timeout")
            if proc.poll() is not None and not sel.get_map():
                break

            events = sel.select(0.2)
            if not events:
                if proc.poll() is not None:
                    break
                continue
            for key, _ in events:
                try:
                    data = os.read(key.fd, 4096)
                except BlockingIOError:
                    continue
                except OSError:
                    sel.unregister(key.fd)
                    os.close(key.fd)
                    continue
                if not data:
                    sel.unregister(key.fd)
                    os.close(key.fd)
                    continue
                t = time.perf_counter()
                raw.extend(data)
                piece = strip_ansi(data.decode("utf-8", errors="replace"))
                clean += piece
                sys.stdout.write(piece)
                sys.stdout.flush()

                matches = list(READY_RE.finditer(clean))
                if matches and ready_t is None:
                    ready_t = t
                    submit_t = time.perf_counter()
                    os.write(master_fd, (PROMPT + "\n").encode("utf-8"))
                    submitted = True
                    gen_deadline = submit_t + args.generation_timeout
                    continue

                if submitted:
                    post_submit_text += piece

                if submitted and first_visible_t is None:
                    tail_start = post_submit_text.rfind(PROMPT)
                    if tail_start >= 0:
                        tail = post_submit_text[tail_start + len(PROMPT):]
                        if not prompt_echo_seen:
                            if "\n" not in tail:
                                continue
                            prompt_echo_seen = True
                            tail = tail.split("\n", 1)[1]
                    else:
                        tail = post_submit_text
                    for marker in READY_RE.finditer(tail):
                        tail = tail.replace(marker.group(0), "")
                    tail = tail.replace("> ", "").strip()
                    if visible_tokens(tail) > 0:
                        first_visible_t = t

                if submitted and first_visible_t is not None:
                    matches = list(READY_RE.finditer(clean))
                    if len(matches) >= 2:
                        second_ready_t = t
                        os.write(master_fd, b"/exit\n")
                        raise StopIteration
        raise RuntimeError("process exited before second ready marker")
    except StopIteration:
        pass
    finally:
        try:
            os.write(master_fd, b"/exit\n")
        except OSError:
            pass
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=20)

    args.raw_output.write_bytes(raw)
    clean_text = strip_ansi(raw.decode("utf-8", errors="replace"))
    stderr_match = STDERR_LOG_RE.search(clean_text)
    record = {
        "ok": ready_t is not None and submit_t is not None and first_visible_t is not None,
        "ser": args.ser,
        "command": cmd,
        "prompt": PROMPT,
        "seed": args.seed,
        "predict_tokens": args.predict_tokens,
        "time_to_type_s": (ready_t - start) if ready_t else None,
        "interactive_ttft_s": (first_visible_t - submit_t) if first_visible_t and submit_t else None,
        "first_visible_s": (first_visible_t - start) if first_visible_t else None,
        "generation_elapsed_s": (second_ready_t - submit_t) if second_ready_t and submit_t else None,
        "stderr_log": stderr_match.group(1) if stderr_match else None,
        "returncode": proc.returncode,
        "raw_output": str(args.raw_output),
    }
    add_stderr_metrics(record)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("record=" + str(args.output), flush=True)
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
