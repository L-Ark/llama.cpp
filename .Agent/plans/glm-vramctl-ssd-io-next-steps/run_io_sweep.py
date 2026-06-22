#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
MODEL = REPO / "models/GLM-5.1-UD-IQ3_XXS/GLM-5.1-UD-IQ3_XXS-00001-of-00007.gguf"
PACK = REPO / "models/GLM-5.1-UD-IQ3_XXS/glm51-iq3xxs.expert-pack"
PROFILE = REPO / "presets/moe/groundtruth/wici-glm51-interactive-n84-expert-top8.runtime.csv"
BIN = REPO / "build-cuda/bin/llama-cli"
PROMPT = "Answer with one short sentence: why does NVMe latency matter for MoE inference?"

TIMING_RE = re.compile(
    r"llama_print_timings:\s+(?P<name>load|prompt eval|eval|total) time =\s+"
    r"(?P<ms>[0-9.]+) ms(?: /(?:\s+(?P<tokens>[0-9]+) (?:tokens|runs))?"
    r".*?(?P<tps>[0-9.]+|-nan|inf) tokens per second)?"
)
PACK_RE = re.compile(
    r"\[moe_stream_batch\] expert pack: hits=(?P<hits>\d+) misses=(?P<misses>\d+) "
    r"read_failures=(?P<read_failures>\d+) direct_reads=(?P<direct_reads>\d+) "
    r"direct_fallbacks=(?P<direct_fallbacks>\d+) iouring_reads=(?P<iouring_reads>\d+) "
    r"iouring_bytes=(?P<iouring_bytes>\d+) iouring_fallbacks=(?P<iouring_fallbacks>\d+) "
    r"iouring_submit_us=(?P<iouring_submit_us>\d+) iouring_wait_us=(?P<iouring_wait_us>\d+) "
    r"iouring_h2d_enqueues=(?P<iouring_h2d_enqueues>\d+)"
)
IO_DETAIL_RE = re.compile(
    r"\[moe_stream_batch\] expert pack iouring detail: batches=(?P<batches>\d+) "
    r"submit_calls=(?P<submit_calls>\d+) wait_calls=(?P<wait_calls>\d+) "
    r"cqes=(?P<cqes>\d+) inflight_avg=(?P<inflight_avg>[0-9.]+) "
    r"inflight_max=(?P<inflight_max>\d+) "
    r"batch_hist=1:(?P<hist_1>\d+),2-4:(?P<hist_2_4>\d+),5-8:(?P<hist_5_8>\d+),"
    r"9-16:(?P<hist_9_16>\d+),17-32:(?P<hist_17_32>\d+),gt32:(?P<hist_gt32>\d+)"
)
VCACHE_RE = re.compile(
    r"\[moe_stream_batch\] VRAM cache: hits=(?P<hits>\d+) misses=(?P<misses>\d+) "
    r"preloads=(?P<preloads>\d+) pinned=(?P<pinned>\d+) hit_rate=(?P<hit_rate>[0-9.]+)%"
)
PIN_RE = re.compile(
    r"\[moe_stream_batch\] pinned staging(?P<name>[^:]*): copies=(?P<copies>\d+) waits=(?P<waits>\d+) "
    r"fallbacks=(?P<fallbacks>\d+) slots=(?P<slots>\d+) slot=(?P<slot_mib>[0-9.]+) MiB"
    r"(?: slot_wait=(?P<slot_wait_ms>[0-9.]+) ms host_stage=(?P<host_stage_ms>[0-9.]+) ms "
    r"enqueue=(?P<enqueue_ms>[0-9.]+) ms h2d=(?P<h2d_ms>[0-9.]+) ms h2d_timed=(?P<h2d_timed>\d+))?"
)
PIN_IO_RE = re.compile(
    r"\[moe_stream_batch\] pinned staging(?P<name>[^:]*) iouring: batches=(?P<batches>\d+) "
    r"jobs=(?P<jobs>\d+) submit_calls=(?P<submit_calls>\d+) wait_calls=(?P<wait_calls>\d+) "
    r"cqes=(?P<cqes>\d+) inflight_avg=(?P<inflight_avg>[0-9.]+) inflight_max=(?P<inflight_max>\d+) "
    r"batch_hist=1:(?P<hist_1>\d+),2-4:(?P<hist_2_4>\d+),5-8:(?P<hist_5_8>\d+),"
    r"9-16:(?P<hist_9_16>\d+),17-32:(?P<hist_17_32>\d+),gt32:(?P<hist_gt32>\d+)"
)


def parse_stderr(stderr: str) -> dict[str, object]:
    record: dict[str, object] = {}
    for match in TIMING_RE.finditer(stderr):
        key = match.group("name").replace(" ", "_")
        record[f"{key}_ms"] = float(match.group("ms"))
        if match.group("tokens"):
            record[f"{key}_tokens"] = int(match.group("tokens"))
        tps = match.group("tps")
        if tps and tps not in {"-nan", "inf"}:
            record[f"{key}_tokens_per_s"] = float(tps)

    pack_matches = list(PACK_RE.finditer(stderr))
    if pack_matches:
        for key, value in pack_matches[-1].groupdict().items():
            record[f"expert_pack_{key}"] = int(value)

    detail_matches = list(IO_DETAIL_RE.finditer(stderr))
    if detail_matches:
        for key, value in detail_matches[-1].groupdict().items():
            record[f"iouring_{key}"] = float(value) if key == "inflight_avg" else int(value)

    cache_matches = list(VCACHE_RE.finditer(stderr))
    if cache_matches:
        cache = cache_matches[-1]
        record["vram_cache_hits"] = int(cache.group("hits"))
        record["vram_cache_misses"] = int(cache.group("misses"))
        record["vram_cache_hit_rate_pct"] = float(cache.group("hit_rate"))

    pinned = {}
    for match in PIN_RE.finditer(stderr):
        name = match.group("name").strip() or "main"
        pinned[name] = {
            key: (float(value) if key.endswith("_ms") or key == "slot_mib" else int(value))
            for key, value in match.groupdict().items()
            if key != "name" and value is not None
        }
    if pinned:
        record["pinned_staging"] = pinned

    pinned_io = {}
    for match in PIN_IO_RE.finditer(stderr):
        name = match.group("name").strip() or "main"
        pinned_io[name] = {
            key: (float(value) if key == "inflight_avg" else int(value))
            for key, value in match.groupdict().items()
            if key != "name"
        }
    if pinned_io:
        record["pinned_staging_iouring"] = pinned_io

    return record


def run_case(args: argparse.Namespace, backend: str, slots: int, depth: int, label: str) -> dict[str, object]:
    run_dir = args.run_dir / label
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"
    status_path = run_dir / "status.txt"
    command_path = run_dir / "command.txt"

    env = [
        "env",
        "CUDA_VISIBLE_DEVICES=0",
        "MLA=3",
        "CPU_MOE=1",
        "IGNORE_EOS=1",
        "GGML_CUDA_NO_PINNED=1",
        "GGML_CUDA_FORCE_MMQ=1",
        "GGML_CUDA_FORCE_CUBLAS=0",
        "GGML_MOE_STREAM=1",
        "GGML_MOE_STREAM_BATCH_ONLY=1",
        "GGML_MOE_STREAM_DEFER=1",
        "GGML_MOE_STREAM_CPU_OPS=1",
        "GGML_MOE_PARALLEL_EXPERTS=1",
        "GGML_MOE_STREAM_FUSED_UP_GATE=1",
        "GGML_MOE_VRAM_CACHE_SPLIT=1",
        "GGML_MOE_STREAM_ONE_CACHE_MIB=0",
        "GGML_MOE_VRAM_CACHE_POLICY=lfu_lru",
        f"GGML_MOE_STAGE_PINNED_SLOTS={slots}",
        "GGML_MOE_PREFETCH_DOWN=0",
        "GGML_MOE_PREFETCH_DOWN_DEPTH=8",
        f"GGML_MOE_EXPERT_PACK={PACK}",
        "GGML_MOE_RAM_TIER_MIB=0",
        "GGML_MOE_RAM_TIER_SKIP=0",
        "GGML_MOE_VRAM_CACHE_MIB=12288",
        "GGML_MOE_VRAM_CACHE_UPGATE_PCT=60",
        f"GGML_MOE_VRAM_PROFILE={PROFILE}",
        "GGML_MOE_VRAM_PROFILE_PROTECT=1",
        "GGML_MOE_VRAM_PROFILE_RESERVE_PCT=10",
        f"GGML_MOE_IO_BACKEND={backend}",
        f"GGML_MOE_IO_DEPTH={depth}",
        "GGML_MOE_IO_BYTES=2097152",
        "GGML_MOE_STAGE_PINNED=0",
        "GGML_MOE_STREAM_UP_GATE_PARALLEL=1",
        "GGML_MOE_STREAM_UP_GATE_PARALLEL_STAGE=1",
        "GGML_MOE_STREAM_UP_GATE_SPLIT_STAGE=1",
        "GGML_MOE_DOWN_PARALLEL_STAGE=1",
    ]
    cmd = [
        *env,
        str(BIN),
        "-m", str(MODEL),
        "-c", "2048",
        "-ngl", "79",
        "-fa", "on",
        "-mla", "3",
        "-cmoe",
        "--defer-experts",
        "--no-warmup",
        "-b", "2048",
        "-n", str(args.predict_tokens),
        "--temp", "0",
        "--top-p", "1.0",
        "--min-p", "0.0",
        "--repeat-last-n", "512",
        "--repeat-penalty", "1.20",
        "--presence-penalty", "0.1",
        "--frequency-penalty", "0.2",
        "-t", "16",
        "-tb", "24",
        "--seed", "42",
        "-p", PROMPT,
    ]
    shell = (
        f"cd {shlex.quote(str(REPO))} && "
        f"exec {shlex.join(cmd)} >{shlex.quote(str(stdout_path))} 2>{shlex.quote(str(stderr_path))}"
    )
    systemd_cmd = [
        "systemd-run", "--user", "--wait", "--collect", "--quiet",
        "-p", f"WorkingDirectory={REPO}",
        "-p", "MemoryMax=2G",
        "-p", "MemorySwapMax=0",
        "bash", "-lc", shell,
    ]
    command_path.write_text(shlex.join(systemd_cmd) + "\n", encoding="utf-8")

    start = time.perf_counter()
    proc = subprocess.run(systemd_cmd, cwd=REPO, text=True, capture_output=True)
    wall_s = time.perf_counter() - start
    status_path.write_text(str(proc.returncode) + "\n", encoding="utf-8")
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""

    record = {
        "label": label,
        "backend": backend,
        "slots": slots,
        "depth": depth,
        "predict_tokens": args.predict_tokens,
        "returncode": proc.returncode,
        "wall_s": wall_s,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "status": str(status_path),
        "command": str(command_path),
        "stdout_prefix": stdout[:200],
        "systemd_stdout": proc.stdout,
        "systemd_stderr": proc.stderr,
    }
    record.update(parse_stderr(stderr))
    (run_dir / "summary.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=["direct", "iouring"], required=True)
    parser.add_argument("--slots", type=int, required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--predict-tokens", type=int, default=84)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    record = run_case(args, args.backend, args.slots, args.depth, args.label)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["returncode"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
