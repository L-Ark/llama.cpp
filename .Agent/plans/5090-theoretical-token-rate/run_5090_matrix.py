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
BIN = REPO / "build-cuda/bin/llama-cli"
TOP8_PROFILE = REPO / "presets/moe/groundtruth/wici-glm51-interactive-n84-expert-top8.runtime.csv"
ROUTE_PROFILE = REPO / "presets/moe/groundtruth/wici-glm51-interactive-n84.route.csv"
PROMPT = "Answer with one short sentence: why does NVMe latency matter for MoE inference?"

TIMING_RE = re.compile(
    r"llama_print_timings:\s+(?P<name>load|prompt eval|eval|total) time =\s+"
    r"(?P<ms>[0-9.]+) ms(?: /(?:\s+(?P<tokens>[0-9]+) (?:tokens|runs))?"
    r".*?(?P<tps>[0-9.]+|-nan|inf) tokens per second)?"
)
PACK_RE = re.compile(
    r"\[moe_stream_batch\] expert pack: hits=(?P<hits>\d+) misses=(?P<misses>\d+) "
    r"read_failures=(?P<read_failures>\d+) direct_reads=(?P<direct_reads>\d+) "
    r"direct_fallbacks=(?P<direct_fallbacks>\d+)"
    r"(?: iouring_reads=(?P<iouring_reads>\d+) iouring_bytes=(?P<iouring_bytes>\d+) "
    r"iouring_fallbacks=(?P<iouring_fallbacks>\d+) iouring_submit_us=(?P<iouring_submit_us>\d+) "
    r"iouring_wait_us=(?P<iouring_wait_us>\d+) iouring_h2d_enqueues=(?P<iouring_h2d_enqueues>\d+))?"
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
RAM_TIER_RE = re.compile(
    r"\[moe_stream_batch\] RAM tier: hits=(?P<hits>\d+) total=(?P<total>\d+) "
    r"hit_rate=(?P<hit_rate>[0-9.]+)% resident=(?P<resident_mib>[0-9.]+) MiB"
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
HOST_PREFETCH_RE = re.compile(
    r"\[moe_stream_batch\] host prefetch: calls=(?P<calls>\d+) matched=(?P<matched>\d+) "
    r"resync=(?P<resync>\d+) submitted=(?P<submitted>\d+) hits=(?P<hits>\d+) "
    r"misses=(?P<misses>\d+) evicted=(?P<evicted>\d+) read_failures=(?P<read_failures>\d+) "
    r"alloc_failures=(?P<alloc_failures>\d+) duplicate_skips=(?P<duplicate_skips>\d+) "
    r"(?:(?:profile_pinned_skips=(?P<profile_pinned_skips>\d+) ))?"
    r"reserved_skips=(?P<reserved_skips>\d+) no_slot=(?P<no_slot>\d+) "
    r"planned_enqueued=(?P<planned_enqueued>\d+) planned_dequeued=(?P<planned_dequeued>\d+) "
    r"planned_duplicate_skips=(?P<planned_duplicate_skips>\d+) scan_passes=(?P<scan_passes>\d+) "
    r"cursor=(?P<cursor>\d+) produce_cursor=(?P<produce_cursor>\d+)/(?P<trace_size>\d+) "
    r"skip=(?P<skip>\d+) used=(?P<used_mib>[0-9.]+) MiB slots=(?P<slots>\d+)"
)
TRACE_PREFETCH_RE = re.compile(
    r"\[moe_stream_batch\] trace prefetch: calls=(?P<calls>\d+) matched=(?P<matched>\d+) "
    r"resync=(?P<resync>\d+) loads=(?P<loads>\d+) cached=(?P<cached>\d+) "
    r"missing_tensor=(?P<missing_tensor>\d+) cache_unavailable=(?P<cache_unavailable>\d+) "
    r"cursor=(?P<cursor>\d+) prefetch_cursor=(?P<prefetch_cursor>\d+)/(?P<trace_size>\d+)"
)


def profile_path(name: str) -> Path:
    if name == "top8":
        return TOP8_PROFILE
    if name == "route":
        return ROUTE_PROFILE
    path = Path(name)
    return path if path.is_absolute() else REPO / path


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
        for key, value in pack_matches[-1].groupdict(default="0").items():
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
        record["vram_cache_preloads"] = int(cache.group("preloads"))
        record["vram_cache_pinned"] = int(cache.group("pinned"))
        record["vram_cache_hit_rate_pct"] = float(cache.group("hit_rate"))

    ram_matches = list(RAM_TIER_RE.finditer(stderr))
    if ram_matches:
        ram = ram_matches[-1]
        record["ram_tier_hits"] = int(ram.group("hits"))
        record["ram_tier_total"] = int(ram.group("total"))
        record["ram_tier_hit_rate_pct"] = float(ram.group("hit_rate"))
        record["ram_tier_resident_mib"] = float(ram.group("resident_mib"))

    pinned = {}
    for match in PIN_RE.finditer(stderr):
        name = match.group("name").strip() or "main"
        pinned[name] = {
            key: (float(value) if key.endswith("_ms") or key == "slot_mib" else int(value))
            for key, value in match.groupdict().items()
            if key != "name" and value is not None
        }
    for match in PIN_IO_RE.finditer(stderr):
        name = match.group("name").strip() or "main"
        pinned.setdefault(name, {})
        pinned[name]["iouring"] = {
            key: (float(value) if key == "inflight_avg" else int(value))
            for key, value in match.groupdict().items()
            if key != "name" and value is not None
        }
    if pinned:
        record["pinned_staging"] = pinned

    host_matches = list(HOST_PREFETCH_RE.finditer(stderr))
    if host_matches:
        host = {}
        for key, value in host_matches[-1].groupdict().items():
            host[key] = float(value) if key == "used_mib" else int(value)
        record["host_prefetch_stats"] = host
    trace_matches = list(TRACE_PREFETCH_RE.finditer(stderr))
    if trace_matches:
        record["trace_prefetch_stats"] = {
            key: int(value)
            for key, value in trace_matches[-1].groupdict().items()
        }
    return record


def env_list(args: argparse.Namespace, run_dir: Path) -> list[str]:
    profile = profile_path(args.profile)
    env = [
        "env",
        "CUDA_VISIBLE_DEVICES=0",
        f"LLAMA_CACHE={run_dir / 'llama-cache'}",
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
        f"GGML_MOE_VRAM_CACHE_POLICY={args.cache_policy}",
        f"GGML_MOE_VRAM_CACHE_PROFILE_AFTER={args.cache_profile_after}",
        f"GGML_MOE_STAGE_PINNED_SLOTS={args.slots}",
        f"GGML_MOE_PREFETCH_DOWN={1 if args.prefetch_down else 0}",
        f"GGML_MOE_PREFETCH_DOWN_DEPTH={args.prefetch_down_depth}",
        f"GGML_MOE_EXPERT_PACK={PACK}",
        f"GGML_MOE_RAM_TIER_MIB={args.ram_tier_mib}",
        f"GGML_MOE_RAM_TIER_SKIP={args.ram_tier_skip}",
        f"GGML_MOE_VRAM_CACHE_MIB={args.vram_cache_mib}",
        f"GGML_MOE_VRAM_CACHE_UPGATE_PCT={args.vram_upgate_pct}",
        f"GGML_MOE_VRAM_PROFILE={profile}",
        "GGML_MOE_VRAM_PROFILE_PROTECT=1",
        f"GGML_MOE_VRAM_PROFILE_RESERVE_PCT={args.profile_reserve_pct}",
        f"GGML_MOE_IO_BACKEND={args.backend}",
        f"GGML_MOE_IO_DEPTH={args.depth}",
        f"GGML_MOE_IO_BYTES={args.io_bytes}",
        f"GGML_MOE_IO_REFILL_BATCH={args.io_refill_batch}",
        f"GGML_MOE_STAGE_PINNED={1 if args.stage_pinned else 0}",
        "GGML_MOE_STREAM_UP_GATE_PARALLEL=1",
        "GGML_MOE_STREAM_UP_GATE_PARALLEL_STAGE=1",
    ]
    if not args.no_down_parallel_stage:
        env.append("GGML_MOE_DOWN_PARALLEL_STAGE=1")
    if args.up_gate_stage_split:
        env.append("GGML_MOE_STREAM_UP_GATE_STAGE_SPLIT=1")
        env.append("GGML_MOE_STREAM_UP_GATE_SPLIT_STAGE=1")
    if args.iouring_sort_offset:
        env.append("GGML_MOE_IO_SORT_OFFSET=1")
    if args.iouring_sqpoll:
        env.append("GGML_MOE_IO_SQPOLL=1")
    if args.ram_tier_profile:
        env.append(f"GGML_MOE_RAM_TIER_PROFILE={profile_path(args.ram_tier_profile)}")
    if args.prompt_profile:
        env.append(f"GGML_MOE_VRAM_PROFILE_PROMPT={profile_path(args.prompt_profile)}")
    if args.ram_tier_no_pin:
        env.append("GGML_MOE_RAM_TIER_PIN=0")
    if args.ram_tier_pin_mib is not None:
        env.append(f"GGML_MOE_RAM_TIER_PIN_MIB={args.ram_tier_pin_mib}")
    if args.gpu_handoff:
        env.append("GGML_MOE_GPU_HANDOFF=1")
    if args.prompt_dynamic_experts:
        env.append("GGML_MOE_PROMPT_DYNAMIC_EXPERTS=1")
    if args.prompt_startup_preload:
        env.append("LLAMA_PROMPT_STARTUP_PROFILE_PRELOAD=1")
    if args.prompt_ttft_markers:
        env.append("LLAMA_PROMPT_TTFT_MARKERS=1")
    if args.eager_cublas:
        env.append("GGML_CUDA_EAGER_CUBLAS=1")
    if args.cuda_disable_graphs:
        env.append("GGML_CUDA_DISABLE_GRAPHS=1")
    if args.decline_debug:
        env.append("GGML_MOE_STREAM_DECLINE_DEBUG=1")
    if args.cache_auto_clamp:
        env.append("GGML_MOE_VRAM_CACHE_AUTO_CLAMP=1")
        env.append(f"GGML_MOE_VRAM_CACHE_SAFETY_MIB={args.cache_safety_mib}")
        if args.cache_graph_reserve_mib > 0:
            env.append(f"GGML_MOE_VRAM_CACHE_GRAPH_RESERVE_MIB={args.cache_graph_reserve_mib}")
    if args.profile_preload_evict:
        env.append("GGML_MOE_VRAM_PROFILE_PRELOAD_EVICT=1")
    if args.route_profile_out:
        env.append("GGML_MOE_BATCH_PROFILE=1")
        env.append(f"GGML_MOE_BATCH_PROFILE_OUT={args.route_profile_out}")
    if args.route_trace_out:
        env.append(f"GGML_MOE_ROUTE_TRACE_OUT={args.route_trace_out}")
    if args.ttft_trace_out:
        env.append(f"GGML_MOE_TTFT_TRACE_OUT={args.ttft_trace_out}")
    if args.score_trace_debug:
        env.append("GGML_MOE_SCORE_TRACE_DEBUG=1")
    if args.score_trace_out:
        env.append(f"GGML_MOE_SCORE_TRACE_OUT={args.score_trace_out}")
    if args.skip_estimate:
        env.append("GGML_MOE_SKIP_ESTIMATE=1")
        env.append(f"GGML_MOE_SKIP_ESTIMATE_KEEP={args.skip_estimate_keep}")
    if args.skip_nonresident:
        env.append("GGML_MOE_SKIP_NONRESIDENT=1")
        env.append(f"GGML_MOE_SKIP_NONRESIDENT_KEEP={args.skip_nonresident_keep}")
    if args.trace_prefetch:
        env.append(f"GGML_MOE_TRACE_PREFETCH={profile_path(args.trace_prefetch)}")
        env.append(f"GGML_MOE_TRACE_PREFETCH_WINDOW={args.trace_prefetch_window}")
        env.append(f"GGML_MOE_TRACE_PREFETCH_MAX_LOADS={args.trace_prefetch_max_loads}")
        env.append(f"GGML_MOE_TRACE_PREFETCH_LEAD_EVENTS={args.trace_prefetch_lead_events}")
    if args.host_prefetch:
        env.append(f"GGML_MOE_HOST_PREFETCH={profile_path(args.host_prefetch)}")
        env.append(f"GGML_MOE_HOST_PREFETCH_LEAD_EVENTS={args.host_prefetch_lead_events}")
        env.append(f"GGML_MOE_HOST_PREFETCH_SKIP_EVENTS={args.host_prefetch_skip_events}")
        env.append(f"GGML_MOE_HOST_PREFETCH_SLOTS={args.host_prefetch_slots}")
        env.append(f"GGML_MOE_HOST_PREFETCH_MAX_MIB={args.host_prefetch_max_mib}")
    if args.planned_host_prefetch:
        env.append("GGML_MOE_PLANNED_HOST_PREFETCH=1")
        env.append(f"GGML_MOE_HOST_PREFETCH_SLOTS={args.host_prefetch_slots}")
        env.append(f"GGML_MOE_HOST_PREFETCH_MAX_MIB={args.host_prefetch_max_mib}")
    return env


def run_case(args: argparse.Namespace) -> dict[str, object]:
    run_dir = (args.run_dir / args.label).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"
    status_path = run_dir / "status.txt"
    command_path = run_dir / "command.txt"
    env_path = run_dir / "env.json"

    env = env_list(args, run_dir)
    cmd = [
        *env,
        str(BIN),
        "-m", str(MODEL),
        "-c", str(args.ctx),
        "-ngl", str(args.ngl),
        "-fa", "on",
        "-mla", "3",
        "-cmoe",
        "--defer-experts",
        "--no-warmup",
        "-b", str(args.batch),
        "-n", str(args.predict_tokens),
        "--temp", "0",
        "--top-p", "1.0",
        "--min-p", "0.0",
        "--repeat-last-n", "512",
        "--repeat-penalty", "1.20",
        "--presence-penalty", "0.1",
        "--frequency-penalty", "0.2",
        "-t", str(args.threads),
        "-tb", str(args.threads_batch),
        "--seed", str(args.seed),
        "-p", args.prompt,
    ]
    if args.ser:
        cmd.extend(["--smart-expert-reduction", args.ser])
    if args.no_graph_reuse:
        cmd.append("--no-graph-reuse")
    shell = (
        f"cd {shlex.quote(str(REPO))} && exec {shlex.join(cmd)} "
        f">{shlex.quote(str(stdout_path))} 2>{shlex.quote(str(stderr_path))}"
    )
    systemd_cmd = [
        "systemd-run", "--user", "--wait", "--collect", "--quiet",
        "-p", f"WorkingDirectory={REPO}",
        "-p", f"MemoryMax={args.memory_max}",
        "-p", f"MemorySwapMax={args.memory_swap_max}",
        "bash", "-lc", shell,
    ]
    command_path.write_text(shlex.join(systemd_cmd) + "\n", encoding="utf-8")
    env_pairs = {}
    for item in env[1:]:
        if "=" in item:
            key, value = item.split("=", 1)
            env_pairs[key] = value
    env_path.write_text(json.dumps(env_pairs, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    start = time.perf_counter()
    proc = subprocess.run(systemd_cmd, cwd=REPO, text=True, capture_output=True)
    wall_s = time.perf_counter() - start
    status_path.write_text(str(proc.returncode) + "\n", encoding="utf-8")
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""

    record: dict[str, object] = {
        "label": args.label,
        "returncode": proc.returncode,
        "wall_s": wall_s,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "status": str(status_path),
        "command": str(command_path),
        "env": str(env_path),
        "stdout_prefix": stdout[:240],
        "systemd_stdout": proc.stdout,
        "systemd_stderr": proc.stderr,
        "strict_2gb": args.memory_max == "2G" and args.memory_swap_max == "0" and args.ram_tier_mib == 0,
        "profile": args.profile,
        "vram_cache_mib": args.vram_cache_mib,
        "ram_tier_mib": args.ram_tier_mib,
        "ram_tier_profile": args.ram_tier_profile,
        "ram_tier_no_pin": args.ram_tier_no_pin,
        "ram_tier_pin_mib": args.ram_tier_pin_mib,
        "backend": args.backend,
        "io_refill_batch": args.io_refill_batch,
        "stage_pinned": args.stage_pinned,
        "gpu_handoff": args.gpu_handoff,
        "prompt_dynamic_experts": args.prompt_dynamic_experts,
        "prompt_startup_preload": args.prompt_startup_preload,
        "predict_tokens": args.predict_tokens,
        "trace_prefetch": args.trace_prefetch,
        "trace_prefetch_window": args.trace_prefetch_window,
        "trace_prefetch_max_loads": args.trace_prefetch_max_loads,
        "trace_prefetch_lead_events": args.trace_prefetch_lead_events,
        "host_prefetch": args.host_prefetch,
        "host_prefetch_lead_events": args.host_prefetch_lead_events,
        "host_prefetch_skip_events": args.host_prefetch_skip_events,
        "host_prefetch_slots": args.host_prefetch_slots,
        "host_prefetch_max_mib": args.host_prefetch_max_mib,
        "planned_host_prefetch": args.planned_host_prefetch,
        "profile_preload_evict": args.profile_preload_evict,
        "iouring_sort_offset": args.iouring_sort_offset,
        "iouring_sqpoll": args.iouring_sqpoll,
        "no_graph_reuse": args.no_graph_reuse,
        "cache_graph_reserve_mib": args.cache_graph_reserve_mib,
    }
    record.update(parse_stderr(stderr))
    (run_dir / "summary.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--profile", default="top8")
    parser.add_argument("--vram-cache-mib", type=int, default=12288)
    parser.add_argument("--vram-upgate-pct", type=int, default=60)
    parser.add_argument("--cache-policy", choices=["lru", "lfu_lru", "profile_lfu_lru", "hybrid_profile_lfu_lru"], default="lfu_lru")
    parser.add_argument("--cache-profile-after", type=int, default=12624)
    parser.add_argument("--profile-reserve-pct", type=int, default=10)
    parser.add_argument("--prompt-profile")
    parser.add_argument("--ram-tier-mib", type=int, default=0)
    parser.add_argument("--ram-tier-skip", type=int, default=0)
    parser.add_argument("--ram-tier-profile")
    parser.add_argument("--ram-tier-no-pin", action="store_true")
    parser.add_argument("--ram-tier-pin-mib", type=int)
    parser.add_argument("--backend", choices=["direct", "iouring"], default="iouring")
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--slots", type=int, default=16)
    parser.add_argument("--io-bytes", type=int, default=2097152)
    parser.add_argument("--io-refill-batch", type=int, default=1)
    parser.add_argument("--prefetch-down", action="store_true")
    parser.add_argument("--prefetch-down-depth", type=int, default=8)
    parser.add_argument("--stage-pinned", action="store_true")
    parser.add_argument("--up-gate-stage-split", action="store_true")
    parser.add_argument("--no-down-parallel-stage", action="store_true")
    parser.add_argument("--iouring-sort-offset", action="store_true")
    parser.add_argument("--iouring-sqpoll", action="store_true")
    parser.add_argument("--gpu-handoff", action="store_true")
    parser.add_argument("--prompt-dynamic-experts", action="store_true")
    parser.add_argument("--prompt-startup-preload", action="store_true")
    parser.add_argument("--prompt-ttft-markers", action="store_true")
    parser.add_argument("--eager-cublas", action="store_true")
    parser.add_argument("--cuda-disable-graphs", action="store_true")
    parser.add_argument("--decline-debug", action="store_true")
    parser.add_argument("--cache-auto-clamp", action="store_true")
    parser.add_argument("--cache-safety-mib", type=int, default=256)
    parser.add_argument("--cache-graph-reserve-mib", type=int, default=0)
    parser.add_argument("--profile-preload-evict", action="store_true")
    parser.add_argument("--route-profile-out")
    parser.add_argument("--route-trace-out")
    parser.add_argument("--ttft-trace-out")
    parser.add_argument("--score-trace-debug", action="store_true")
    parser.add_argument("--score-trace-out")
    parser.add_argument("--skip-estimate", action="store_true")
    parser.add_argument("--skip-estimate-keep", type=int, default=7)
    parser.add_argument("--skip-nonresident", action="store_true")
    parser.add_argument("--skip-nonresident-keep", type=int, default=7)
    parser.add_argument("--trace-prefetch")
    parser.add_argument("--trace-prefetch-window", type=int, default=96)
    parser.add_argument("--trace-prefetch-max-loads", type=int, default=8)
    parser.add_argument("--trace-prefetch-lead-events", type=int, default=0)
    parser.add_argument("--host-prefetch")
    parser.add_argument("--host-prefetch-lead-events", type=int, default=2048)
    parser.add_argument("--host-prefetch-skip-events", type=int, default=0)
    parser.add_argument("--host-prefetch-slots", type=int, default=64)
    parser.add_argument("--host-prefetch-max-mib", type=int, default=512)
    parser.add_argument("--planned-host-prefetch", action="store_true")
    parser.add_argument("--memory-max", default="2G")
    parser.add_argument("--memory-swap-max", default="0")
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--ngl", type=int, default=79)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--predict-tokens", type=int, default=84)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--threads-batch", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--ser")
    parser.add_argument("--no-graph-reuse", action="store_true")
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    record = run_case(args)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["returncode"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
