#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[3]
MATRIX_RUNNER = REPO / ".Agent/plans/5090-theoretical-token-rate/run_5090_matrix.py"
ORACLE_PROFILE = REPO / ".Agent/plans/5090-theoretical-token-rate/oracle-n84.route.csv"


def load_matrix_runner():
    spec = importlib.util.spec_from_file_location("run_5090_matrix", MATRIX_RUNNER)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load runner: {MATRIX_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_case(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        run_dir=args.run_dir,
        label=args.label,
        profile=str(args.profile),
        vram_cache_mib=args.vram_cache_mib,
        vram_upgate_pct=args.vram_upgate_pct,
        cache_policy="lfu_lru",
        cache_profile_after=12624,
        profile_reserve_pct=10,
        prompt_profile=None,
        ram_tier_mib=0,
        ram_tier_skip=0,
        ram_tier_profile=None,
        ram_tier_no_pin=False,
        ram_tier_pin_mib=None,
        backend="iouring",
        depth=16,
        slots=16,
        io_bytes=2097152,
        io_refill_batch=1,
        prefetch_down=False,
        prefetch_down_depth=8,
        stage_pinned=False,
        up_gate_stage_split=False,
        no_down_parallel_stage=False,
        iouring_sort_offset=args.iouring_sort_offset,
        iouring_sqpoll=False,
        gpu_handoff=True,
        prompt_dynamic_experts=True,
        prompt_startup_preload=True,
        prompt_ttft_markers=True,
        eager_cublas=True,
        cuda_disable_graphs=False,
        decline_debug=False,
        cache_auto_clamp=True,
        cache_safety_mib=256,
        cache_graph_reserve_mib=0,
        profile_preload_evict=False,
        route_profile_out=None,
        route_trace_out=None,
        ttft_trace_out=None,
        score_trace_debug=False,
        score_trace_out=None,
        skip_estimate=False,
        skip_estimate_keep=7,
        skip_nonresident=False,
        skip_nonresident_keep=7,
        trace_prefetch=None,
        trace_prefetch_window=96,
        trace_prefetch_max_loads=8,
        trace_prefetch_lead_events=0,
        host_prefetch=None,
        host_prefetch_lead_events=2048,
        host_prefetch_skip_events=0,
        host_prefetch_slots=64,
        host_prefetch_max_mib=512,
        planned_host_prefetch=False,
        memory_max="2G",
        memory_swap_max="0",
        ctx=args.ctx,
        ngl=79,
        batch=2048,
        predict_tokens=args.predict_tokens,
        threads=16,
        threads_batch=24,
        seed=42,
        prompt="Answer with one short sentence: why does NVMe latency matter for MoE inference?",
        ser=None,
        no_graph_reuse=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict 2GB GLM n84/n8 with a chosen expert-pack path.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--expert-pack", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=ORACLE_PROFILE)
    parser.add_argument("--predict-tokens", type=int, default=84)
    parser.add_argument("--vram-cache-mib", type=int, default=20480)
    parser.add_argument("--vram-upgate-pct", type=int, default=65)
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--iouring-sort-offset", action="store_true")
    args = parser.parse_args()
    if not args.expert_pack.is_file():
        raise SystemExit(f"missing expert pack: {args.expert_pack}")
    if not args.profile.is_file():
        raise SystemExit(f"missing profile: {args.profile}")
    return args


def main() -> int:
    args = parse_args()
    runner = load_matrix_runner()
    runner.PACK = args.expert_pack.resolve()
    record = runner.run_case(make_case(args))
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record.get("returncode") == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
