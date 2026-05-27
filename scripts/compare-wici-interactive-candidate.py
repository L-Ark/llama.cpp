#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_first_jsonl(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    raise SystemExit(f"empty jsonl: {path}")


def pct_improvement(baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None or baseline <= 0:
        return None
    return 100.0 * (baseline - candidate) / baseline


def pct_change(baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None or baseline <= 0:
        return None
    return 100.0 * (candidate - baseline) / baseline


def as_float(record: dict[str, Any], key: str) -> float | None:
    value = record.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def command_text(record: dict[str, Any]) -> str:
    command = record.get("command")
    if isinstance(command, list):
        return " ".join(str(item) for item in command)
    return str(command or "")


def summarize_ttft_trace(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    rows: list[dict[str, str]]
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}

    start = next((i for i, row in enumerate(rows) if row.get("op") == "mark_submit"), 0)
    end = next((i for i, row in enumerate(rows[start + 1 :], start + 1) if row.get("op") == "mark_first_token"), len(rows))
    window = rows[start:end]
    by_op: dict[str, float] = {}
    for row in window:
        op = row.get("op") or ""
        if not op or op.startswith("mark_"):
            continue
        by_op[op] = by_op.get(op, 0.0) + float(row.get("copy_ms") or 0.0)

    return {
        "prompt_compute_ms": by_op.get("cpu_up_compute", 0.0)
        + by_op.get("cpu_up_dyn_compute", 0.0)
        + by_op.get("cpu_up_dynamic_compute", 0.0)
        + by_op.get("cpu_up_many_compute", 0.0)
        + by_op.get("cpu_up_hybrid_compute", 0.0)
        + by_op.get("cpu_down_compute", 0.0)
        + by_op.get("cpu_down_dyn_compute", 0.0)
        + by_op.get("cpu_down_dynamic_comput", 0.0)
        + by_op.get("cpu_down_dynamic_compute", 0.0)
        + by_op.get("cpu_down_many_compute", 0.0)
        + by_op.get("cpu_down_hybrid_compute", 0.0),
        "prompt_total_ms": by_op.get("cpu_prompt_upgate", 0.0) + by_op.get("cpu_prompt_down", 0.0),
        "major_faults": by_op.get("cpu_up_majflt", 0.0) + by_op.get("cpu_down_majflt", 0.0),
        "preload_ms": by_op.get("preload_load", 0.0) + by_op.get("runtime_load", 0.0),
    }


def mechanism_id(record: dict[str, Any]) -> str:
    text = command_text(record)
    if "LLAMA_CHAT_TTFT_EXPERT_PACK=" in text:
        return "ttft_mmap_pack"
    if "LLAMA_CHAT_PREFIX_PREFILL=1" in text:
        return "chat_prefix_prefill"
    if "GGML_MOE_PROMPT_UP_GATE_HYBRID=1" in text or "GGML_MOE_PROMPT_DOWN_HYBRID=1" in text:
        return "cpu_prompt_hybrid_many"
    if "GGML_MOE_PROMPT_UP_GATE_MANY=1" in text or "GGML_MOE_PROMPT_DOWN_MANY=1" in text:
        return "cpu_prompt_many"
    if "GGML_MOE_PREFETCH_DOWN_FROM_UPGATE=1" in text:
        return "same_layer_down_prefetch"
    if "GGML_MOE_STREAM_PROMPT_UP_GATE=unsafe-q8-1" in text:
        return "unsafe_prompt_cuda_upgate"
    if "GGML_MOE_STREAM_PROMPT_UP_GATE=exact-q8-k" in text:
        return "exact_prompt_cuda_upgate"
    if "GGML_MOE_STREAM_PROMPT_UP_GATE=1" in text:
        return "decode_cuda_upgate_control"
    return "unknown"


def next_after_rejection(
    mechanism: str,
    baseline_trace: dict[str, Any],
    candidate_trace: dict[str, Any],
    candidate: dict[str, Any],
) -> str:
    prompt_compute = float(baseline_trace.get("prompt_compute_ms") or 0.0)
    candidate_compute = float(candidate_trace.get("prompt_compute_ms") or 0.0)
    compute_delta = pct_improvement(prompt_compute, candidate_compute) or 0.0
    failed_text = str(candidate.get("error") or "").lower()
    cache_failed = bool(candidate.get("moe_vram_cache_failed"))

    if mechanism == "ttft_mmap_pack" and compute_delta < 5.0:
        return "speculative_expert_prefetch_accuracy_probe"
    if mechanism == "same_layer_down_prefetch":
        return "exact_chunked_prompt_upgate_kernel"
    if mechanism == "unsafe_prompt_cuda_upgate" and (cache_failed or "timeout" in failed_text):
        return "exact_prompt_cuda_upgate_with_headroom_model"
    if mechanism == "exact_prompt_cuda_upgate" and candidate.get("n36_canary_ok") is False:
        return "reject_exact_prompt_cuda_semantics_and_optimize_cpu_prompt_moe"
    if mechanism == "exact_prompt_cuda_upgate":
        return "reject_exact_prompt_cuda_latency_and_optimize_cpu_prompt_moe"
    if mechanism == "chat_prefix_prefill" and candidate.get("n36_canary_ok") is False:
        return "reject_split_prefill_until_exact_logits_match"
    if mechanism == "cpu_prompt_hybrid_many":
        return "reject_hybrid_threading_and_prioritize_row_batched_prompt_moe"
    if mechanism == "cpu_prompt_many":
        return "keep_as_guarded_small_win_not_primary_breakthrough"
    if prompt_compute > 0:
        return "exact_row_batched_prompt_moe"
    return "expand_critical_path_trace"


def compare(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    baseline_trace: dict[str, Any] | None = None,
    candidate_trace: dict[str, Any] | None = None,
    min_ttft_improvement_pct: float = 15.0,
    max_eval_tps_regression_pct: float = 5.0,
) -> dict[str, Any]:
    baseline_trace = baseline_trace or {}
    candidate_trace = candidate_trace or {}
    mechanism = mechanism_id(candidate)

    ttft_gain = pct_improvement(as_float(baseline, "interactive_ttft_s"), as_float(candidate, "interactive_ttft_s"))
    time_to_type_gain = pct_improvement(as_float(baseline, "time_to_type_s"), as_float(candidate, "time_to_type_s"))
    eval_tps_change = pct_change(as_float(baseline, "eval_tokens_per_s"), as_float(candidate, "eval_tokens_per_s"))
    prompt_compute_gain = pct_improvement(
        float(baseline_trace.get("prompt_compute_ms") or 0.0),
        float(candidate_trace.get("prompt_compute_ms") or 0.0),
    )

    canary_ok = candidate.get("n36_canary_ok")
    if canary_ok is None:
        canary_ok = True
    ok = bool(candidate.get("ok")) and bool(canary_ok)
    latency_gate = (ttft_gain is not None and ttft_gain >= min_ttft_improvement_pct) or (
        time_to_type_gain is not None and time_to_type_gain >= min_ttft_improvement_pct
    )
    token_gate = eval_tps_change is None or eval_tps_change >= -max_eval_tps_regression_pct
    accepted = ok and latency_gate and token_gate

    result = {
        "schema": "wici_interactive_candidate_decision_v1",
        "mechanism": mechanism,
        "accepted": accepted,
        "metrics": {
            "interactive_ttft_improvement_pct": ttft_gain,
            "time_to_type_improvement_pct": time_to_type_gain,
            "eval_tokens_per_s_change_pct": eval_tps_change,
            "prompt_compute_improvement_pct": prompt_compute_gain,
        },
        "gates": {
            "process_and_canary_ok": ok,
            "latency_gate": latency_gate,
            "token_rate_gate": token_gate,
            "min_ttft_or_ttt_improvement_pct": min_ttft_improvement_pct,
            "max_eval_tps_regression_pct": max_eval_tps_regression_pct,
        },
        "baseline_trace": baseline_trace,
        "candidate_trace": candidate_trace,
    }
    if not accepted:
        result["next_experiment"] = next_after_rejection(mechanism, baseline_trace, candidate_trace, candidate)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare one Wici interactive candidate against a baseline without sweeping.")
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline JSONL from bench-wici-glm51-interactive-latency.py.")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate JSONL from bench-wici-glm51-interactive-latency.py.")
    parser.add_argument("--baseline-ttft-trace", type=Path, help="Optional baseline TTFT trace CSV.")
    parser.add_argument("--candidate-ttft-trace", type=Path, help="Optional candidate TTFT trace CSV.")
    parser.add_argument("--min-ttft-improvement-pct", type=float, default=15.0)
    parser.add_argument("--max-eval-tps-regression-pct", type=float, default=5.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    decision = compare(
        read_first_jsonl(args.baseline),
        read_first_jsonl(args.candidate),
        summarize_ttft_trace(args.baseline_ttft_trace),
        summarize_ttft_trace(args.candidate_ttft_trace),
        args.min_ttft_improvement_pct,
        args.max_eval_tps_regression_pct,
    )
    text = json.dumps(decision, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
