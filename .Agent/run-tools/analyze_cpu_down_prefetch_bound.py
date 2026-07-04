#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import pathlib
import subprocess
import time


ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNS = ROOT / ".Agent" / "runs" / "20260704-vendor-ds4-coldstart"


def sha256_path(path):
    path = pathlib.Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    path = pathlib.Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rel(path):
    path = pathlib.Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def input_record(path):
    path = pathlib.Path(path)
    rec = {"path": rel(path), "exists": path.exists()}
    if path.exists() and path.is_file():
        rec["sha256"] = sha256_path(path)
    return rec


def git_out(args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def compact_summary(path):
    path = pathlib.Path(path)
    rec = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return rec
    rec["sha256"] = sha256_path(path)
    data = load_json(path)
    for key in (
        "eval_tok_s",
        "prompt_tok_s",
        "ttft_estimate_ms",
        "first_answer_ms",
        "elapsed_seconds",
        "correctness_ok",
        "ram_ok",
        "ram_limit_killed",
        "oom_seen",
        "memory_peak_bytes",
        "memory_file_bytes",
        "pgmajfault",
        "workingset_refault_file",
    ):
        if key in data:
            rec[key] = data[key]
    return rec


def role_from_tensor(name):
    if "ffn_down_exps" in name:
        return "down"
    if "ffn_up_exps" in name:
        return "up"
    if "ffn_gate_exps" in name:
        return "gate"
    return "other"


def parse_fallback_csv(path):
    out = {
        "path": str(path),
        "exists": pathlib.Path(path).exists(),
        "by_role_phase": {},
        "by_role": {},
        "expert_bytes": {},
    }
    if not out["exists"]:
        return out
    out["sha256"] = sha256_path(path)
    with pathlib.Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            role = role_from_tensor(row.get("tensor", ""))
            phase = row.get("phase", "unknown")
            fallback_us = int(row.get("fallback_us") or 0)
            count = int(row.get("count") or 0)
            calls = int(row.get("calls") or 0)
            expert_bytes = int(row.get("expert_bytes") or 0)
            key = f"{role}:{phase}"
            for bucket_name in ("by_role_phase", "by_role"):
                bucket_key = key if bucket_name == "by_role_phase" else role
                bucket = out[bucket_name].setdefault(
                    bucket_key,
                    {
                        "entries": 0,
                        "count": 0,
                        "calls": 0,
                        "fallback_ms": 0.0,
                        "unique_expert_payload_bytes": 0,
                        "call_weighted_payload_bytes": 0,
                    },
                )
                bucket["entries"] += 1
                bucket["count"] += count
                bucket["calls"] += calls
                bucket["fallback_ms"] += fallback_us / 1000.0
                bucket["unique_expert_payload_bytes"] += expert_bytes
                bucket["call_weighted_payload_bytes"] += expert_bytes * calls
            out["expert_bytes"].setdefault(str(expert_bytes), 0)
            out["expert_bytes"][str(expert_bytes)] += 1
    return out


def ceiling_from_removal(base_eval_tok_s, elapsed_seconds, ttft_ms, removable_ms):
    generation_window_s = elapsed_seconds - ttft_ms / 1000.0
    estimated_decoded_tokens = base_eval_tok_s * generation_window_s
    new_generation_window_s = max(0.001, generation_window_s - removable_ms / 1000.0)
    return {
        "base_eval_tok_s": base_eval_tok_s,
        "generation_window_s_estimate": generation_window_s,
        "estimated_decoded_tokens_from_reported_rate": estimated_decoded_tokens,
        "removable_ms": removable_ms,
        "new_generation_window_s_estimate": new_generation_window_s,
        "optimistic_eval_tok_s_no_overhead": estimated_decoded_tokens / new_generation_window_s,
    }


def required_saving_ms_for_target(base_eval_tok_s, elapsed_seconds, ttft_ms, target_tok_s):
    generation_window_s = elapsed_seconds - ttft_ms / 1000.0
    estimated_decoded_tokens = base_eval_tok_s * generation_window_s
    target_window_s = estimated_decoded_tokens / target_tok_s
    return max(0.0, (generation_window_s - target_window_s) * 1000.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(RUNS / "cpu-down-prefetch-overlap-bound.json"),
        help="output JSON artifact",
    )
    args = ap.parse_args()

    accepted_path = RUNS / "gate-prefill-top3000-pushed-repro-result.json"
    phasec_path = RUNS / "phaseC-cold-vs-nodrop-comparison.json"
    fallback_path = RUNS / "current-sota-cpu-fallback-profile.json"
    walltrace_path = RUNS / "cpu-moe-walltrace-result.json"
    screening_path = RUNS / "exact-updown-candidate-screening.json"
    source_bound_path = RUNS / "source-movement-async-bound.json"

    accepted = load_json(accepted_path)
    phasec = load_json(phasec_path)
    fallback = load_json(fallback_path)
    walltrace = load_json(walltrace_path)

    cold_case = pathlib.Path(phasec["cold_case"])
    nodrop_case = pathlib.Path(phasec["nodrop_case"])
    cold_profile = parse_fallback_csv(cold_case / "fallback-profile.csv")
    nodrop_profile = parse_fallback_csv(nodrop_case / "fallback-profile.csv")

    def delta_ms(key):
        return (
            cold_profile["by_role_phase"].get(key, {}).get("fallback_ms", 0.0)
            - nodrop_profile["by_role_phase"].get(key, {}).get("fallback_ms", 0.0)
        )

    down_decode_delta_ms = delta_ms("down:decode")
    down_prompt_delta_ms = delta_ms("down:prompt")
    up_decode_delta_ms = delta_ms("up:decode")
    up_prompt_delta_ms = delta_ms("up:prompt")
    down_total_delta_ms = (
        cold_profile["by_role"].get("down", {}).get("fallback_ms", 0.0)
        - nodrop_profile["by_role"].get("down", {}).get("fallback_ms", 0.0)
    )

    summary = accepted["summary"]
    base_eval_tok_s = float(summary["eval_tok_s"])
    elapsed_seconds = float(summary["elapsed_seconds"])
    ttft_ms = float(summary["ttft_estimate_ms"])

    bounds = {
        "hide_down_decode_delta": ceiling_from_removal(
            base_eval_tok_s,
            elapsed_seconds,
            ttft_ms,
            down_decode_delta_ms,
        ),
        "hide_down_total_delta_if_it_all_landed_in_generation_window": ceiling_from_removal(
            base_eval_tok_s,
            elapsed_seconds,
            ttft_ms,
            down_total_delta_ms,
        ),
        "hide_half_down_decode_delta": ceiling_from_removal(
            base_eval_tok_s,
            elapsed_seconds,
            ttft_ms,
            down_decode_delta_ms * 0.5,
        ),
        "targets": {
            "saving_ms_for_4_5_tok_s": required_saving_ms_for_target(
                base_eval_tok_s, elapsed_seconds, ttft_ms, 4.5
            ),
            "saving_ms_for_5_0_tok_s": required_saving_ms_for_target(
                base_eval_tok_s, elapsed_seconds, ttft_ms, 5.0
            ),
            "saving_ms_for_10_0_tok_s": required_saving_ms_for_target(
                base_eval_tok_s, elapsed_seconds, ttft_ms, 10.0
            ),
        },
    }

    willneed_summaries = {
        "current_tensor_cpu_willneed_probe": compact_summary(
            "/root/lfz/runs/vendor-ds4-16gb/20260703T045821Z-20260703T045821Z-odirect-sota-cpu-willneed-probe/france-cpu40-vram0gb/summary.json"
        ),
        "stream_opwide_willneed_gate_cache13568": compact_summary(
            "/root/lfz/runs/vendor-ds4-16gb/20260701T183053Z-20260702_stream-opwide-willneed-gate-cache13568/france-cpu40-vram0gb/summary.json"
        ),
        "late10_cpu_willneed_no_trace": compact_summary(
            "/root/lfz/runs/vendor-ds4-16gb/20260702T034123Z-20260702_late10_cpu_willneed_no_trace/france-cpu40-vram0gb/summary.json"
        ),
    }

    down_decode = cold_profile["by_role_phase"].get("down:decode", {})
    down_total = cold_profile["by_role"].get("down", {})
    prefetch_scale = {
        "down_decode_entries": down_decode.get("entries"),
        "down_decode_calls": down_decode.get("calls"),
        "down_decode_unique_payload_gib": down_decode.get("unique_expert_payload_bytes", 0) / (1024**3),
        "down_decode_call_weighted_payload_gib": down_decode.get("call_weighted_payload_bytes", 0) / (1024**3),
        "down_total_entries": down_total.get("entries"),
        "down_total_calls": down_total.get("calls"),
        "down_total_unique_payload_gib": down_total.get("unique_expert_payload_bytes", 0) / (1024**3),
        "down_total_call_weighted_payload_gib": down_total.get("call_weighted_payload_bytes", 0) / (1024**3),
    }

    ceiling = bounds["hide_down_decode_delta"]["optimistic_eval_tok_s_no_overhead"]
    passes_bound = ceiling > base_eval_tok_s + 0.1
    artifact = {
        "artifact": "cpu-down-prefetch-overlap-bound",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_head": git_out(["rev-parse", "HEAD"]),
        "source_branch": git_out(["rev-parse", "--abbrev-ref", "HEAD"]),
        "inputs": {
            "accepted_sota": input_record(accepted_path),
            "phaseC_cold_vs_nodrop": input_record(phasec_path),
            "current_sota_cpu_fallback_profile": input_record(fallback_path),
            "cpu_moe_walltrace": input_record(walltrace_path),
            "exact_updown_screening": input_record(screening_path),
            "source_movement_async_bound": input_record(source_bound_path),
            "cold_fallback_profile_csv": input_record(cold_case / "fallback-profile.csv"),
            "nodrop_fallback_profile_csv": input_record(nodrop_case / "fallback-profile.csv"),
        },
        "accepted_sota": {
            "eval_tok_s": summary["eval_tok_s"],
            "prompt_tok_s": summary["prompt_tok_s"],
            "ttft_ms": summary["ttft_estimate_ms"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "memory_peak_bytes": summary["memory_peak_bytes"],
            "memory_file_bytes": summary["memory_file_bytes"],
            "ram_ok": summary["ram_ok"],
            "correctness_ok": summary["correctness_ok"],
            "run_dir": accepted["run_dir"],
        },
        "bottleneck": {
            "active_path": "gate-only one-stream cache; up/down ffn_exps remain CPU fallback",
            "walltrace_gate_all_streamed": walltrace["derived"]["gate_all_streamed"],
            "walltrace_up_down_all_declined": walltrace["derived"]["up_down_all_declined"],
            "fallback_key_numbers": fallback["analysis"]["key_numbers"],
        },
        "cold_vs_nodrop_fallback_deltas_ms": {
            "down_decode": down_decode_delta_ms,
            "down_prompt": down_prompt_delta_ms,
            "down_total": down_total_delta_ms,
            "up_decode": up_decode_delta_ms,
            "up_prompt": up_prompt_delta_ms,
            "phasec_fallback_total_delta_abs": abs(phasec["metrics"]["fallback_total_ms"]["delta"]),
            "phasec_pgmajfault_delta_abs": abs(phasec["metrics"]["pgmajfault"]["delta"]),
        },
        "prefetch_scale_estimate": prefetch_scale,
        "theoretical_bounds": bounds,
        "prior_willneed_rejections": willneed_summaries,
        "candidate": {
            "name": "route_specific_cpu_down_prefetch_from_up",
            "code_path": "ggml/src/ggml-cpu/ggml-cpu.c:ggml_compute_forward_mul_mat_id",
            "exactness": "page-timing only; must not change tensor values, routing, top-k, dot products, accumulation order, or CPU/GPU compute split",
            "implementation_mode": "default-off env GGML_MOE_CPU_PREFETCH_DOWN_FROM_UP=1",
            "intended_overlap": "issue MADV_WILLNEED for the matching routed ffn_down_exps experts after ffn_up_exps routing is known, while up fallback and later activation work are still running",
            "vram_impact": "none; must preserve GGML_MOE_STREAM_ONE_CACHE_MIB=13568 gate cache",
            "host_ram_impact": "page cache remains charged to the strict 16GB cgroup; candidate must be killed/rejected on any cgroup memory or OOM violation",
            "top1_verifier_requirement": "not required if patch is page-timing-only, but France semantic correctness remains mandatory; any compute/layout/logit change requires top1 verifier before performance",
        },
        "decision": {
            "verdict": "allow_one_default_off_probe" if passes_bound else "do_not_implement_bound_too_small",
            "passes_bound_gate": passes_bound,
            "reason": (
                "Down decode cold-vs-nodrop delta gives a no-overhead ceiling above the accepted 4.4 tok/s line, but the path is only an incremental SOTA candidate and prior broad WILLNEED probes regressed."
                if passes_bound
                else "No-overhead down decode ceiling is not materially above the accepted 4.4 tok/s line."
            ),
            "not_path_to_10_tok_s_by_itself": True,
            "required_next_step": (
                "Implement the smallest default-off page-timing patch with counters, then run exactly one strict cold France benchmark."
                if passes_bound
                else "Do not patch runtime source for this candidate; return to exact compute/offload redesign or a compatible DeepSeek draft/MTP artifact."
            ),
            "reject_probe_if": [
                "eval_tok_s <= 4.4",
                "TTFT > 33617.688744 ms for a promoted result",
                "memory_peak_bytes > 16000000000 or cgroup page cache escapes accounting",
                "oom_seen, oom_kill, or ram_limit_killed",
                "France output is incomplete, incoherent, or semantically wrong",
                "gate pack direct failures appear or gate cache counters collapse",
                "prefetch counters show missing down tensor registration for most decode calls",
            ],
        },
    }

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
        f.write("\n")
    print(str(out_path))


if __name__ == "__main__":
    main()
