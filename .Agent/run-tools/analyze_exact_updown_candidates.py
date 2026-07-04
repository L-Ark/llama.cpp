#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import time


ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNS = ROOT / ".Agent" / "runs" / "20260704-vendor-ds4-coldstart"
RUNS_20260703 = ROOT / ".Agent" / "runs" / "20260703-vendor-ds4-coldstart"


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


def maybe_load_json(path):
    path = pathlib.Path(path)
    if not path.exists():
        return None
    return load_json(path)


def git_out(args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def compact_summary(path):
    d = maybe_load_json(path)
    if not d:
        return {"path": str(path), "exists": False}
    keys = [
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
    ]
    out = {k: d.get(k) for k in keys if k in d}
    out["path"] = str(path)
    out["exists"] = True
    if "answer" in d:
        out["answer_prefix"] = d["answer"][:220]
    return out


def rel(p):
    p = pathlib.Path(p)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def input_record(path):
    path = pathlib.Path(path)
    rec = {"path": rel(path), "exists": path.exists()}
    if path.exists() and path.is_file():
        rec["sha256"] = sha256_path(path)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(RUNS / "exact-updown-candidate-screening.json"),
        help="output JSON artifact",
    )
    args = ap.parse_args()

    accepted = load_json(RUNS / "gate-prefill-top3000-pushed-repro-result.json")
    fallback = load_json(RUNS / "current-sota-cpu-fallback-profile.json")
    hard = load_json(RUNS / "phaseB-current-sota-hard-bound-table.json")
    hot = load_json(RUNS / "ds4-hot-dispatch-h0-bound-analysis.json")
    skip = load_json(RUNS / "true-skip-zero-row-bound-analysis.json")
    drift = load_json(RUNS / "gpu-updown-output-drift-audit.json")
    alg = load_json(RUNS / "algorithmic-support-inspection.json")
    source = load_json(RUNS / "source-movement-async-bound.json")
    verifier = load_json(RUNS / "llama-results-top1-selfcheck-light-result.json")
    chunk = load_json(RUNS / "current-sota-cpu-chunk-trace-full-result.json")
    repack_micro = load_json(RUNS_20260703 / "mxfp4-transient-repack-harness-summary.json")

    summary_paths = {
        "cuda_graph_enabled": pathlib.Path("/root/lfz/runs/vendor-ds4-16gb/20260703T091200Z-20260703T091200Z-cuda-graphs-enabled-sota-probe/france-cpu40-vram0gb/summary.json"),
        "no_repack": pathlib.Path("/root/lfz/runs/vendor-ds4-16gb/20260703T054431Z-20260703T054431Z-sota-no-repack-diagnostic/france-cpu40-vram0gb/summary.json"),
        "transient_down_repack_full_model": pathlib.Path("/root/lfz/runs/vendor-ds4-16gb/20260703T105753Z-20260703T-transient-down-repack-probe/france-cpu40-vram0gb/summary.json"),
        "chunk32": pathlib.Path("/root/lfz/runs/vendor-ds4-16gb/20260703T123416Z-20260703_moe_cpu_chunk32_probe/france-cpu40-vram0gb/summary.json"),
        "single_row_chunks40": pathlib.Path("/root/lfz/runs/vendor-ds4-16gb/20260703T052405Z-20260703T052405Z-single-row-chunks40-candidate/france-cpu40-vram0gb/summary.json"),
        "thread18": pathlib.Path("/root/lfz/runs/vendor-ds4-16gb/20260703T050107Z-20260703T050107Z-odirect-sota-thread18-probe/france-cpu40-vram0gb/summary.json"),
        "thread22": pathlib.Path("/root/lfz/runs/vendor-ds4-16gb/20260703T050239Z-20260703T050239Z-odirect-sota-thread22-probe/france-cpu40-vram0gb/summary.json"),
        "packdirect_top128_updown": pathlib.Path("/root/lfz/runs/vendor-ds4-16gb/20260703T212905Z-20260704_cpu_fallback_packdirect_top128_updown/france-cpu40-vram0gb/summary.json"),
        "packmmap_top128_updown": pathlib.Path("/root/lfz/runs/vendor-ds4-16gb/20260703T210008Z-20260704_cpu_fallback_packmmap_top128_updown/france-cpu40-vram0gb/summary.json"),
        "packmmap_top256_updown": pathlib.Path("/root/lfz/runs/vendor-ds4-16gb/20260703T205623Z-20260704_cpu_fallback_packmmap_top256_updown/france-cpu40-vram0gb/summary.json"),
    }
    summaries = {name: compact_summary(path) for name, path in summary_paths.items()}

    hard_by_name = {x["candidate"]: x for x in hard.get("hard_bounds", [])}
    accepted_summary = accepted["summary"]
    fallback_key = fallback["analysis"]["key_numbers"]

    hot_best_preserve = hot["best_partial_layer_knapsack"][0]
    hot_best_492 = hot["best_partial_layer_knapsack"][1]
    skip_zero_bound = skip["bounds"]["perfect_existing_zero_group_elimination"]
    all_fallback_ref = skip["bounds"]["remove_all_remaining_updown_cpu_fallback_for_reference_only"]

    candidate_classes = [
        {
            "name": "current_ds4_hot_dispatch",
            "code_path": ["src/llama-deepseek4-hot.*", "src/models/deepseek4.cpp"],
            "exactness": "intended exact hot/cold split, but still requires top1 verifier before performance",
            "expected_removable_ms": hot_best_preserve["optimistic_fallback_seconds_saved"] * 1000.0,
            "vram_cost": "best preserving accepted gate cache: %.2f MiB used under %.0f MiB free" % (
                hot_best_preserve["best_gpu_mib"],
                hot_best_preserve["budget_mib"],
            ),
            "steals_gate_cache": False,
            "host_ram_impact": "no model run; H0 bound only",
            "theoretical_tok_s_ceiling": hot_best_preserve["optimistic_eval_tok_s_no_overhead"],
            "verifier_before_perf": True,
            "evidence": [
                rel(RUNS / "ds4-hot-dispatch-h0-bound-analysis.json"),
                "If 492 MiB is freed by risking gate cache, ceiling is %.3f tok/s only" % hot_best_492["optimistic_eval_tok_s_no_overhead"],
            ],
            "decision": "rejected_before_model_run",
            "reason": hot["decision"]["reason"],
        },
        {
            "name": "true_skip_zero_row",
            "code_path": ["ggml/src/ggml-cpu/ggml-cpu.c:ggml_compute_forward_mul_mat_id"],
            "exactness": "exact only for rows already pruned by current keep_topk policy",
            "expected_removable_ms": skip_zero_bound["seconds_saved"] * 1000.0,
            "vram_cost": "none",
            "steals_gate_cache": False,
            "host_ram_impact": "none",
            "theoretical_tok_s_ceiling": skip_zero_bound["optimistic_eval_tok_s_no_overhead"],
            "verifier_before_perf": False,
            "evidence": [rel(RUNS / "true-skip-zero-row-bound-analysis.json")],
            "decision": "rejected_bound_too_small",
            "reason": skip["decision"]["reason"],
        },
        {
            "name": "existing_one_stream_gpu_updown_cache_classes",
            "code_path": ["ggml/src/ggml-cuda/*moe*", "ggml/src/ggml.c graph path"],
            "exactness": "can change generation trajectory; must pass full-output/top1 correctness",
            "expected_removable_ms": None,
            "vram_cost": "uses same VRAM pressure class as gate cache and can collapse cache/source behavior",
            "steals_gate_cache": True,
            "host_ram_impact": "strict cold runs stayed within 16GB but refault/cache counters worsened",
            "theoretical_tok_s_ceiling": None,
            "verifier_before_perf": True,
            "evidence": [
                rel(RUNS / "gpu-updown-output-drift-audit.json"),
                "hot64 measured %.1f tok/s with manual semantic failure" % drift["cases"]["rejected_hot_updown64_require_profile"]["summary"]["eval_tok_s"],
                "nofilter measured %.1f tok/s with incomplete generation" % drift["cases"]["rejected_updown_stream_nofilter"]["summary"]["eval_tok_s"],
            ],
            "decision": "rejected_correctness_and_performance",
            "reason": drift["answers_to_plan_questions"]["q4_worth_another_run"],
        },
        {
            "name": "cuda_graph_enabled_on_current_sota",
            "code_path": ["ggml/src/ggml-cuda/common.cuh graph capture path", ".Agent/run-tools/llama-cli-cuda-graphs-enabled.sh"],
            "exactness": "should not change logits; performance-only launch overhead probe",
            "expected_removable_ms": None,
            "vram_cost": "none expected",
            "steals_gate_cache": False,
            "host_ram_impact": "strict cold run within 16GB",
            "theoretical_tok_s_ceiling": summaries["cuda_graph_enabled"].get("eval_tok_s"),
            "verifier_before_perf": False,
            "evidence": [summaries["cuda_graph_enabled"]],
            "decision": "rejected_measured_tie_regress",
            "reason": "strict cold run measured 4.1 tok/s, below accepted 4.4 tok/s",
        },
        {
            "name": "cpu_repack_and_transient_repack",
            "code_path": [
                ".Agent/run-tools/mxfp4_transient_repack_harness.cpp",
                "ggml/src/ggml-cpu/ggml-cpu.c MXFP4 fallback path",
            ],
            "exactness": "microbench exact, but runtime variants either slower or disturbed cold page/cache behavior",
            "expected_removable_ms": None,
            "vram_cost": "none persistent for transient; persistent/hotset repack has unacceptable memory/page-cache tradeoff",
            "steals_gate_cache": False,
            "host_ram_impact": "transient full-model run stayed within 16GB but increased page/refault pressure",
            "theoretical_tok_s_ceiling": None,
            "verifier_before_perf": True,
            "evidence": [
                rel(RUNS_20260703 / "mxfp4-transient-repack-harness-summary.json"),
                summaries["no_repack"],
                summaries["transient_down_repack_full_model"],
            ],
            "decision": "rejected_microbench_or_full_model",
            "reason": repack_micro["decision"] + " Full-model transient down-repack measured %.1f tok/s." % summaries["transient_down_repack_full_model"].get("eval_tok_s"),
        },
        {
            "name": "page_source_prewarm_packmmap_packdirect",
            "code_path": [
                "one-stream expert pack direct-read path",
                "CPU fallback pack mmap/direct diagnostic paths",
                "madvise/page-touch prewarm diagnostics",
            ],
            "exactness": "performance-only if routing and math unchanged",
            "expected_removable_ms": None,
            "vram_cost": "none; host page cache or direct-read byte volume is the issue",
            "steals_gate_cache": False,
            "host_ram_impact": "must fit 16GB including page cache; tested variants did but tied/regressed",
            "theoretical_tok_s_ceiling": None,
            "verifier_before_perf": False,
            "evidence": [
                rel(RUNS / "source-movement-async-bound.json"),
                summaries["packdirect_top128_updown"],
                summaries["packmmap_top128_updown"],
                summaries["packmmap_top256_updown"],
            ],
            "decision": "rejected_negative_io_bound",
            "reason": source["conclusion"]["reason"],
        },
        {
            "name": "chunk_affinity_scheduler_only",
            "code_path": ["ggml/src/ggml-cpu/ggml-cpu.c thread/chunk partitioning", ".Agent/run-tools/llama-cli-taskset-0-19.sh"],
            "exactness": "performance-only if chunking preserves arithmetic order per output",
            "expected_removable_ms": chunk["thread_totals"]["max_ms"] - chunk["thread_totals"]["min_ms"],
            "vram_cost": "none",
            "steals_gate_cache": False,
            "host_ram_impact": "single-row chunks40 violated RAM; chunk32/affinity stayed within 16GB but regressed",
            "theoretical_tok_s_ceiling": None,
            "verifier_before_perf": False,
            "evidence": [
                rel(RUNS / "current-sota-cpu-chunk-trace-full-result.json"),
                summaries["chunk32"],
                summaries["single_row_chunks40"],
                summaries["thread18"],
                summaries["thread22"],
            ],
            "decision": "rejected_bound_or_ram",
            "reason": chunk["conclusion"]["next_recommendation"],
        },
        {
            "name": "algorithmic_multi_token_without_local_draft_or_mtp",
            "code_path": ["examples/speculative*", "examples/lookahead", "DeepSeek4 MTP/NextN runtime hooks"],
            "exactness": "can change target sequence unless verified by target model",
            "expected_removable_ms": None,
            "vram_cost": "unknown; no compatible local artifact",
            "steals_gate_cache": "unknown",
            "host_ram_impact": "unknown; no compatible local artifact",
            "theoretical_tok_s_ceiling": None,
            "verifier_before_perf": True,
            "evidence": [rel(RUNS / "algorithmic-support-inspection.json")],
            "decision": "closed_missing_compatible_artifact",
            "reason": alg["decision"]["no_source_algorithmic_candidate_available_now"],
        },
    ]

    passing = [c for c in candidate_classes if not c["decision"].startswith(("rejected", "closed"))]
    artifact = {
        "artifact": "exact-updown-candidate-screening",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {
            "branch": git_out(["branch", "--show-current"]),
            "head": git_out(["rev-parse", "HEAD"]),
            "remote_ssd_vendor_deepseek_token_rate_16gb": git_out(["ls-remote", "ssd", "refs/heads/vendor/deepseek-token-rate-16gb"]),
        },
        "accepted_sota": {
            "eval_tok_s": accepted_summary["eval_tok_s"],
            "prompt_tok_s": accepted_summary["prompt_tok_s"],
            "ttft_ms": accepted_summary["ttft_estimate_ms"],
            "ttft_limit_ms": accepted_summary["ttft_limit_ms"],
            "elapsed_seconds": accepted_summary["elapsed_seconds"],
            "memory_peak_bytes": accepted_summary["memory_peak_bytes"],
            "memory_file_bytes": accepted_summary["memory_file_bytes"],
            "ram_ok": accepted_summary["ram_ok"],
            "correctness_ok": accepted_summary["manual_correctness_ok"],
            "run_dir": accepted["run_dir"],
            "pushed_branch": accepted["pushed_branch"],
            "answer": accepted_summary["answer"],
        },
        "current_bottleneck": {
            "primary": fallback["analysis"]["primary_bottleneck"],
            "total_fallback_ms": fallback_key["total_fallback_ms"],
            "up_fallback_ms": fallback_key["up_fallback_ms"],
            "down_fallback_ms": fallback_key["down_fallback_ms"],
            "decode_fallback_ms": fallback_key["decode_fallback_ms"],
            "prompt_fallback_ms": fallback_key["prompt_fallback_ms"],
            "fallback_counts": {
                "up": fallback["by_role"]["up"]["count"],
                "down": fallback["by_role"]["down"]["count"],
            },
            "chunk_trace_conclusion": chunk["conclusion"],
        },
        "hard_bounds": {
            "all_decode_updown_removed": hard_by_name.get("all CPU up/down decode fallback removed"),
            "all_decode_updown_and_one_stream_source_removed": hard_by_name.get("all CPU up/down decode fallback + all one-stream source time removed"),
            "all_remaining_updown_fallback_reference": all_fallback_ref,
        },
        "correctness_gate": {
            "lightweight_top1_verifier_status": verifier["status"],
            "same_top1": verifier["summary"]["same_top1"],
            "n_tokens": verifier["summary"]["n_tokens"],
            "first_mismatch_pos": verifier["summary"]["first_mismatch_pos"],
            "baseline_top1_report_sha256": verifier["summary"]["baseline_top1_report_sha256"],
            "artifact": rel(RUNS / "llama-results-top1-selfcheck-light-result.json"),
        },
        "candidate_classes": candidate_classes,
        "screening_result": {
            "concrete_runtime_candidate_passes_gate": bool(passing),
            "passing_candidates": [c["name"] for c in passing],
            "verdict": "No already-defined concrete runtime candidate passes the plan's screening gate. Do not run another full model benchmark until a new exact compute/offload/layout mechanism has a hard bound above tie and a correctness-verification path.",
            "next_required_work": [
                "Inspect the active CPU fallback code path to identify a new exact mechanism that reduces actual up/down source/dot work instead of only changing scheduling.",
                "A viable design must explain how it avoids the rejected hot-dispatch VRAM/gate-cache tradeoff and rejected repack/page-refault regressions.",
                "If the design can change logits, run the lightweight top1 verifier before any strict cold token-rate run.",
            ],
        },
        "inputs": {
            "tracked_json": [
                input_record(RUNS / "gate-prefill-top3000-pushed-repro-result.json"),
                input_record(RUNS / "current-sota-cpu-fallback-profile.json"),
                input_record(RUNS / "phaseB-current-sota-hard-bound-table.json"),
                input_record(RUNS / "ds4-hot-dispatch-h0-bound-analysis.json"),
                input_record(RUNS / "true-skip-zero-row-bound-analysis.json"),
                input_record(RUNS / "gpu-updown-output-drift-audit.json"),
                input_record(RUNS / "algorithmic-support-inspection.json"),
                input_record(RUNS / "source-movement-async-bound.json"),
                input_record(RUNS / "llama-results-top1-selfcheck-light-result.json"),
                input_record(RUNS / "current-sota-cpu-chunk-trace-full-result.json"),
                input_record(RUNS_20260703 / "mxfp4-transient-repack-harness-summary.json"),
            ],
            "external_summary_json": [
                input_record(p) for p in summary_paths.values()
            ],
        },
    }

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    print(str(out))
    print(sha256_path(out))


if __name__ == "__main__":
    main()
