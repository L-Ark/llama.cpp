#!/usr/bin/env python3
import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(RUNS / "cpu-down-prefetch-dedup-bound.json"),
        help="output JSON artifact",
    )
    args = ap.parse_args()

    accepted_path = RUNS / "gate-prefill-top3000-pushed-repro-result.json"
    bound_path = RUNS / "cpu-down-prefetch-overlap-bound.json"
    candidate_path = RUNS / "cpu-down-prefetch-from-up-candidate-result.json"
    rejected_path = RUNS / "cpu-down-prefetch-from-up-pushed-repro-rejected.json"

    accepted = load_json(accepted_path)
    bound = load_json(bound_path)
    candidate = load_json(candidate_path)
    rejected = load_json(rejected_path)

    pushed_prefetch = rejected["counters"]["prefetch"]
    advised_bytes = int(pushed_prefetch["advised_bytes"])
    advised_experts = int(pushed_prefetch["advised_experts"])
    down_total_unique_gib = float(bound["prefetch_scale_estimate"]["down_total_unique_payload_gib"])
    down_decode_unique_gib = float(bound["prefetch_scale_estimate"]["down_decode_unique_payload_gib"])
    down_total_call_weighted_gib = float(bound["prefetch_scale_estimate"]["down_total_call_weighted_payload_gib"])
    down_decode_call_weighted_gib = float(bound["prefetch_scale_estimate"]["down_decode_call_weighted_payload_gib"])
    down_total_entries = int(bound["prefetch_scale_estimate"]["down_total_entries"])
    down_decode_entries = int(bound["prefetch_scale_estimate"]["down_decode_entries"])

    advised_gib = advised_bytes / (1024**3)
    down_total_unique_bytes = down_total_unique_gib * (1024**3)
    down_decode_unique_bytes = down_decode_unique_gib * (1024**3)

    repeated_to_unique_total_ratio = advised_gib / down_total_unique_gib
    repeated_to_unique_decode_ratio = advised_gib / down_decode_unique_gib
    max_advice_reduction_total = 1.0 - (down_total_unique_bytes / advised_bytes)
    max_advice_reduction_decode = 1.0 - (down_decode_unique_bytes / advised_bytes)
    expert_reduction_total = 1.0 - (down_total_entries / advised_experts)
    expert_reduction_decode = 1.0 - (down_decode_entries / advised_experts)

    artifact = {
        "artifact": "cpu-down-prefetch-dedup-bound",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_head": git_out(["rev-parse", "HEAD"]),
        "source_branch": git_out(["rev-parse", "--abbrev-ref", "HEAD"]),
        "inputs": {
            "accepted_sota": input_record(accepted_path),
            "route_specific_prefetch_bound": input_record(bound_path),
            "dirty_source_candidate": input_record(candidate_path),
            "pushed_repro_rejected": input_record(rejected_path),
        },
        "accepted_sota": {
            "eval_tok_s": accepted["summary"]["eval_tok_s"],
            "ttft_ms": accepted["summary"]["ttft_estimate_ms"],
            "memory_peak_bytes": accepted["summary"]["memory_peak_bytes"],
            "memory_file_bytes": accepted["summary"]["memory_file_bytes"],
            "correctness_ok": accepted["summary"]["correctness_ok"],
            "run_dir": accepted["run_dir"],
        },
        "previous_prefetch_result": {
            "dirty_source_eval_tok_s": candidate["summary"]["eval_tok_s"],
            "pushed_repro_eval_tok_s": rejected["summary"]["eval_tok_s"],
            "pushed_repro_verdict": rejected["verdict"],
            "pushed_prefetch_advised_experts": advised_experts,
            "pushed_prefetch_advised_gib": advised_gib,
            "pushed_prefetch_madvise_failures": int(pushed_prefetch["madvise_failures"]),
            "pushed_prefetch_matched_calls": int(pushed_prefetch["matched"]),
            "pushed_prefetch_missing_calls": int(pushed_prefetch["missing"]),
        },
        "dedup_model": {
            "mechanism": "route-specific future down prefetch, but each layer/expert is advised at most once per run",
            "correctness": "page-timing only; no routing, top-k, tensor value, dot-product, accumulation, or CPU/GPU compute split changes",
            "additional_state": "small per-layer/expert advised bitmap; no copied expert payload and no VRAM consumption",
            "down_total_unique_payload_gib": down_total_unique_gib,
            "down_decode_unique_payload_gib": down_decode_unique_gib,
            "down_total_call_weighted_payload_gib_reference": down_total_call_weighted_gib,
            "down_decode_call_weighted_payload_gib_reference": down_decode_call_weighted_gib,
            "previous_advised_gib": advised_gib,
            "repeated_to_unique_total_ratio": repeated_to_unique_total_ratio,
            "repeated_to_unique_decode_ratio": repeated_to_unique_decode_ratio,
            "max_advice_byte_reduction_vs_rejected_total_unique": max_advice_reduction_total,
            "max_advice_byte_reduction_vs_rejected_decode_unique": max_advice_reduction_decode,
            "down_total_unique_entries": down_total_entries,
            "down_decode_unique_entries": down_decode_entries,
            "max_advice_expert_reduction_vs_rejected_total_unique": expert_reduction_total,
            "max_advice_expert_reduction_vs_rejected_decode_unique": expert_reduction_decode,
        },
        "theoretical_bounds_inherited": bound["theoretical_bounds"],
        "decision": {
            "verdict": "allow_one_default_off_dedup_probe",
            "reason": (
                "The previous page-timing probe had a positive no-overhead bound but tied after pushed reproduction. "
                "Its counters show 84.56GB of repeated advice. A dedup bitmap can cut advice volume to at most the observed unique down payload "
                "without changing math or consuming VRAM, so it is materially different from the rejected probe."
            ),
            "not_path_to_10_tok_s_by_itself": True,
            "required_next_step": "Update the plan, then implement a default-off GGML_MOE_CPU_PREFETCH_DOWN_FROM_UP_DEDUP=1 probe with counters and run one strict cold France benchmark.",
            "reject_probe_if": [
                "eval_tok_s <= 4.4 after pushed-source reproduction",
                "TTFT > 33617.688744 ms for a promoted result",
                "memory_peak_bytes > 16000000000 or page cache is not charged to the cgroup",
                "any OOM/kill/ram-limit flag appears",
                "France output is incomplete, incoherent, or semantically wrong",
                "gate pack direct failures appear or gate cache counters collapse",
                "dedup counters show repeated advice remains near the rejected 84.56GB scale",
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
