#!/usr/bin/env python3
import csv
import hashlib
import json
import pathlib
import subprocess
from datetime import datetime, timezone


ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNS = ROOT / ".Agent" / "runs" / "20260704-vendor-ds4-coldstart"


def load_json(path):
    with pathlib.Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_path(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_out(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def rel(path):
    path = pathlib.Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def tok_rate_after(base_decoded_tokens, base_decode_window_s, removable_ms):
    new_window = base_decode_window_s - removable_ms / 1000.0
    if new_window <= 0:
        return None
    return base_decoded_tokens / new_window


def add_bound(rows, name, removable_ms, base_decoded_tokens, base_decode_window_s, **extra):
    rows.append({
        "name": name,
        "removable_ms": removable_ms,
        "theoretical_tok_s_no_overhead": tok_rate_after(base_decoded_tokens, base_decode_window_s, removable_ms),
        **extra,
    })


def read_csv_rows(path):
    with pathlib.Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def group_profile_rows(rows, predicate):
    out = {
        "entries": 0,
        "calls": 0,
        "count": 0,
        "fallback_ms": 0.0,
        "payload_bytes_unique": 0,
    }
    seen_payload = set()
    for row in rows:
        if not predicate(row):
            continue
        out["entries"] += 1
        out["calls"] += int(row.get("calls") or 0)
        out["count"] += int(row.get("count") or 0)
        out["fallback_ms"] += float(row.get("fallback_us") or 0) / 1000.0
        key = (row.get("tensor"), row.get("expert_idx"))
        if key not in seen_payload:
            seen_payload.add(key)
            out["payload_bytes_unique"] += int(row.get("expert_bytes") or 0)
    out["payload_mib_unique"] = out["payload_bytes_unique"] / 1024.0 / 1024.0
    return out


def layer_of(row):
    tensor = row.get("tensor") or ""
    marker = "blk."
    if marker not in tensor:
        return None
    rest = tensor.split(marker, 1)[1]
    try:
        return int(rest.split(".", 1)[0])
    except ValueError:
        return None


def main():
    accepted = load_json(RUNS / "gate-prefill-top3000-pushed-repro-result.json")
    fallback = load_json(RUNS / "current-sota-cpu-fallback-profile.json")
    phase_b = load_json(RUNS / "phaseB-current-sota-bottleneck-summary.json")
    phase_c = load_json(RUNS / "phaseC-nodrop-bottleneck-summary.json")
    cold_vs_nodrop = load_json(RUNS / "phaseC-cold-vs-nodrop-comparison.json")
    source = load_json(RUNS / "source-movement-async-bound.json")
    chunk = load_json(RUNS / "current-sota-cpu-chunk-trace-analysis.json")
    screening = load_json(RUNS / "exact-updown-candidate-screening.json")
    convert = load_json(RUNS / "mulmatid-convert-skip-bound.json")

    accepted_summary = accepted["summary"]
    base_tok_s = float(accepted_summary["eval_tok_s"])
    base_elapsed_s = float(accepted_summary["elapsed_seconds"])
    base_ttft_ms = float(accepted_summary["ttft_estimate_ms"])
    base_decode_window_s = base_elapsed_s - base_ttft_ms / 1000.0
    base_decoded_tokens = base_tok_s * base_decode_window_s

    profile_path = pathlib.Path(fallback["case_dir"]) / "fallback_profile.csv"
    profile_rows = read_csv_rows(profile_path)

    rows = []
    by_role_phase = fallback["by_role_phase"]
    add_bound(rows, "decode up fallback removed", by_role_phase["up:decode"]["fallback_ms"], base_decoded_tokens, base_decode_window_s,
              class_="exact_compute_or_offload", source="current-sota-cpu-fallback-profile")
    add_bound(rows, "decode down fallback removed", by_role_phase["down:decode"]["fallback_ms"], base_decoded_tokens, base_decode_window_s,
              class_="exact_compute_or_offload", source="current-sota-cpu-fallback-profile")
    add_bound(rows, "all decode up/down fallback removed", fallback["by_phase"]["decode"]["fallback_ms"], base_decoded_tokens, base_decode_window_s,
              class_="exact_compute_or_offload_ideal", source="current-sota-cpu-fallback-profile")
    add_bound(rows, "all prompt up/down fallback removed", fallback["by_phase"]["prompt"]["fallback_ms"], base_decoded_tokens, base_decode_window_s,
              class_="ttft_only_mostly", source="current-sota-cpu-fallback-profile")
    add_bound(rows, "all prompt+decode up/down fallback removed", fallback["analysis"]["key_numbers"]["total_fallback_ms"], base_decoded_tokens, base_decode_window_s,
              class_="full_run_ideal_reference", source="current-sota-cpu-fallback-profile")

    phase_delta = cold_vs_nodrop["metrics"]
    add_bound(rows, "cold-vs-nodrop decode fallback delta removed", abs(phase_delta["fallback_decode_ms"]["delta"]), base_decoded_tokens, base_decode_window_s,
              class_="source_page_stall_reference_not_promotable", source="phaseC-cold-vs-nodrop-comparison")
    add_bound(rows, "cold-vs-nodrop total fallback delta removed", abs(phase_delta["fallback_total_ms"]["delta"]), base_decoded_tokens, base_decode_window_s,
              class_="source_page_stall_reference_not_promotable", source="phaseC-cold-vs-nodrop-comparison")

    one_total_ms = accepted.get("one_trace_aggregate", {}).get("all_gate", {}).get("total_ms")
    one_src0_ms = accepted.get("one_trace_aggregate", {}).get("all_gate", {}).get("src0_ms")
    if one_total_ms is not None:
        add_bound(rows, "gate one-stream total time removed", one_total_ms, base_decoded_tokens, base_decode_window_s,
                  class_="gate_source_or_stream_reference", source="accepted one_trace_aggregate")
    if one_src0_ms is not None:
        add_bound(rows, "gate one-stream src0 time removed", one_src0_ms, base_decoded_tokens, base_decode_window_s,
                  class_="gate_source_reference", source="accepted one_trace_aggregate")
    if one_total_ms is not None:
        add_bound(rows, "all decode up/down fallback + gate one-stream total removed", fallback["by_phase"]["decode"]["fallback_ms"] + one_total_ms, base_decoded_tokens, base_decode_window_s,
                  class_="stacked_reference_to_10", source="current fallback + accepted one_trace_aggregate")

    add_bound(rows, "chunk scheduling/tail gap removed", source["inputs"]["scheduling_gap_s"] * 1000.0, base_decoded_tokens, base_decode_window_s,
              class_="scheduler_reference_closed", source="source-movement-async-bound")
    add_bound(rows, "mul_mat_id src1 conversion removed", convert["bound_result"]["total_convert_ms_upper_bound"], base_decoded_tokens, base_decode_window_s,
              class_="micro_closed", source="mulmatid-convert-skip-bound")

    top_bands = {}
    for band_name, lo, hi in [
        ("layers 0-2", 0, 2),
        ("layers 3-9", 3, 9),
        ("layers 10-19", 10, 19),
        ("layers 20-29", 20, 29),
        ("layers 30-39", 30, 39),
    ]:
        grouped = group_profile_rows(profile_rows, lambda r, lo=lo, hi=hi: (layer_of(r) is not None and lo <= layer_of(r) <= hi))
        top_bands[band_name] = grouped
        add_bound(rows, f"{band_name} up/down fallback removed", grouped["fallback_ms"], base_decoded_tokens, base_decode_window_s,
                  class_="layer_band_reference", source="fallback_profile.csv", payload_mib_unique=grouped["payload_mib_unique"])

    hotset_rows = []
    sorted_decode = sorted([r for r in profile_rows if r.get("phase") == "decode"], key=lambda r: float(r.get("fallback_us") or 0), reverse=True)
    for k in [64, 128, 256, 512, 1024, 2048]:
        subset = sorted_decode[:k]
        fallback_ms = sum(float(r.get("fallback_us") or 0) for r in subset) / 1000.0
        payload = sum(int(r.get("expert_bytes") or 0) for r in subset)
        rec = {
            "top_entries": k,
            "fallback_ms": fallback_ms,
            "payload_mib_naive": payload / 1024.0 / 1024.0,
            "theoretical_tok_s_no_overhead": tok_rate_after(base_decoded_tokens, base_decode_window_s, fallback_ms),
        }
        hotset_rows.append(rec)

    rows_sorted = sorted(rows, key=lambda r: (r["theoretical_tok_s_no_overhead"] or 0), reverse=True)
    viable = []
    for row in rows_sorted:
        if (row["theoretical_tok_s_no_overhead"] or 0) > base_tok_s:
            if row["class_"] in ("exact_compute_or_offload", "exact_compute_or_offload_ideal", "stacked_reference_to_10"):
                viable.append(row["name"])

    out = {
        "schema": "vendor-ds4-post-prefetch-bottleneck-v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source": {
            "head": git_out("rev-parse", "HEAD"),
            "dirty": bool(git_out("status", "--short")),
            "branch": git_out("branch", "--show-current"),
            "remote_branch": "https://github.com/wici-ai/ssd-llama.git vendor/deepseek-token-rate-16gb",
        },
        "accepted_sota": {
            "eval_tok_s": base_tok_s,
            "prompt_tok_s": accepted_summary["prompt_tok_s"],
            "elapsed_seconds": base_elapsed_s,
            "ttft_ms": base_ttft_ms,
            "decode_window_seconds": base_decode_window_s,
            "decoded_tokens_estimate": base_decoded_tokens,
            "run_dir": accepted["run_dir"],
            "ram_ok": accepted_summary["ram_ok"],
            "correctness_ok": accepted_summary["correctness_ok"],
            "memory_peak_bytes": accepted_summary["memory_peak_bytes"],
            "memory_file_bytes": accepted_summary["memory_file_bytes"],
        },
        "inputs": {
            rel(RUNS / "gate-prefill-top3000-pushed-repro-result.json"): sha256_path(RUNS / "gate-prefill-top3000-pushed-repro-result.json"),
            rel(RUNS / "current-sota-cpu-fallback-profile.json"): sha256_path(RUNS / "current-sota-cpu-fallback-profile.json"),
            rel(RUNS / "phaseB-current-sota-bottleneck-summary.json"): sha256_path(RUNS / "phaseB-current-sota-bottleneck-summary.json"),
            rel(RUNS / "phaseC-nodrop-bottleneck-summary.json"): sha256_path(RUNS / "phaseC-nodrop-bottleneck-summary.json"),
            rel(RUNS / "phaseC-cold-vs-nodrop-comparison.json"): sha256_path(RUNS / "phaseC-cold-vs-nodrop-comparison.json"),
            rel(RUNS / "source-movement-async-bound.json"): sha256_path(RUNS / "source-movement-async-bound.json"),
            rel(RUNS / "current-sota-cpu-chunk-trace-analysis.json"): sha256_path(RUNS / "current-sota-cpu-chunk-trace-analysis.json"),
            rel(RUNS / "exact-updown-candidate-screening.json"): sha256_path(RUNS / "exact-updown-candidate-screening.json"),
            rel(RUNS / "mulmatid-convert-skip-bound.json"): sha256_path(RUNS / "mulmatid-convert-skip-bound.json"),
            rel(profile_path): sha256_path(profile_path),
        },
        "bottleneck_rows_sorted_by_ceiling": rows_sorted,
        "decode_hotset_reference": hotset_rows,
        "layer_band_payload_reference": top_bands,
        "cold_vs_nodrop_metrics": phase_delta,
        "closed_candidate_summary": screening["screening_result"],
        "interpretation": {
            "primary_remaining_bottleneck": "CPU up/down fallback dominates. Under the accepted 4.4 tok/s run window, ideal removal of all decode up/down fallback has an estimated ceiling above 10 tok/s, so the next design must target exact full or near-full decode fallback reduction rather than millisecond-scale micro-optimizations.",
            "do_not_repeat": [
                "synchronous madvise/page-touch down prefetch",
                "packmmap/packdirect source movement without new bandwidth model",
                "chunk-size/affinity-only probes",
                "mul_mat_id conversion skip",
                "existing one-stream up/down cache expansion without new correctness mechanism",
            ],
            "next_required_design": "Inspect active CPU fallback code and find a new exact compute/offload/layout mechanism. Any arithmetic/offload change must pass the existing top1 verifier before strict cold benchmark.",
            "candidate_rows_worth_design": viable,
        },
    }

    out_path = RUNS / "post-prefetch-bottleneck-hard-bound.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_path)
    print(json.dumps({
        "accepted_eval_tok_s": base_tok_s,
        "top_bounds": rows_sorted[:8],
        "candidate_rows_worth_design": viable,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
