#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


LAYER_RE = re.compile(r"blk\.(\d+)\.ffn_(up|down|gate)_exps")


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_text(cmd: list[str], cwd: Path) -> str:
    return subprocess.check_output(cmd, cwd=cwd, text=True).strip()


def role_layer_band(tensor: str | None) -> tuple[str, int | None, str]:
    m = LAYER_RE.search(tensor or "")
    if not m:
        return "other", None, "other"
    layer = int(m.group(1))
    role = m.group(2)
    if layer <= 2:
        band = "0-2"
    elif layer <= 9:
        band = "3-9"
    elif layer <= 19:
        band = "10-19"
    elif layer <= 29:
        band = "20-29"
    else:
        band = "30-39"
    return role, layer, band


def new_bucket() -> dict[str, int]:
    return {
        "entries": 0,
        "count": 0,
        "calls": 0,
        "fallback_us": 0,
        "touch_us": 0,
        "expert_bytes_call_sum": 0,
        "pack_mmap_calls": 0,
        "gguf_calls": 0,
    }


def add_bucket(bucket: dict[str, int], row: dict[str, str]) -> None:
    calls = int(row.get("calls") or 0)
    bucket["entries"] += 1
    bucket["count"] += int(row.get("count") or 0)
    bucket["calls"] += calls
    bucket["fallback_us"] += int(row.get("fallback_us") or 0)
    bucket["touch_us"] += int(row.get("touch_us") or 0)
    bucket["expert_bytes_call_sum"] += int(row.get("expert_bytes") or 0) * calls
    bucket["pack_mmap_calls"] += int(row.get("pack_mmap_calls") or 0)
    bucket["gguf_calls"] += int(row.get("gguf_calls") or 0)


def finalize_bucket(bucket: dict[str, int]) -> dict:
    total_us = bucket["fallback_us"] + bucket["touch_us"]
    calls = bucket["calls"]
    out = dict(bucket)
    out["fallback_ms"] = bucket["fallback_us"] / 1000.0
    out["touch_ms"] = bucket["touch_us"] / 1000.0
    out["total_touch_plus_fallback_ms"] = total_us / 1000.0
    out["avg_fallback_us_per_call"] = bucket["fallback_us"] / calls if calls else 0.0
    out["avg_touch_us_per_call"] = bucket["touch_us"] / calls if calls else 0.0
    out["touch_fraction_of_touch_plus_fallback"] = bucket["touch_us"] / total_us if total_us else 0.0
    out["expert_call_gib"] = bucket["expert_bytes_call_sum"] / (1024**3)
    return out


def parse_fallback(path: Path) -> dict:
    aggs = {
        "all": new_bucket(),
        "by_phase": defaultdict(new_bucket),
        "by_role": defaultdict(new_bucket),
        "by_role_phase": defaultdict(new_bucket),
        "by_band": defaultdict(new_bucket),
        "by_band_phase": defaultdict(new_bucket),
    }
    rows = []

    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            role, layer, band = role_layer_band(row.get("tensor"))
            phase = row.get("phase", "")
            add_bucket(aggs["all"], row)
            add_bucket(aggs["by_phase"][phase], row)
            add_bucket(aggs["by_role"][role], row)
            add_bucket(aggs["by_role_phase"][f"{role}:{phase}"], row)
            add_bucket(aggs["by_band"][band], row)
            add_bucket(aggs["by_band_phase"][f"{band}:{phase}"], row)

            converted = dict(row)
            converted["role"] = role
            converted["layer"] = layer
            converted["band"] = band
            for key in ["rank", "count", "calls", "fallback_us", "touch_us", "expert_bytes", "expert_idx", "pack_mmap_calls", "gguf_calls"]:
                converted[key] = int(converted[key])
            rows.append(converted)

    def finish_map(mapping: defaultdict) -> dict:
        return {k: finalize_bucket(v) for k, v in sorted(mapping.items())}

    return {
        "path": str(path),
        "all": finalize_bucket(aggs["all"]),
        "by_phase": finish_map(aggs["by_phase"]),
        "by_role": finish_map(aggs["by_role"]),
        "by_role_phase": finish_map(aggs["by_role_phase"]),
        "by_band": finish_map(aggs["by_band"]),
        "by_band_phase": finish_map(aggs["by_band_phase"]),
        "top_fallback_entries": sorted(rows, key=lambda r: r["fallback_us"], reverse=True)[:20],
        "top_touch_entries": sorted(rows, key=lambda r: r["touch_us"], reverse=True)[:20],
    }


def parse_one_trace(path: Path) -> dict:
    fields = ["src0_ms", "src1_ms", "kernel_ms", "d2h_ms", "sync_ms", "scatter_ms", "dontneed_ms", "total_ms"]
    aggs = defaultdict(lambda: {"rows": 0, **{field: 0.0 for field in fields}})
    first_row = None
    last_t_ms = 0.0

    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seq = int(row["seq"])
            hit = row["cache_hit"]
            if first_row is None:
                first_row = dict(row)
            last_t_ms = max(last_t_ms, float(row["t_ms"]))
            keys = ["all", f"cache_hit_{hit}", "seq0_prefill_marker" if seq == 0 else "excluding_seq0"]
            for key in keys:
                aggs[key]["rows"] += 1
                for field in fields:
                    aggs[key][field] += float(row[field])

    out = {}
    for key, value in aggs.items():
        item = dict(value)
        if item["rows"]:
            for field in fields:
                item[f"avg_{field}"] = item[field] / item["rows"]
        out[key] = item

    return {
        "path": str(path),
        "last_t_ms": last_t_ms,
        "first_row": first_row,
        "aggregates": out,
    }


def parse_stderr_counters(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    counters = {}
    m = re.search(
        r"one expert pack: hits=(\d+) misses=(\d+) reads=(\d+) bytes=(\d+) failures=(\d+) "
        r"entries=(\d+) direct_enabled=(\d+) direct_reads=(\d+) direct_failures=(\d+) direct_fallbacks=(\d+)",
        text,
    )
    if m:
        keys = "hits misses reads bytes failures entries direct_enabled direct_reads direct_failures direct_fallbacks".split()
        counters["pack"] = {key: int(value) for key, value in zip(keys, m.groups())}
    m = re.search(
        r"one prefill: enabled=(\d+) loaded=(\d+) limit=(\d+) attempted=(\d+) inserted=(\d+) "
        r"pack_misses=(\d+) read_failures=(\d+) bytes=(\d+) elapsed_ms=([0-9.]+)",
        text,
    )
    if m:
        keys = "enabled loaded limit attempted inserted pack_misses read_failures bytes elapsed_ms".split()
        counters["prefill"] = {
            key: (float(value) if key == "elapsed_ms" else int(value))
            for key, value in zip(keys, m.groups())
        }
    m = re.search(r"VRAM cache: hits=(\d+) misses=(\d+) hit_rate=([0-9.]+)%", text)
    if m:
        counters["vram_cache"] = {
            "hits": int(m.group(1)),
            "misses": int(m.group(2)),
            "hit_rate_pct": float(m.group(3)),
        }
    return counters


def summarize_hashes(prefix: str, case_dir: Path, files: list[str]) -> dict[str, str | None]:
    return {f"{prefix}{name}": sha256(case_dir / name) for name in files if (case_dir / name).exists()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--accepted-case", type=Path, required=True)
    parser.add_argument("--invalid-case", type=Path)
    parser.add_argument("--full-case", type=Path, required=True)
    parser.add_argument("--touch-case", type=Path, required=True)
    parser.add_argument("--chunk-analysis", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()

    accepted_summary = load_json(args.accepted_case / "summary.json")
    full_summary = load_json(args.full_case / "summary.json")
    touch_summary = load_json(args.touch_case / "summary.json")
    invalid_summary = load_json(args.invalid_case / "summary.json") if args.invalid_case and (args.invalid_case / "summary.json").exists() else None

    full_fallback = parse_fallback(args.full_case / "fallback_profile.csv")
    touch_fallback = parse_fallback(args.touch_case / "fallback_profile.csv")
    full_one = parse_one_trace(args.full_case / "one_trace.csv")
    touch_one = parse_one_trace(args.touch_case / "one_trace.csv")
    chunk = load_json(args.chunk_analysis)

    accepted_decode_ms = accepted_summary["elapsed_seconds"] * 1000.0 - accepted_summary["ttft_estimate_ms"]
    decoded_tokens_est = accepted_summary["eval_tok_s"] * accepted_decode_ms / 1000.0
    target_decode_ms = decoded_tokens_est / 10.0 * 1000.0

    full_decode_fallback_ms = full_fallback["by_phase"]["decode"]["fallback_ms"]
    touch_decode_touch_ms = touch_fallback["by_phase"]["decode"]["touch_ms"]
    touch_decode_hot_ms = touch_fallback["by_phase"]["decode"]["fallback_ms"]
    touch_decode_total_ms = touch_fallback["by_phase"]["decode"]["total_touch_plus_fallback_ms"]
    decode_source_page_delta_ms = max(0.0, full_decode_fallback_ms - touch_decode_hot_ms)
    chunk_ideal20_ms = chunk["sum_chunk_ms"] / 20.0
    chunk_tail_gap_ms = max(0.0, chunk["max_thread_ms"] - chunk_ideal20_ms)

    def bound(removable_ms: float) -> dict:
        new_ms = accepted_decode_ms - removable_ms
        return {
            "removable_ms": removable_ms,
            "decode_ms_after_zero_overhead": new_ms,
            "tok_s_zero_overhead": decoded_tokens_est / (new_ms / 1000.0) if new_ms > 0 else None,
            "margin_to_10_tok_s_ms": target_decode_ms - new_ms,
        }

    artifact_hashes = {}
    common_files = [
        "summary.json",
        "stderr.txt",
        "stdout.txt",
        "fallback_profile.csv",
        "one_trace.csv",
        "environment.txt",
        "exact_command.txt",
    ]
    artifact_hashes.update(summarize_hashes("full_trace/", args.full_case, common_files + ["cpu_chunk_trace.csv"]))
    artifact_hashes.update(summarize_hashes("touch_split/", args.touch_case, common_files))
    artifact_hashes["chunk_analysis/current-head-sota44-cpu-chunk-analysis.json"] = sha256(args.chunk_analysis)

    analysis = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "fresh-sota44-current-head-bottleneck-analysis",
        "source": {
            "repo": str(args.repo),
            "branch": run_text(["git", "rev-parse", "--abbrev-ref", "HEAD"], args.repo),
            "head": run_text(["git", "rev-parse", "HEAD"], args.repo),
            "status": run_text(["git", "status", "--short"], args.repo),
        },
        "runs": {
            "accepted_sota_reference": {"case_dir": str(args.accepted_case), "summary": accepted_summary},
            "invalid_relative_binary_run": {
                "case_dir": str(args.invalid_case) if args.invalid_case else None,
                "summary": invalid_summary,
                "decision": "invalid_runner_error_exit_127_not_used",
            },
            "full_component_trace": {"case_dir": str(args.full_case), "summary": full_summary, "diagnostic_only": True},
            "touch_split_trace": {"case_dir": str(args.touch_case), "summary": touch_summary, "diagnostic_only": True},
        },
        "validity": {
            "full_component_trace_valid": bool(full_summary.get("exit_status") == 0 and full_summary.get("systemd_status") == 0 and full_summary.get("ram_ok") and full_summary.get("correctness_ok") and not full_summary.get("oom_seen")),
            "touch_split_trace_valid": bool(touch_summary.get("exit_status") == 0 and touch_summary.get("systemd_status") == 0 and touch_summary.get("ram_ok") and touch_summary.get("correctness_ok") and not touch_summary.get("oom_seen")),
            "ttft_trace_present": (args.full_case / "ttft_trace.csv").exists(),
            "ttft_trace_note": "GGML_MOE_TTFT_TRACE_OUT did not produce a file for this one-stream accepted path; use first_answer_ms and one_trace/prefill counters instead.",
        },
        "component_profiles": {
            "full_trace_fallback_profile": full_fallback,
            "touch_split_fallback_profile": touch_fallback,
            "full_trace_one_stream": full_one,
            "touch_split_one_stream": touch_one,
            "full_trace_cpu_chunk_analysis": chunk,
            "full_trace_stderr_counters": parse_stderr_counters(args.full_case / "stderr.txt"),
            "touch_split_stderr_counters": parse_stderr_counters(args.touch_case / "stderr.txt"),
        },
        "key_numbers": {
            "accepted_decode_window_ms": accepted_decode_ms,
            "accepted_decoded_tokens_estimate": decoded_tokens_est,
            "target_10tok_decode_ms": target_decode_ms,
            "required_decode_saving_ms_for_10tok": accepted_decode_ms - target_decode_ms,
            "full_trace_decode_fallback_ms": full_decode_fallback_ms,
            "full_trace_prompt_fallback_ms": full_fallback["by_phase"]["prompt"]["fallback_ms"],
            "touch_split_decode_touch_ms": touch_decode_touch_ms,
            "touch_split_raw_touch_ms_note": "Raw touch_us is an active sequential page-touch profiler cost and can exceed the accepted decode window; use full_trace_decode_fallback_ms - touch_split_decode_hot_fallback_ms_after_touch as the source/page exposure estimate.",
            "touch_split_decode_hot_fallback_ms_after_touch": touch_decode_hot_ms,
            "touch_split_decode_touch_plus_hot_ms": touch_decode_total_ms,
            "decode_source_page_delta_ms_estimate": decode_source_page_delta_ms,
            "full_trace_chunk_sum_ms": chunk["sum_chunk_ms"],
            "full_trace_chunk_ideal20_ms": chunk_ideal20_ms,
            "full_trace_chunk_max_thread_ms": chunk["max_thread_ms"],
            "full_trace_chunk_median_thread_ms": chunk["median_thread_ms"],
            "full_trace_chunk_tail_gap_max_minus_ideal20_ms": chunk_tail_gap_ms,
            "full_trace_gate_one_stream_excluding_seq0_total_ms": full_one["aggregates"].get("excluding_seq0", {}).get("total_ms"),
            "full_trace_gate_one_stream_seq0_prefill_marker_total_ms": full_one["aggregates"].get("seq0_prefill_marker", {}).get("total_ms"),
        },
        "hard_bounds_zero_overhead": {
            "remove_full_trace_decode_cpu_fallback": bound(full_decode_fallback_ms),
            "remove_estimated_decode_source_page_delta": bound(decode_source_page_delta_ms),
            "remove_touch_split_decode_hot_compute_only_after_touch": bound(touch_decode_hot_ms),
            "remove_chunk_tail_gap_only": bound(chunk_tail_gap_ms),
        },
        "decision": {
            "sota_changed": False,
            "current_accepted_sota_tok_s": 4.4,
            "primary_bottleneck": "decode CPU up/down fallback remains the only single component with enough removable time to approach 10 tok/s.",
            "closed_from_this_trace": [
                "scheduler/tail-only optimization: max-thread minus ideal20 is far below the required decode saving",
                "gate-only optimization: accepted path already has 94.6% gate cache hit rate and gate one-stream excluding seq0 is too small to reach 10 tok/s alone",
                "diagnostic trace token rates are not SOTA candidates because profiling perturbs runtime",
            ],
            "next_required_plan": "Design the next candidate around exact decode up/down CPU fallback removal or overlap under the 1.64s overhead budget, avoiding the already closed top768 resident-source route.",
        },
        "artifact_hashes_sha256": artifact_hashes,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    print(json.dumps({
        "key_numbers": analysis["key_numbers"],
        "hard_bounds_zero_overhead": analysis["hard_bounds_zero_overhead"],
        "decision": analysis["decision"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
