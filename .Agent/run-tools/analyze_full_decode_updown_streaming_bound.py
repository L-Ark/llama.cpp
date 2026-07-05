#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
import subprocess
from collections import defaultdict
from pathlib import Path


GIB = 1024 ** 3
MIB = 1024 ** 2


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def tensor_kind_layer(name: str):
    m = re.match(r"blk\.(\d+)\.ffn_(up|down|gate)_exps\.weight", name)
    if not m:
        return -1, "other"
    return int(m.group(1)), m.group(2)


def read_profile(path: Path):
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    by_phase_kind = defaultdict(lambda: {
        "rows": 0,
        "calls": 0,
        "fallback_us": 0,
        "touch_us": 0,
        "call_bytes": 0,
        "unique": set(),
    })
    by_phase_layer = defaultdict(lambda: {
        "rows": 0,
        "calls": 0,
        "fallback_us": 0,
        "touch_us": 0,
        "call_bytes": 0,
        "unique": set(),
    })
    for r in rows:
        phase = r["phase"]
        layer, kind = tensor_kind_layer(r["tensor"])
        calls = int(r["calls"])
        fallback_us = int(r["fallback_us"])
        touch_us = int(r["touch_us"])
        expert_bytes = int(r["expert_bytes"])
        ident = (r["tensor"], int(r["expert_idx"]))
        for d in (by_phase_kind[(phase, kind)], by_phase_layer[(phase, layer)]):
            d["rows"] += 1
            d["calls"] += calls
            d["fallback_us"] += fallback_us
            d["touch_us"] += touch_us
            d["call_bytes"] += calls * expert_bytes
            d["unique"].add(ident)

    def finish(d):
        out = dict(d)
        out["unique_count"] = len(out.pop("unique"))
        out["payload_bytes"] = out["unique_count"] * 4456448
        out["fallback_ms"] = out["fallback_us"] / 1000.0
        out["touch_ms"] = out["touch_us"] / 1000.0
        out["call_gib"] = out["call_bytes"] / GIB
        out["payload_gib"] = out["payload_bytes"] / GIB
        return out

    return {
        "by_phase_kind": {f"{k[0]}:{k[1]}": finish(v) for k, v in by_phase_kind.items()},
        "by_phase_layer": {f"{k[0]}:{k[1]}": finish(v) for k, v in by_phase_layer.items()},
    }


def ms_for_gib(gib: float, gib_s: float) -> float:
    return (gib / gib_s) * 1000.0


def tok_s(decoded_tokens: float, decode_ms: float) -> float:
    return decoded_tokens / (decode_ms / 1000.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/root/lfz/vendor/llama.cpp-deepseek-v4")
    ap.add_argument("--summary-out", required=True)
    args = ap.parse_args()

    repo = Path(args.repo)
    runs = repo / ".Agent/runs/20260705-vendor-ds4-coldstart"
    bottleneck = load_json(runs / "fresh-sota44-current-head-bottleneck-analysis.json")
    screen = load_json(runs / "current-head-exact-decode-updown-candidate-screen.json")
    mmvq = load_json(runs / "mmvq-fused-transfer-hard-bound.json")

    full_trace_profile = Path("/root/lfz/runs/vendor-ds4-16gb/20260705T032634Z-20260705_current_head_sota44_component_trace_absbin/france-trace-cpu40-vram0gb/fallback_profile.csv")
    touch_profile = Path("/root/lfz/runs/vendor-ds4-16gb/20260705T033155Z-20260705_current_head_sota44_touch_split/france-touch-cpu40-vram0gb/fallback_profile.csv")
    full_prof = read_profile(full_trace_profile)
    touch_prof = read_profile(touch_profile)

    req = screen["requirements"]
    accepted_decode_ms = float(req["accepted_decode_ms"])
    target_decode_ms = float(req["target_decode_ms_for_10tok"])
    decoded_tokens = float(req["decoded_tokens_estimate"])
    full_removal_margin_ms = screen["candidate_rows"]["source_page_plus_full_fallback_removed"]["margin_to_10tok_ms"]
    total_call_gib = float(req["total_decode_call_weighted_gib"])
    source_page_delta_ms = float(req["source_page_delta_ms_estimate"])
    raw_exact_gib_s = float(req["raw_warp_gpu_gib_s"])
    transposed_exact_gib_s = float(req["transposed_warp_gpu_gib_s"])
    cpu_micro_gib_s = float(req["best_cpu_microprobe_gib_s"])
    mmvq_gib_s = float(mmvq["requirements"]["mmvq_probe_kernel_gib_s"])
    direct_prefill_gib_s = 1.850
    current_source_effective_gib_s = total_call_gib / (source_page_delta_ms / 1000.0)

    decode_up = touch_prof["by_phase_kind"]["decode:up"]
    decode_down = touch_prof["by_phase_kind"]["decode:down"]
    unique_payload_gib = decode_up["payload_gib"] + decode_down["payload_gib"]
    prompt_unique_payload_gib = (
        touch_prof["by_phase_kind"].get("prompt:up", {}).get("payload_gib", 0.0)
        + touch_prof["by_phase_kind"].get("prompt:down", {}).get("payload_gib", 0.0)
    )

    kernel_rows = []
    for name, bw in [
        ("exact_raw_warp_probe", raw_exact_gib_s),
        ("exact_transposed_warp_probe", transposed_exact_gib_s),
        ("best_cpu_microprobe_equivalent", cpu_micro_gib_s),
        ("non_exact_mmvq_probe", mmvq_gib_s),
    ]:
        kernel_ms = ms_for_gib(total_call_gib, bw)
        rem_ms = full_removal_margin_ms - kernel_ms
        row = {
            "name": name,
            "bandwidth_gib_s": bw,
            "kernel_ms_for_call_weighted_decode": kernel_ms,
            "remaining_margin_after_kernel_ms": rem_ms,
            "decode_ms_if_zero_source_plus_kernel": accepted_decode_ms - source_page_delta_ms - float(req["hot_compute_ms_after_touch"]) + kernel_ms,
        }
        row["tok_s_if_zero_source_plus_kernel"] = tok_s(decoded_tokens, row["decode_ms_if_zero_source_plus_kernel"])
        if rem_ms > 0:
            row["required_unique_source_gib_s_after_kernel"] = unique_payload_gib / (rem_ms / 1000.0)
            row["required_call_weighted_source_gib_s_after_kernel"] = total_call_gib / (rem_ms / 1000.0)
        else:
            row["required_unique_source_gib_s_after_kernel"] = None
            row["required_call_weighted_source_gib_s_after_kernel"] = None
        kernel_rows.append(row)

    source_rows = []
    for name, bw in [
        ("measured_direct_prefill_odirect", direct_prefill_gib_s),
        ("current_page_source_effective_call_weighted", current_source_effective_gib_s),
        ("pcie_24gib_s_ideal", 24.0),
        ("pcie_50gib_s_ideal", 50.0),
    ]:
        source_rows.append({
            "name": name,
            "bandwidth_gib_s": bw,
            "unique_payload_ms": ms_for_gib(unique_payload_gib, bw),
            "call_weighted_ms": ms_for_gib(total_call_gib, bw),
        })

    # Per-layer residency estimate: one extra GPU MoE layer costs all gate/up/down
    # tensors. Use full-trace layer fallback as the optimistic saved time, then
    # subtract the gate-cache capacity penalty used in previous screens.
    full_layers = full_prof["by_phase_layer"]
    touch_layers = touch_prof["by_phase_layer"]
    layer_rows = []
    gate_slot_mib = 4456448 / MIB
    per_tensor_layer_mib = 256 * 4456448 / MIB
    per_full_layer_mib = 3 * per_tensor_layer_mib
    cuda_free_mib_reference = float(req["cuda_free_mib_reference"])
    extra_mib_needed = max(0.0, per_full_layer_mib - cuda_free_mib_reference)
    gate_slots_stolen = math.ceil(extra_mib_needed / gate_slot_mib)
    lost_gate_score_est = gate_slots_stolen * 2
    gate_penalty_ms = lost_gate_score_est * float(req["gate_miss_cost_ms"])

    for key, d in full_layers.items():
        phase, layer_s = key.split(":")
        if phase != "decode":
            continue
        layer = int(layer_s)
        td = touch_layers.get(key, {})
        layer_rows.append({
            "layer": layer,
            "full_trace_fallback_ms": d["fallback_ms"],
            "touch_split_compute_ms": td.get("fallback_ms", 0.0),
            "touch_split_touch_ms": td.get("touch_ms", 0.0),
            "call_gib": d["call_gib"],
            "unique_payload_gib": d["payload_gib"],
            "optimistic_full_layer_saved_ms": d["fallback_ms"],
            "estimated_gate_penalty_ms_for_one_extra_gpu_layer": gate_penalty_ms,
            "optimistic_net_ms_after_gate_penalty": d["fallback_ms"] - gate_penalty_ms,
        })
    layer_rows.sort(key=lambda x: x["optimistic_full_layer_saved_ms"], reverse=True)

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    remote = subprocess.check_output(["git", "rev-parse", "ssd/vendor/deepseek-token-rate-16gb"], cwd=repo, text=True).strip()

    summary = {
        "status": "full_decode_updown_cuda_streaming_rejected_before_runtime_source_edit",
        "source_head": head,
        "remote_head": remote,
        "inputs": {
            "bottleneck": ".Agent/runs/20260705-vendor-ds4-coldstart/fresh-sota44-current-head-bottleneck-analysis.json",
            "candidate_screen": ".Agent/runs/20260705-vendor-ds4-coldstart/current-head-exact-decode-updown-candidate-screen.json",
            "mmvq_bound": ".Agent/runs/20260705-vendor-ds4-coldstart/mmvq-fused-transfer-hard-bound.json",
            "full_trace_profile": str(full_trace_profile),
            "touch_split_profile": str(touch_profile),
        },
        "requirements": {
            "accepted_decode_ms": accepted_decode_ms,
            "target_decode_ms_for_10tok": target_decode_ms,
            "decoded_tokens_estimate": decoded_tokens,
            "full_fallback_removal_margin_ms": full_removal_margin_ms,
            "source_page_delta_ms_estimate": source_page_delta_ms,
            "hot_compute_ms_after_touch": float(req["hot_compute_ms_after_touch"]),
            "total_decode_call_weighted_gib": total_call_gib,
            "decode_unique_updown_payload_gib": unique_payload_gib,
            "prompt_unique_updown_payload_gib": prompt_unique_payload_gib,
            "current_source_effective_call_weighted_gib_s": current_source_effective_gib_s,
            "measured_direct_prefill_odirect_gib_s": direct_prefill_gib_s,
        },
        "profile_totals": {
            "touch_split_decode_up": decode_up,
            "touch_split_decode_down": decode_down,
        },
        "kernel_bound_rows": kernel_rows,
        "source_bound_rows": source_rows,
        "resident_layer_tradeoff": {
            "per_tensor_layer_mib": per_tensor_layer_mib,
            "per_full_gate_up_down_layer_mib": per_full_layer_mib,
            "cuda_free_mib_reference": cuda_free_mib_reference,
            "extra_mib_needed_for_one_layer": extra_mib_needed,
            "gate_slots_stolen_est": gate_slots_stolen,
            "lost_gate_score_est": lost_gate_score_est,
            "gate_penalty_ms_est": gate_penalty_ms,
            "top_layers": layer_rows[:12],
            "historical_evidence": [
                "cpu_moe=39/38 strict runs are already recorded as rejected in the plan.",
                "cpu39/cache10496 only tied about 1.6 tok/s and late10 cpu39/cache13000 hit a complete stream-cache cliff.",
            ],
        },
        "decision": {
            "full_exact_gpu_kernel": "rejected: exact raw/transposed probes require about 2.0s kernel time for the full 148.916 GiB call-weighted decode payload, exceeding the 1.729s total margin before source or integration overhead.",
            "full_mmvq_gpu_kernel": "rejected before source edit: the measured non-exact MMVQ kernel leaves about 1.19s for source and integration, requiring at least about 18 GiB/s even with perfect per-run unique expert reuse or about 125 GiB/s without reuse; measured O_DIRECT is about 1.85 GiB/s and current page-source effective rate is about 9.17 GiB/s.",
            "full_resident_payload": "rejected: the decode unique up/down payload is about 21.81 GiB, larger than available VRAM and larger than the 16GB host/page-cache budget; preloading it at measured O_DIRECT speed would add about 11.8s TTFT.",
            "extra_gpu_layers": "rejected: moving one full MoE layer to GPU costs about 3264 MiB and steals roughly 712 gate-cache slots, with an estimated gate penalty larger than the optimistic per-layer fallback saving; historical cpu_moe=39/38 runs already regressed.",
            "sota_changed": False,
            "strict_cold_benchmark_allowed": False,
        },
        "next_step": "Do not implement a full decode up/down streaming source path from this bound. The remaining path must be a new algorithmic route that reduces both source bytes and exact compute substantially, or a fresh bottleneck search for smaller accepted-SOTA improvements with the same correctness/RAM/TTFT gates.",
    }
    out = Path(args.summary_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
