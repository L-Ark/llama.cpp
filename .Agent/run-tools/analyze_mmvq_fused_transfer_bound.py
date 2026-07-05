#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
from pathlib import Path


PAIR_COUNTS = [16, 32, 48, 64, 80, 96, 128]
SLOT_MIB = 4.25
CUDA_FREE_MIB_ACCEPTED = 238.0
GATE_MISS_COST_MS = 1.024052203321064
BEST_CPU_MICROPROBE_GIB_S = 74.719
N_EMBD = 2048
N_FF = 4096


def run_text(cmd: list[str], cwd: Path) -> str:
    return subprocess.check_output(cmd, cwd=cwd, text=True).strip()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def tensor_role(tensor: str) -> str | None:
    if "ffn_up_exps" in tensor:
        return "up"
    if "ffn_down_exps" in tensor:
        return "down"
    return None


def tensor_layer(tensor: str) -> int:
    m = re.search(r"blk\.(\d+)\.", tensor)
    if not m:
        raise ValueError(f"cannot parse layer from {tensor}")
    return int(m.group(1))


def gate_scores(path: Path) -> list[int]:
    scores: list[int] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                try:
                    scores.append(int(parts[2]))
                except ValueError:
                    pass
    scores.sort(reverse=True)
    return scores


def gate_penalty_ms(scores: list[int], payload_mib: float) -> dict:
    extra_mib = max(0.0, payload_mib - CUDA_FREE_MIB_ACCEPTED)
    stolen_slots = int(math.ceil(extra_mib / SLOT_MIB)) if extra_mib > 0 else 0
    start = max(0, 3000 - stolen_slots)
    lost_score = sum(scores[start:3000]) if stolen_slots > 0 else 0
    return {
        "payload_mib": payload_mib,
        "cuda_free_mib_reference": CUDA_FREE_MIB_ACCEPTED,
        "extra_mib_needed": extra_mib,
        "gate_slots_stolen_est": stolen_slots,
        "lost_gate_score_est": lost_score,
        "gate_penalty_ms_est": lost_score * GATE_MISS_COST_MS,
    }


def decode_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            role = tensor_role(row["tensor"])
            if row["phase"] != "decode" or role is None:
                continue
            calls = int(row["calls"])
            expert_bytes = int(row["expert_bytes"])
            rows.append({
                "tensor": row["tensor"],
                "layer": tensor_layer(row["tensor"]),
                "expert": int(row["expert_idx"]),
                "role": role,
                "calls": calls,
                "expert_bytes": expert_bytes,
                "hot_ms": int(row["fallback_us"]) / 1000.0,
                "call_weighted_gib": calls * expert_bytes / (1024**3),
                "payload_mib": expert_bytes / (1024**2),
            })
    return rows


def tok_s(tokens: float, decode_ms: float) -> float:
    return tokens / (decode_ms / 1000.0)


def bound(tokens: float, target_ms: float, decode_ms: float) -> dict:
    return {
        "decode_ms": decode_ms,
        "tok_s": tok_s(tokens, decode_ms),
        "margin_to_10tok_ms": target_ms - decode_ms,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--bottleneck", type=Path, required=True)
    ap.add_argument("--mmvq-validation", type=Path, required=True)
    ap.add_argument("--gate-profile", type=Path, required=True)
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args()

    bottleneck = load_json(args.bottleneck)
    key = bottleneck["key_numbers"]
    touch_case = Path(bottleneck["runs"]["touch_split_trace"]["case_dir"])
    rows = decode_rows(touch_case / "fallback_profile.csv")
    scores = gate_scores(args.gate_profile)
    mmvq = load_json(args.mmvq_validation)
    mmvq_sample = mmvq["runs"]["fixed_text_compare512"]["sample"]

    mmvq_gib_s = float(mmvq_sample["kernel_source_bandwidth_gib_s"])
    sample_h2d_us = float(mmvq_sample["timing_us"]["h2d"])
    sample_d2h_us = float(mmvq_sample["timing_us"]["d2h"])
    sample_ready_rows = float(mmvq_sample["ready_rows"])
    sample_src0_bytes = float(mmvq_sample["src0_bytes"])

    # The compare probe H2D/D2H timings are pageable, per-op diagnostic timings.
    # They are useful as an upper bound for CPU-backend skip/write, not as a graph-level lower bound.
    # Convert to bytes/ms rates from the fixed-text sample.
    sample_f32_h2d_bytes = 0
    sample_d2h_bytes = float(mmvq_sample.get("out_bytes", mmvq_sample["output_elements"] * 4))
    # Sample ready rows are split almost evenly between up/down in the validation artifact.
    # Use the exact by-kind counts if present; otherwise approximate from ready rows.
    by_kind = mmvq["runs"]["fixed_text_compare512"].get("by_kind", {})
    sample_up_rows = float(by_kind.get("up", {}).get("ready_rows", sample_ready_rows / 2.0))
    sample_down_rows = float(by_kind.get("down", {}).get("ready_rows", sample_ready_rows / 2.0))
    sample_f32_h2d_bytes = sample_up_rows * N_EMBD * 4 + sample_down_rows * N_FF * 4
    h2d_bytes_per_ms = sample_f32_h2d_bytes / (sample_h2d_us / 1000.0) if sample_h2d_us > 0 else None
    d2h_bytes_per_ms = sample_d2h_bytes / (sample_d2h_us / 1000.0) if sample_d2h_us > 0 else None

    pairs: dict[tuple[int, int], dict] = {}
    for row in rows:
        pair = pairs.setdefault((row["layer"], row["expert"]), {
            "layer": row["layer"],
            "expert": row["expert"],
            "up": None,
            "down": None,
        })
        pair[row["role"]] = row

    complete_pairs = []
    for pair in pairs.values():
        if pair["up"] is None or pair["down"] is None:
            continue
        up = pair["up"]
        down = pair["down"]
        # In routed decode these should normally match. If they do not, the strict fused lower bound can
        # only fuse min(up, down) rows and leaves the excess to non-fused handling.
        fused_calls = min(up["calls"], down["calls"])
        complete_pairs.append({
            "layer": pair["layer"],
            "expert": pair["expert"],
            "up_calls": up["calls"],
            "down_calls": down["calls"],
            "fused_calls": fused_calls,
            "hot_ms": up["hot_ms"] + down["hot_ms"],
            "up_hot_ms": up["hot_ms"],
            "down_hot_ms": down["hot_ms"],
            "call_weighted_gib_updown": up["call_weighted_gib"] + down["call_weighted_gib"],
            "call_weighted_gib_gate_recompute": up["call_weighted_gib"],
            "payload_mib_updown": up["payload_mib"] + down["payload_mib"],
            "payload_mib_gate_updown": up["payload_mib"] * 3,
            "cpu_backend_fused_h2d_bytes": fused_calls * N_EMBD * 4,
            "cpu_backend_fused_d2h_bytes": fused_calls * N_EMBD * 4,
            "standalone_h2d_bytes": up["calls"] * N_EMBD * 4 + down["calls"] * N_FF * 4,
            "standalone_d2h_bytes": up["calls"] * N_FF * 4 + down["calls"] * N_EMBD * 4,
        })
    complete_pairs.sort(key=lambda r: r["hot_ms"], reverse=True)

    accepted_decode_ms = key["accepted_decode_window_ms"]
    decoded_tokens = key["accepted_decoded_tokens_estimate"]
    target_decode_ms = key["target_10tok_decode_ms"]
    source_delta_ms = key["decode_source_page_delta_ms_estimate"]
    hot_ms = key["touch_split_decode_hot_fallback_ms_after_touch"]
    total_call_gib = sum(r["call_weighted_gib"] for r in rows)
    base_after_source_and_full_hot_removed = accepted_decode_ms - source_delta_ms - hot_ms

    pair_rows = []
    for count in PAIR_COUNTS:
        selected = complete_pairs[:count]
        hot_save = sum(r["hot_ms"] for r in selected)
        selected_gib = sum(r["call_weighted_gib_updown"] for r in selected)
        selected_gate_gib = sum(r["call_weighted_gib_gate_recompute"] for r in selected)
        payload_mib = sum(r["payload_mib_updown"] for r in selected)
        payload_gate_mib = sum(r["payload_mib_gate_updown"] for r in selected)
        remaining_gib = total_call_gib - selected_gib
        remaining_cpu_ms = remaining_gib / BEST_CPU_MICROPROBE_GIB_S * 1000.0
        penalty = gate_penalty_ms(scores, payload_mib)
        penalty_gate = gate_penalty_ms(scores, payload_gate_mib)
        ideal_zero_transfer_decode = (
            base_after_source_and_full_hot_removed +
            remaining_cpu_ms +
            penalty["gate_penalty_ms_est"]
        )
        updown_kernel_ms = selected_gib / mmvq_gib_s * 1000.0
        gate_recompute_kernel_ms = (selected_gib + selected_gate_gib) / mmvq_gib_s * 1000.0

        fused_h2d_bytes = sum(r["cpu_backend_fused_h2d_bytes"] for r in selected)
        fused_d2h_bytes = sum(r["cpu_backend_fused_d2h_bytes"] for r in selected)
        standalone_h2d_bytes = sum(r["standalone_h2d_bytes"] for r in selected)
        standalone_d2h_bytes = sum(r["standalone_d2h_bytes"] for r in selected)
        fused_transfer_ms = 0.0
        standalone_transfer_ms = 0.0
        if h2d_bytes_per_ms and d2h_bytes_per_ms:
            fused_transfer_ms = fused_h2d_bytes / h2d_bytes_per_ms + fused_d2h_bytes / d2h_bytes_per_ms
            standalone_transfer_ms = standalone_h2d_bytes / h2d_bytes_per_ms + standalone_d2h_bytes / d2h_bytes_per_ms

        pair_rows.append({
            "pair_count": count,
            "complete_pairs_available": len(complete_pairs),
            "hot_save_ms_zero_kernel": hot_save,
            "payload_mib_updown": payload_mib,
            "payload_mib_gate_updown": payload_gate_mib,
            "call_weighted_gib_updown": selected_gib,
            "call_weighted_gib_gate_recompute_extra": selected_gate_gib,
            "remaining_cpu_microprobe_ms": remaining_cpu_ms,
            "gate_penalty_updown_payload": penalty,
            "gate_penalty_gate_updown_payload": penalty_gate,
            "allowed_fused_ms_for_10tok_zero_transfer": target_decode_ms - ideal_zero_transfer_decode,
            "graph_zero_transfer_existing_gate_output": bound(
                decoded_tokens,
                target_decode_ms,
                ideal_zero_transfer_decode + updown_kernel_ms,
            ),
            "graph_zero_transfer_gate_recompute": bound(
                decoded_tokens,
                target_decode_ms,
                base_after_source_and_full_hot_removed +
                remaining_cpu_ms +
                penalty_gate["gate_penalty_ms_est"] +
                gate_recompute_kernel_ms,
            ),
            "cpu_backend_fused_transfer_bytes": {
                "h2d": fused_h2d_bytes,
                "d2h": fused_d2h_bytes,
            },
            "cpu_backend_fused_transfer_ms_at_probe_rates": fused_transfer_ms,
            "cpu_backend_standalone_transfer_bytes": {
                "h2d": standalone_h2d_bytes,
                "d2h": standalone_d2h_bytes,
            },
            "cpu_backend_standalone_transfer_ms_at_probe_rates": standalone_transfer_ms,
            "cpu_backend_fused_existing_gate_output": bound(
                decoded_tokens,
                target_decode_ms,
                ideal_zero_transfer_decode + updown_kernel_ms + fused_transfer_ms,
            ),
            "kernel_ms": {
                "updown_at_mmvq_probe_bandwidth": updown_kernel_ms,
                "gate_updown_recompute_at_mmvq_probe_bandwidth": gate_recompute_kernel_ms,
            },
            "top_pairs_preview": selected[:5],
        })

    artifact = {
        "experiment": "mmvq-fused-transfer-hard-bound",
        "source": {
            "repo": str(args.repo),
            "branch": run_text(["git", "rev-parse", "--abbrev-ref", "HEAD"], args.repo),
            "head": run_text(["git", "rev-parse", "HEAD"], args.repo),
            "status": run_text(["git", "status", "--short"], args.repo),
        },
        "inputs": {
            "bottleneck": str(args.bottleneck),
            "touch_split_case": str(touch_case),
            "mmvq_validation": str(args.mmvq_validation),
            "gate_profile": str(args.gate_profile),
        },
        "requirements": {
            "accepted_decode_ms": accepted_decode_ms,
            "decoded_tokens_estimate": decoded_tokens,
            "target_decode_ms_for_10tok": target_decode_ms,
            "source_page_delta_ms_estimate": source_delta_ms,
            "hot_compute_ms_after_touch": hot_ms,
            "total_decode_call_weighted_gib": total_call_gib,
            "best_cpu_microprobe_gib_s": BEST_CPU_MICROPROBE_GIB_S,
            "mmvq_probe_kernel_gib_s": mmvq_gib_s,
            "probe_h2d_bytes_per_ms": h2d_bytes_per_ms,
            "probe_d2h_bytes_per_ms": d2h_bytes_per_ms,
        },
        "pair_rows": pair_rows,
    }

    best_graph = max(pair_rows, key=lambda r: r["graph_zero_transfer_existing_gate_output"]["tok_s"])
    best_cpu_fused = max(pair_rows, key=lambda r: r["cpu_backend_fused_existing_gate_output"]["tok_s"])
    artifact["decision"] = {
        "sota_changed": False,
        "strict_cold_benchmark_allowed": False,
        "best_graph_zero_transfer_existing_gate_output": {
            "pair_count": best_graph["pair_count"],
            **best_graph["graph_zero_transfer_existing_gate_output"],
        },
        "best_cpu_backend_fused_existing_gate_output": {
            "pair_count": best_cpu_fused["pair_count"],
            **best_cpu_fused["cpu_backend_fused_existing_gate_output"],
        },
        "reason": (
            "This is a hard-bound screen only. A CPU-backend fused skip/write still pays measured H2D/D2H "
            "and cannot be promoted unless it exceeds 10 tok/s with correctness. A graph-level zero-transfer "
            "route is only allowed if it can reuse existing gate output on GPU; recomputing gate or adding "
            "gate payload must be separately bounded."
        ),
        "next_allowed_source_edit": (
            "Only a default-off graph/probe that proves hot-path gate/up/down can stay on GPU without "
            "intermediate H2D/D2H and without sacrificing the accepted gate cache. No standalone CPU-backend "
            "MMVQ skip/write benchmark is allowed from this artifact alone."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
