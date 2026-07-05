#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path


TOP_N = [64, 128, 256, 384, 512, 768, 1024]
SLOT_MIB = 4.25
CUDA_FREE_MIB_ACCEPTED = 238.0
GATE_MISS_COST_MS = 1.024052203321064
BEST_CPU_MICROPROBE_GIB_S = 74.719
RAW_WARP_GPU_GIB_S = 74.590
TRANSPOSED_WARP_GPU_GIB_S = 72.434
REPACK_SPEEDUP = {"up": 1.425, "down": 1.499}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_text(cmd: list[str], cwd: Path) -> str:
    return subprocess.check_output(cmd, cwd=cwd, text=True).strip()


def tok_s(tokens: float, decode_ms: float) -> float:
    return tokens / (decode_ms / 1000.0)


def gate_scores(path: Path) -> list[int]:
    scores = []
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
    # Accepted path preloads top3000 gate entries. Stealing slots drops the tail of that list.
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


def read_decode_rows(path: Path, hot_field: str = "fallback_us") -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tensor = row["tensor"]
            if row["phase"] != "decode" or ("ffn_up_exps" not in tensor and "ffn_down_exps" not in tensor):
                continue
            role = "up" if "ffn_up_exps" in tensor else "down"
            calls = int(row["calls"])
            expert_bytes = int(row["expert_bytes"])
            rows.append({
                "tensor": tensor,
                "role": role,
                "expert_idx": int(row["expert_idx"]),
                "calls": calls,
                "expert_bytes": expert_bytes,
                "hot_ms": int(row[hot_field]) / 1000.0,
                "touch_ms": int(row.get("touch_us", 0)) / 1000.0,
                "call_weighted_gib": calls * expert_bytes / (1024**3),
                "payload_mib": expert_bytes / (1024**2),
            })
    rows.sort(key=lambda r: r["hot_ms"], reverse=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--bottleneck", type=Path, required=True)
    ap.add_argument("--gate-profile", type=Path, required=True)
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args()

    bottleneck = load_json(args.bottleneck)
    key = bottleneck["key_numbers"]
    full_case = Path(bottleneck["runs"]["full_component_trace"]["case_dir"])
    touch_case = Path(bottleneck["runs"]["touch_split_trace"]["case_dir"])

    full_rows = read_decode_rows(full_case / "fallback_profile.csv")
    touch_rows = read_decode_rows(touch_case / "fallback_profile.csv")
    scores = gate_scores(args.gate_profile)

    accepted_decode_ms = key["accepted_decode_window_ms"]
    decoded_tokens = key["accepted_decoded_tokens_estimate"]
    target_decode_ms = key["target_10tok_decode_ms"]
    source_delta_ms = key["decode_source_page_delta_ms_estimate"]
    hot_ms = key["touch_split_decode_hot_fallback_ms_after_touch"]
    total_call_gib = sum(r["call_weighted_gib"] for r in touch_rows)
    hot_by_role = {}
    for role in ["up", "down"]:
        role_ms = sum(r["hot_ms"] for r in touch_rows if r["role"] == role)
        hot_by_role[role] = role_ms

    base_after_source_and_full_hot_removed = accepted_decode_ms - source_delta_ms - hot_ms

    def bound_from_decode(decode_ms: float) -> dict:
        return {
            "decode_ms": decode_ms,
            "tok_s": tok_s(decoded_tokens, decode_ms),
            "margin_to_10tok_ms": target_decode_ms - decode_ms,
        }

    all_cpu_microprobe_hot_ms = total_call_gib / BEST_CPU_MICROPROBE_GIB_S * 1000.0
    repack_hot_ms = (
        hot_by_role["up"] / REPACK_SPEEDUP["up"] +
        hot_by_role["down"] / REPACK_SPEEDUP["down"]
    )

    topn_rows = []
    for n in TOP_N:
        selected = touch_rows[:n]
        hot_save = sum(r["hot_ms"] for r in selected)
        call_gib = sum(r["call_weighted_gib"] for r in selected)
        payload_mib = sum(r["payload_mib"] for r in selected)
        remaining_gib = total_call_gib - call_gib
        remaining_cpu_microprobe_ms = remaining_gib / BEST_CPU_MICROPROBE_GIB_S * 1000.0
        penalty = gate_penalty_ms(scores, payload_mib)

        ideal_no_top_kernel_ms = (
            base_after_source_and_full_hot_removed +
            remaining_cpu_microprobe_ms +
            penalty["gate_penalty_ms_est"]
        )
        allowed_top_kernel_ms = target_decode_ms - ideal_no_top_kernel_ms
        required_top_kernel_gib_s = call_gib / (allowed_top_kernel_ms / 1000.0) if allowed_top_kernel_ms > 0 else None

        raw_gpu_kernel_ms = call_gib / RAW_WARP_GPU_GIB_S * 1000.0
        transposed_gpu_kernel_ms = call_gib / TRANSPOSED_WARP_GPU_GIB_S * 1000.0
        topn_rows.append({
            "top_n": n,
            "hot_save_ms_zero_kernel": hot_save,
            "call_weighted_gib": call_gib,
            "payload_mib": payload_mib,
            "gate_penalty": penalty,
            "remaining_cpu_microprobe_ms": remaining_cpu_microprobe_ms,
            "ideal_zero_top_kernel": bound_from_decode(ideal_no_top_kernel_ms),
            "allowed_top_kernel_ms_for_10tok": allowed_top_kernel_ms,
            "required_top_kernel_gib_s": required_top_kernel_gib_s,
            "raw_warp_gpu_kernel_ms_at_74_590_gib_s": raw_gpu_kernel_ms,
            "raw_warp_gpu_path": bound_from_decode(ideal_no_top_kernel_ms + raw_gpu_kernel_ms),
            "transposed_warp_gpu_kernel_ms_at_72_434_gib_s": transposed_gpu_kernel_ms,
            "transposed_warp_gpu_path": bound_from_decode(ideal_no_top_kernel_ms + transposed_gpu_kernel_ms),
        })

    candidate_rows = {
        "source_page_only": bound_from_decode(accepted_decode_ms - source_delta_ms),
        "source_page_plus_all_cpu_microprobe_best": bound_from_decode(
            accepted_decode_ms - source_delta_ms - hot_ms + all_cpu_microprobe_hot_ms
        ),
        "source_page_plus_persistent_repack_hot_compute": bound_from_decode(
            accepted_decode_ms - source_delta_ms - hot_ms + repack_hot_ms
        ),
        "source_page_plus_full_fallback_removed": bound_from_decode(
            accepted_decode_ms - source_delta_ms - hot_ms
        ),
    }

    artifact = {
        "experiment": "current-head-exact-decode-updown-candidate-screen",
        "source": {
            "repo": str(args.repo),
            "branch": run_text(["git", "rev-parse", "--abbrev-ref", "HEAD"], args.repo),
            "head": run_text(["git", "rev-parse", "HEAD"], args.repo),
            "status": run_text(["git", "status", "--short"], args.repo),
        },
        "inputs": {
            "bottleneck": str(args.bottleneck),
            "full_component_case": str(full_case),
            "touch_split_case": str(touch_case),
            "gate_profile": str(args.gate_profile),
        },
        "requirements": {
            "accepted_decode_ms": accepted_decode_ms,
            "decoded_tokens_estimate": decoded_tokens,
            "target_decode_ms_for_10tok": target_decode_ms,
            "required_saving_ms": key["required_decode_saving_ms_for_10tok"],
            "source_page_delta_ms_estimate": source_delta_ms,
            "hot_compute_ms_after_touch": hot_ms,
            "total_decode_call_weighted_gib": total_call_gib,
            "best_cpu_microprobe_gib_s": BEST_CPU_MICROPROBE_GIB_S,
            "raw_warp_gpu_gib_s": RAW_WARP_GPU_GIB_S,
            "transposed_warp_gpu_gib_s": TRANSPOSED_WARP_GPU_GIB_S,
            "cuda_free_mib_reference": CUDA_FREE_MIB_ACCEPTED,
            "gate_miss_cost_ms": GATE_MISS_COST_MS,
        },
        "candidate_rows": candidate_rows,
        "topn_hot_residual_rows": topn_rows,
        "screening_decision": {
            "sota_changed": False,
            "immediate_runtime_patch_allowed": False,
            "reason": (
                "No candidate has enough measured evidence to satisfy the 10 tok/s hard bound. "
                "Source/page-only reaches only 9.22 tok/s. Source plus best existing CPU microprobe or persistent repack remains below 10. "
                "The closest mixed topN path requires a new exact top128-class kernel around 245 GiB/s plus source/page elimination and CPU microprobe-level remaining compute; existing raw/transposed exact GPU probes are only about 72-75 GiB/s."
            ),
            "only_reopen_conditions": [
                "Prove a new exact top128/top256 hot-residual kernel above the required GiB/s threshold with fixed-text top1 stability.",
                "Recover enough VRAM for at least top128 payload without material gate-cache penalty, or include the measured gate penalty in the bound.",
                "Prove an exact source/page elimination or overlap mechanism that does not repeat the closed synchronous prefetch/page-touch/packmmap/direct families.",
                "Keep total integration overhead inside the remaining 10 tok/s margin."
            ],
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    print(json.dumps({
        "candidate_rows": candidate_rows,
        "best_topn_rows": topn_rows[:4],
        "decision": artifact["screening_decision"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
