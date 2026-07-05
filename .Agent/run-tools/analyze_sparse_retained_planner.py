#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_shape(shape: str) -> list[int]:
    return [int(x) for x in shape.split("x") if x]


def read_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mib(n: float) -> float:
    return n / (1024.0 * 1024.0)


def find_pair_row(bound: dict, pair_count: int) -> dict | None:
    for row in bound.get("pair_rows", []):
        if int(row.get("pair_count", -1)) == pair_count:
            return row
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile-json", required=True, type=Path)
    ap.add_argument("--manifest-csv", required=True, type=Path)
    ap.add_argument("--bound-json", required=True, type=Path)
    ap.add_argument("--summary-out", required=True, type=Path)
    ap.add_argument("--n-picks", type=int, default=6)
    ap.add_argument("--copy-gib-s", type=float, default=24.0)
    ap.add_argument("--fast-copy-gib-s", type=float, default=50.0)
    args = ap.parse_args()

    profile = load_json(args.profile_json)
    bound = load_json(args.bound_json)
    manifest = read_manifest(args.manifest_csv)
    pairs = profile.get("pairs", [])
    if not pairs or not manifest:
        raise SystemExit("empty profile or manifest")

    per_layer: dict[int, set[int]] = defaultdict(set)
    for p in pairs:
        per_layer[int(p["layer"])].add(int(p["expert"]))

    up_entries = [r for r in manifest if r["role"] == "up"]
    down_entries = [r for r in manifest if r["role"] == "down"]
    if not up_entries or not down_entries:
        raise SystemExit("manifest must include up and down entries")

    up_expert_bytes = int(up_entries[0]["nbytes"])
    down_expert_bytes = int(down_entries[0]["nbytes"])
    gate_expert_bytes = up_expert_bytes
    down_shape = parse_shape(down_entries[0]["shape"])
    # GGUF down tensor shape is n_ff x n_embd x n_expert for DS4.
    n_embd = down_shape[1] if len(down_shape) >= 2 else 4096
    n_picks = args.n_picks

    active_layers = len(per_layer)
    pair_count = len(pairs)
    ideal_updown_payload = pair_count * (up_expert_bytes + down_expert_bytes)
    ideal_gate_updown_payload = pair_count * (gate_expert_bytes + up_expert_bytes + down_expert_bytes)

    # Existing DS4_HOT-style rectangular mul_mat_id requires K real experts per layer
    # plus P per-pick dummy experts and one prefetch padding slot per active layer.
    rectangular_entries = sum(len(experts) + n_picks + 1 for experts in per_layer.values())
    rectangular_real_entries = sum(len(experts) for experts in per_layer.values())
    rectangular_dummy_entries = rectangular_entries - rectangular_real_entries
    rectangular_updown_payload = rectangular_entries * (up_expert_bytes + down_expert_bytes)
    rectangular_gate_updown_payload = rectangular_entries * (gate_expert_bytes + up_expert_bytes + down_expert_bytes)

    graph_best = bound.get("decision", {}).get("best_graph_zero_transfer_existing_gate_output", {})
    graph_margin_ms = float(graph_best.get("margin_to_10tok_ms", 18.947062554683725))
    graph_tok_s = float(graph_best.get("tok_s", 10.013888843633731))
    graph_decode_ms = float(graph_best.get("decode_ms", 13641.929489845314))
    decoded_tokens_est = graph_tok_s * graph_decode_ms / 1000.0

    # If hot and cold branches live on different backends, at least one branch's
    # [n_embd, P, T] output must cross a backend boundary before add/sum.
    combine_copy_bytes_per_layer_token = n_embd * n_picks * 4
    combine_copy_bytes_decode = combine_copy_bytes_per_layer_token * active_layers * decoded_tokens_est
    combine_copy_gib = combine_copy_bytes_decode / (1024.0 ** 3)
    combine_copy_ms_at_copy = 1000.0 * combine_copy_gib / args.copy_gib_s
    combine_copy_ms_at_fast = 1000.0 * combine_copy_gib / args.fast_copy_gib_s

    pair_bound = find_pair_row(bound, pair_count) or {}
    gate_recompute = pair_bound.get("graph_zero_transfer_gate_recompute", {})
    graph_existing_gate = pair_bound.get("graph_zero_transfer_existing_gate_output", {})

    payload_extra_mib_vs_free = max(0.0, mib(ideal_updown_payload) - 238.0)
    rectangular_extra_mib_vs_free = max(0.0, mib(rectangular_updown_payload) - 238.0)

    rejects: list[str] = []
    if rectangular_updown_payload > ideal_updown_payload:
        rejects.append("current rectangular DS4_HOT-style IDs require per-layer dummy/padding experts and inflate payload above the 544 MiB sparse bound")
    if gate_recompute and float(gate_recompute.get("tok_s", 0.0)) < 10.0:
        rejects.append("gate recompute remains below 10 tok/s in the hard-bound artifact")
    if combine_copy_ms_at_copy >= graph_margin_ms:
        rejects.append("hot/cold backend combine copy lower bound consumes the full graph-zero-transfer margin at conservative bandwidth")
    elif combine_copy_ms_at_fast >= graph_margin_ms:
        rejects.append("hot/cold backend combine copy lower bound consumes the full graph-zero-transfer margin even at fast bandwidth")

    viable_conditions = [
        "gate output is reused from an existing retained CUDA tensor; no gate recompute and no gate weight payload",
        "hot picks are compacted so cold picks do not require P dummy experts per active layer",
        "hot/cold combine is either copy-free or the single unavoidable output copy is proven below the remaining margin including scheduler overhead",
        "all placement/copy facts are verified before any logit-changing benchmark",
    ]

    status = "planner_rejects_current_graph_shapes"
    if not rejects:
        status = "planner_allows_next_probe"

    summary = {
        "artifact": str(args.summary_out),
        "status": status,
        "accepted_sota_unchanged": True,
        "inputs": {
            "profile_json": str(args.profile_json),
            "manifest_csv": str(args.manifest_csv),
            "bound_json": str(args.bound_json),
        },
        "model_shape_inferred": {
            "n_embd": n_embd,
            "n_picks": n_picks,
            "up_expert_bytes": up_expert_bytes,
            "down_expert_bytes": down_expert_bytes,
            "gate_expert_bytes_assumed": gate_expert_bytes,
        },
        "sparse_profile": {
            "pair_count": pair_count,
            "active_layers": active_layers,
            "max_pairs_per_layer": max(len(v) for v in per_layer.values()),
            "per_layer_pair_counts": {str(k): len(v) for k, v in sorted(per_layer.items())},
        },
        "payloads": {
            "ideal_updown_bytes": ideal_updown_payload,
            "ideal_updown_mib": mib(ideal_updown_payload),
            "ideal_gate_updown_bytes": ideal_gate_updown_payload,
            "ideal_gate_updown_mib": mib(ideal_gate_updown_payload),
            "rectangular_entries": rectangular_entries,
            "rectangular_real_entries": rectangular_real_entries,
            "rectangular_dummy_padding_entries": rectangular_dummy_entries,
            "rectangular_updown_bytes": rectangular_updown_payload,
            "rectangular_updown_mib": mib(rectangular_updown_payload),
            "rectangular_gate_updown_bytes": rectangular_gate_updown_payload,
            "rectangular_gate_updown_mib": mib(rectangular_gate_updown_payload),
            "ideal_updown_extra_mib_vs_238_mib_free": payload_extra_mib_vs_free,
            "rectangular_updown_extra_mib_vs_238_mib_free": rectangular_extra_mib_vs_free,
        },
        "hard_bound": {
            "graph_existing_gate_tok_s": graph_existing_gate.get("tok_s", graph_tok_s),
            "graph_existing_gate_margin_ms": graph_existing_gate.get("margin_to_10tok_ms", graph_margin_ms),
            "gate_recompute_tok_s": gate_recompute.get("tok_s"),
            "gate_recompute_margin_ms": gate_recompute.get("margin_to_10tok_ms"),
            "decoded_tokens_est": decoded_tokens_est,
        },
        "hot_cold_combine_copy_lower_bound": {
            "bytes_per_active_layer_per_decode_token": combine_copy_bytes_per_layer_token,
            "decode_bytes": int(round(combine_copy_bytes_decode)),
            "decode_gib": combine_copy_gib,
            "copy_ms_at_gib_s": {
                str(args.copy_gib_s): combine_copy_ms_at_copy,
                str(args.fast_copy_gib_s): combine_copy_ms_at_fast,
            },
            "margin_ms": graph_margin_ms,
            "margin_left_after_copy_ms": {
                str(args.copy_gib_s): graph_margin_ms - combine_copy_ms_at_copy,
                str(args.fast_copy_gib_s): graph_margin_ms - combine_copy_ms_at_fast,
            },
            "note": "This is a lower bound for one backend crossing of the [n_embd, P, T] hot/cold expert output. It excludes scheduler split, launch, event, allocation, and cache effects.",
        },
        "rejects_current_shapes": rejects,
        "conditions_for_next_probe": viable_conditions,
        "next_allowed_source_edit": (
            "Only a default-off DS4 compact sparse-hot placement probe/new op skeleton that proves compact hot picks, "
            "gate-output reuse, and hot/cold combine copy count. Do not use current DS4_HOT rectangular dummy shape."
        ),
    }

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
