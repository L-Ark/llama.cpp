#!/usr/bin/env python3
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_json(path):
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-csv", required=True)
    ap.add_argument("--default-top1-summary", required=True)
    ap.add_argument("--probe-top1-summary", required=True)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--source-head", required=True)
    ap.add_argument("--summary-out", required=True)
    args = ap.parse_args()

    probe_csv = Path(args.probe_csv)
    rows = list(csv.DictReader(probe_csv.open("r", encoding="utf-8")))
    if not rows:
        raise SystemExit(f"empty probe csv: {probe_csv}")

    def i(row, key):
        value = row.get(key, "")
        return int(value) if value not in ("", None) else 0

    row_count = len(rows)
    layers = sorted({i(r, "il") for r in rows})
    active_layers = sorted({i(r, "il") for r in rows if i(r, "compact_candidate")})
    mix_tokens = Counter(i(r, "mix_tokens") for r in rows)
    selected_ne0 = Counter(i(r, "selected_ne0") for r in rows)
    full_gate_buft = Counter(r["full_gate_buft"] for r in rows)
    full_up_buft = Counter(r["full_up_buft"] for r in rows)
    full_down_buft = Counter(r["full_down_buft"] for r in rows)
    cur_buft = Counter(r["cur_buft"] for r in rows)
    selected_buft = Counter(r["selected_buft"] for r in rows)
    weights_buft = Counter(r["weights_buft"] for r in rows)

    bool_sums = {
        key: sum(i(r, key) for r in rows)
        for key in [
            "hot_active",
            "hot_ready",
            "hot_dispatch_enabled",
            "dispatch_dual",
            "selected_matches_hot_picks",
            "graph_gate_output_input_available",
            "current_hot_path_computes_gate",
            "compact_candidate",
        ]
    }

    combine_bytes_total = sum(i(r, "combine_bytes_call") for r in rows)
    combine_bytes_decode_like = sum(i(r, "combine_bytes_call") for r in rows if i(r, "mix_tokens") == 1)
    rectangular_updown_bytes_max = max(i(r, "rectangular_updown_bytes_layer") for r in rows)
    rectangular_entries_max = max(i(r, "rectangular_entries_layer") for r in rows)
    full_up_expert_bytes = Counter(i(r, "full_up_expert_bytes") for r in rows)
    full_down_expert_bytes = Counter(i(r, "full_down_expert_bytes") for r in rows)

    default_top1 = read_json(args.default_top1_summary)
    probe_top1 = read_json(args.probe_top1_summary)
    top1_ok = (
        default_top1
        and probe_top1
        and default_top1.get("status") == "pass"
        and probe_top1.get("status") == "pass"
        and default_top1.get("same_top1") == default_top1.get("n_tokens")
        and probe_top1.get("same_top1") == probe_top1.get("n_tokens")
    )

    verdict_reasons = []
    if bool_sums["graph_gate_output_input_available"] == 0:
        verdict_reasons.append("build_expert_mix receives no retained graph-level gate output tensor")
    if bool_sums["hot_active"] == 0:
        verdict_reasons.append("accepted path has no active DS4 hot manager state")
    non_cuda_full_up = row_count - full_up_buft.get("CUDA0", 0)
    non_cuda_full_down = row_count - full_down_buft.get("CUDA0", 0)
    if non_cuda_full_up or non_cuda_full_down:
        verdict_reasons.append(
            "accepted path still uses non-CUDA full up/down expert tensors for many observed rows "
            f"(up_non_cuda={non_cuda_full_up}, down_non_cuda={non_cuda_full_down})"
        )
    if rectangular_updown_bytes_max == 0:
        verdict_reasons.append("no current hot rectangular payload is active in this accepted-path probe")

    status = "validated_probe_rejects_current_accepted_graph_as_sparse_retained_route"
    if not top1_ok:
        status = "invalid_probe_correctness_failed"

    summary = {
        "status": status,
        "source_head": args.source_head,
        "run_root": args.run_root,
        "probe_csv": str(probe_csv),
        "row_count": row_count,
        "layers_observed": {"count": len(layers), "first": layers[:5], "last": layers[-5:]},
        "active_compact_candidate_layers": {"count": len(active_layers), "first": active_layers[:5], "last": active_layers[-5:]},
        "mix_tokens_counts": dict(sorted(mix_tokens.items())),
        "selected_ne0_counts": dict(sorted(selected_ne0.items())),
        "backend_counts": {
            "cur_buft": dict(cur_buft),
            "selected_buft": dict(selected_buft),
            "weights_buft": dict(weights_buft),
            "full_gate_buft": dict(full_gate_buft),
            "full_up_buft": dict(full_up_buft),
            "full_down_buft": dict(full_down_buft),
        },
        "bool_sums": bool_sums,
        "full_up_expert_bytes_counts": dict(sorted(full_up_expert_bytes.items())),
        "full_down_expert_bytes_counts": dict(sorted(full_down_expert_bytes.items())),
        "rectangular_entries_max": rectangular_entries_max,
        "rectangular_updown_bytes_max": rectangular_updown_bytes_max,
        "combine_bytes_total": combine_bytes_total,
        "combine_bytes_decode_like_mix1": combine_bytes_decode_like,
        "default_top1": default_top1,
        "probe_top1": probe_top1,
        "top1_ok": bool(top1_ok),
        "verdict_reasons": verdict_reasons,
        "decision": (
            "Keep the default-off probe as diagnostic infrastructure. The accepted graph path still has no retained "
            "gate tensor and most observed full up/down expert tensors are CPU_Mapped/CUDA_Host rather than CUDA0, "
            "so this does not yet prove a compact zero-transfer sparse retained route. The next source work must "
            "introduce or prove a real retained CUDA gate/up/down branch before any logit-changing benchmark."
        ),
    }

    out = Path(args.summary_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if top1_ok else 1)


if __name__ == "__main__":
    main()
