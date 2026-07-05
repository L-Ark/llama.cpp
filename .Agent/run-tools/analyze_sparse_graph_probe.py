#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def as_int(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    if value == "":
        return 0
    return int(value)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-csv", required=True, type=Path)
    ap.add_argument("--profile-json", required=True, type=Path)
    ap.add_argument("--summary-out", required=True, type=Path)
    args = ap.parse_args()

    profile = json.loads(args.profile_json.read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = []
    if args.probe_csv.exists():
        with args.probe_csv.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    roles = Counter(row.get("tensor_role", "unknown") for row in rows)
    src1_h2d_bytes = sum(as_int(row, "src1_h2d_bytes") for row in rows)
    dst_d2h_bytes = sum(as_int(row, "dst_d2h_bytes") for row in rows)
    scatter_rows = sum(1 for row in rows if as_int(row, "scatter_to_cpu") != 0)
    retained_rows = sum(1 for row in rows if as_int(row, "retains_gpu_output") != 0)
    handle_rows = sum(1 for row in rows if as_int(row, "returns_gpu_handle") != 0)
    zero_ready_rows = sum(1 for row in rows if as_int(row, "zero_transfer_ready") != 0)
    cache_hit_rows = sum(1 for row in rows if as_int(row, "src0_from_cache") != 0)
    cache_insert_rows = sum(1 for row in rows if as_int(row, "src0_cache_inserted") != 0)

    reasons: list[str] = []
    if not rows:
        status = "inconclusive_no_stream_rows"
        reasons.append("Probe CSV has no rows, so current dataflow could not be audited.")
    else:
        if src1_h2d_bytes > 0:
            reasons.append("one-stream still copies F32 src1 activations from host to GPU.")
        if dst_d2h_bytes > 0:
            reasons.append("one-stream still copies dst output from GPU to host.")
        if scatter_rows > 0:
            reasons.append("one-stream still scatters GPU output into CPU dst.")
        if retained_rows == 0:
            reasons.append("probe observed no retained reusable GPU output.")
        if handle_rows == 0:
            reasons.append("probe observed no returned GPU output handle.")
        if roles.get("up", 0) == 0 or roles.get("down", 0) == 0:
            reasons.append("accepted path does not stream sparse up/down pair rows through this helper.")
        status = "passes_zero_transfer_probe" if not reasons and zero_ready_rows == len(rows) else "rejected_current_dataflow"

    summary = {
        "artifact": str(args.summary_out),
        "probe_csv": str(args.probe_csv),
        "profile_json": str(args.profile_json),
        "status": status,
        "accepted_sota_unchanged": True,
        "profile": {
            "format": profile.get("format"),
            "top_n": profile.get("top_n"),
            "pairs": len(profile.get("pairs", [])),
            "payload_bytes": profile.get("payload_bytes"),
            "payload_mib": profile.get("payload_mib"),
            "active_layers": profile.get("active_layers"),
            "max_pairs_per_layer": profile.get("max_pairs_per_layer"),
        },
        "probe_rows": len(rows),
        "roles": dict(sorted(roles.items())),
        "src1_h2d_bytes": src1_h2d_bytes,
        "dst_d2h_bytes": dst_d2h_bytes,
        "scatter_rows": scatter_rows,
        "retained_gpu_output_rows": retained_rows,
        "returned_gpu_handle_rows": handle_rows,
        "zero_transfer_ready_rows": zero_ready_rows,
        "src0_cache_hit_rows": cache_hit_rows,
        "src0_cache_insert_rows": cache_insert_rows,
        "zero_transfer_ready": status == "passes_zero_transfer_probe",
        "reasons": reasons,
        "decision": (
            "Do not implement a logit-changing sparse pair graph path on top of the current CPU-backend "
            "one-stream helper. A future route must first add a real graph-level retained GPU tensor or "
            "equivalent zero-transfer buffer for gate/up/down."
        ),
    }

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if status != "inconclusive_no_stream_rows" else 1


if __name__ == "__main__":
    raise SystemExit(main())
