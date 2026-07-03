#!/usr/bin/env python3
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_CASE = Path(
    "/root/lfz/runs/vendor-ds4-16gb/20260703T100011Z-20260703T100011Z-cpu-fallback-fine-trace-limit2m/france-cpu40-vram0gb"
)

# Harness-derived warm timings from .Agent/run-tools/run_mxfp4_dot_harness.sh 200.
# Repack cost is reported separately and is charged only for rows covered by full 8-row groups.
HARNESS = {
    "up": {"k": 4096, "rows": 2048, "repack_ms": 3.951, "speedup": 1.418},
    "down": {"k": 2048, "rows": 4096, "repack_ms": 2.920, "speedup": 1.485},
}

def kind_for_tensor(tensor: str) -> str:
    if "ffn_up" in tensor:
        return "up"
    if "ffn_down" in tensor:
        return "down"
    return "other"

def main() -> int:
    ap = argparse.ArgumentParser(description="Estimate strict-cold transient MXFP4 8-row repack bound from cpu_chunk_trace.csv")
    ap.add_argument("case_dir", nargs="?", default=str(DEFAULT_CASE))
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    case = Path(args.case_dir)
    trace = case / "cpu_chunk_trace.csv"
    if not trace.exists():
        raise SystemExit(f"missing trace: {trace}")

    by_kind = defaultdict(lambda: {
        "chunks": 0,
        "ms": 0.0,
        "rows": 0,
        "eligible_rows8": 0,
        "tail_rows": 0,
        "groups8": 0,
    })
    chunk_rows = Counter()

    with trace.open(newline="") as f:
        for row in csv.DictReader(f):
            kind = kind_for_tensor(row["tensor"])
            if kind == "other":
                continue
            cne1 = int(row["cne1"])
            rows = int(row["ir0_end"]) - int(row["ir0_start"])
            groups8 = rows // 8
            tail = rows % 8
            key = (kind, "cne1=1" if cne1 == 1 else "cne1>1")
            d = by_kind[key]
            d["chunks"] += 1
            d["ms"] += float(row["ms"])
            d["rows"] += rows
            d["eligible_rows8"] += groups8 * 8
            d["tail_rows"] += tail
            d["groups8"] += groups8
            chunk_rows[rows] += 1

    result = {
        "case_dir": str(case),
        "threads": args.threads,
        "harness_basis": HARNESS,
        "chunk_row_hist_top": chunk_rows.most_common(20),
        "by_kind_cne": {},
        "candidate_totals": {},
    }

    totals = defaultdict(lambda: {"compute_saved_ms_threadsum": 0.0, "repack_ms_threadsum": 0.0, "net_saved_ms_threadsum": 0.0})

    for (kind, cne), d in sorted(by_kind.items()):
        h = HARNESS[kind]
        speedup = h["speedup"]
        repack_per_row = h["repack_ms"] / h["rows"]
        eligible_fraction = d["eligible_rows8"] / d["rows"] if d["rows"] else 0.0
        covered_ms = d["ms"] * eligible_fraction
        compute_saved = covered_ms * (1.0 - 1.0 / speedup)
        repack_ms = d["eligible_rows8"] * repack_per_row
        net = compute_saved - repack_ms
        entry = dict(d)
        entry.update({
            "eligible_fraction": eligible_fraction,
            "covered_ms_threadsum_est": covered_ms,
            "compute_saved_ms_threadsum_est": compute_saved,
            "repack_ms_threadsum_est": repack_ms,
            "net_saved_ms_threadsum_est": net,
            "net_saved_wall_ms_est_threads": net / args.threads,
        })
        result["by_kind_cne"][f"{kind}:{cne}"] = entry
        for name in (kind, "all"):
            totals[name]["compute_saved_ms_threadsum"] += compute_saved
            totals[name]["repack_ms_threadsum"] += repack_ms
            totals[name]["net_saved_ms_threadsum"] += net

    for name, d in sorted(totals.items()):
        result["candidate_totals"][name] = {
            **d,
            "net_saved_wall_ms_est_threads": d["net_saved_ms_threadsum"] / args.threads,
        }

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
