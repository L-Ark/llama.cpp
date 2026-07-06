#!/usr/bin/env python3
import argparse
import csv
import json
import pathlib
import re


IOURING_RE = re.compile(r"iouring_bytes=(\d+).*?iouring_wait_us=(\d+)")
PINNED_RE = re.compile(r"copies=(\d+).*?h2d=([0-9.]+) ms")


def gib(nbytes):
    return nbytes / 1024**3


def load_metrics(run_dir):
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


def tensor_expert_bytes(run_dir):
    out = {}
    with (run_dir / "route-profile.csv").open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            out[row["tensor"]] = int(row["expert_bytes"])
    return out


def parse_profile(run_dir, bytes_by_tensor):
    rows = []
    with (run_dir / "up-gate-profile.csv").open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            up_jobs = int(row.get("up_stage_jobs", 0) or 0)
            gate_jobs = int(row.get("gate_stage_jobs", 0) or 0)
            up_tensor = row["up_tensor"]
            gate_tensor = row["gate_tensor"]
            up_bytes_each = bytes_by_tensor.get(up_tensor, 0)
            gate_bytes_each = bytes_by_tensor.get(gate_tensor, 0)
            current_jobs = up_jobs + gate_jobs
            fused_jobs = max(up_jobs, gate_jobs)
            current_bytes = up_jobs * up_bytes_each + gate_jobs * gate_bytes_each
            # Fusing same expert up+gate does not reduce semantic bytes; it only
            # makes each submitted IO/H2D unit larger and cuts per-job overhead.
            fused_bytes = current_bytes
            parallel = int(row.get("parallel_stage", 0) or 0) or int(row.get("parallel_up_gate", 0) or 0)
            up_ms = float(row.get("up_ms", 0.0) or 0.0)
            gate_ms = float(row.get("gate_ms", 0.0) or 0.0)
            exposed_ms = max(up_ms, gate_ms) if parallel else up_ms + gate_ms
            rows.append({
                "seq": int(row["seq"]),
                "mode": row.get("mode", ""),
                "up_tensor": up_tensor,
                "gate_tensor": gate_tensor,
                "up_stage_jobs": up_jobs,
                "gate_stage_jobs": gate_jobs,
                "current_jobs": current_jobs,
                "fused_jobs": fused_jobs,
                "job_reduction": current_jobs - fused_jobs,
                "current_bytes": current_bytes,
                "fused_bytes": fused_bytes,
                "up_ms": up_ms,
                "gate_ms": gate_ms,
                "wall_ms": float(row.get("wall_ms", 0.0) or 0.0),
                "exposed_upgate_ms": exposed_ms,
                "parallel": bool(parallel),
            })
    return rows


def parse_iouring(metrics):
    text = str(metrics.get("expert_pack_0", ""))
    match = IOURING_RE.search(text)
    if not match:
        return 0, 0
    return int(match.group(1)), int(match.group(2))


def parse_h2d(metrics):
    down = str(metrics.get("pinned_staging_4", ""))
    gate = str(metrics.get("pinned_staging_6", ""))
    out = {}
    for name, text in (("down", down), ("gate", gate)):
        match = PINNED_RE.search(text)
        if match:
            out[name] = {"copies": int(match.group(1)), "h2d_ms": float(match.group(2))}
    return out


def summarize(rows, metrics, peak_gib_s):
    current_jobs = sum(r["current_jobs"] for r in rows)
    fused_jobs = sum(r["fused_jobs"] for r in rows)
    current_bytes = sum(r["current_bytes"] for r in rows)
    exposed_ms = sum(r["exposed_upgate_ms"] for r in rows)
    wall_ms = sum(r["wall_ms"] for r in rows)
    paired_rows = sum(1 for r in rows if r["up_stage_jobs"] > 0 and r["gate_stage_jobs"] > 0)
    any_stage_rows = sum(1 for r in rows if r["current_jobs"] > 0)
    ideal_ms_at_peak = (gib(current_bytes) / peak_gib_s) * 1000.0 if peak_gib_s > 0 else 0.0
    current_exposed_gib_s = gib(current_bytes) / (exposed_ms / 1000.0) if exposed_ms > 0 else 0.0
    ideal_saved_ms = max(0.0, exposed_ms - ideal_ms_at_peak)
    decode_ms = float(metrics.get("decode_ms", 0.0) or 0.0)
    decode_runs = int(metrics.get("decode_runs", 0) or 0)
    current_tps = float(metrics.get("token_rate", 0.0) or 0.0)
    ideal_decode_ms = max(0.0, decode_ms - ideal_saved_ms)
    ideal_tps = decode_runs / (ideal_decode_ms / 1000.0) if ideal_decode_ms > 0 else 0.0
    return {
        "prompt_id": metrics.get("prompt_id", ""),
        "quality": metrics.get("quality", ""),
        "decode_ms": decode_ms,
        "decode_runs": decode_runs,
        "current_token_rate": current_tps,
        "current_jobs": current_jobs,
        "fused_jobs": fused_jobs,
        "job_reduction": current_jobs - fused_jobs,
        "job_reduction_pct": 100.0 * (current_jobs - fused_jobs) / current_jobs if current_jobs else 0.0,
        "current_upgate_gib": gib(current_bytes),
        "exposed_upgate_ms": exposed_ms,
        "upgate_wall_sum_ms": wall_ms,
        "current_exposed_upgate_gib_s": current_exposed_gib_s,
        "ideal_upgate_ms_at_peak": ideal_ms_at_peak,
        "ideal_saved_ms_at_peak": ideal_saved_ms,
        "ideal_token_rate_at_peak": ideal_tps,
        "paired_stage_rows": paired_rows,
        "any_stage_rows": any_stage_rows,
        "paired_stage_row_pct": 100.0 * paired_rows / any_stage_rows if any_stage_rows else 0.0,
    }


def write_report(out, run_dir, summary, metrics, rows, peak_gib_s):
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_dir": str(run_dir),
        "peak_gib_s": peak_gib_s,
        "summary": summary,
        "metrics_subset": {
            "expert_pack_0": metrics.get("expert_pack_0"),
            "expert_pack_iouring_0": metrics.get("expert_pack_iouring_0"),
            "pinned_staging_4": metrics.get("pinned_staging_4"),
            "pinned_staging_6": metrics.get("pinned_staging_6"),
            "vram_upgate_0": metrics.get("vram_upgate_0"),
            "memory.peak": metrics.get("memory.peak"),
        },
        "top_rows_by_jobs": sorted(rows, key=lambda r: r["job_reduction"], reverse=True)[:40],
    }
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Kimi fused up/gate IO bound",
        "",
        "This is a dev-only offline bound. It does not use held-out test prompts and does not change runtime code.",
        "",
        f"- run_dir: `{run_dir}`",
        f"- prompt: `{summary['prompt_id']}`",
        f"- quality: `{summary['quality']}`",
        f"- current token rate: `{summary['current_token_rate']:.2f} tok/s`",
        f"- decode: `{summary['decode_ms']:.2f} ms / {summary['decode_runs']} tokens`",
        f"- peak bandwidth assumption: `{peak_gib_s:.2f} GiB/s`",
        "",
        "## Bound",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| current up+gate staged bytes | {summary['current_upgate_gib']:.2f} GiB |",
        f"| current up+gate staged jobs | {summary['current_jobs']} |",
        f"| fused up+gate jobs | {summary['fused_jobs']} |",
        f"| job reduction | {summary['job_reduction']} ({summary['job_reduction_pct']:.1f}%) |",
        f"| rows with both up and gate staged | {summary['paired_stage_rows']}/{summary['any_stage_rows']} ({summary['paired_stage_row_pct']:.1f}%) |",
        f"| current exposed up+gate bandwidth | {summary['current_exposed_upgate_gib_s']:.2f} GiB/s |",
        f"| current exposed up+gate time | {summary['exposed_upgate_ms']:.2f} ms |",
        f"| ideal up+gate time at peak | {summary['ideal_upgate_ms_at_peak']:.2f} ms |",
        f"| ideal saved decode time at peak | {summary['ideal_saved_ms_at_peak']:.2f} ms |",
        f"| ideal token rate if all saved | {summary['ideal_token_rate_at_peak']:.2f} tok/s |",
        "",
        "## Interpretation",
        "",
        "- Fusing up+gate can reduce job count but cannot reduce moved bytes.",
        "- The `ideal token rate` row is an optimistic upper bound: it assumes all exposed up+gate time above pure IO peak disappears from decode.",
        "- If this upper bound remains far below 5 tok/s, fused up/gate pack is not enough as a primary path.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Offline bound for fusing same-expert up+gate pack reads.")
    parser.add_argument("--run-dir", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--peak-gib-s", type=float, default=10.4)
    args = parser.parse_args()

    metrics = load_metrics(args.run_dir)
    rows = parse_profile(args.run_dir, tensor_expert_bytes(args.run_dir))
    summary = summarize(rows, metrics, args.peak_gib_s)
    write_report(args.out, args.run_dir, summary, metrics, rows, args.peak_gib_s)
    print(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
