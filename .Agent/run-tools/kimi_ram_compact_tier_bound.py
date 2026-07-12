#!/usr/bin/env python3
"""Bound whether a compact RAM tier can matter before runtime work.

Input is a report from kimi_ram_candidate_multidev_screen.py plus an optional
baseline metrics.txt.  The bound is intentionally simple and conservative in
decision making:

- "all_selected" is an optimistic upper bound where every selected candidate
  byte removes exposed SSD/io_uring wait.
- "ram_dominant" is the scheduler-realistic bound where only RAM-dominant
  batches are counted.
- compact_ratio scales RAM H2D bytes, not SSD bytes saved.

If even the optimistic bound cannot reach the target token rate, the RAM
candidate should not proceed to runtime implementation.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


GIB = 1024 ** 3


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen-report", required=True, type=Path)
    ap.add_argument("--baseline-metrics", type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--out-md", required=True, type=Path)
    ap.add_argument("--baseline-decode-ms", type=float)
    ap.add_argument("--decode-runs", type=int)
    ap.add_argument("--target-tok-s", type=float, default=2.0)
    ap.add_argument("--ssd-gib-s", type=float, default=10.4)
    ap.add_argument("--ram-h2d-gib-s", type=float, default=24.0)
    ap.add_argument("--ratios", default="1.0,0.75,0.5,0.35,0.25,0.125")
    return ap.parse_args()


def load_metrics(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def parse_float_metric(metrics: dict[str, str], key: str) -> float | None:
    raw = metrics.get(key)
    if raw is None:
        return None
    m = re.search(r"[-+]?[0-9]*\.?[0-9]+", raw)
    return float(m.group(0)) if m else None


def parse_int_metric(metrics: dict[str, str], key: str) -> int | None:
    raw = metrics.get(key)
    if raw is None:
        return None
    m = re.search(r"\d+", raw)
    return int(m.group(0)) if m else None


def scenario_row(
    name: str,
    baseline_decode_ms: float,
    decode_runs: int,
    source_gib: float,
    ratio: float,
    ssd_gib_s: float,
    ram_h2d_gib_s: float,
) -> dict[str, float | str]:
    ssd_saved_ms = 1000.0 * source_gib / ssd_gib_s
    ram_h2d_ms = 1000.0 * source_gib * ratio / ram_h2d_gib_s
    net_saved_ms = max(0.0, ssd_saved_ms - ram_h2d_ms)
    decode_ms = max(1.0, baseline_decode_ms - net_saved_ms)
    return {
        "scenario": name,
        "source_gib": source_gib,
        "compact_ratio": ratio,
        "ssd_saved_ms_optimistic": ssd_saved_ms,
        "ram_h2d_ms": ram_h2d_ms,
        "net_saved_ms": net_saved_ms,
        "bounded_decode_ms": decode_ms,
        "bounded_tok_s": decode_runs / (decode_ms / 1000.0),
    }


def main() -> int:
    args = parse_args()
    screen = json.loads(args.screen_report.read_text())
    metrics = load_metrics(args.baseline_metrics)
    baseline_decode_ms = args.baseline_decode_ms or parse_float_metric(metrics, "decode_ms")
    decode_runs = args.decode_runs or parse_int_metric(metrics, "decode_runs")
    if baseline_decode_ms is None or decode_runs is None:
        raise SystemExit("Need --baseline-decode-ms/--decode-runs or a metrics.txt with decode_ms/decode_runs")

    batch_summary = screen.get("batch_summary", {})
    selected_gib = float(screen.get("selected_weighted_gib", 0.0))
    hit_candidate_gib = float(batch_summary.get("hit_candidate_gib", selected_gib))
    dominant_gib = float(batch_summary.get("ram_dominant_candidate_gib", 0.0))
    ram_only_gib = float(batch_summary.get("ram_only_candidate_gib", 0.0))
    ratios = [float(x) for x in args.ratios.split(",") if x.strip()]

    rows = []
    for ratio in ratios:
        rows.append(scenario_row(
            "all_selected_optimistic",
            baseline_decode_ms,
            decode_runs,
            selected_gib,
            ratio,
            args.ssd_gib_s,
            args.ram_h2d_gib_s,
        ))
        rows.append(scenario_row(
            "hit_batches_only",
            baseline_decode_ms,
            decode_runs,
            hit_candidate_gib,
            ratio,
            args.ssd_gib_s,
            args.ram_h2d_gib_s,
        ))
        rows.append(scenario_row(
            "ram_dominant_scheduler_realistic",
            baseline_decode_ms,
            decode_runs,
            dominant_gib,
            ratio,
            args.ssd_gib_s,
            args.ram_h2d_gib_s,
        ))

    target_decode_ms = 1000.0 * decode_runs / args.target_tok_s
    needed_saved_ms = max(0.0, baseline_decode_ms - target_decode_ms)
    best_all = max((r for r in rows if r["scenario"] == "all_selected_optimistic"), key=lambda r: float(r["bounded_tok_s"]))
    best_dom = max((r for r in rows if r["scenario"] == "ram_dominant_scheduler_realistic"), key=lambda r: float(r["bounded_tok_s"]))
    decision = "reject_runtime_ab"
    if float(best_all["bounded_tok_s"]) >= args.target_tok_s and float(best_dom["bounded_tok_s"]) >= args.target_tok_s:
        decision = "candidate_can_advance_to_runtime_ab"
    elif float(best_all["bounded_tok_s"]) >= args.target_tok_s:
        decision = "reject_until_batch_dominance_improves"

    result = {
        "kind": "kimi_ram_compact_tier_bound",
        "screen_report": str(args.screen_report),
        "baseline_metrics": str(args.baseline_metrics) if args.baseline_metrics else None,
        "baseline_decode_ms": baseline_decode_ms,
        "decode_runs": decode_runs,
        "baseline_tok_s": decode_runs / (baseline_decode_ms / 1000.0),
        "target_tok_s": args.target_tok_s,
        "target_decode_ms": target_decode_ms,
        "needed_saved_ms": needed_saved_ms,
        "ssd_gib_s": args.ssd_gib_s,
        "ram_h2d_gib_s": args.ram_h2d_gib_s,
        "selected_weighted_gib": selected_gib,
        "hit_candidate_gib": hit_candidate_gib,
        "ram_dominant_candidate_gib": dominant_gib,
        "ram_only_candidate_gib": ram_only_gib,
        "best_all_selected": best_all,
        "best_ram_dominant": best_dom,
        "decision": decision,
        "rows": rows,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Kimi RAM compact-tier bound",
        "",
        "This is an offline bound. It does not change runtime behavior or claim SOTA.",
        "",
        f"- screen report: `{args.screen_report}`",
        f"- baseline decode: `{baseline_decode_ms:.2f} ms / {decode_runs}`",
        f"- baseline token rate: `{result['baseline_tok_s']:.2f}`",
        f"- target token rate: `{args.target_tok_s:.2f}`",
        f"- needed decode saving: `{needed_saved_ms:.2f} ms`",
        f"- SSD bandwidth assumption: `{args.ssd_gib_s:.2f} GiB/s`",
        f"- RAM H2D bandwidth assumption: `{args.ram_h2d_gib_s:.2f} GiB/s`",
        "",
        "## Candidate Bytes",
        "",
        f"- selected weighted bytes: `{selected_gib:.3f} GiB`",
        f"- hit-batch candidate bytes: `{hit_candidate_gib:.3f} GiB`",
        f"- RAM-dominant candidate bytes: `{dominant_gib:.3f} GiB`",
        f"- RAM-only candidate bytes: `{ram_only_gib:.3f} GiB`",
        "",
        "## Bound",
        "",
        "| scenario | ratio | source GiB | SSD saved ms | RAM H2D ms | net saved ms | bounded tok/s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['scenario']}` | `{float(row['compact_ratio']):.3f}` | "
            f"`{float(row['source_gib']):.3f}` | `{float(row['ssd_saved_ms_optimistic']):.2f}` | "
            f"`{float(row['ram_h2d_ms']):.2f}` | `{float(row['net_saved_ms']):.2f}` | "
            f"`{float(row['bounded_tok_s']):.2f}` |"
        )
    lines += [
        "",
        "## Decision",
        "",
        f"- `{decision}`",
        "",
        "Interpretation:",
        "",
        "- If `all_selected_optimistic` cannot reach the target, the profile is too small even under ideal scheduling.",
        "- If only `all_selected_optimistic` reaches the target, improve batch dominance before runtime work.",
        "- A runtime A/B is justified only when the RAM-dominant bound clears the target with margin.",
        "",
    ]
    args.out_md.write_text("\n".join(lines))

    print(f"decision={decision}")
    print(f"baseline_tok_s={result['baseline_tok_s']:.3f} target={args.target_tok_s:.3f}")
    print(f"selected_gib={selected_gib:.3f} dominant_gib={dominant_gib:.3f}")
    print(f"best_all_tok_s={float(best_all['bounded_tok_s']):.3f}")
    print(f"best_dominant_tok_s={float(best_dom['bounded_tok_s']):.3f}")
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
