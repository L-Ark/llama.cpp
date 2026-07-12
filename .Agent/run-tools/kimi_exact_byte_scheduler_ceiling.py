#!/usr/bin/env python3
"""Estimate the ceiling of exact-byte scheduling without expert byte reduction."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, median
from typing import Any


IOURING_BYTES_RE = re.compile(r"iouring_bytes=(\d+)")


def parse_expert_pack_bytes(metrics: dict[str, Any]) -> int:
    text = str(metrics.get("expert_pack_0", ""))
    match = IOURING_BYTES_RE.search(text)
    return int(match.group(1)) if match else 0


def load_metrics(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "metrics.json"
    if not path.exists():
        raise RuntimeError(f"missing metrics.json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_ttft_trace(path: Path) -> dict[str, Any]:
    out = {
        "runtime_load_events": 0,
        "runtime_load_bytes": 0,
        "runtime_load_copy_ms_sum": 0.0,
        "call_upgate_ms_sum": 0.0,
        "call_down_ms_sum": 0.0,
        "call_upgate_events": 0,
        "call_down_events": 0,
    }
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            op = row.get("op", "")
            try:
                expert_bytes = int(row.get("expert_bytes", "0") or 0)
                copy_ms = float(row.get("copy_ms", "0") or 0.0)
            except ValueError:
                continue
            if op == "runtime_load":
                if int(row.get("cache_hit", "0") or 0) == 0:
                    out["runtime_load_events"] += 1
                    out["runtime_load_bytes"] += expert_bytes
                    out["runtime_load_copy_ms_sum"] += copy_ms
            elif op == "call_upgate":
                out["call_upgate_events"] += 1
                out["call_upgate_ms_sum"] += copy_ms
            elif op == "call_down":
                out["call_down_events"] += 1
                out["call_down_ms_sum"] += copy_ms
    return out


def prompt_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "metrics.json").exists())


def tok_s(ms_per_token: float) -> float:
    return 1000.0 / ms_per_token if ms_per_token > 0 else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = (len(vals) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - idx) + vals[hi] * (idx - lo)


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    target = result["target_tok_s"]
    lines = [
        "# Kimi exact-byte scheduler ceiling",
        "",
        "This is an offline ceiling analysis. It does not change runtime behavior or claim SOTA.",
        "",
        f"- profile root: `{result['profile_root']}`",
        f"- movement bandwidth ceiling: `{result['bandwidth_gib_s']:.2f} GiB/s`",
        f"- all-hit MoE floor: `{result['all_hit_floor_ms_per_token']:.1f} ms/token`",
        "",
        "## Summary",
        "",
        f"- prompts: `{result['summary']['prompts']}`",
        f"- measured mean token rate: `{result['summary']['measured_mean_tok_s']:.3f} tok/s`",
        f"- transfer-only mean ceiling: `{result['summary']['transfer_only_mean_tok_s']:.3f} tok/s`",
        f"- floor+transfer mean ceiling: `{result['summary']['floor_plus_transfer_mean_tok_s']:.3f} tok/s`",
        f"- best prompt floor+transfer ceiling: `{result['summary']['floor_plus_transfer_max_tok_s']:.3f} tok/s`",
        f"- worst prompt floor+transfer ceiling: `{result['summary']['floor_plus_transfer_min_tok_s']:.3f} tok/s`",
        f"- mean byte ratio needed for {target:.1f} tok/s after floor: `{result['summary']['mean_required_byte_ratio_for_target_tps']:.3f}x`",
        "",
        "## Per Prompt",
        "",
        f"| prompt | measured tok/s | moved GiB/token | transfer-only tok/s | floor+transfer tok/s | required ratio for {target:.1f} tok/s | call wall ms/token | runtime-load copy ms/token |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["prompts"]:
        lines.append(
            f"| `{row['prompt_id']}` | `{row['measured_tok_s']:.3f}` | "
            f"`{row['moved_gib_per_token']:.3f}` | `{row['transfer_only_tok_s']:.3f}` | "
            f"`{row['floor_plus_transfer_tok_s']:.3f}` | `{row['required_byte_ratio_for_target_tps']:.3f}` | "
            f"`{row['call_wall_ms_per_token']:.1f}` | `{row['runtime_load_copy_ms_per_token']:.1f}` |"
        )
    lines += [
        "",
        "## Decision",
        "",
        result["decision"],
        "",
        "## Reproduce",
        "",
        "```bash",
        result["reproduce_command"],
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--bandwidth-gib-s", type=float, default=10.4)
    parser.add_argument("--all-hit-floor-ms-per-token", type=float, default=40.1)
    parser.add_argument("--target-tok-s", type=float, default=5.0)
    args = parser.parse_args()

    bw_bytes_per_ms = args.bandwidth_gib_s * (1024 ** 3) / 1000.0
    target_ms = 1000.0 / args.target_tok_s
    transfer_budget_ms = max(0.0, target_ms - args.all_hit_floor_ms_per_token)

    rows = []
    for run_dir in prompt_dirs(args.profile_root):
        metrics = load_metrics(run_dir)
        decode_runs = int(metrics.get("decode_runs", 0) or 0)
        decode_ms = float(metrics.get("decode_ms", 0.0) or 0.0)
        measured_tok_s = float(metrics.get("token_rate", 0.0) or 0.0)
        moved_bytes = parse_expert_pack_bytes(metrics)
        ttft = parse_ttft_trace(run_dir / "ttft-trace.csv")
        tokens = max(decode_runs, 1)
        moved_gib_per_token = moved_bytes / tokens / (1024 ** 3)
        transfer_ms_per_token = (moved_bytes / tokens) / bw_bytes_per_ms
        floor_plus_transfer_ms = args.all_hit_floor_ms_per_token + transfer_ms_per_token
        required_bytes_per_token = transfer_budget_ms * bw_bytes_per_ms
        required_ratio = required_bytes_per_token / max(moved_bytes / tokens, 1)
        call_wall = ttft["call_upgate_ms_sum"] + ttft["call_down_ms_sum"]
        rows.append({
            "prompt_id": run_dir.name,
            "decode_runs": decode_runs,
            "decode_ms": decode_ms,
            "measured_tok_s": measured_tok_s,
            "moved_bytes": moved_bytes,
            "moved_gib_per_token": moved_gib_per_token,
            "transfer_only_ms_per_token": transfer_ms_per_token,
            "transfer_only_tok_s": tok_s(transfer_ms_per_token),
            "floor_plus_transfer_ms_per_token": floor_plus_transfer_ms,
            "floor_plus_transfer_tok_s": tok_s(floor_plus_transfer_ms),
            "required_byte_ratio_for_target_tps": required_ratio,
            "runtime_load_events_per_token": ttft["runtime_load_events"] / tokens,
            "runtime_load_copy_ms_per_token": ttft["runtime_load_copy_ms_sum"] / tokens,
            "call_wall_ms_per_token": call_wall / tokens,
            "call_events_per_token": (ttft["call_upgate_events"] + ttft["call_down_events"]) / tokens,
        })

    if not rows:
        raise RuntimeError(f"no prompt metrics found under {args.profile_root}")

    floor_rates = [r["floor_plus_transfer_tok_s"] for r in rows]
    required_ratios = [r["required_byte_ratio_for_target_tps"] for r in rows]
    measured_rates = [r["measured_tok_s"] for r in rows]
    transfer_rates = [r["transfer_only_tok_s"] for r in rows]
    decision = (
        f"Exact-byte scheduling without byte reduction is not a primary {args.target_tok_s:.1f} tok/s path: "
        f"even at {args.bandwidth_gib_s:.1f} GiB/s and a {args.all_hit_floor_ms_per_token:.1f} ms/token "
        f"all-hit floor, the best held-out prompt ceiling is {max(floor_rates):.2f} tok/s and "
        f"the mean ceiling is {mean(floor_rates):.2f} tok/s. Continue with structural byte reduction."
    )
    result = {
        "kind": "kimi_exact_byte_scheduler_ceiling",
        "profile_root": str(args.profile_root),
        "bandwidth_gib_s": args.bandwidth_gib_s,
        "all_hit_floor_ms_per_token": args.all_hit_floor_ms_per_token,
        "target_tok_s": args.target_tok_s,
        "prompts": rows,
        "summary": {
            "prompts": len(rows),
            "measured_mean_tok_s": mean(measured_rates),
            "measured_median_tok_s": median(measured_rates),
            "transfer_only_mean_tok_s": mean(transfer_rates),
            "transfer_only_median_tok_s": median(transfer_rates),
            "floor_plus_transfer_mean_tok_s": mean(floor_rates),
            "floor_plus_transfer_median_tok_s": median(floor_rates),
            "floor_plus_transfer_min_tok_s": min(floor_rates),
            "floor_plus_transfer_max_tok_s": max(floor_rates),
            "mean_required_byte_ratio_for_target_tps": mean(required_ratios),
            "median_required_byte_ratio_for_target_tps": median(required_ratios),
            "p10_required_byte_ratio_for_target_tps": percentile(required_ratios, 0.10),
        },
        "decision": decision,
        "reproduce_command": " ".join(__import__("sys").argv),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.out_md, result)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
