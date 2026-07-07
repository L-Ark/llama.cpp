#!/usr/bin/env python3
"""Decode ceiling for predicted up/gate prefetch recall.

This is an offline bound. It asks how much token rate would improve if a
predictor could hide a given fraction of exposed decode up/gate wall time.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def parse_policy_recalls(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    data = load_json(path)
    out = {}
    for name in ("cap1_all", "cap2_all", "cap4_all", "cap8_all"):
        try:
            out[name] = float(data["policies"][name]["summary"]["recall"])
        except KeyError:
            pass
    return out


def analyze_prompt(path: Path, recalls: dict[str, float], target_tps: float, default_floor_ms: float) -> dict[str, Any]:
    metrics = load_json(path / "metrics.json")
    decode_ms = float(metrics["decode_ms"])
    decode_runs = int(metrics["decode_runs"])
    up_path = path / "up-gate-profile.csv"
    rows = []
    with up_path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        raise RuntimeError(f"empty up-gate profile: {up_path}")

    allhit = [
        float(r["wall_ms"])
        for r in rows
        if int(r["up_cache_misses"]) == 0 and int(r["gate_cache_misses"]) == 0
    ]
    allhit_floor_ms = mean(allhit) if allhit else default_floor_ms
    wall_sum_ms = sum(float(r["wall_ms"]) for r in rows)
    exposed_ms = sum(max(0.0, float(r["wall_ms"]) - allhit_floor_ms) for r in rows)

    policies = {}
    for name, recall in {**recalls, "perfect_upgate_hide": 1.0}.items():
        saved_ms = exposed_ms * max(0.0, min(1.0, recall))
        new_decode_ms = max(1e-9, decode_ms - saved_ms)
        policies[name] = {
            "recall": recall,
            "saved_ms": saved_ms,
            "new_decode_ms": new_decode_ms,
            "token_rate": decode_runs / (new_decode_ms / 1000.0),
            "speedup": decode_ms / new_decode_ms,
        }

    target_decode_ms = decode_runs / target_tps * 1000.0
    required_recall = (decode_ms - target_decode_ms) / exposed_ms if exposed_ms > 0 else math.inf

    return {
        "prompt": path.name,
        "prompt_id": metrics.get("prompt_id", path.name),
        "quality": metrics.get("quality", ""),
        "decode_ms": decode_ms,
        "decode_runs": decode_runs,
        "current_token_rate": decode_runs / (decode_ms / 1000.0),
        "reported_token_rate": float(metrics.get("token_rate", 0.0) or 0.0),
        "upgate_calls": len(rows),
        "allhit_rows": len(allhit),
        "allhit_floor_ms": allhit_floor_ms,
        "upgate_wall_sum_ms": wall_sum_ms,
        "upgate_exposed_ms": exposed_ms,
        "upgate_exposed_decode_fraction": exposed_ms / decode_ms if decode_ms else 0.0,
        "required_recall_for_target_tps": required_recall,
        "policies": policies,
    }


def aggregate(rows: list[dict[str, Any]], policy_names: list[str], target_tps: float) -> dict[str, Any]:
    decode_ms = sum(r["decode_ms"] for r in rows)
    decode_runs = sum(r["decode_runs"] for r in rows)
    exposed_ms = sum(r["upgate_exposed_ms"] for r in rows)
    out: dict[str, Any] = {
        "prompts": len(rows),
        "decode_ms": decode_ms,
        "decode_runs": decode_runs,
        "current_token_rate": decode_runs / (decode_ms / 1000.0),
        "upgate_exposed_ms": exposed_ms,
        "upgate_exposed_decode_fraction": exposed_ms / decode_ms if decode_ms else 0.0,
        "required_recall_for_target_tps": (
            (decode_ms - decode_runs / target_tps * 1000.0) / exposed_ms
            if exposed_ms > 0 else math.inf
        ),
        "policies": {},
    }
    for name in policy_names:
        recall = rows[0]["policies"][name]["recall"] if rows and name in rows[0]["policies"] else 0.0
        saved = exposed_ms * max(0.0, min(1.0, recall))
        new_decode = max(1e-9, decode_ms - saved)
        out["policies"][name] = {
            "recall": recall,
            "saved_ms": saved,
            "new_decode_ms": new_decode,
            "token_rate": decode_runs / (new_decode / 1000.0),
            "speedup": decode_ms / new_decode,
        }
    return out


def write_md(path: Path, result: dict[str, Any]) -> None:
    agg = result["aggregate"]
    lines = [
        "# Kimi prediction-recall decode ceiling",
        "",
        "This is an offline bound. It does not change runtime behavior or claim SOTA.",
        "",
        f"- profile root: `{result['profile_root']}`",
        f"- policy source: `{result.get('policy_source') or ''}`",
        f"- target token rate: `{result['target_tps']:.2f} tok/s`",
        f"- prompts: `{agg['prompts']}`",
        f"- current aggregate token rate: `{agg['current_token_rate']:.3f} tok/s`",
        f"- exposed up/gate wall fraction: `{agg['upgate_exposed_decode_fraction']:.3f}`",
        f"- required up/gate hide recall for target: `{agg['required_recall_for_target_tps']:.3f}`",
        "",
        "## Aggregate Policies",
        "",
        "| policy | recall | saved s | token rate | speedup |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in agg["policies"].items():
        lines.append(
            f"| `{name}` | `{row['recall']:.4f}` | `{row['saved_ms'] / 1000.0:.2f}` | "
            f"`{row['token_rate']:.3f}` | `{row['speedup']:.3f}x` |"
        )

    lines += [
        "",
        "## Per Prompt",
        "",
        "| prompt | current tok/s | exposed upgate % decode | req recall for 5 | perfect-hide tok/s | cap8 tok/s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["prompts"]:
        perfect = row["policies"].get("perfect_upgate_hide", {})
        cap8 = row["policies"].get("cap8_all", {})
        lines.append(
            f"| `{row['prompt_id']}` | `{row['current_token_rate']:.3f}` | "
            f"`{row['upgate_exposed_decode_fraction']:.3f}` | "
            f"`{row['required_recall_for_target_tps']:.3f}` | "
            f"`{perfect.get('token_rate', 0.0):.3f}` | "
            f"`{cap8.get('token_rate', 0.0):.3f}` |"
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
    parser.add_argument("--policy-json", type=Path)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--target-tps", type=float, default=5.0)
    parser.add_argument("--default-allhit-floor-ms", type=float, default=0.70)
    args = parser.parse_args()

    recalls = parse_policy_recalls(args.policy_json)
    prompt_rows = []
    for prompt_dir in sorted(p for p in args.profile_root.iterdir() if p.is_dir()):
        if (prompt_dir / "metrics.json").exists() and (prompt_dir / "up-gate-profile.csv").exists():
            prompt_rows.append(analyze_prompt(prompt_dir, recalls, args.target_tps, args.default_allhit_floor_ms))
    if not prompt_rows:
        raise RuntimeError(f"no prompt profiles under {args.profile_root}")

    policy_names = list(prompt_rows[0]["policies"].keys())
    agg = aggregate(prompt_rows, policy_names, args.target_tps)
    perfect = agg["policies"]["perfect_upgate_hide"]["token_rate"]
    if perfect < args.target_tps:
        decision = (
            "Even perfect up/gate hide cannot reach the target token rate. "
            "Prediction/prefetch is secondary and must be combined with byte reduction or a representation change."
        )
    else:
        decision = (
            "Perfect up/gate hide could reach the target in this optimistic bound. "
            "A runtime predictor would still need dev-only training and held-out validation."
        )

    result = {
        "kind": "kimi_prediction_recall_decode_ceiling",
        "profile_root": str(args.profile_root),
        "policy_source": str(args.policy_json) if args.policy_json else None,
        "target_tps": args.target_tps,
        "aggregate": agg,
        "prompts": prompt_rows,
        "decision": decision,
        "reproduce_command": " ".join(os.sys.argv),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_md(args.out_md, result)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
