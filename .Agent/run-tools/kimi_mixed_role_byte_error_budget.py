#!/usr/bin/env python3
"""Combine down and fused up/gate compression screens under global byte budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def role_rows(screen: dict[str, Any], prefix: str) -> list[dict[str, Any]]:
    rows = []
    for name, row in screen[prefix].items():
        rows.append({
            "candidate": name,
            "ratio": float(row["mean_byte_ratio"]),
            "mean_rel_l2": float(row["mean_rel_l2"]),
            "max_rel_l2": float(row["max_rel_l2"]),
            "rows": int(row["rows"]),
        })
    return rows


def aggregate_role_rows(screen: dict[str, Any], role: str) -> list[dict[str, Any]]:
    rows = []
    needle = role + ":"
    for name, row in screen["aggregate"].items():
        if not name.startswith(needle):
            continue
        rows.append({
            "candidate": name,
            "ratio": float(row["mean_byte_ratio"]),
            "mean_rel_l2": float(row["mean_rel_l2"]),
            "max_rel_l2": float(row["max_rel_l2"]),
            "rows": int(row["rows"]),
        })
    return rows


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Kimi mixed-role byte/error budget",
        "",
        "This is an offline budget screen. It does not change runtime behavior or claim SOTA.",
        "",
        f"- source screen: `{result['source_screen']}`",
        f"- down weight: `{result['weights']['down']:.3f}`",
        f"- fused up/gate weight: `{result['weights']['fused_up_gate']:.3f}`",
        f"- target global byte ratio: `<= {result['target_global_ratio']:.2f}x`",
        f"- target mean rel L2 per component: `<= {result['target_mean_rel_l2']:.2f}`",
        f"- combinations under byte target: `{len(result['under_budget'])}`",
        f"- passing combinations: `{len(result['passing'])}`",
        "",
        "## Best Under Budget",
        "",
        "| down | fused up/gate | global ratio | down rel L2 | fused rel L2 | worst rel L2 | decision |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in result["under_budget"][:40]:
        lines.append(
            f"| `{row['down_candidate']}` | `{row['fused_candidate']}` | "
            f"`{row['global_ratio']:.4f}` | `{row['down_mean_rel_l2']:.6f}` | "
            f"`{row['fused_mean_rel_l2']:.6f}` | `{row['worst_mean_rel_l2']:.6f}` | "
            f"{row['decision']} |"
        )
    if not result["under_budget"]:
        lines.append("| none | none | - | - | - | - | reject |")

    lines += [
        "",
        "## Conclusion",
        "",
        result["conclusion"],
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
    parser.add_argument("--screen-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--down-weight", type=float, default=0.376)
    parser.add_argument("--upgate-weight", type=float, default=0.624)
    parser.add_argument("--target-global-ratio", type=float, default=0.40)
    parser.add_argument("--target-mean-rel-l2", type=float, default=0.10)
    args = parser.parse_args()

    screen = json.loads(args.screen_json.read_text(encoding="utf-8"))
    down_rows = aggregate_role_rows(screen, "down")
    fused_rows = role_rows(screen, "fused_up_gate")

    combos = []
    for down in down_rows:
        for fused in fused_rows:
            global_ratio = args.down_weight * down["ratio"] + args.upgate_weight * fused["ratio"]
            if global_ratio > args.target_global_ratio:
                continue
            worst = max(down["mean_rel_l2"], fused["mean_rel_l2"])
            passing = (
                down["mean_rel_l2"] <= args.target_mean_rel_l2 and
                fused["mean_rel_l2"] <= args.target_mean_rel_l2
            )
            combos.append({
                "down_candidate": down["candidate"],
                "fused_candidate": fused["candidate"],
                "global_ratio": global_ratio,
                "down_ratio": down["ratio"],
                "fused_ratio": fused["ratio"],
                "down_mean_rel_l2": down["mean_rel_l2"],
                "fused_mean_rel_l2": fused["mean_rel_l2"],
                "worst_mean_rel_l2": worst,
                "decision": "advance" if passing else "reject",
            })

    combos.sort(key=lambda row: (row["worst_mean_rel_l2"], row["global_ratio"]))
    passing = [row for row in combos if row["decision"] == "advance"]
    if passing:
        conclusion = "At least one mixed-role combination passes the offline byte/error gate; next step is dev-train/freeze and held-out validation before any runtime path."
    elif combos:
        best = combos[0]
        conclusion = (
            "No mixed-role combination passes. Even the best under-budget pair has "
            f"worst mean rel L2 {best['worst_mean_rel_l2']:.3f}, so GP68-GP70 "
            "blockwise residual tuning should stop as a primary 5 tok/s path."
        )
    else:
        conclusion = "No combination from the source screen is under the global byte target."

    result = {
        "kind": "kimi_mixed_role_byte_error_budget",
        "source_screen": str(args.screen_json),
        "weights": {
            "down": args.down_weight,
            "fused_up_gate": args.upgate_weight,
        },
        "target_global_ratio": args.target_global_ratio,
        "target_mean_rel_l2": args.target_mean_rel_l2,
        "under_budget": combos,
        "passing": passing,
        "conclusion": conclusion,
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
