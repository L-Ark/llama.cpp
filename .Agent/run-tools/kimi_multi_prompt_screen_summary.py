#!/usr/bin/env python3
"""Aggregate per-prompt Kimi activation compression screens."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


class Acc:
    def __init__(self) -> None:
        self.rows = 0
        self.rel_sum = 0.0
        self.rel_max = 0.0
        self.abs_sum = 0.0
        self.abs_max = 0.0
        self.ratio_sum = 0.0

    def add(self, row: dict[str, Any]) -> None:
        n = int(row.get("rows", 0) or 0)
        if n <= 0:
            return
        self.rows += n
        self.rel_sum += float(row["mean_rel_l2"]) * n
        self.rel_max = max(self.rel_max, float(row["max_rel_l2"]))
        self.abs_sum += float(row["mean_abs_error"]) * n
        self.abs_max = max(self.abs_max, float(row["max_abs_error"]))
        self.ratio_sum += float(row["mean_byte_ratio"]) * n

    def row(self) -> dict[str, Any]:
        if self.rows <= 0:
            return {
                "rows": 0,
                "mean_rel_l2": 0.0,
                "max_rel_l2": 0.0,
                "mean_abs_error": 0.0,
                "max_abs_error": 0.0,
                "mean_byte_ratio": 0.0,
            }
        return {
            "rows": self.rows,
            "mean_rel_l2": self.rel_sum / self.rows,
            "max_rel_l2": self.rel_max,
            "mean_abs_error": self.abs_sum / self.rows,
            "max_abs_error": self.abs_max,
            "mean_byte_ratio": self.ratio_sum / self.rows,
        }


def load_screen(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    prompt_id = path.parent.name
    return {"prompt_id": prompt_id, "path": str(path), "data": data}


def aggregate_section(screens: list[dict[str, Any]], section: str) -> dict[str, dict[str, Any]]:
    accs: dict[str, Acc] = defaultdict(Acc)
    for screen in screens:
        rows = screen["data"].get(section, {})
        for key, row in rows.items():
            accs[key].add(row)
    return {key: acc.row() for key, acc in sorted(accs.items())}


def mixed_role_budget(
        aggregate: dict[str, dict[str, Any]],
        fused: dict[str, dict[str, Any]],
        down_weight: float,
        upgate_weight: float,
        max_ratio: float,
        max_rel_l2: float) -> list[dict[str, Any]]:
    down_rows = [(k, v) for k, v in aggregate.items() if k.startswith("down:")]
    fused_rows = [(k, v) for k, v in fused.items()]
    out = []
    for down_name, down in down_rows:
        for fused_name, fused_row in fused_rows:
            global_ratio = down_weight * down["mean_byte_ratio"] + upgate_weight * fused_row["mean_byte_ratio"]
            if global_ratio > max_ratio:
                continue
            worst = max(down["mean_rel_l2"], fused_row["mean_rel_l2"])
            out.append({
                "down_candidate": down_name,
                "fused_candidate": fused_name,
                "global_ratio": global_ratio,
                "down_mean_rel_l2": down["mean_rel_l2"],
                "fused_mean_rel_l2": fused_row["mean_rel_l2"],
                "worst_mean_rel_l2": worst,
                "decision": "advance" if worst <= max_rel_l2 else "reject",
            })
    out.sort(key=lambda r: (r["decision"] != "advance", r["worst_mean_rel_l2"], r["global_ratio"]))
    return out


def write_md(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Kimi multi-prompt representation screen",
        "",
        "This is a dev-only offline screen. It does not change runtime behavior or claim SOTA.",
        "",
        f"- screens: `{len(result['screens'])}`",
        f"- aggregate matvec candidates: `{len(result['aggregate'])}`",
        f"- aggregate fused candidates: `{len(result['fused_up_gate'])}`",
        f"- passing matvec candidates: `{len(result['passing_matvec'])}`",
        f"- passing fused candidates: `{len(result['passing_fused'])}`",
        f"- passing mixed-role candidates: `{len(result['passing_mixed_role'])}`",
        "",
        "## Best Mixed-Role Under Budget",
        "",
        "| down | fused up/gate | global ratio | down rel L2 | fused rel L2 | worst rel L2 | decision |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in result["mixed_role"][:40]:
        lines.append(
            f"| `{row['down_candidate']}` | `{row['fused_candidate']}` | "
            f"`{row['global_ratio']:.4f}` | `{row['down_mean_rel_l2']:.6f}` | "
            f"`{row['fused_mean_rel_l2']:.6f}` | `{row['worst_mean_rel_l2']:.6f}` | "
            f"{row['decision']} |"
        )
    if not result["mixed_role"]:
        lines.append("| none | none | - | - | - | - | reject |")

    lines += [
        "",
        "## Best Fused Up/Gate",
        "",
        "| candidate | rows | byte ratio | mean rel L2 | max rel L2 |",
        "|---|---:|---:|---:|---:|",
    ]
    fused_sorted = sorted(result["fused_up_gate"].items(), key=lambda kv: (kv[1]["mean_rel_l2"], kv[1]["mean_byte_ratio"]))
    for key, row in fused_sorted[:30]:
        lines.append(
            f"| `{key}` | `{row['rows']}` | `{row['mean_byte_ratio']:.4f}` | "
            f"`{row['mean_rel_l2']:.6f}` | `{row['max_rel_l2']:.6f}` |"
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
    parser.add_argument("--screen-json", type=Path, action="append", required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--target-byte-ratio", type=float, default=0.40)
    parser.add_argument("--target-mean-rel-l2", type=float, default=0.10)
    parser.add_argument("--down-weight", type=float, default=0.376)
    parser.add_argument("--upgate-weight", type=float, default=0.624)
    args = parser.parse_args()

    screens = [load_screen(p) for p in args.screen_json]
    aggregate = aggregate_section(screens, "aggregate")
    fused = aggregate_section(screens, "fused_up_gate")
    passing_matvec = [
        {"candidate": key, **row}
        for key, row in aggregate.items()
        if row["mean_byte_ratio"] <= args.target_byte_ratio and row["mean_rel_l2"] <= args.target_mean_rel_l2
    ]
    passing_fused = [
        {"candidate": key, **row}
        for key, row in fused.items()
        if row["mean_byte_ratio"] <= args.target_byte_ratio and row["mean_rel_l2"] <= args.target_mean_rel_l2
    ]
    mixed = mixed_role_budget(
        aggregate,
        fused,
        args.down_weight,
        args.upgate_weight,
        args.target_byte_ratio,
        args.target_mean_rel_l2,
    )
    passing_mixed = [row for row in mixed if row["decision"] == "advance"]
    if passing_mixed:
        decision = "At least one candidate passes the multi-prompt offline gate; freeze it before any held-out validation or runtime work."
    else:
        decision = (
            "No candidate passes the multi-prompt offline gate. Do not implement this blockwise "
            "low-bit residual family as a runtime path; move to a qualitatively different representation."
        )

    result = {
        "kind": "kimi_multi_prompt_screen_summary",
        "screens": [{"prompt_id": s["prompt_id"], "path": s["path"]} for s in screens],
        "target_byte_ratio": args.target_byte_ratio,
        "target_mean_rel_l2": args.target_mean_rel_l2,
        "weights": {"down": args.down_weight, "fused_up_gate": args.upgate_weight},
        "aggregate": aggregate,
        "fused_up_gate": fused,
        "passing_matvec": passing_matvec,
        "passing_fused": passing_fused,
        "mixed_role": mixed,
        "passing_mixed_role": passing_mixed,
        "decision": decision,
        "reproduce_command": " ".join(__import__("sys").argv),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_md(args.out_md, result)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
