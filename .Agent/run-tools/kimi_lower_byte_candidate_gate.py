#!/usr/bin/env python3
"""Generate the Kimi lower-byte candidate gate report from existing evidence.

This is a planning/evidence tool. It does not inspect held-out routes for
candidate selection and it does not change runtime behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_float(text: str) -> float:
    return float(text.replace(",", ""))


def parse_quant_table(path: Path) -> dict[str, Any]:
    rows = []
    for line in read_text(path).splitlines():
        if not line.startswith("| `blk."):
            continue
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cols) < 10:
            continue
        tensor = cols[0].strip("`")
        rows.append({
            "tensor": tensor,
            "expert": int(cols[1]),
            "type": cols[2],
            "current_mib": parse_float(cols[3]),
            "bits": int(cols[4]),
            "block": int(cols[5]),
            "ratio": parse_float(cols[6]),
            "rel_l2": parse_float(cols[7]),
            "rel_max": parse_float(cols[8]),
            "finite": cols[9] == "True",
            "role": "gate" if "ffn_gate" in tensor else "up" if "ffn_up" in tensor else "down",
        })
    if not rows:
        raise RuntimeError(f"no quant rows parsed from {path}")

    by_bits: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_bits.setdefault(row["bits"], []).append(row)

    def summarize(items: list[dict[str, Any]]) -> dict[str, float]:
        return {
            "ratio_min": min(r["ratio"] for r in items),
            "ratio_max": max(r["ratio"] for r in items),
            "rel_l2_min": min(r["rel_l2"] for r in items),
            "rel_l2_max": max(r["rel_l2"] for r in items),
        }

    return {
        "path": str(path),
        "rows": len(rows),
        "by_bits": {str(bits): summarize(items) for bits, items in sorted(by_bits.items())},
        "target_ratio_rows": [
            row for row in rows if row["ratio"] <= 0.40
        ],
    }


def parse_cluster_report(path: Path) -> dict[str, Any]:
    text = read_text(path)
    tensor = re.search(r"- tensor: `([^`]+)`", text)
    typ = re.search(r"- type: `([^`]+)`", text)
    rows = []
    for line in text.splitlines():
        if not line.startswith("| ") or line.startswith("| ---") or "clusters" in line:
            continue
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cols) != 6 or not cols[0].isdigit():
            continue
        rows.append({
            "clusters": int(cols[0]),
            "raw_residual_weight": parse_float(cols[1]),
            "rank64_error_weight": parse_float(cols[2]),
            "rank128_error_weight": parse_float(cols[3]),
            "rank128_residual_ratio": parse_float(cols[4]),
            "base_bf16_mib": parse_float(cols[5]),
        })
    if not rows:
        raise RuntimeError(f"no cluster rows parsed from {path}")
    row16 = next((r for r in rows if r["clusters"] == 16), rows[-1])
    return {
        "path": str(path),
        "tensor": tensor.group(1) if tensor else "",
        "type": typ.group(1) if typ else "",
        "cluster16": row16,
    }


def parse_shadow_csv(path: Path) -> dict[str, Any]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    global_rows = {row["threshold"]: row for row in rows if row["scope"] == "global"}
    if not global_rows:
        raise RuntimeError(f"no global rows parsed from {path}")
    out = {"path": str(path), "thresholds": {}}
    for th, row in sorted(global_rows.items(), key=lambda kv: float(kv[0])):
        out["thresholds"][th] = {
            "skip_block_ratio": parse_float(row["skip_block_ratio"]),
            "skip_value_ratio": parse_float(row["skip_value_ratio"]),
            "mean_l2_rel": parse_float(row["mean_l2_rel"]),
            "max_l2_rel": parse_float(row["max_l2_rel"]),
            "mean_abs_err": parse_float(row["mean_abs_err"]),
            "max_abs_err": parse_float(row["max_abs_err"]),
        }
    return out


def regex_float(text: str, pattern: str, default: float | None = None) -> float:
    m = re.search(pattern, text, re.MULTILINE)
    if not m:
        if default is None:
            raise RuntimeError(f"pattern not found: {pattern}")
        return default
    return parse_float(m.group(1))


def parse_iq1_report(path: Path) -> dict[str, Any]:
    text = read_text(path)
    return {
        "path": str(path),
        "all_dev_hybrid_ratio": regex_float(text, r"hybrid byte ratio: `([0-9.]+)`"),
        "budget_worst_62": regex_float(text, r"`62\.000 GiB`: worst `[^`]+` `([0-9.]+)`"),
        "runtime_nbytes_mismatch": "nbytes mismatch" in text or "nbytes compatible" in text,
    }


def parse_v2_hotset_report(path: Path) -> dict[str, Any]:
    text = read_text(path)
    return {
        "path": str(path),
        "best_packed_ratio": regex_float(text, r"best_packed_ratio=([0-9.]+)"),
        "best_hybrid_byte_ratio": regex_float(text, r"best_hybrid_byte_ratio=([0-9.]+)"),
        "best_ideal_transfer_only_tps": regex_float(text, r"best_ideal_transfer_only_tps=([0-9.]+)"),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    quant = parse_quant_table(args.quant_report)
    down_cluster = parse_cluster_report(args.d2moe_down_report)
    gate_cluster = parse_cluster_report(args.d2moe_gate_report)
    shadow = parse_shadow_csv(args.shadow_error_csv)
    iq1 = parse_iq1_report(args.iq1_report)
    v2 = parse_v2_hotset_report(args.v2_hotset_report)

    decisions = [
        {
            "candidate": "naive blockwise 1-bit re-encode",
            "decision": "reject",
            "reason": "Some rows approach the byte target, but weight rel L2 is far above a quality-preserving range.",
        },
        {
            "candidate": "naive blockwise 2-bit re-encode",
            "decision": "reject",
            "reason": "Weight error remains large and byte ratio is above the 0.30x-0.40x target.",
        },
        {
            "candidate": "D2MoE clustered/base residual",
            "decision": "reject as primary",
            "reason": "Clustered bases still leave large rank128 error and require too much resident base memory.",
        },
        {
            "candidate": "selected IQ1_S/v2 hotsets",
            "decision": "reject as prompt-general primary path",
            "reason": "Realistic ratios are above target or require nearly complete coverage with no runtime overhead margin.",
        },
        {
            "candidate": "down activation block skipping",
            "decision": "reject as primary",
            "reason": "Down-only skip has too much output error at useful skip rates and cannot reduce up/gate bytes.",
        },
    ]

    return {
        "kind": "kimi_lower_byte_candidate_gate",
        "target_total_moved_byte_ratio": [0.30, 0.40],
        "inputs": {
            "quant_report": str(args.quant_report),
            "d2moe_down_report": str(args.d2moe_down_report),
            "d2moe_gate_report": str(args.d2moe_gate_report),
            "shadow_error_csv": str(args.shadow_error_csv),
            "iq1_report": str(args.iq1_report),
            "v2_hotset_report": str(args.v2_hotset_report),
        },
        "quant": quant,
        "d2moe": {
            "down": down_cluster,
            "gate": gate_cluster,
        },
        "down_activation_shadow": shadow,
        "iq1": iq1,
        "v2_hotset": v2,
        "decisions": decisions,
        "next_step": "Measure activation-output error for compressed up/gate/down candidates on real hidden states before writing runtime support.",
        "reproduce_command": " ".join(sys.argv),
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    q1 = result["quant"]["by_bits"].get("1", {})
    q2 = result["quant"]["by_bits"].get("2", {})
    d2_down = result["d2moe"]["down"]["cluster16"]
    d2_gate = result["d2moe"]["gate"]["cluster16"]
    sh = result["down_activation_shadow"]["thresholds"]
    th01 = sh.get("0.1") or sh.get("0.10")
    th02 = sh.get("0.2") or sh.get("0.20")
    iq1 = result["iq1"]
    v2 = result["v2_hotset"]

    lines = [
        "# Kimi lower-byte candidate gate",
        "",
        "This report is generated from committed planning artifacts. It does not change runtime behavior and does not use held-out prompts for candidate selection.",
        "",
        f"- target total moved-byte ratio: `{result['target_total_moved_byte_ratio'][0]:.2f}x-{result['target_total_moved_byte_ratio'][1]:.2f}x`",
        "",
        "## Candidate Results",
        "",
        "| candidate | evidence | decision |",
        "|---|---|---|",
        (
            "| naive blockwise 1-bit re-encode | "
            f"ratio `{q1.get('ratio_min', 0.0):.3f}x-{q1.get('ratio_max', 0.0):.3f}x`, "
            f"rel L2 `{q1.get('rel_l2_min', 0.0):.3f}-{q1.get('rel_l2_max', 0.0):.3f}` | reject |"
        ),
        (
            "| naive blockwise 2-bit re-encode | "
            f"ratio `{q2.get('ratio_min', 0.0):.3f}x-{q2.get('ratio_max', 0.0):.3f}x`, "
            f"rel L2 `{q2.get('rel_l2_min', 0.0):.3f}-{q2.get('rel_l2_max', 0.0):.3f}` | reject |"
        ),
        (
            "| D2MoE clustered base, top32 sample | "
            f"16-cluster rank128 error down `{d2_down['rank128_error_weight']:.4f}`, "
            f"gate `{d2_gate['rank128_error_weight']:.4f}`, base `{d2_down['base_bf16_mib']:.0f} MiB` per tensor | reject as primary |"
        ),
        (
            "| selected IQ1_S hotsets | "
            f"all-dev candidate hybrid ratio `{iq1['all_dev_hybrid_ratio']:.4f}x`; "
            f"small-budget worst ratio `{iq1['budget_worst_62']:.4f}x` | reject as direct 5 tok/s path |"
        ),
        (
            "| selected v2 hotsets | "
            f"best transfer-only row needs packed `{v2['best_packed_ratio']:.3f}x`, "
            f"hybrid `{v2['best_hybrid_byte_ratio']:.3f}x`, ideal `{v2['best_ideal_transfer_only_tps']:.3f} tok/s` | reject as runtime path without more margin |"
        ),
        (
            "| down activation block skipping | "
            f"threshold `0.1`: skip `{th01['skip_block_ratio']:.3f}`, mean rel error `{th01['mean_l2_rel']:.3f}`; "
            f"threshold `0.2`: skip `{th02['skip_block_ratio']:.3f}`, mean rel error `{th02['mean_l2_rel']:.3f}` | reject as primary |"
        ),
        "",
        "## Decision",
        "",
        "Do not implement runtime support for the rejected candidates as the next primary optimization.",
        "",
        "The next viable experiment must measure activation-output error for compressed up/gate/down compute on real hidden states and must reduce up/gate and down movement together.",
        "",
        "## Evidence Inputs",
        "",
    ]
    for key, value in result["inputs"].items():
        lines.append(f"- {key}: `{value}`")
    lines += [
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
    parser = argparse.ArgumentParser(description="Generate Kimi lower-byte candidate gate report.")
    parser.add_argument("--quant-report", type=Path, default=Path(".Agent/runs/20260707-gp11-quant-reencode-bound/report.md"))
    parser.add_argument("--d2moe-down-report", type=Path, default=Path(".Agent/runs/20260706-kimi-d2moe-phase0/cluster-base-blk56-down-top32.md"))
    parser.add_argument("--d2moe-gate-report", type=Path, default=Path(".Agent/runs/20260706-kimi-d2moe-phase0/cluster-base-blk56-gate-top32.md"))
    parser.add_argument("--shadow-error-csv", type=Path, default=Path(".Agent/runs/20260707-gp62-shadow-error-multiprompt-n32/summary.csv"))
    parser.add_argument("--iq1-report", type=Path, default=Path(".Agent/runs/20260707-gp34-iq1s-budgeted-hotset-bound/report.md"))
    parser.add_argument("--v2-hotset-report", type=Path, default=Path(".Agent/runs/20260707-gp46-v2-hotset-sweep/report.md"))
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    result = build_report(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.out_md, result)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
