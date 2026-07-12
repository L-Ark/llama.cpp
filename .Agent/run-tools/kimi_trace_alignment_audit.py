#!/usr/bin/env python3
"""Audit Kimi route-score, route-detail and activation-dump alignment."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def tensor_kind(tensor: str) -> str:
    if ".ffn_up_" in tensor:
        return "up"
    if ".ffn_gate_" in tensor:
        return "gate"
    if ".ffn_down_" in tensor:
        return "down"
    return "other"


def layer_of(tensor: str) -> int:
    match = re.search(r"blk\.(\d+)\.", tensor or "")
    return int(match.group(1)) if match else -1


def summarize_run(run: Path) -> dict[str, Any]:
    route_score = read_csv(run / "route-score-trace.csv")
    route_detail = read_csv(run / "route-detail.csv")
    activations = read_csv(run / "activations.csv")
    metrics = {}
    if (run / "metrics.json").exists():
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8", errors="replace"))

    score_by_call: dict[int, list[int]] = defaultdict(list)
    for row in route_score:
        score_by_call[int(row["call"])].append(int(row["layer"]))

    detail_mode = Counter(row.get("mode", "") for row in route_detail)
    detail_kind = Counter(row.get("kind", "") for row in route_detail)
    detail_token = Counter(row.get("token_id", "") for row in route_detail)
    detail_prompt_token = Counter(row.get("token_id", "") for row in route_detail if row.get("mode") == "prompt")
    detail_decode_token = Counter(row.get("token_id", "") for row in route_detail if row.get("mode") == "decode")

    activation_role = Counter(row.get("role", "") for row in activations)
    activation_mode = Counter(row.get("mode", "") for row in activations)
    activation_tensor_kind = Counter(tensor_kind(row.get("tensor", "")) for row in activations)
    activation_token = Counter(row.get("token_id", "") for row in activations)
    activation_mismatch = sum(1 for row in activations if row.get("role", "") != tensor_kind(row.get("tensor", "")))

    activation_call_layers: dict[int, Counter[tuple[str, str, int]]] = defaultdict(Counter)
    for row in activations:
        call = int(row["call"])
        activation_call_layers[call][(row.get("role", ""), tensor_kind(row.get("tensor", "")), layer_of(row.get("tensor", "")))] += 1

    return {
        "run": str(run),
        "metrics": {
            "exit": metrics.get("exit"),
            "token_rate": metrics.get("token_rate"),
            "ttft_ms": metrics.get("ttft_ms"),
            "memory_peak": metrics.get("memory.peak"),
            "quality": metrics.get("quality"),
            "quality_reason": metrics.get("quality_reason"),
        },
        "route_score": {
            "rows": len(route_score),
            "calls": len(score_by_call),
            "positions": dict(Counter(row.get("pos", "") for row in route_score).most_common()),
            "n_tokens": dict(Counter(row.get("n_tokens", "") for row in route_score).most_common()),
            "first_calls": {
                str(call): {
                    "count": len(layers),
                    "first_layers": layers[:5],
                    "last_layers": layers[-5:],
                }
                for call, layers in sorted(score_by_call.items())[:6]
            },
        },
        "route_detail": {
            "rows": len(route_detail),
            "mode": dict(detail_mode.most_common()),
            "kind": dict(detail_kind.most_common()),
            "token_id": dict(detail_token.most_common(12)),
            "prompt_token_id": dict(detail_prompt_token.most_common(12)),
            "decode_token_id": dict(detail_decode_token.most_common(12)),
        },
        "activations": {
            "rows": len(activations),
            "role": dict(activation_role.most_common()),
            "mode": dict(activation_mode.most_common()),
            "tensor_kind": dict(activation_tensor_kind.most_common()),
            "token_id": dict(activation_token.most_common(12)),
            "role_tensor_mismatch": activation_mismatch,
            "first_calls": {
                str(call): [
                    {"role": key[0], "tensor_kind": key[1], "layer": key[2], "count": count}
                    for key, count in counter.most_common(6)
                ]
                for call, counter in sorted(activation_call_layers.items())[:8]
            },
        },
    }


def write_md(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Kimi trace alignment audit",
        "",
        "This is a default-off instrumentation audit. It does not claim SOTA.",
        "",
        "## Runs",
        "",
    ]
    for label, run in result["runs"].items():
        lines += [
            f"### {label}",
            "",
            f"- run: `{run['run']}`",
            f"- exit: `{run['metrics'].get('exit')}`",
            f"- token_rate: `{run['metrics'].get('token_rate')}`",
            f"- TTFT ms: `{run['metrics'].get('ttft_ms')}`",
            f"- memory peak: `{run['metrics'].get('memory_peak')}`",
            f"- quality: `{run['metrics'].get('quality')}`",
            "",
            "Route score:",
            "",
            f"- rows: `{run['route_score']['rows']}`",
            f"- calls: `{run['route_score']['calls']}`",
            f"- positions: `{run['route_score']['positions']}`",
            "",
            "Route detail:",
            "",
            f"- rows: `{run['route_detail']['rows']}`",
            f"- mode: `{run['route_detail']['mode']}`",
            f"- kind: `{run['route_detail']['kind']}`",
            f"- token_id top: `{run['route_detail']['token_id']}`",
            f"- prompt token_id top: `{run['route_detail']['prompt_token_id']}`",
            f"- decode token_id top: `{run['route_detail']['decode_token_id']}`",
            "",
            "Activations:",
            "",
            f"- rows: `{run['activations']['rows']}`",
            f"- role: `{run['activations']['role']}`",
            f"- mode: `{run['activations']['mode']}`",
            f"- tensor_kind: `{run['activations']['tensor_kind']}`",
            f"- token_id top: `{run['activations']['token_id']}`",
            f"- role/tensor mismatches: `{run['activations']['role_tensor_mismatch']}`",
            "",
        ]

    after = result["runs"].get("after")
    before = result["runs"].get("before")
    lines += [
        "## Decision",
        "",
    ]
    if after and after["activations"]["role_tensor_mismatch"] == 0 and after["activations"]["mode"].get("prompt", 0) == 0:
        lines.append(
            "Pass: after the instrumentation fix, activation dump rows are decode-only for this smoke, and role names match tensor kinds."
        )
    else:
        lines.append("Fail: after-run activation dump still has role/mode alignment issues.")
    if before:
        lines.append(
            f"Before fix, role/tensor mismatches were `{before['activations']['role_tensor_mismatch']}`; after fix they are `{after['activations']['role_tensor_mismatch'] if after else 'missing'}`."
        )
    lines += [
        "",
        "Remaining limitation:",
        "",
        "- activation `token_id` is still the per-ubatch tensor-column index. In decode it is expected to be `0`; global decode position should be recovered from `route-score-trace.csv` (`pos`) and layer/order alignment.",
        "- This fix makes a future full N96 hidden/router admission corpus feasible, but it is not itself a token-rate optimization.",
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
    parser = argparse.ArgumentParser(description="Audit Kimi trace alignment runs.")
    parser.add_argument("--before", type=Path)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    runs = {"after": summarize_run(args.after)}
    if args.before:
        runs = {"before": summarize_run(args.before), **runs}
    result = {
        "kind": "kimi_trace_alignment_audit",
        "runs": runs,
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
