#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def float_field(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value else 0.0


def find_window(rows: list[dict[str, str]]) -> tuple[int, int]:
    submit = next((i for i, row in enumerate(rows) if row.get("op") == "mark_submit"), None)
    if submit is None:
        raise SystemExit("trace has no mark_submit row")
    first = next((i for i, row in enumerate(rows[submit + 1 :], submit + 1) if row.get("op") == "mark_first_token"), None)
    if first is None:
        raise SystemExit("trace has no mark_first_token row after mark_submit")
    return submit, first


def summarize(path: Path, top_n: int) -> dict[str, Any]:
    rows = read_rows(path)
    submit, first = find_window(rows)
    window = [row for row in rows[submit + 1 : first] if not row.get("op", "").startswith("mark_")]

    by_op: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0.0, "elapsed_ms": 0.0})
    for row in window:
        op = row.get("op", "")
        by_op[op]["count"] += 1
        by_op[op]["elapsed_ms"] += float_field(row, "copy_ms")

    elapsed = {op: values["elapsed_ms"] for op, values in by_op.items()}
    def elapsed_any(*ops: str) -> float:
        return sum(elapsed.get(op, 0.0) for op in ops)
    prompt_moe_subspans = {
        "group": elapsed_any("cpu_up_group", "cpu_prompt_upgate_group") + elapsed_any("cpu_down_group", "cpu_prompt_down_group"),
        "prefetch_preload": elapsed_any(
            "cpu_up_prep",
            "cpu_prompt_upgate_prefetch_preload",
            "cpu_prompt_upgate_prefe",
        ) + elapsed_any(
            "cpu_down_prep",
            "cpu_prompt_down_prefetch_preload",
            "cpu_prompt_down_prefetc",
        ),
        "cuda_probe": elapsed_any("cpu_up_cuda_probe", "cpu_prompt_upgate_cuda_probe"),
        "compute": elapsed_any(
            "cpu_up_compute",
            "cpu_up_dyn_compute",
            "cpu_up_dynamic_compute",
            "cpu_up_many_compute",
            "cpu_prompt_upgate_compute",
            "cpu_prompt_upgate_compu",
        ) + elapsed_any("cpu_down_compute", "cpu_down_dyn_compute", "cpu_down_dynamic_compute", "cpu_down_dynamic_comput", "cpu_prompt_down_compute"),
    }
    prompt_moe_faults = {
        "minor": elapsed_any("cpu_up_minflt") + elapsed_any("cpu_down_minflt"),
        "major": elapsed_any("cpu_up_majflt") + elapsed_any("cpu_down_majflt"),
    }
    route_keys: set[tuple[str, str]] = set()
    route_unique_bytes = 0
    route_events = 0
    for row in window:
        if row.get("op") not in ("cpu_up_route", "cpu_gate_route", "cpu_down_route"):
            continue
        route_events += 1
        key = (row.get("tensor") or "", row.get("expert_idx") or "")
        if key in route_keys:
            continue
        route_keys.add(key)
        route_unique_bytes += int(row.get("expert_bytes") or 0)

    top_events = sorted(window, key=lambda row: float_field(row, "copy_ms"), reverse=True)[:top_n]
    return {
        "schema": "wici_ttft_trace_summary_v1",
        "trace": str(path),
        "submit_to_first_token_ms": float_field(rows[first], "t_ms") - float_field(rows[submit], "t_ms"),
        "events": len(window),
        "prompt_moe_subspans": {key: round(value, 3) for key, value in prompt_moe_subspans.items()},
        "prompt_moe_faults": {key: round(value, 3) for key, value in prompt_moe_faults.items()},
        "prompt_moe_route_events": route_events,
        "prompt_moe_route_unique_experts": len(route_keys),
        "prompt_moe_route_unique_gib": round(route_unique_bytes / float(1024 ** 3), 3),
        "by_op": {
            op: {"count": int(values["count"]), "elapsed_ms": round(values["elapsed_ms"], 3)}
            for op, values in sorted(by_op.items(), key=lambda item: item[1]["elapsed_ms"], reverse=True)
        },
        "top_events": [
            {
                "op": row.get("op"),
                "tensor": row.get("tensor"),
                "expert_idx": int(row.get("expert_idx") or -1),
                "expert_bytes": int(row.get("expert_bytes") or 0),
                "copy_ms": float_field(row, "copy_ms"),
                "ram_hit": row.get("ram_hit") == "1",
            }
            for row in top_events
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Wici MoE TTFT trace CSV between submit and first token.")
    parser.add_argument("trace", type=Path)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.top <= 0:
        raise SystemExit("--top must be positive")
    return args


def main() -> int:
    args = parse_args()
    summary = summarize(args.trace, args.top)
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
