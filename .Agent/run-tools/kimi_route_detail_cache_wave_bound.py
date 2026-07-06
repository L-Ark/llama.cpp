#!/usr/bin/env python3
"""Prompt-general cache-wave bound from Kimi route-detail traces.

This is an offline planning tool. It estimates whether a prompt-general VRAM
expert cache allocation can reduce per-layer IO waves versus a global LFU
baseline under the same approximate entry budget. It does not change runtime
behavior and it must use dev traces only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TENSOR_RE = re.compile(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight")


def key_for(row: dict[str, str]) -> str:
    return f"{row['tensor']}#{row['expert_idx']}"


def parse_prompt_dir(path: Path) -> dict[str, Any]:
    detail = path / "route-detail.csv"
    metrics_path = path / "metrics.txt"
    prompt = path.name
    calls: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    if not detail.exists():
        raise FileNotFoundError(detail)
    with detail.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["mode"] != "decode":
                continue
            kind = row["kind"]
            tensor = row["tensor"]
            layer = int(row["layer"])
            m = TENSOR_RE.search(tensor)
            if not m:
                continue
            call_key = (kind, int(row["call"]), layer, tensor)
            slot = calls.setdefault(
                call_key,
                {
                    "prompt": prompt,
                    "kind": kind,
                    "layer": layer,
                    "tensor": tensor,
                    "keys": {},
                },
            )
            slot["keys"][key_for(row)] = int(row["expert_bytes"])

    metrics: dict[str, str] = {}
    if metrics_path.exists():
        for line in metrics_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                metrics[k] = v

    return {
        "prompt": prompt,
        "calls": list(calls.values()),
        "metrics": metrics,
        "detail_rows": sum(1 for _ in detail.open("r", encoding="utf-8", errors="replace")) - 1,
    }


def pool_for_kind(kind: str) -> str:
    return "down" if kind == "down" else "upgate"


def split_calls(calls: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_pool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for call in calls:
        by_pool[pool_for_kind(call["kind"])].append(call)
    return by_pool


def key_stats(calls: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for call in calls:
        for key, nbytes in call["keys"].items():
            slot = stats.setdefault(
                key,
                {
                    "key": key,
                    "count": 0,
                    "bytes": nbytes,
                    "total_bytes": 0,
                    "layer": call["layer"],
                    "kind": call["kind"],
                },
            )
            slot["count"] += 1
            slot["total_bytes"] += nbytes
    return stats


def select_lfu(calls: list[dict[str, Any]], slots: int, allowed_layers: set[int] | None = None) -> set[str]:
    stats = key_stats(calls)
    rows = list(stats.values())
    if allowed_layers is not None:
        rows = [r for r in rows if r["layer"] in allowed_layers]
    rows.sort(key=lambda r: (r["total_bytes"], r["count"], -r["layer"], r["key"]), reverse=True)
    return {r["key"] for r in rows[:slots]}


def select_front_then_lfu(calls: list[dict[str, Any]], slots: int, front_n: int) -> set[str]:
    front_layers = set(range(1, front_n + 1))
    selected = select_lfu(calls, slots, front_layers)
    if len(selected) >= slots:
        return set(list(selected)[:slots])
    stats = key_stats(calls)
    rows = [r for r in stats.values() if r["key"] not in selected]
    rows.sort(key=lambda r: (r["total_bytes"], r["count"], -r["layer"], r["key"]), reverse=True)
    for row in rows:
        if len(selected) >= slots:
            break
        selected.add(row["key"])
    return selected


def select_setcover_fullhit(calls: list[dict[str, Any]], slots: int) -> set[str]:
    """Greedy set-completion strategy.

    A single expert does not reduce waves when a call has 8 active experts and
    IO depth is 8. This strategy therefore chooses missing sets that complete
    many call groups per added key.
    """
    selected: set[str] = set()
    call_sets = [set(call["keys"]) for call in calls]
    call_bytes = [sum(call["keys"].values()) for call in calls]
    remaining = set(range(len(call_sets)))

    while len(selected) < slots and remaining:
        best_missing: set[str] | None = None
        best_score = -1.0
        best_idx = -1
        room = slots - len(selected)
        for idx in list(remaining):
            missing = call_sets[idx] - selected
            if not missing:
                remaining.discard(idx)
                continue
            cost = len(missing)
            if cost > room:
                continue
            # Reward completing a call, weighted lightly by bytes so larger
            # expert groups win ties. Reuse is captured because missing shrinks.
            score = (1.0 + call_bytes[idx] / (1024**3)) / cost
            if score > best_score:
                best_score = score
                best_missing = missing
                best_idx = idx
        if not best_missing:
            break
        selected.update(best_missing)
        remaining.discard(best_idx)
    return selected


def score(calls_by_prompt: dict[str, list[dict[str, Any]]], selected: set[str], io_depth: int) -> dict[str, Any]:
    per_prompt: dict[str, Any] = {}
    aggregate = {
        "calls": 0,
        "full_hit_calls": 0,
        "missed_bytes": 0,
        "hit_bytes": 0,
        "waves": 0,
        "active_entries": 0,
        "miss_entries": 0,
    }
    for prompt, calls in calls_by_prompt.items():
        row = {
            "calls": 0,
            "full_hit_calls": 0,
            "missed_bytes": 0,
            "hit_bytes": 0,
            "waves": 0,
            "active_entries": 0,
            "miss_entries": 0,
        }
        for call in calls:
            keys = set(call["keys"])
            miss = keys - selected
            hit = keys & selected
            miss_count = len(miss)
            row["calls"] += 1
            row["active_entries"] += len(keys)
            row["miss_entries"] += miss_count
            row["full_hit_calls"] += int(miss_count == 0)
            row["waves"] += math.ceil(miss_count / io_depth) if miss_count else 0
            row["missed_bytes"] += sum(call["keys"][key] for key in miss)
            row["hit_bytes"] += sum(call["keys"][key] for key in hit)
        row["full_hit_rate"] = row["full_hit_calls"] / row["calls"] if row["calls"] else 0.0
        row["miss_entry_rate"] = row["miss_entries"] / row["active_entries"] if row["active_entries"] else 0.0
        row["missed_gib"] = row["missed_bytes"] / (1024**3)
        row["hit_gib"] = row["hit_bytes"] / (1024**3)
        per_prompt[prompt] = row
        for key in aggregate:
            aggregate[key] += row[key]
    aggregate["full_hit_rate"] = aggregate["full_hit_calls"] / aggregate["calls"] if aggregate["calls"] else 0.0
    aggregate["miss_entry_rate"] = aggregate["miss_entries"] / aggregate["active_entries"] if aggregate["active_entries"] else 0.0
    aggregate["missed_gib"] = aggregate["missed_bytes"] / (1024**3)
    aggregate["hit_gib"] = aggregate["hit_bytes"] / (1024**3)
    return {"aggregate": aggregate, "per_prompt": per_prompt}


def relative_to(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "aggregate_wave_reduction": reduction(candidate["aggregate"]["waves"], baseline["aggregate"]["waves"]),
        "aggregate_miss_byte_reduction": reduction(candidate["aggregate"]["missed_bytes"], baseline["aggregate"]["missed_bytes"]),
        "worst_prompt_wave_reduction": 1.0,
        "worst_prompt_miss_byte_change": -1.0,
    }
    for prompt, row in candidate["per_prompt"].items():
        base = baseline["per_prompt"][prompt]
        wr = reduction(row["waves"], base["waves"])
        mb_change = (row["missed_bytes"] - base["missed_bytes"]) / base["missed_bytes"] if base["missed_bytes"] else 0.0
        out["worst_prompt_wave_reduction"] = min(out["worst_prompt_wave_reduction"], wr)
        out["worst_prompt_miss_byte_change"] = max(out["worst_prompt_miss_byte_change"], mb_change)
    return out


def reduction(new_value: float, old_value: float) -> float:
    return (old_value - new_value) / old_value if old_value else 0.0


def make_calls_by_prompt(prompt_items: list[dict[str, Any]], pool: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for item in prompt_items:
        out[item["prompt"]] = [c for c in item["calls"] if pool_for_kind(c["kind"]) == pool]
    return out


def write_md(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Kimi route-detail cache-wave bound",
        "",
        "This is a dev-only offline diagnostic. It is not SOTA and does not use held-out test prompts.",
        "",
        "## Inputs",
        "",
        f"- run_dir: `{result['run_dir']}`",
        f"- prompts: `{len(result['prompts'])}`",
        f"- io_depth: `{result['io_depth']}`",
        f"- current upgate slots: `{result['budgets']['current']['upgate']}`",
        f"- current down slots: `{result['budgets']['current']['down']}`",
        "",
        "## Runtime Metrics",
        "",
        "| prompt | quality | tok/s | TTFT ms | decode ms | memory peak | route rows |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in result["prompts"]:
        m = item["metrics"]
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} |".format(
                item["prompt"],
                m.get("quality", ""),
                m.get("token_rate", ""),
                m.get("ttft_ms", ""),
                m.get("decode_ms", ""),
                m.get("memory.peak", ""),
                item["detail_rows"],
            )
        )
    lines.extend(
        [
            "",
            "## Strategy Summary",
            "",
            "| pool | strategy | slots | waves | full-hit rate | missed GiB | aggregate wave reduction vs LFU | worst-prompt wave reduction | worst miss-byte change | pass gate |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in result["summary_rows"]:
        rel = row.get("relative_to_lfu", {})
        agg = row["score"]["aggregate"]
        lines.append(
            "| {pool} | {strategy} | {slots} | {waves} | {fhr:.4f} | {miss:.2f} | {awr:.4f} | {wwr:.4f} | {wmb:.4f} | {gate} |".format(
                pool=row["pool"],
                strategy=row["strategy"],
                slots=row["slots"],
                waves=agg["waves"],
                fhr=agg["full_hit_rate"],
                miss=agg["missed_gib"],
                awr=rel.get("aggregate_wave_reduction", 0.0),
                wwr=rel.get("worst_prompt_wave_reduction", 0.0),
                wmb=rel.get("worst_prompt_miss_byte_change", 0.0),
                gate="yes" if row.get("passes_gate") else "no",
            )
        )
    lines.extend(
        [
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
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    parser.add_argument("--io-depth", type=int, default=8)
    parser.add_argument("--upgate-slots", type=int, default=1735)
    parser.add_argument("--down-slots", type=int, default=766)
    parser.add_argument("--front-layers", default="2,4,8,12,16")
    parser.add_argument("--slot-upgate-mib", type=float, default=5.36)
    parser.add_argument("--slot-down-mib", type=float, default=7.44)
    args = parser.parse_args()

    prompt_items = [parse_prompt_dir(path) for path in sorted(args.run_dir.iterdir()) if path.is_dir()]
    front_layers = [int(x) for x in args.front_layers.split(",") if x.strip()]
    budgets = {
        "current": {"upgate": args.upgate_slots, "down": args.down_slots},
        "split_sweeps": {},
    }
    total_mib = args.upgate_slots * args.slot_upgate_mib + args.down_slots * args.slot_down_mib
    for pct in [62, 68, 72, 76, 80]:
        budgets["split_sweeps"][str(pct)] = {
            "upgate": int((total_mib * pct / 100) // args.slot_upgate_mib),
            "down": int((total_mib * (100 - pct) / 100) // args.slot_down_mib),
        }

    summary_rows: list[dict[str, Any]] = []
    all_scores: dict[str, Any] = {}

    for pool in ["upgate", "down"]:
        calls_by_prompt = make_calls_by_prompt(prompt_items, pool)
        calls = [call for calls in calls_by_prompt.values() for call in calls]
        base_slots = budgets["current"][pool]
        lfu_selected = select_lfu(calls, base_slots)
        lfu_score = score(calls_by_prompt, lfu_selected, args.io_depth)
        all_scores[f"{pool}:global_lfu"] = lfu_score
        summary_rows.append({
            "pool": pool,
            "strategy": "global_lfu",
            "slots": base_slots,
            "selected_count": len(lfu_selected),
            "score": lfu_score,
            "relative_to_lfu": relative_to(lfu_score, lfu_score),
            "passes_gate": False,
        })

        for n in front_layers:
            selected = select_front_then_lfu(calls, base_slots, n)
            sc = score(calls_by_prompt, selected, args.io_depth)
            rel = relative_to(sc, lfu_score)
            summary_rows.append({
                "pool": pool,
                "strategy": f"front_layers_1_{n}",
                "slots": base_slots,
                "selected_count": len(selected),
                "score": sc,
                "relative_to_lfu": rel,
                "passes_gate": rel["worst_prompt_wave_reduction"] >= 0.20 and rel["worst_prompt_miss_byte_change"] <= 0.05,
            })

        selected = select_setcover_fullhit(calls, base_slots)
        sc = score(calls_by_prompt, selected, args.io_depth)
        rel = relative_to(sc, lfu_score)
        summary_rows.append({
            "pool": pool,
            "strategy": "setcover_fullhit",
            "slots": base_slots,
            "selected_count": len(selected),
            "score": sc,
            "relative_to_lfu": rel,
            "passes_gate": rel["worst_prompt_wave_reduction"] >= 0.20 and rel["worst_prompt_miss_byte_change"] <= 0.05,
        })

        for pct, budget in budgets["split_sweeps"].items():
            slots = budget[pool]
            selected = select_lfu(calls, slots)
            sc = score(calls_by_prompt, selected, args.io_depth)
            rel = relative_to(sc, lfu_score)
            summary_rows.append({
                "pool": pool,
                "strategy": f"split_{pct}_lfu",
                "slots": slots,
                "selected_count": len(selected),
                "score": sc,
                "relative_to_lfu": rel,
                "passes_gate": rel["worst_prompt_wave_reduction"] >= 0.20 and rel["worst_prompt_miss_byte_change"] <= 0.05,
            })

    passing = [row for row in summary_rows if row["passes_gate"]]
    decision = (
        "At least one offline strategy passes the GP30 gate; freeze the best dev-only candidate and design a default-off runtime policy before held-out testing."
        if passing
        else "No offline strategy passes the GP30 gate. Reject front-layer/full-hit cache reallocation as the next primary runtime path under the current entry budget."
    )

    result = {
        "note": "dev-only cache-wave bound; not SOTA",
        "run_dir": str(args.run_dir),
        "io_depth": args.io_depth,
        "budgets": budgets,
        "prompts": [
            {
                "prompt": item["prompt"],
                "metrics": item["metrics"],
                "detail_rows": item["detail_rows"],
                "decode_calls": len(item["calls"]),
            }
            for item in prompt_items
        ],
        "summary_rows": summary_rows,
        "passing_count": len(passing),
        "decision": decision,
        "reproduce_command": " ".join([
            ".Agent/run-tools/kimi_route_detail_cache_wave_bound.py",
            "--run-dir", str(args.run_dir),
            "--out-json", str(args.out_json),
            "--out-md", str(args.out_md),
        ]),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_md(args.out_md, result)
    print(json.dumps({"passing_count": len(passing), "decision": decision}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
