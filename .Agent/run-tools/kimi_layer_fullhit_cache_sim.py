#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


LAYER_RE = re.compile(r"^blk\.(\d+)\.ffn_(up|gate|down)_exps\.weight$")


@dataclass(frozen=True)
class ExpertKey:
    tensor: str
    expert_idx: int
    expert_bytes: int


@dataclass
class Call:
    prompt: str
    call_idx: int
    tensor: str
    role: str
    layer: int
    keys: tuple[ExpertKey, ...]


def role_layer(tensor: str) -> tuple[str, int]:
    m = LAYER_RE.match(tensor)
    if not m:
        return "other", -1
    return m.group(2), int(m.group(1))


def pool_of(role: str) -> str:
    if role in ("up", "gate"):
        return "upgate"
    if role == "down":
        return "down"
    return "other"


def gib(n: float) -> float:
    return n / 1024.0 / 1024.0 / 1024.0


def pct(num: float, den: float) -> float:
    return 100.0 * num / den if den else 0.0


def load_calls(path: Path) -> list[Call]:
    prompt = path.parent.name
    rows = []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            tensor = row["tensor"]
            role, layer = role_layer(tensor)
            if role == "other":
                continue
            rows.append({
                "tensor": tensor,
                "role": role,
                "layer": layer,
                "key": ExpertKey(tensor, int(row["expert_idx"]), int(row["expert_bytes"])),
            })

    calls: list[Call] = []
    i = 0
    call_idx = 0
    while i < len(rows):
        tensor = rows[i]["tensor"]
        role = rows[i]["role"]
        layer = rows[i]["layer"]
        keys = []
        while i < len(rows) and rows[i]["tensor"] == tensor:
            keys.append(rows[i]["key"])
            i += 1
        call_idx += 1
        calls.append(Call(prompt, call_idx, tensor, role, layer, tuple(keys)))
    return calls


def load_prompt_calls(route_root: Path) -> dict[str, list[Call]]:
    prompts = {}
    for path in sorted(route_root.glob("*/route-trace.csv")):
        calls = load_calls(path)
        if calls:
            prompts[path.parent.name] = calls
    if not prompts:
        raise RuntimeError(f"no route traces found under {route_root}")
    return prompts


def key_bytes(keys: Iterable[ExpertKey]) -> int:
    return sum(k.expert_bytes for k in set(keys))


def select_by_traffic(calls: list[Call], budget_by_pool: dict[str, int]) -> dict[str, set[ExpertKey]]:
    traffic: dict[str, Counter[ExpertKey]] = {"upgate": Counter(), "down": Counter()}
    for call in calls:
        pool = pool_of(call.role)
        if pool not in traffic:
            continue
        for key in call.keys:
            traffic[pool][key] += key.expert_bytes

    selected = {"upgate": set(), "down": set()}
    used = {"upgate": 0, "down": 0}
    for pool, counter in traffic.items():
        ranked = sorted(counter.items(), key=lambda kv: (kv[1], kv[0].expert_bytes), reverse=True)
        for key, _score in ranked:
            if key in selected[pool]:
                continue
            if used[pool] + key.expert_bytes > budget_by_pool[pool]:
                continue
            selected[pool].add(key)
            used[pool] += key.expert_bytes
    return selected


def select_front_layers(calls: list[Call], budget_by_pool: dict[str, int], max_layer: int) -> dict[str, set[ExpertKey]]:
    filtered = [call for call in calls if 0 <= call.layer <= max_layer]
    return select_by_traffic(filtered, budget_by_pool)


def signature_key(call: Call) -> tuple[str, tuple[ExpertKey, ...]]:
    return (call.tensor, tuple(sorted(set(call.keys), key=lambda k: (k.expert_idx, k.expert_bytes))))


def select_fullhit_greedy(calls: list[Call], budget_by_pool: dict[str, int]) -> dict[str, set[ExpertKey]]:
    selected = {"upgate": set(), "down": set()}
    used = {"upgate": 0, "down": 0}
    by_pool_sig: dict[str, Counter[tuple[str, tuple[ExpertKey, ...]]]] = {"upgate": Counter(), "down": Counter()}
    for call in calls:
        pool = pool_of(call.role)
        if pool in by_pool_sig:
            by_pool_sig[pool][signature_key(call)] += 1

    for pool, sig_counts in by_pool_sig.items():
        ranked = []
        for sig, count in sig_counts.items():
            keys = set(sig[1])
            cost = key_bytes(keys)
            ranked.append((count / max(cost, 1), count, cost, keys))
        ranked.sort(reverse=True)
        # Fast approximation: completing a repeated 8-expert tensor-call is what
        # removes an IO wave at depth=8. Recompute missing cost as previous
        # selections may have partially covered later signatures.
        for _score, _count, _cost, keys in ranked:
            missing = keys - selected[pool]
            if not missing:
                continue
            cost = key_bytes(missing)
            if used[pool] + cost > budget_by_pool[pool]:
                continue
            selected[pool].update(missing)
            used[pool] += cost
    return selected


def eval_calls(calls: list[Call], selected: dict[str, set[ExpertKey]], io_depth: int) -> dict[str, float]:
    out = {
        "calls": 0,
        "full_hit_calls": 0,
        "baseline_waves": 0,
        "waves": 0,
        "waves_saved": 0,
        "bytes_total": 0,
        "bytes_missed": 0,
        "bytes_saved": 0,
        "upgate_calls": 0,
        "upgate_full_hit_calls": 0,
        "upgate_waves_saved": 0,
        "down_calls": 0,
        "down_full_hit_calls": 0,
        "down_waves_saved": 0,
    }
    for call in calls:
        pool = pool_of(call.role)
        if pool not in selected:
            continue
        keys = set(call.keys)
        hits = keys & selected[pool]
        miss_count = len(keys) - len(hits)
        call_bytes = key_bytes(keys)
        missed_bytes = key_bytes(keys - hits)
        base_waves = math.ceil(len(keys) / io_depth)
        waves = math.ceil(miss_count / io_depth)
        saved = base_waves - waves
        out["calls"] += 1
        out["baseline_waves"] += base_waves
        out["waves"] += waves
        out["waves_saved"] += saved
        out["bytes_total"] += call_bytes
        out["bytes_missed"] += missed_bytes
        out["bytes_saved"] += call_bytes - missed_bytes
        if miss_count == 0:
            out["full_hit_calls"] += 1
        prefix = "upgate" if pool == "upgate" else "down"
        out[f"{prefix}_calls"] += 1
        out[f"{prefix}_waves_saved"] += saved
        if miss_count == 0:
            out[f"{prefix}_full_hit_calls"] += 1
    return out


def selected_summary(selected: dict[str, set[ExpertKey]]) -> dict[str, dict[str, float]]:
    return {
        pool: {
            "entries": len(keys),
            "bytes": key_bytes(keys),
            "gib": gib(key_bytes(keys)),
        }
        for pool, keys in selected.items()
    }


def run_leave_one_out(
    prompts: dict[str, list[Call]],
    budget_by_pool: dict[str, int],
    io_depth: int,
    front_layers: list[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    names = sorted(prompts)
    for heldout in names:
        train = [call for name in names if name != heldout for call in prompts[name]]
        validation = prompts[heldout]
        strategies: list[tuple[str, dict[str, set[ExpertKey]]]] = [
            ("global_hotset", select_by_traffic(train, budget_by_pool)),
            ("fullhit_greedy", select_fullhit_greedy(train, budget_by_pool)),
        ]
        for max_layer in front_layers:
            strategies.append((f"front_layers_0_{max_layer}", select_front_layers(train, budget_by_pool, max_layer)))
        for strategy, selected in strategies:
            metrics = eval_calls(validation, selected, io_depth)
            summary = selected_summary(selected)
            rows.append({
                "heldout_prompt": heldout,
                "strategy": strategy,
                **metrics,
                "full_hit_pct": pct(metrics["full_hit_calls"], metrics["calls"]),
                "waves_saved_pct": pct(metrics["waves_saved"], metrics["baseline_waves"]),
                "bytes_saved_pct": pct(metrics["bytes_saved"], metrics["bytes_total"]),
                "upgate_full_hit_pct": pct(metrics["upgate_full_hit_calls"], metrics["upgate_calls"]),
                "down_full_hit_pct": pct(metrics["down_full_hit_calls"], metrics["down_calls"]),
                "selected": summary,
            })
    return rows


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_strategy: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_strategy[str(row["strategy"])].append(row)
    out = []
    for strategy, items in sorted(by_strategy.items()):
        mean_waves_saved = sum(float(r["waves_saved_pct"]) for r in items) / len(items)
        worst_waves_saved = min(float(r["waves_saved_pct"]) for r in items)
        mean_full_hit = sum(float(r["full_hit_pct"]) for r in items) / len(items)
        worst_full_hit = min(float(r["full_hit_pct"]) for r in items)
        mean_bytes_saved = sum(float(r["bytes_saved_pct"]) for r in items) / len(items)
        mean_upgate_full = sum(float(r["upgate_full_hit_pct"]) for r in items) / len(items)
        mean_down_full = sum(float(r["down_full_hit_pct"]) for r in items) / len(items)
        out.append({
            "strategy": strategy,
            "prompts": len(items),
            "mean_waves_saved_pct": mean_waves_saved,
            "worst_waves_saved_pct": worst_waves_saved,
            "mean_full_hit_pct": mean_full_hit,
            "worst_full_hit_pct": worst_full_hit,
            "mean_bytes_saved_pct": mean_bytes_saved,
            "mean_upgate_full_hit_pct": mean_upgate_full,
            "mean_down_full_hit_pct": mean_down_full,
        })
    out.sort(key=lambda r: (r["mean_waves_saved_pct"], r["worst_waves_saved_pct"]), reverse=True)
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "heldout_prompt",
        "strategy",
        "calls",
        "full_hit_calls",
        "full_hit_pct",
        "baseline_waves",
        "waves",
        "waves_saved",
        "waves_saved_pct",
        "bytes_total",
        "bytes_saved",
        "bytes_saved_pct",
        "upgate_full_hit_pct",
        "down_full_hit_pct",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_md(path: Path, result: dict[str, object]) -> None:
    lines = [
        "# Kimi layer full-hit cache simulation",
        "",
        "This is a dev-only offline simulation. It does not change runtime behavior or claim SOTA.",
        "",
        f"- route root: `{result['route_root']}`",
        f"- prompts: `{result['prompt_count']}`",
        f"- io depth: `{result['io_depth']}`",
        f"- upgate budget GiB: `{result['budget_gib']['upgate']:.2f}`",
        f"- down budget GiB: `{result['budget_gib']['down']:.2f}`",
        "",
        "## Aggregate",
        "",
        "| strategy | mean wave saved | worst wave saved | mean full-hit calls | worst full-hit calls | mean bytes saved | up/gate full-hit | down full-hit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["aggregate"]:
        lines.append(
            f"| `{row['strategy']}` | `{row['mean_waves_saved_pct']:.2f}%` | "
            f"`{row['worst_waves_saved_pct']:.2f}%` | `{row['mean_full_hit_pct']:.2f}%` | "
            f"`{row['worst_full_hit_pct']:.2f}%` | `{row['mean_bytes_saved_pct']:.2f}%` | "
            f"`{row['mean_upgate_full_hit_pct']:.2f}%` | `{row['mean_down_full_hit_pct']:.2f}%` |"
        )
    lines += [
        "",
        "## Decision",
        "",
        str(result["decision"]),
        "",
        "## Reproduce",
        "",
        "```bash",
        str(result["reproduce_command"]),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--route-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--io-depth", type=int, default=8)
    ap.add_argument("--upgate-budget-gib", type=float, default=9.1)
    ap.add_argument("--down-budget-gib", type=float, default=5.6)
    ap.add_argument("--front-layers", default="1,2,4,8,12,16")
    args = ap.parse_args()

    prompts = load_prompt_calls(args.route_root)
    budget_by_pool = {
        "upgate": int(args.upgate_budget_gib * 1024**3),
        "down": int(args.down_budget_gib * 1024**3),
    }
    front_layers = [int(x) for x in args.front_layers.split(",") if x.strip()]
    rows = run_leave_one_out(prompts, budget_by_pool, args.io_depth, front_layers)
    agg = aggregate(rows)
    best = agg[0]
    global_row = next((row for row in agg if row["strategy"] == "global_hotset"), None)
    if global_row and best["strategy"] != "global_hotset" and best["mean_waves_saved_pct"] > global_row["mean_waves_saved_pct"] + 1.0:
        decision = (
            f"`{best['strategy']}` beats global_hotset on dev leave-one-out IO waves. "
            "Generate a static profile or env-gated admission policy before held-out testing."
        )
    else:
        decision = (
            "No non-global strategy beats global_hotset by more than 1pp mean wave savings. "
            "Do not implement a layer/front/full-hit cache policy from this simulation alone."
        )

    result = {
        "kind": "kimi_layer_fullhit_cache_sim",
        "route_root": str(args.route_root),
        "prompt_count": len(prompts),
        "io_depth": args.io_depth,
        "budget_gib": {"upgate": args.upgate_budget_gib, "down": args.down_budget_gib},
        "front_layers": front_layers,
        "aggregate": agg,
        "rows": rows,
        "decision": decision,
        "reproduce_command": " ".join(__import__("sys").argv),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.out_dir / "leave-one-out.csv", rows)
    write_md(args.out_dir / "summary.md", result)
    print(f"wrote {args.out_dir / 'summary.json'}")
    print(f"wrote {args.out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
