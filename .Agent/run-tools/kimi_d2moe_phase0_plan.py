#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


TENSOR_RE = re.compile(r"^blk\.(?P<layer>\d+)\.ffn_(?P<kind>up|gate|down)_exps\.weight$")


def git_head(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def parse_int(value: str | None, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(float(value))


def parse_float(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def read_inventory(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            tensor = row["tensor"]
            shape = [int(x) for x in row["shape"].split(",") if x]
            if len(shape) < 3:
                raise RuntimeError(f"{tensor}: expected expert tensor shape with expert dim, got {shape}")
            match = TENSOR_RE.match(tensor)
            if not match:
                continue
            rows, cols = int(shape[0]), int(shape[1])
            out[tensor] = {
                "tensor": tensor,
                "layer": int(match.group("layer")),
                "kind": match.group("kind"),
                "type": row["type"],
                "shape": shape,
                "matrix_rows": rows,
                "matrix_cols": cols,
                "tensor_bytes": parse_int(row["tensor_bytes"]),
                "expert_bytes": parse_int(row["expert_bytes"]),
                "n_experts": parse_int(row["n_experts"]),
                "shard_index": parse_int(row["shard_index"]),
                "data_offset": parse_int(row["data_offset"]),
                "shard_path": row["shard_path"],
            }
    if not out:
        raise RuntimeError(f"no expert tensors found in inventory: {path}")
    return out


def iter_profile_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tensor = row.get("tensor") or row.get("up_tensor") or row.get("gate_tensor") or ""
                if tensor:
                    rows.append({**row, "_source_profile": str(path), "_tensor": tensor})
                if row.get("up_tensor") and row.get("gate_tensor"):
                    gate_row = dict(row)
                    gate_row["_source_profile"] = str(path)
                    gate_row["_tensor"] = row["gate_tensor"]
                    rows.append(gate_row)
    return rows


def aggregate_profiles(profile_rows: list[dict[str, Any]], inventory: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    by_layer: dict[int, dict[str, Any]] = defaultdict(lambda: {
        "layer": 0,
        "count": 0,
        "fallback_us": 0.0,
        "weighted_bytes": 0,
        "tensors": defaultdict(lambda: {"count": 0, "fallback_us": 0.0, "weighted_bytes": 0}),
    })
    missing_tensors: dict[str, int] = defaultdict(int)

    for row in profile_rows:
        tensor = row.get("_tensor", "")
        info = inventory.get(tensor)
        if info is None:
            if tensor:
                missing_tensors[tensor] += 1
            continue
        expert_idx = parse_int(row.get("expert_idx"), -1)
        count = parse_int(row.get("count"), 1)
        fallback_us = parse_float(row.get("fallback_us"), 0.0)
        expert_bytes = int(info["expert_bytes"])
        key = (tensor, expert_idx)
        item = by_key.setdefault(key, {
            "tensor": tensor,
            "expert_idx": expert_idx,
            "count": 0,
            "fallback_us": 0.0,
            "expert_bytes": expert_bytes,
            "layer": info["layer"],
            "kind": info["kind"],
            "type": info["type"],
        })
        item["count"] += count
        item["fallback_us"] += fallback_us
        layer = int(info["layer"])
        by_layer[layer]["layer"] = layer
        by_layer[layer]["count"] += count
        by_layer[layer]["fallback_us"] += fallback_us
        by_layer[layer]["weighted_bytes"] += count * expert_bytes
        tensor_stats = by_layer[layer]["tensors"][tensor]
        tensor_stats["count"] += count
        tensor_stats["fallback_us"] += fallback_us
        tensor_stats["weighted_bytes"] += count * expert_bytes

    layer_rows = []
    for layer, stats in by_layer.items():
        tensors = {
            tensor: {
                "count": item["count"],
                "fallback_us": item["fallback_us"],
                "weighted_bytes": item["weighted_bytes"],
            }
            for tensor, item in stats["tensors"].items()
        }
        layer_rows.append({
            "layer": layer,
            "count": stats["count"],
            "fallback_us": stats["fallback_us"],
            "weighted_bytes": stats["weighted_bytes"],
            "tensors": tensors,
        })
    layer_rows.sort(key=lambda r: (r["fallback_us"], r["weighted_bytes"], r["count"]), reverse=True)
    key_rows = sorted(by_key.values(), key=lambda r: (r["fallback_us"], r["count"] * r["expert_bytes"]), reverse=True)
    return {
        "by_layer": layer_rows,
        "by_key": key_rows,
        "missing_tensors": dict(sorted(missing_tensors.items())),
    }


def select_layers(layer_rows: list[dict[str, Any]], inventory: dict[str, dict[str, Any]], top_layers: int) -> list[int]:
    all_layers = sorted({int(v["layer"]) for v in inventory.values()})
    traffic_layers = [int(r["layer"]) for r in layer_rows[:top_layers]]
    selected = set(traffic_layers)
    if all_layers:
        selected.add(all_layers[0])
        selected.add(all_layers[len(all_layers) // 2])
        selected.add(all_layers[-1])
    # Include one early, one middle, and one late layer from the traffic distribution when available.
    if layer_rows:
        thirds = [0, len(layer_rows) // 2, len(layer_rows) - 1]
        for idx in thirds:
            selected.add(int(layer_rows[idx]["layer"]))
    return sorted(selected)


def rank_estimates(info: dict[str, Any], ranks: list[int], bytes_per_value: int) -> list[dict[str, Any]]:
    rows = int(info["matrix_rows"])
    cols = int(info["matrix_cols"])
    full_bytes = int(info["expert_bytes"])
    estimates = []
    for rank in ranks:
        delta_bytes = rank * (rows + cols) * bytes_per_value
        estimates.append({
            "rank": rank,
            "delta_bytes_bf16": delta_bytes,
            "delta_vs_full_ratio": delta_bytes / full_bytes if full_bytes else None,
            "full_expert_bytes": full_bytes,
        })
    return estimates


def build_plan(
    inventory: dict[str, dict[str, Any]],
    aggregate: dict[str, Any],
    selected_layers: list[int],
    ranks: list[int],
    bytes_per_value: int,
    top_experts_per_tensor: int,
) -> dict[str, Any]:
    by_tensor_experts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in aggregate["by_key"]:
        by_tensor_experts[item["tensor"]].append(item)

    selected_tensors = []
    base_resident_bytes_bf16 = 0
    for tensor, info in sorted(inventory.items(), key=lambda kv: (kv[1]["layer"], kv[1]["kind"])):
        if int(info["layer"]) not in selected_layers:
            continue
        rows = int(info["matrix_rows"])
        cols = int(info["matrix_cols"])
        base_bytes_bf16 = rows * cols * bytes_per_value
        base_resident_bytes_bf16 += base_bytes_bf16
        experts = by_tensor_experts.get(tensor, [])[:top_experts_per_tensor]
        selected_tensors.append({
            "tensor": tensor,
            "layer": info["layer"],
            "kind": info["kind"],
            "type": info["type"],
            "shape": info["shape"],
            "expert_bytes": info["expert_bytes"],
            "base_resident_bytes_bf16": base_bytes_bf16,
            "rank_estimates": rank_estimates(info, ranks, bytes_per_value),
            "top_profile_experts": [
                {
                    "expert_idx": e["expert_idx"],
                    "count": e["count"],
                    "fallback_us": e["fallback_us"],
                }
                for e in experts
            ],
            "source": {
                "shard_path": info["shard_path"],
                "data_offset": info["data_offset"],
            },
        })

    return {
        "selected_layers": selected_layers,
        "selected_tensors": selected_tensors,
        "base_resident_bytes_bf16": base_resident_bytes_bf16,
        "base_resident_gib_bf16": base_resident_bytes_bf16 / (1024 ** 3),
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Kimi D2MoE Phase 0 Bound Input Plan",
        "",
        f"- repo_head: `{summary['repo_head']}`",
        f"- inventory: `{summary['inputs']['inventory_tsv']}`",
        f"- selected_layers: `{', '.join(map(str, summary['plan']['selected_layers']))}`",
        f"- selected_tensors: `{len(summary['plan']['selected_tensors'])}`",
        f"- bf16_base_resident_gib: `{summary['plan']['base_resident_gib_bf16']:.3f}`",
        "",
        "## Decision",
        "",
        "This artifact does not claim D2MoE is viable yet. It selects the first",
        "Kimi tensors and experts for a real residual-SVD pass and estimates the",
        "payload ratios for candidate delta ranks.",
        "",
        "## Top Profile Layers",
        "",
        "| layer | count | fallback_ms | weighted_gib |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in summary["aggregate"]["by_layer"][:20]:
        lines.append(
            f"| {row['layer']} | {row['count']} | {row['fallback_us'] / 1000.0:.3f} | "
            f"{row['weighted_bytes'] / (1024 ** 3):.3f} |"
        )
    lines += [
        "",
        "## Selected Tensor Rank Estimates",
        "",
        "| tensor | type | expert MiB | base MiB | rank64 ratio | rank128 ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in summary["plan"]["selected_tensors"]:
        by_rank = {r["rank"]: r for r in item["rank_estimates"]}
        r64 = by_rank.get(64, {}).get("delta_vs_full_ratio")
        r128 = by_rank.get(128, {}).get("delta_vs_full_ratio")
        lines.append(
            f"| `{item['tensor']}` | {item['type']} | {item['expert_bytes'] / (1024 ** 2):.2f} | "
            f"{item['base_resident_bytes_bf16'] / (1024 ** 2):.2f} | "
            f"{r64:.3f} | {r128:.3f} |"
        )
    lines += [
        "",
        "## Next Step",
        "",
        "Implement the real residual reader/dequantizer for the selected tensor set,",
        "then compute weighted base tensors and residual SVD curves. Keep this path",
        "offline and default-off until a real Kimi n96 cold-start run improves over",
        "the same-host baseline under the existing quality and memory gates.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Kimi D2MoE Phase 0 residual-SVD bound inputs.")
    parser.add_argument("--inventory-tsv", type=Path, required=True)
    parser.add_argument("--profile-csv", type=Path, action="append", required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--top-layers", type=int, default=6)
    parser.add_argument("--top-experts-per-tensor", type=int, default=8)
    parser.add_argument("--ranks", default="16,32,64,96,128,192,256")
    parser.add_argument("--bytes-per-value", type=int, default=2, help="BF16/F16 byte size for base and delta estimates.")
    args = parser.parse_args()

    ranks = [int(x) for x in args.ranks.split(",") if x.strip()]
    repo = Path(__file__).resolve().parents[2]
    inventory = read_inventory(args.inventory_tsv)
    profile_rows = iter_profile_rows(args.profile_csv)
    aggregate = aggregate_profiles(profile_rows, inventory)
    layers = select_layers(aggregate["by_layer"], inventory, args.top_layers)
    plan = build_plan(inventory, aggregate, layers, ranks, args.bytes_per_value, args.top_experts_per_tensor)

    summary = {
        "kind": "kimi_d2moe_phase0_bound_input_plan",
        "repo_head": git_head(repo),
        "inputs": {
            "inventory_tsv": str(args.inventory_tsv),
            "profile_csv": [str(p) for p in args.profile_csv],
            "ranks": ranks,
            "bytes_per_value": args.bytes_per_value,
        },
        "aggregate": {
            "by_layer": aggregate["by_layer"],
            "top_keys": aggregate["by_key"][:100],
            "missing_tensors": aggregate["missing_tensors"],
        },
        "plan": plan,
        "limitations": [
            "This is a payload and input-selection bound only; it does not dequantize tensors or compute residual SVD yet.",
            "Rank byte estimates assume BF16 low-rank U/V factors.",
            "A real quality decision requires residual reconstruction and Kimi cold-start semantic validation.",
        ],
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(args.out_md, summary)

    print(f"selected_layers={','.join(map(str, layers))}")
    print(f"selected_tensors={len(plan['selected_tensors'])}")
    print(f"base_resident_gib_bf16={plan['base_resident_gib_bf16']:.3f}")
    print(f"out_json={args.out_json}")
    print(f"out_md={args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
