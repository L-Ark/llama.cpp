#!/usr/bin/env python3
"""Oracle bound for keeping only the largest active expert contributions exact.

This is an offline dev-only screen. It uses dumped activation vectors and the
exact GGUF experts to estimate whether a partial-exact representation could keep
output error low while reducing moved expert bytes enough for the 5 tok/s target.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_residual_helper() -> Any:
    helper_path = Path(__file__).with_name("kimi_d2moe_residual_rank.py")
    spec = importlib.util.spec_from_file_location("kimi_d2moe_residual_rank", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper module from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


R = load_residual_helper()
H = R.H


def parse_csv_ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def parse_csv_floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def load_rows(path: Path, max_records: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            row["record_id"] = int(row["record_id"])
            row["call"] = int(row["call"])
            row["type"] = int(row["type"])
            row["expert_idx"] = int(row["expert_idx"])
            row["active_slot"] = int(row["active_slot"])
            row["dst_id"] = int(row["dst_id"])
            row["token_id"] = int(row["token_id"])
            row["ne00"] = int(row["ne00"])
            row["ne01"] = int(row["ne01"])
            row["expert_bytes"] = int(row["expert_bytes"])
            row["offset_bytes"] = int(row["offset_bytes"])
            row["nbytes"] = int(row["nbytes"])
            rows.append(row)
            if max_records > 0 and len(rows) >= max_records:
                break
    if not rows:
        raise RuntimeError(f"no activation rows in {path}")
    return rows


def load_vector(torch: Any, bin_path: Path, row: dict[str, Any]) -> Any:
    with bin_path.open("rb") as f:
        f.seek(row["offset_bytes"])
        raw = f.read(row["nbytes"])
    expected = row["ne00"] * 4
    if len(raw) != expected:
        raise RuntimeError(f"short activation read for record {row['record_id']}: {len(raw)} != {expected}")
    return torch.frombuffer(bytearray(raw), dtype=torch.float32).clone()


def expert_matvec(torch: Any, matrix: Any, vec: Any, row: dict[str, Any]) -> Any:
    if matrix.shape[1] == row["ne00"] and matrix.shape[0] == row["ne01"]:
        return torch.mv(matrix, vec)
    if matrix.shape[0] == row["ne00"] and matrix.shape[1] == row["ne01"]:
        return torch.mv(matrix.t().contiguous(), vec)
    raise RuntimeError(
        f"shape mismatch for {row['tensor']} expert {row['expert_idx']}: "
        f"matrix={tuple(matrix.shape)} activation ne00/ne01={row['ne00']}/{row['ne01']}"
    )


def silu(torch: Any, x: Any) -> Any:
    return x * torch.sigmoid(x)


def layer_name(tensor: str) -> str:
    # blk.60.ffn_up_exps.weight -> blk.60
    parts = tensor.split(".")
    if len(parts) >= 2 and parts[0] == "blk":
        return f"{parts[0]}.{parts[1]}"
    return tensor


def contribution_rows_for_prompt(
        torch: Any,
        rows: list[dict[str, Any]],
        activation_bin: Path,
        inventory: dict[str, Any],
        lib: Any) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[(row["tensor"], row["expert_idx"])].append(row)

    vectors = {row["record_id"]: load_vector(torch, activation_bin, row) for row in rows}
    outputs: dict[int, Any] = {}
    metadata = {"unique_tensor_experts": len(by_key), "dequantized": 0}

    for (tensor, expert_idx), key_rows in sorted(by_key.items()):
        if tensor not in inventory:
            raise RuntimeError(f"tensor not in inventory: {tensor}")
        matrix, _ = R.dequant_to_torch(lib, inventory[tensor], expert_idx, torch)
        matrix = matrix.to(torch.float32)
        for row in key_rows:
            outputs[row["record_id"]] = expert_matvec(torch, matrix, vectors[row["record_id"]], row)
        metadata["dequantized"] += 1
        del matrix

    contributions: list[dict[str, Any]] = []

    down_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["role"] == "down":
            key = ("down", row["mode"], row["call"], layer_name(row["tensor"]), row["tensor"], row["token_id"])
            down_groups[key].append(row)

    for key, group_rows in sorted(down_groups.items()):
        for row in group_rows:
            out = outputs[row["record_id"]]
            contributions.append({
                "group_key": "|".join(map(str, key)),
                "role": "down",
                "tensor": row["tensor"],
                "expert_idx": row["expert_idx"],
                "active_slot": row["active_slot"],
                "bytes": row["expert_bytes"],
                "norm2": float(torch.sum(out * out).item()),
                "sum_vector": out,
            })

    pair_index: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["role"] not in ("up", "gate"):
            continue
        pair_key = (
            row["mode"],
            row["call"],
            layer_name(row["tensor"]),
            row["expert_idx"],
            row["active_slot"],
            row["dst_id"],
            row["token_id"],
        )
        pair_index[pair_key][row["role"]] = row

    fused_groups: dict[tuple[Any, ...], list[tuple[dict[str, Any], dict[str, Any], Any]]] = defaultdict(list)
    for pair in pair_index.values():
        if "up" not in pair or "gate" not in pair:
            continue
        up = pair["up"]
        gate = pair["gate"]
        fused = outputs[up["record_id"]] * silu(torch, outputs[gate["record_id"]])
        key = ("fused_up_gate", up["mode"], up["call"], layer_name(up["tensor"]), up["token_id"])
        fused_groups[key].append((up, gate, fused))

    for key, pairs in sorted(fused_groups.items()):
        for up, gate, fused in pairs:
            contributions.append({
                "group_key": "|".join(map(str, key)),
                "role": "fused_up_gate",
                "tensor": layer_name(up["tensor"]),
                "expert_idx": up["expert_idx"],
                "active_slot": up["active_slot"],
                "bytes": up["expert_bytes"] + gate["expert_bytes"],
                "norm2": float(torch.sum(fused * fused).item()),
                "sum_vector": fused,
            })

    return contributions, metadata


def summarize_groups(
        torch: Any,
        contributions: list[dict[str, Any]],
        keep_counts: list[int],
        energy_targets: list[float],
        max_rel_l2: float) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in contributions:
        grouped[item["group_key"]].append(item)

    count_acc: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    ratio_acc: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    exact_needed: dict[str, list[int]] = defaultdict(list)
    exact_needed_ratio: dict[str, list[float]] = defaultdict(list)
    target_needed: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    target_needed_ratio: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    group_rows: list[dict[str, Any]] = []

    for group_key, items in sorted(grouped.items()):
        if not items:
            continue
        role = items[0]["role"]
        items_sorted = sorted(items, key=lambda x: x["norm2"], reverse=True)
        total_norm2 = sum(max(0.0, item["norm2"]) for item in items_sorted)
        total_bytes = sum(item["bytes"] for item in items_sorted)
        if total_norm2 <= 0.0 or total_bytes <= 0:
            continue
        prefix_norm2 = []
        prefix_bytes = []
        acc_norm = 0.0
        acc_bytes = 0
        for item in items_sorted:
            acc_norm += max(0.0, item["norm2"])
            acc_bytes += item["bytes"]
            prefix_norm2.append(acc_norm)
            prefix_bytes.append(acc_bytes)

        best_needed = len(items_sorted)
        for idx, kept_norm2 in enumerate(prefix_norm2):
            rel = math.sqrt(max(0.0, total_norm2 - kept_norm2) / total_norm2)
            if rel <= max_rel_l2:
                best_needed = idx + 1
                break
        exact_needed[role].append(best_needed)
        exact_needed_ratio[role].append(prefix_bytes[best_needed - 1] / total_bytes)

        for target in energy_targets:
            label = f"{target:.3f}".rstrip("0").rstrip(".")
            needed = len(items_sorted)
            for idx, kept_norm2 in enumerate(prefix_norm2):
                if kept_norm2 / total_norm2 >= target:
                    needed = idx + 1
                    break
            target_needed[role][label].append(needed)
            target_needed_ratio[role][label].append(prefix_bytes[needed - 1] / total_bytes)

        row: dict[str, Any] = {
            "group_key": group_key,
            "role": role,
            "active": len(items_sorted),
            "total_bytes": total_bytes,
            "top1_energy": prefix_norm2[0] / total_norm2,
            "needed_for_rel_l2": best_needed,
            "needed_for_rel_l2_ratio": prefix_bytes[best_needed - 1] / total_bytes,
        }
        for keep in keep_counts:
            kept = min(keep, len(items_sorted))
            kept_norm2 = prefix_norm2[kept - 1]
            rel = math.sqrt(max(0.0, total_norm2 - kept_norm2) / total_norm2)
            ratio = prefix_bytes[kept - 1] / total_bytes
            count_acc[role][keep].append(rel)
            ratio_acc[role][keep].append(ratio)
            row[f"keep{keep}_rel_l2"] = rel
            row[f"keep{keep}_byte_ratio"] = ratio
        group_rows.append({k: v for k, v in row.items() if k != "sum_vector"})

    def stats(vals: list[float]) -> dict[str, float]:
        if not vals:
            return {"n": 0, "mean": 0.0, "max": 0.0, "p50": 0.0, "p90": 0.0}
        vals_sorted = sorted(vals)
        def pct(p: float) -> float:
            pos = min(len(vals_sorted) - 1, max(0, int(round((len(vals_sorted) - 1) * p))))
            return vals_sorted[pos]
        return {
            "n": len(vals),
            "mean": sum(vals) / len(vals),
            "max": max(vals),
            "p50": pct(0.50),
            "p90": pct(0.90),
        }

    by_role: dict[str, Any] = {}
    roles = sorted(set(item["role"] for item in contributions))
    for role in roles:
        by_role[role] = {
            "groups": len([row for row in group_rows if row["role"] == role]),
            "keep_counts": {
                str(keep): {
                    "rel_l2": stats(count_acc[role][keep]),
                    "byte_ratio": stats(ratio_acc[role][keep]),
                }
                for keep in keep_counts
            },
            "needed_for_rel_l2": stats([float(v) for v in exact_needed[role]]),
            "needed_for_rel_l2_byte_ratio": stats(exact_needed_ratio[role]),
            "needed_for_energy": {
                target: {
                    "count": stats([float(v) for v in target_needed[role][target]]),
                    "byte_ratio": stats(target_needed_ratio[role][target]),
                }
                for target in sorted(target_needed[role])
            },
        }

    return {"by_role": by_role, "groups": group_rows}


def write_md(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Kimi partial-exact contribution oracle",
        "",
        "This is a dev-only offline oracle. It does not change runtime behavior or claim SOTA.",
        "",
        f"- generated_at: `{result['generated_at']}`",
        f"- prompts: `{len(result['prompts'])}`",
        f"- total activation records: `{result['total_records']}`",
        f"- max rel L2 gate: `{result['max_rel_l2']}`",
        "",
        "## Role Summary",
        "",
        "| role | groups | keep | mean rel L2 | p90 rel L2 | max rel L2 | mean exact byte ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for role, role_row in result["summary"]["by_role"].items():
        for keep, keep_row in role_row["keep_counts"].items():
            lines.append(
                f"| `{role}` | `{role_row['groups']}` | `{keep}` | "
                f"`{keep_row['rel_l2']['mean']:.6f}` | `{keep_row['rel_l2']['p90']:.6f}` | "
                f"`{keep_row['rel_l2']['max']:.6f}` | `{keep_row['byte_ratio']['mean']:.4f}` |"
            )

    lines += [
        "",
        "## Exact Experts Needed For Error Gate",
        "",
        "| role | mean count | p90 count | max count | mean exact byte ratio | p90 byte ratio | max byte ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for role, role_row in result["summary"]["by_role"].items():
        count = role_row["needed_for_rel_l2"]
        ratio = role_row["needed_for_rel_l2_byte_ratio"]
        lines.append(
            f"| `{role}` | `{count['mean']:.2f}` | `{count['p90']:.2f}` | `{count['max']:.2f}` | "
            f"`{ratio['mean']:.4f}` | `{ratio['p90']:.4f}` | `{ratio['max']:.4f}` |"
        )

    lines += [
        "",
        "## Energy Targets",
        "",
        "| role | energy | mean count | mean exact byte ratio | p90 exact byte ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for role, role_row in result["summary"]["by_role"].items():
        for target, target_row in role_row["needed_for_energy"].items():
            lines.append(
                f"| `{role}` | `{target}` | `{target_row['count']['mean']:.2f}` | "
                f"`{target_row['byte_ratio']['mean']:.4f}` | `{target_row['byte_ratio']['p90']:.4f}` |"
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
    parser.add_argument("--prompt-root", type=Path, action="append", required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--libggml-base", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--keep-counts", default="1,2,4,6,8")
    parser.add_argument("--energy-targets", default="0.90,0.95,0.99")
    parser.add_argument("--max-records-per-prompt", type=int, default=72)
    parser.add_argument("--max-rel-l2", type=float, default=0.10)
    parser.add_argument("--target-byte-ratio", type=float, default=0.40)
    parser.add_argument("--torch-threads", type=int, default=8)
    args = parser.parse_args()

    import torch

    torch.set_num_threads(args.torch_threads)
    inventory = H.load_inventory(args.inventory)
    lib = H.load_ggml(args.libggml_base)
    keep_counts = parse_csv_ints(args.keep_counts)
    energy_targets = parse_csv_floats(args.energy_targets)

    all_contribs: list[dict[str, Any]] = []
    prompts: list[dict[str, Any]] = []
    total_records = 0
    for root in args.prompt_root:
        act_csv = root / "act" / "activations.csv"
        act_bin = root / "act" / "activations.f32"
        rows = load_rows(act_csv, args.max_records_per_prompt)
        contribs, metadata = contribution_rows_for_prompt(torch, rows, act_bin, inventory, lib)
        prompt_id = root.name
        for item in contribs:
            item["group_key"] = f"{prompt_id}|{item['group_key']}"
        all_contribs.extend(contribs)
        total_records += len(rows)
        prompts.append({
            "prompt_id": prompt_id,
            "root": str(root),
            "records": len(rows),
            **metadata,
        })

    summary = summarize_groups(torch, all_contribs, keep_counts, energy_targets, args.max_rel_l2)
    advance = False
    for role_row in summary["by_role"].values():
        ratio = role_row["needed_for_rel_l2_byte_ratio"]["mean"]
        max_count = role_row["needed_for_rel_l2"]["max"]
        if ratio <= args.target_byte_ratio and max_count <= max(keep_counts):
            advance = True
    if advance:
        decision = (
            "At least one role shows oracle partial-exact retention inside the byte target. "
            "Do not use held-out yet; design a dev-only mixed exact/residual candidate and rerun the activation-output gate."
        )
    else:
        decision = (
            "The oracle requires too many exact active expert contributions to meet the rel L2 gate inside the "
            "0.30x-0.40x byte budget. Do not implement partial-exact topK retention as a primary runtime path."
        )

    result = {
        "kind": "kimi_partial_exact_contribution_oracle",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prompts": prompts,
        "total_records": total_records,
        "keep_counts": keep_counts,
        "energy_targets": energy_targets,
        "max_rel_l2": args.max_rel_l2,
        "target_byte_ratio": args.target_byte_ratio,
        "summary": summary,
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
