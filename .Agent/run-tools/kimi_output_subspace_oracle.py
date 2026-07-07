#!/usr/bin/env python3
"""Oracle bound for low-rank active expert output subspaces.

This is a dev-only offline screen. It fits an SVD basis on the exact active
expert contribution outputs for each call, then measures whether a small rank can
reconstruct the summed contribution. Because the basis is fit on the same call,
this is an optimistic upper bound, not a runtime implementation.
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
    parts = tensor.split(".")
    if len(parts) >= 2 and parts[0] == "blk":
        return f"{parts[0]}.{parts[1]}"
    return tensor


def contribution_groups_for_prompt(
        torch: Any,
        rows: list[dict[str, Any]],
        activation_bin: Path,
        inventory: dict[str, Any],
        lib: Any,
        prompt_id: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
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

    groups: list[dict[str, Any]] = []

    down_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["role"] == "down":
            key = (prompt_id, "down", row["mode"], row["call"], layer_name(row["tensor"]), row["tensor"], row["token_id"])
            down_groups[key].append(row)
    for key, group_rows in sorted(down_groups.items()):
        group_rows.sort(key=lambda r: r["active_slot"])
        mat = torch.stack([outputs[row["record_id"]] for row in group_rows], dim=0)
        groups.append({
            "group_key": "|".join(map(str, key)),
            "prompt_id": prompt_id,
            "role": "down",
            "active": len(group_rows),
            "bytes": sum(row["expert_bytes"] for row in group_rows),
            "matrix": mat,
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

    fused_groups: dict[tuple[Any, ...], list[tuple[dict[str, Any], Any]]] = defaultdict(list)
    for pair in pair_index.values():
        if "up" not in pair or "gate" not in pair:
            continue
        up = pair["up"]
        gate = pair["gate"]
        fused = outputs[up["record_id"]] * silu(torch, outputs[gate["record_id"]])
        key = (prompt_id, "fused_up_gate", up["mode"], up["call"], layer_name(up["tensor"]), up["token_id"])
        fused_groups[key].append((up, fused))
    for key, pairs in sorted(fused_groups.items()):
        pairs.sort(key=lambda p: p[0]["active_slot"])
        mat = torch.stack([fused for _, fused in pairs], dim=0)
        groups.append({
            "group_key": "|".join(map(str, key)),
            "prompt_id": prompt_id,
            "role": "fused_up_gate",
            "active": len(pairs),
            "bytes": sum(row["expert_bytes"] * 2 for row, _ in pairs),
            "matrix": mat,
        })

    return groups, metadata


class Acc:
    def __init__(self) -> None:
        self.n = 0
        self.sum_rel = 0.0
        self.max_rel = 0.0
        self.sum_fro = 0.0
        self.max_fro = 0.0
        self.sum_ratio = 0.0

    def add(self, rel: float, fro: float, ratio: float) -> None:
        self.n += 1
        self.sum_rel += rel
        self.max_rel = max(self.max_rel, rel)
        self.sum_fro += fro
        self.max_fro = max(self.max_fro, fro)
        self.sum_ratio += ratio

    def row(self) -> dict[str, Any]:
        return {
            "groups": self.n,
            "mean_sum_rel_l2": self.sum_rel / self.n if self.n else 0.0,
            "max_sum_rel_l2": self.max_rel,
            "mean_fro_rel_l2": self.sum_fro / self.n if self.n else 0.0,
            "max_fro_rel_l2": self.max_fro,
            "mean_rank_ratio": self.sum_ratio / self.n if self.n else 0.0,
        }


def analyze_groups(torch: Any, groups: list[dict[str, Any]], ranks: list[int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    accs: dict[tuple[str, int], Acc] = defaultdict(Acc)
    group_rows: list[dict[str, Any]] = []
    for group in groups:
        mat = group["matrix"].to(torch.float32)
        active = int(group["active"])
        exact_sum = mat.sum(dim=0)
        exact_sum_norm = float(torch.linalg.vector_norm(exact_sum).item())
        fro_norm = float(torch.linalg.matrix_norm(mat, ord="fro").item())
        u, s, vh = torch.linalg.svd(mat, full_matrices=False)
        row = {
            "group_key": group["group_key"],
            "prompt_id": group["prompt_id"],
            "role": group["role"],
            "active": active,
            "bytes": group["bytes"],
        }
        for rank in ranks:
            usable = max(1, min(rank, active, s.numel()))
            recon = (u[:, :usable] * s[:usable]) @ vh[:usable, :]
            sum_rel = float(torch.linalg.vector_norm(recon.sum(dim=0) - exact_sum).item()) / max(exact_sum_norm, 1e-30)
            fro_rel = float(torch.linalg.matrix_norm(recon - mat, ord="fro").item()) / max(fro_norm, 1e-30)
            ratio = usable / max(active, 1)
            accs[(group["role"], rank)].add(sum_rel, fro_rel, ratio)
            row[f"rank{rank}_sum_rel_l2"] = sum_rel
            row[f"rank{rank}_fro_rel_l2"] = fro_rel
            row[f"rank{rank}_ratio"] = ratio
        group_rows.append(row)
    summary: dict[str, Any] = {}
    for (role, rank), acc in sorted(accs.items()):
        summary[f"{role}:rank{rank}"] = {"role": role, "rank": rank, **acc.row()}
    return summary, group_rows


def write_md(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Kimi output-subspace oracle",
        "",
        "This is a dev-only offline oracle. It does not change runtime behavior or claim SOTA.",
        "",
        f"- generated_at: `{result['generated_at']}`",
        f"- prompts: `{len(result['prompts'])}`",
        f"- total activation records: `{result['total_records']}`",
        f"- ranks: `{result['ranks']}`",
        "",
        "## Summary",
        "",
        "| role | rank | rank ratio | groups | mean sum rel L2 | max sum rel L2 | mean row Fro rel L2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["summary"].values():
        lines.append(
            f"| `{row['role']}` | `{row['rank']}` | `{row['mean_rank_ratio']:.4f}` | `{row['groups']}` | "
            f"`{row['mean_sum_rel_l2']:.6f}` | `{row['max_sum_rel_l2']:.6f}` | "
            f"`{row['mean_fro_rel_l2']:.6f}` |"
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
    parser.add_argument("--ranks", default="1,2,3,4")
    parser.add_argument("--max-records-per-prompt", type=int, default=72)
    parser.add_argument("--target-rank", type=int, default=3)
    parser.add_argument("--target-mean-rel-l2", type=float, default=0.10)
    parser.add_argument("--torch-threads", type=int, default=8)
    args = parser.parse_args()

    import torch

    torch.set_num_threads(args.torch_threads)
    inventory = H.load_inventory(args.inventory)
    lib = H.load_ggml(args.libggml_base)
    ranks = parse_csv_ints(args.ranks)

    all_groups: list[dict[str, Any]] = []
    prompts: list[dict[str, Any]] = []
    total_records = 0
    for root in args.prompt_root:
        prompt_id = root.name
        rows = load_rows(root / "act" / "activations.csv", args.max_records_per_prompt)
        groups, metadata = contribution_groups_for_prompt(
            torch,
            rows,
            root / "act" / "activations.f32",
            inventory,
            lib,
            prompt_id,
        )
        all_groups.extend(groups)
        total_records += len(rows)
        prompts.append({
            "prompt_id": prompt_id,
            "root": str(root),
            "records": len(rows),
            "groups": len(groups),
            **metadata,
        })

    summary, group_rows = analyze_groups(torch, all_groups, ranks)
    passing = [
        row for row in summary.values()
        if int(row["rank"]) <= args.target_rank and float(row["mean_sum_rel_l2"]) <= args.target_mean_rel_l2
    ]
    roles_passing = {row["role"] for row in passing}
    if {"down", "fused_up_gate"}.issubset(roles_passing):
        decision = (
            "A low-rank output-subspace oracle passes the dev gate for both roles. "
            "Do not use held-out yet; design a concrete dev-only runtime/storage model before validation."
        )
    else:
        decision = (
            "Rank <=3 dynamic output-subspace reconstruction does not pass the dev gate for both roles. "
            "Do not implement output-subspace runtime kernels as the next primary path."
        )

    result = {
        "kind": "kimi_output_subspace_oracle",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prompts": prompts,
        "total_records": total_records,
        "ranks": ranks,
        "target_rank": args.target_rank,
        "target_mean_rel_l2": args.target_mean_rel_l2,
        "summary": summary,
        "groups": group_rows,
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
