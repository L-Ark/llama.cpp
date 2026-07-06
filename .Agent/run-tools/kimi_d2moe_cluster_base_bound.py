#!/usr/bin/env python3
"""Offline clustered-base D2MoE feasibility bound for Kimi experts.

This tests the idea "multiple base experts per layer/tensor, residual deltas per
expert" without changing runtime behavior. It dequantizes a hot expert sample,
clusters experts by deterministic weight sketches, builds one weighted base per
cluster, and measures both raw residual size and low-rank residual curves.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import statistics
import sys
import time
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


def load_phase0(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def choose_tensor(phase0: dict[str, Any], preferred_kind: str) -> str:
    for item in phase0.get("plan", {}).get("selected_tensors", []):
        if item.get("kind") == preferred_kind:
            return item["tensor"]
    selected = phase0.get("plan", {}).get("selected_tensors", [])
    if not selected:
        raise RuntimeError("phase0 plan has no selected_tensors")
    return selected[0]["tensor"]


def experts_from_phase0(phase0: dict[str, Any], tensor: str, max_experts: int) -> list[dict[str, Any]]:
    for item in phase0.get("plan", {}).get("selected_tensors", []):
        if item.get("tensor") == tensor:
            out = []
            for expert in (item.get("top_profile_experts") or [])[:max_experts]:
                out.append(
                    {
                        "expert_idx": int(expert["expert_idx"]),
                        "count": int(expert.get("count", 1)),
                        "fallback_us": float(expert.get("fallback_us", 0.0)),
                    }
                )
            return out
    return []


def experts_from_route_profile(path: Path, tensor: str, max_experts: int) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("tensor") != tensor:
                continue
            expert_idx = int(row["expert_idx"])
            if expert_idx in seen:
                continue
            seen.add(expert_idx)
            rows.append(
                {
                    "expert_idx": expert_idx,
                    "count": max(int(row.get("count", 1)), 1),
                    "fallback_us": 0.0,
                    "rank": int(row.get("rank", len(rows) + 1)),
                }
            )
            if len(rows) >= max_experts:
                break
    return rows


def normalize_features(torch: Any, features: Any) -> Any:
    features = features - features.mean(dim=1, keepdim=True)
    norm = torch.linalg.vector_norm(features, dim=1, keepdim=True).clamp_min(1e-12)
    return features / norm


def farthest_first_init(torch: Any, features: Any, weights: Any, k: int) -> Any:
    n = features.shape[0]
    centers = [int(torch.argmax(weights).item())]
    while len(centers) < k:
        selected = features[torch.tensor(centers, dtype=torch.long)]
        dist = torch.cdist(features, selected).min(dim=1).values
        dist[torch.tensor(centers, dtype=torch.long)] = -1
        centers.append(int(torch.argmax(dist).item()))
    return features[torch.tensor(centers, dtype=torch.long)].clone()


def weighted_kmeans(torch: Any, features: Any, weights: Any, k: int, iters: int) -> list[int]:
    n = features.shape[0]
    if k <= 1:
        return [0] * n
    if k >= n:
        return list(range(n))
    centers = farthest_first_init(torch, features, weights, k)
    assignments = torch.zeros(n, dtype=torch.long)
    for _ in range(iters):
        distances = torch.cdist(features, centers)
        assignments = torch.argmin(distances, dim=1)
        for cid in range(k):
            mask = assignments == cid
            if not mask.any():
                continue
            w = weights[mask].reshape(-1, 1)
            centers[cid] = (features[mask] * w).sum(dim=0) / w.sum().clamp_min(1e-12)
        centers = normalize_features(torch, centers)
    return [int(x) for x in assignments.tolist()]


def rank_curve(torch: Any, residual: Any, weight_norm: float, ranks: list[int], oversample: int, niter: int) -> dict[str, Any]:
    residual_fro2 = float(torch.sum(residual * residual).item())
    if residual_fro2 == 0:
        return {
            "raw_residual_vs_weight": 0.0,
            "ranks": [
                {
                    "rank": rank,
                    "residual_norm_ratio": 0.0,
                    "total_error_vs_weight": 0.0,
                    "explained_residual_energy_ratio": 1.0,
                }
                for rank in ranks
            ],
        }
    q = min(min(residual.shape), max(ranks) + oversample)
    _, s, _ = torch.svd_lowrank(residual, q=q, niter=niter)
    s2 = s * s
    rows = []
    for rank in ranks:
        usable = min(rank, s2.numel())
        explained = float(torch.sum(s2[:usable]).item()) / residual_fro2
        explained = min(max(explained, 0.0), 1.0)
        remaining = residual_fro2 * max(0.0, 1.0 - explained)
        rows.append(
            {
                "rank": rank,
                "residual_norm_ratio": math.sqrt(max(0.0, 1.0 - explained)),
                "total_error_vs_weight": math.sqrt(remaining) / weight_norm if weight_norm else 0.0,
                "explained_residual_energy_ratio": explained,
            }
        )
    return {
        "raw_residual_vs_weight": math.sqrt(residual_fro2) / weight_norm if weight_norm else 0.0,
        "ranks": rows,
    }


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Kimi D2MoE clustered-base bound",
        "",
        f"- generated_at: `{result['generated_at']}`",
        f"- tensor: `{result['tensor']}`",
        f"- experts: `{len(result['experts'])}`",
        f"- type: `{result['type']}`",
        f"- shape: `{result['shape']}`",
        "",
        "| clusters | raw residual/weight | rank64 error/weight | rank128 error/weight | rank128 residual ratio | base bf16 MiB |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["cluster_results"]:
        rank64 = row["rank_summary"].get("64", {})
        rank128 = row["rank_summary"].get("128", {})
        lines.append(
            "| {k} | {raw:.4f} | {r64:.4f} | {r128:.4f} | {rr128:.4f} | {base:.2f} |".format(
                k=row["clusters"],
                raw=row["raw_residual_vs_weight_mean"],
                r64=rank64.get("total_error_vs_weight_mean", float("nan")),
                r128=rank128.get("total_error_vs_weight_mean", float("nan")),
                rr128=rank128.get("residual_norm_ratio_mean", float("nan")),
                base=row["base_resident_bf16_mib"],
            )
        )
    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            result["reproduce_command"],
            "```",
            "",
            "This is an offline bound only. It is not a runtime token-rate or quality result.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--phase0-plan", required=True, type=Path)
    parser.add_argument("--route-profile", type=Path)
    parser.add_argument("--libggml-base", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    parser.add_argument("--tensor")
    parser.add_argument("--preferred-kind", default="down")
    parser.add_argument("--max-experts", type=int, default=32)
    parser.add_argument("--clusters", default="1,4,8,16")
    parser.add_argument("--ranks", default="64,128")
    parser.add_argument("--sketch-dim", type=int, default=8192)
    parser.add_argument("--kmeans-iters", type=int, default=20)
    parser.add_argument("--max-svd-experts", type=int, default=8)
    parser.add_argument("--oversample", type=int, default=8)
    parser.add_argument("--niter", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    import torch

    torch.set_num_threads(args.torch_threads)
    inventory = H.load_inventory(args.inventory)
    phase0 = load_phase0(args.phase0_plan)
    tensor = args.tensor or choose_tensor(phase0, args.preferred_kind)
    row = inventory[tensor]
    if args.route_profile:
        experts = experts_from_route_profile(args.route_profile, tensor, args.max_experts)
    else:
        experts = experts_from_phase0(phase0, tensor, args.max_experts)
    if not experts:
        raise SystemExit(f"no experts selected for {tensor}")

    cluster_values = [int(x) for x in args.clusters.split(",") if x]
    ranks = [int(x) for x in args.ranks.split(",") if x]
    lib = H.load_ggml(args.libggml_base)

    matrices = []
    dequant_elapsed = {}
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.seed)
    n_elements = math.prod(row.shape[:2])
    sketch_dim = min(args.sketch_dim, n_elements)
    sketch_idx = torch.randperm(n_elements, generator=gen)[:sketch_dim]

    features = []
    norms = []
    for expert in experts:
        matrix, elapsed_s = R.dequant_to_torch(lib, row, expert["expert_idx"], torch)
        matrices.append(matrix)
        dequant_elapsed[expert["expert_idx"]] = elapsed_s
        flat = matrix.reshape(-1)
        features.append(flat[sketch_idx].clone())
        norms.append(float(torch.linalg.vector_norm(matrix).item()))

    feature_tensor = normalize_features(torch, torch.stack(features, dim=0).to(torch.float32))
    weight_tensor = torch.tensor([max(expert["count"], 1) for expert in experts], dtype=torch.float32)

    svd_order = sorted(range(len(experts)), key=lambda i: experts[i]["count"], reverse=True)[: args.max_svd_experts]
    rows, cols = row.shape[:2]
    result = {
        "kind": "kimi_d2moe_cluster_base_bound",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "inventory": str(args.inventory),
        "phase0_plan": str(args.phase0_plan),
        "route_profile": str(args.route_profile) if args.route_profile else None,
        "libggml_base": str(args.libggml_base),
        "tensor": tensor,
        "layer": row.layer,
        "tensor_kind": row.kind,
        "type": row.type_name,
        "shape": row.shape,
        "expert_bytes": row.expert_bytes,
        "max_experts": args.max_experts,
        "sketch_dim": sketch_dim,
        "kmeans_iters": args.kmeans_iters,
        "max_svd_experts": args.max_svd_experts,
        "ranks": ranks,
        "reproduce_command": " ".join(sys.argv),
        "experts": [
            {
                **expert,
                "dequant_elapsed_s": dequant_elapsed[expert["expert_idx"]],
                "weight_norm": norms[i],
            }
            for i, expert in enumerate(experts)
        ],
        "cluster_results": [],
    }

    for k in cluster_values:
        assignments = weighted_kmeans(torch, feature_tensor, weight_tensor, k, args.kmeans_iters)
        actual_k = len(set(assignments))

        bases = []
        for cid in range(actual_k):
            base = torch.zeros_like(matrices[0])
            denom = 0.0
            for i, matrix in enumerate(matrices):
                if assignments[i] != cid:
                    continue
                w = float(weight_tensor[i].item())
                base.add_(matrix, alpha=w)
                denom += w
            if denom:
                base.mul_(1.0 / denom)
            bases.append(base)

        raw_ratios = []
        svd_details = []
        for i, matrix in enumerate(matrices):
            residual = matrix - bases[assignments[i]]
            raw = float(torch.linalg.vector_norm(residual).item()) / norms[i] if norms[i] else 0.0
            raw_ratios.append(raw)
            if i in svd_order:
                curve = rank_curve(torch, residual, norms[i], ranks, args.oversample, args.niter)
                svd_details.append(
                    {
                        "expert_idx": experts[i]["expert_idx"],
                        "count": experts[i]["count"],
                        "cluster": assignments[i],
                        "curve": curve,
                    }
                )
            del residual

        rank_summary = {}
        for rank in ranks:
            rank_rows = []
            for detail in svd_details:
                for rank_row in detail["curve"]["ranks"]:
                    if rank_row["rank"] == rank:
                        rank_rows.append(rank_row)
            rank_summary[str(rank)] = {
                "total_error_vs_weight_mean": mean([x["total_error_vs_weight"] for x in rank_rows]),
                "residual_norm_ratio_mean": mean([x["residual_norm_ratio"] for x in rank_rows]),
                "explained_residual_energy_ratio_mean": mean(
                    [x["explained_residual_energy_ratio"] for x in rank_rows]
                ),
            }

        base_resident_bf16 = actual_k * rows * cols * 2
        result["cluster_results"].append(
            {
                "clusters": k,
                "actual_clusters": actual_k,
                "assignments": [
                    {"expert_idx": experts[i]["expert_idx"], "cluster": assignments[i], "count": experts[i]["count"]}
                    for i in range(len(experts))
                ],
                "raw_residual_vs_weight_mean": mean(raw_ratios),
                "raw_residual_vs_weight_max": max(raw_ratios) if raw_ratios else 0.0,
                "base_resident_bf16_bytes": int(base_resident_bf16),
                "base_resident_bf16_mib": base_resident_bf16 / (1024 * 1024),
                "rank_summary": rank_summary,
                "svd_details": svd_details,
            }
        )
        del bases

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.out_md, result)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
