#!/usr/bin/env python3
"""Compute a small D2MoE residual-rank curve for Kimi experts.

This is an offline feasibility tool. It dequantizes a small set of same-layer
experts, builds a route-frequency weighted base, and estimates low-rank residual
compressibility with CPU torch. It does not modify runtime behavior.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any


def load_dequant_helper() -> Any:
    helper_path = Path(__file__).with_name("kimi_d2moe_dequant_sample.py")
    spec = importlib.util.spec_from_file_location("kimi_d2moe_dequant_sample", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper module from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


H = load_dequant_helper()


def load_phase0(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def choose_default_tensor(phase0: dict[str, Any], preferred_kind: str) -> str:
    selected = phase0.get("plan", {}).get("selected_tensors", [])
    for item in selected:
        if item.get("kind") == preferred_kind:
            return item["tensor"]
    if selected:
        return selected[0]["tensor"]
    raise RuntimeError("phase0 plan has no selected_tensors")


def select_experts(phase0: dict[str, Any], tensor: str, max_experts: int) -> list[dict[str, Any]]:
    selected = phase0.get("plan", {}).get("selected_tensors", [])
    for item in selected:
        if item.get("tensor") == tensor:
            experts = item.get("top_profile_experts") or [{"expert_idx": i, "count": 1} for i in range(max_experts)]
            out = []
            for expert in experts[:max_experts]:
                count = int(expert.get("count", 1))
                out.append(
                    {
                        "expert_idx": int(expert["expert_idx"]),
                        "count": max(count, 1),
                        "fallback_us": float(expert.get("fallback_us", 0.0)),
                    }
                )
            return out
    raise RuntimeError(f"tensor not found in phase0 selected_tensors: {tensor}")


def dequant_to_torch(lib: Any, row: Any, expert_idx: int, torch: Any) -> tuple[Any, float]:
    read_offset, quant_data = H.read_expert_bytes(row, expert_idx)
    _, traits = H.get_traits(lib, row.type_name)
    n_elements = math.prod(row.shape[:2])
    expected_bytes = n_elements * traits.type_size // traits.blck_size
    if expected_bytes != len(quant_data):
        raise RuntimeError(f"expert bytes mismatch for {row.tensor} expert {expert_idx}")

    src = ctypes.create_string_buffer(quant_data)
    out = torch.empty(n_elements, dtype=torch.float32)
    fn = H.TO_FLOAT(traits.to_float)
    t0 = time.perf_counter()
    fn(ctypes.cast(src, ctypes.c_void_p), ctypes.cast(out.data_ptr(), ctypes.POINTER(ctypes.c_float)), n_elements)
    elapsed_s = time.perf_counter() - t0
    del quant_data
    return out.reshape(tuple(row.shape[:2])), elapsed_s


def low_rank_curve(torch: Any, residual: Any, ranks: list[int], oversample: int, niter: int) -> dict[str, Any]:
    max_rank = max(ranks)
    q = min(min(residual.shape), max_rank + oversample)
    fro2 = float(torch.sum(residual * residual).item())
    if fro2 == 0.0:
        return {
            "fro_norm": 0.0,
            "svd_q": q,
            "ranks": [{"rank": r, "residual_norm_ratio": 0.0, "explained_energy_ratio": 1.0} for r in ranks],
        }
    u, s, v = torch.svd_lowrank(residual, q=q, niter=niter)
    s2 = s * s
    rank_rows = []
    for rank in ranks:
        usable = min(rank, s2.numel())
        explained = float(torch.sum(s2[:usable]).item()) / fro2
        explained = min(max(explained, 0.0), 1.0)
        rank_rows.append(
            {
                "rank": rank,
                "residual_norm_ratio": math.sqrt(max(0.0, 1.0 - explained)),
                "explained_energy_ratio": explained,
            }
        )
    return {
        "fro_norm": math.sqrt(fro2),
        "svd_q": q,
        "ranks": rank_rows,
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Kimi D2MoE residual rank sample",
        "",
        f"- generated_at: `{result['generated_at']}`",
        f"- tensor: `{result['tensor']}`",
        f"- experts: `{len(result['experts'])}`",
        f"- libggml_base: `{result['libggml_base']}`",
        "",
        "| expert | weight | dequant_s | r16 | r32 | r64 | r128 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for expert in result["experts"]:
        rank_map = {row["rank"]: row["residual_norm_ratio"] for row in expert["curve"]["ranks"]}
        lines.append(
            "| {expert} | {weight:.4f} | {dequant:.6f} | {r16:.4f} | {r32:.4f} | {r64:.4f} | {r128:.4f} |".format(
                expert=expert["expert_idx"],
                weight=expert["weight"],
                dequant=expert["dequant_elapsed_s"],
                r16=rank_map.get(16, float("nan")),
                r32=rank_map.get(32, float("nan")),
                r64=rank_map.get(64, float("nan")),
                r128=rank_map.get(128, float("nan")),
            )
        )
    lines.extend(
        [
            "",
            "## Byte Estimate",
            "",
            "| rank | bf16_delta_bytes | delta_vs_full_quant |",
            "| ---: | ---: | ---: |",
        ]
    )
    for row in result["rank_byte_estimates"]:
        lines.append(
            f"| {row['rank']} | {row['bf16_delta_bytes']} | {row['delta_vs_full_quant']:.4f} |"
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
            "This is an offline residual-compressibility bound. It is not a runtime token-rate or quality claim.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--phase0-plan", required=True, type=Path)
    parser.add_argument("--libggml-base", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    parser.add_argument("--tensor")
    parser.add_argument("--preferred-kind", default="down")
    parser.add_argument("--max-experts", type=int, default=4)
    parser.add_argument("--ranks", default="16,32,64,128")
    parser.add_argument("--oversample", type=int, default=8)
    parser.add_argument("--niter", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=8)
    args = parser.parse_args()

    import torch

    torch.set_num_threads(args.torch_threads)
    inventory = H.load_inventory(args.inventory)
    phase0 = load_phase0(args.phase0_plan)
    tensor = args.tensor or choose_default_tensor(phase0, args.preferred_kind)
    row = inventory[tensor]
    experts = select_experts(phase0, tensor, args.max_experts)
    ranks = [int(x) for x in args.ranks.split(",") if x]
    total_count = sum(expert["count"] for expert in experts)
    lib = H.load_ggml(args.libggml_base)

    weighted_base = torch.zeros(tuple(row.shape[:2]), dtype=torch.float32)
    dequant_elapsed: dict[int, float] = {}
    for expert in experts:
        expert_idx = expert["expert_idx"]
        matrix, elapsed_s = dequant_to_torch(lib, row, expert_idx, torch)
        dequant_elapsed[expert_idx] = elapsed_s
        weighted_base.add_(matrix, alpha=expert["count"] / total_count)
        del matrix

    result = {
        "kind": "kimi_d2moe_residual_rank_sample",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "inventory": str(args.inventory),
        "phase0_plan": str(args.phase0_plan),
        "libggml_base": str(args.libggml_base),
        "tensor": tensor,
        "layer": row.layer,
        "tensor_kind": row.kind,
        "type": row.type_name,
        "shape": row.shape,
        "expert_bytes": row.expert_bytes,
        "torch_threads": args.torch_threads,
        "ranks": ranks,
        "reproduce_command": " ".join(os.sys.argv),
        "experts": [],
        "rank_byte_estimates": [],
    }

    rows, cols = row.shape[:2]
    for rank in ranks:
        bf16_delta_bytes = rank * (rows + cols) * 2
        result["rank_byte_estimates"].append(
            {
                "rank": rank,
                "bf16_delta_bytes": int(bf16_delta_bytes),
                "delta_vs_full_quant": bf16_delta_bytes / row.expert_bytes,
            }
        )

    for expert in experts:
        expert_idx = expert["expert_idx"]
        matrix, elapsed_s = dequant_to_torch(lib, row, expert_idx, torch)
        residual = matrix - weighted_base
        curve = low_rank_curve(torch, residual, ranks, args.oversample, args.niter)
        result["experts"].append(
            {
                "expert_idx": expert_idx,
                "count": expert["count"],
                "fallback_us": expert["fallback_us"],
                "weight": expert["count"] / total_count,
                "dequant_elapsed_s": elapsed_s + dequant_elapsed.get(expert_idx, 0.0),
                "curve": curve,
            }
        )
        del matrix, residual

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.out_md, result)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
