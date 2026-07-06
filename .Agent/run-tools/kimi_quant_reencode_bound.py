#!/usr/bin/env python3
"""Offline lower-byte re-encoding error screen for Kimi experts.

This tool dequantizes current GGUF expert weights through ggml and then applies
simple symmetric blockwise quantization to the dequantized values. It is only an
error and byte-ratio screen; it does not create a runtime pack.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
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


def parse_csv_ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def load_samples(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise RuntimeError(f"sample file has no samples: {path}")
    return samples


def quantize_blockwise(torch: Any, values: Any, bits: int, block: int) -> dict[str, Any]:
    flat = values.reshape(-1).to(torch.float32)
    n = flat.numel()
    pad = (block - (n % block)) % block
    if pad:
        flat_padded = torch.nn.functional.pad(flat, (0, pad))
    else:
        flat_padded = flat
    grouped = flat_padded.reshape(-1, block)
    scales = grouped.abs().amax(dim=1).clamp_min(1e-12)

    if bits == 1:
        # Optimistic sign quantization. This is a lower-bound style screen and
        # does not account for the quality loss from losing exact zeros.
        q = torch.sign(grouped)
        recon = q * scales[:, None]
    else:
        qmax = (1 << (bits - 1)) - 1
        q = torch.round(grouped / scales[:, None] * qmax).clamp(-qmax, qmax)
        recon = q * (scales[:, None] / qmax)

    recon = recon.reshape(-1)[:n]
    diff = recon - flat
    diff2 = float(torch.sum(diff * diff).item())
    norm2 = float(torch.sum(flat * flat).item())
    abs_max = float(flat.abs().max().item())
    max_err = float(diff.abs().max().item())
    mean_abs = float(diff.abs().mean().item())
    scale_bytes = scales.numel() * 2
    payload_bytes = math.ceil(n * bits / 8)
    return {
        "bits": bits,
        "block": block,
        "payload_bytes": int(payload_bytes),
        "scale_bytes": int(scale_bytes),
        "total_bytes": int(payload_bytes + scale_bytes),
        "rel_l2_error": math.sqrt(diff2 / norm2) if norm2 > 0 else 0.0,
        "rel_max_error": max_err / abs_max if abs_max > 0 else 0.0,
        "mean_abs_error": mean_abs,
        "finite": bool(torch.isfinite(recon).all().item()),
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Kimi quant re-encode bound",
        "",
        "This is an offline error screen. It does not use held-out test prompts and does not create a runtime pack.",
        "",
        f"- generated_at: `{result['generated_at']}`",
        f"- sample_json: `{result['sample_json']}`",
        f"- inventory: `{result['inventory']}`",
        f"- libggml_base: `{result['libggml_base']}`",
        "",
        "| tensor | expert | type | current MiB | bits | block | ratio | rel L2 | rel max | finite |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for sample in result["samples"]:
        for row in sample["candidates"]:
            lines.append(
                f"| `{sample['tensor']}` | {sample['expert_idx']} | {sample['type']} | "
                f"{sample['expert_bytes'] / (1024 * 1024):.2f} | {row['bits']} | {row['block']} | "
                f"{row['ratio_vs_current']:.3f} | {row['rel_l2_error']:.4f} | "
                f"{row['rel_max_error']:.4f} | {row['finite']} |"
            )
    lines.extend([
        "",
        "## Best By Sample Under Ratio",
        "",
        "| tensor | expert | max ratio | bits | block | ratio | rel L2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for target in result["ratio_targets"]:
        for sample in result["samples"]:
            candidates = [r for r in sample["candidates"] if r["ratio_vs_current"] <= target]
            if not candidates:
                continue
            best = min(candidates, key=lambda r: r["rel_l2_error"])
            lines.append(
                f"| `{sample['tensor']}` | {sample['expert_idx']} | {target:.2f} | "
                f"{best['bits']} | {best['block']} | {best['ratio_vs_current']:.3f} | "
                f"{best['rel_l2_error']:.4f} |"
            )
    lines.extend([
        "",
        "## Reproduce",
        "",
        "```bash",
        result["reproduce_command"],
        "```",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline Kimi lower-byte expert re-encoding error screen.")
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--sample-json", required=True, type=Path)
    parser.add_argument("--libggml-base", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    parser.add_argument("--bits", default="1,2,3,4")
    parser.add_argument("--blocks", default="64,128,256")
    parser.add_argument("--ratio-targets", default="0.55,0.50,0.40")
    parser.add_argument("--torch-threads", type=int, default=8)
    args = parser.parse_args()

    import torch

    torch.set_num_threads(args.torch_threads)
    inventory = H.load_inventory(args.inventory)
    samples = load_samples(args.sample_json)
    lib = H.load_ggml(args.libggml_base)
    bits_values = parse_csv_ints(args.bits)
    block_values = parse_csv_ints(args.blocks)
    ratio_targets = [float(part) for part in args.ratio_targets.split(",") if part.strip()]

    result = {
        "kind": "kimi_quant_reencode_bound",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "inventory": str(args.inventory),
        "sample_json": str(args.sample_json),
        "libggml_base": str(args.libggml_base),
        "bits": bits_values,
        "blocks": block_values,
        "ratio_targets": ratio_targets,
        "reproduce_command": " ".join(os.sys.argv),
        "samples": [],
    }

    for sample in samples:
        tensor = sample["tensor"]
        expert_idx = int(sample["expert_idx"])
        row = inventory[tensor]
        matrix, dequant_s = R.dequant_to_torch(lib, row, expert_idx, torch)
        sample_result = {
            "tensor": tensor,
            "expert_idx": expert_idx,
            "count": int(sample.get("count", 0)),
            "layer": row.layer,
            "kind": row.kind,
            "type": row.type_name,
            "shape": row.shape,
            "expert_bytes": row.expert_bytes,
            "dequant_s": dequant_s,
            "candidates": [],
        }
        for bits in bits_values:
            for block in block_values:
                candidate = quantize_blockwise(torch, matrix, bits, block)
                candidate["ratio_vs_current"] = candidate["total_bytes"] / row.expert_bytes
                sample_result["candidates"].append(candidate)
        result["samples"].append(sample_result)
        del matrix

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.out_md, result)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
