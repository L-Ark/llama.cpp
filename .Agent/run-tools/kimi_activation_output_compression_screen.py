#!/usr/bin/env python3
"""Screen compressed expert compute candidates with real MoE activations.

Input comes from the default-off runtime dump controlled by
GGML_MOE_ACTIVATION_DUMP_DIR. The tool dequantizes the selected GGUF expert,
re-encodes it with simple blockwise low-bit candidates, and compares matvec
outputs on the dumped activation vectors.

This is an offline screen. It does not create a runtime pack and does not claim
token-rate improvement.
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


def load_activation_rows(path: Path, max_records: int) -> list[dict[str, Any]]:
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


def quantize_blockwise(torch: Any, values: Any, bits: int, block: int) -> tuple[Any, dict[str, Any]]:
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
        q = torch.sign(grouped)
        recon = q * scales[:, None]
    else:
        qmax = (1 << (bits - 1)) - 1
        q = torch.round(grouped / scales[:, None] * qmax).clamp(-qmax, qmax)
        recon = q * (scales[:, None] / qmax)
    recon = recon.reshape(-1)[:n].reshape_as(values)
    payload_bytes = math.ceil(n * bits / 8)
    scale_bytes = scales.numel() * 2
    info = {
        "bits": bits,
        "block": block,
        "payload_bytes": int(payload_bytes),
        "scale_bytes": int(scale_bytes),
        "total_bytes": int(payload_bytes + scale_bytes),
    }
    return recon, info


class Acc:
    def __init__(self) -> None:
        self.rows = 0
        self.rel_sum = 0.0
        self.rel_max = 0.0
        self.abs_sum = 0.0
        self.abs_max = 0.0
        self.out_norm_sum = 0.0
        self.ratio_sum = 0.0

    def add(self, rel: float, abs_mean: float, abs_max: float, out_norm: float, ratio: float) -> None:
        self.rows += 1
        self.rel_sum += rel
        self.rel_max = max(self.rel_max, rel)
        self.abs_sum += abs_mean
        self.abs_max = max(self.abs_max, abs_max)
        self.out_norm_sum += out_norm
        self.ratio_sum += ratio

    def row(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "mean_rel_l2": self.rel_sum / self.rows if self.rows else 0.0,
            "max_rel_l2": self.rel_max,
            "mean_abs_error": self.abs_sum / self.rows if self.rows else 0.0,
            "max_abs_error": self.abs_max,
            "mean_output_l2": self.out_norm_sum / self.rows if self.rows else 0.0,
            "mean_byte_ratio": self.ratio_sum / self.rows if self.rows else 0.0,
        }


def error_metrics(torch: Any, exact: Any, cand: Any) -> tuple[float, float, float, float]:
    diff = cand - exact
    diff_l2 = float(torch.linalg.vector_norm(diff).item())
    exact_l2 = float(torch.linalg.vector_norm(exact).item())
    rel = diff_l2 / max(exact_l2, 1e-30)
    return rel, float(diff.abs().mean().item()), float(diff.abs().max().item()), exact_l2


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Screen compressed Kimi expert outputs on dumped real activations.")
    parser.add_argument("--activation-csv", type=Path, required=True)
    parser.add_argument("--activation-bin", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--libggml-base", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--bits", default="1,2")
    parser.add_argument("--blocks", default="64,128,256")
    parser.add_argument("--max-records", type=int, default=64)
    parser.add_argument("--torch-threads", type=int, default=8)
    args = parser.parse_args()

    import torch

    torch.set_num_threads(args.torch_threads)
    inventory = H.load_inventory(args.inventory)
    lib = H.load_ggml(args.libggml_base)
    rows = load_activation_rows(args.activation_csv, args.max_records)
    bits_values = parse_csv_ints(args.bits)
    block_values = parse_csv_ints(args.blocks)

    by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[(row["tensor"], row["expert_idx"])].append(row)

    aggregate: dict[str, Acc] = defaultdict(Acc)
    per_role: dict[str, dict[str, Any]] = {}
    exact_outputs: dict[int, Any] = {}
    cand_outputs: dict[tuple[int, int, int], Any] = {}
    candidate_bytes: dict[tuple[str, int, int, int], int] = {}

    for (tensor, expert_idx), key_rows in sorted(by_key.items()):
        if tensor not in inventory:
            raise RuntimeError(f"tensor not in inventory: {tensor}")
        inv = inventory[tensor]
        matrix, dequant_s = R.dequant_to_torch(lib, inv, expert_idx, torch)
        matrix = matrix.to(torch.float32)
        if not (
            (matrix.shape[1] == key_rows[0]["ne00"] and matrix.shape[0] == key_rows[0]["ne01"]) or
            (matrix.shape[0] == key_rows[0]["ne00"] and matrix.shape[1] == key_rows[0]["ne01"])
        ):
            raise RuntimeError(
                f"shape mismatch for {tensor} expert {expert_idx}: matrix={tuple(matrix.shape)} "
                f"activation ne00/ne01={key_rows[0]['ne00']}/{key_rows[0]['ne01']}"
            )

        vectors = {row["record_id"]: load_vector(torch, args.activation_bin, row) for row in key_rows}
        exact_by_record = {
            row["record_id"]: expert_matvec(torch, matrix, vectors[row["record_id"]], row)
            for row in key_rows
        }
        exact_outputs.update(exact_by_record)

        for bits in bits_values:
            for block in block_values:
                recon, qinfo = quantize_blockwise(torch, matrix, bits, block)
                candidate_bytes[(tensor, expert_idx, bits, block)] = qinfo["total_bytes"]
                ratio = qinfo["total_bytes"] / key_rows[0]["expert_bytes"]
                for row in key_rows:
                    rid = row["record_id"]
                    cand = expert_matvec(torch, recon, vectors[rid], row)
                    cand_outputs[(rid, bits, block)] = cand
                    rel, abs_mean, abs_max, out_norm = error_metrics(torch, exact_by_record[rid], cand)
                    key = f"{row['role']}:bits{bits}:block{block}"
                    aggregate[key].add(rel, abs_mean, abs_max, out_norm, ratio)
        per_role_key = f"{tensor}:{expert_idx}"
        per_role[per_role_key] = {
            "tensor": tensor,
            "expert_idx": expert_idx,
            "role": key_rows[0]["role"],
            "records": len(key_rows),
            "type": inv.type_name,
            "expert_bytes": inv.expert_bytes,
            "dequant_s": dequant_s,
        }
        del matrix

    rows_by_id = {row["record_id"]: row for row in rows}
    pair_acc: dict[str, Acc] = defaultdict(Acc)
    pair_index: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["role"] not in ("up", "gate"):
            continue
        pair_key = (row["mode"], row["call"], row["expert_idx"], row["active_slot"], row["dst_id"], row["token_id"])
        pair_index[pair_key][row["role"]] = row

    for pair in pair_index.values():
        if "up" not in pair or "gate" not in pair:
            continue
        up = pair["up"]
        gate = pair["gate"]
        exact = exact_outputs[up["record_id"]] * silu(torch, exact_outputs[gate["record_id"]])
        for bits in bits_values:
            for block in block_values:
                up_c = cand_outputs.get((up["record_id"], bits, block))
                gate_c = cand_outputs.get((gate["record_id"], bits, block))
                if up_c is None or gate_c is None:
                    continue
                cand = up_c * silu(torch, gate_c)
                rel, abs_mean, abs_max, out_norm = error_metrics(torch, exact, cand)
                up_bytes = candidate_bytes[(up["tensor"], up["expert_idx"], bits, block)]
                gate_bytes = candidate_bytes[(gate["tensor"], gate["expert_idx"], bits, block)]
                ratio = (up_bytes + gate_bytes) / max(up["expert_bytes"] + gate["expert_bytes"], 1)
                pair_acc[f"fused_up_gate:bits{bits}:block{block}"].add(rel, abs_mean, abs_max, out_norm, ratio)

    aggregate_rows = {key: acc.row() for key, acc in sorted(aggregate.items())}
    fused_rows = {key: acc.row() for key, acc in sorted(pair_acc.items())}
    target_ratio = 0.40
    max_mean_rel_l2 = 0.10
    passing_matvec = [
        {"candidate": key, **row}
        for key, row in aggregate_rows.items()
        if row["mean_byte_ratio"] <= target_ratio and row["mean_rel_l2"] <= max_mean_rel_l2
    ]
    passing_fused = [
        {"candidate": key, **row}
        for key, row in fused_rows.items()
        if row["mean_byte_ratio"] <= target_ratio and row["mean_rel_l2"] <= max_mean_rel_l2
    ]
    result = {
        "kind": "kimi_activation_output_compression_screen",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "activation_csv": str(args.activation_csv),
        "activation_bin": str(args.activation_bin),
        "inventory": str(args.inventory),
        "libggml_base": str(args.libggml_base),
        "records_loaded": len(rows),
        "unique_tensor_experts": len(by_key),
        "bits": bits_values,
        "blocks": block_values,
        "aggregate": aggregate_rows,
        "fused_up_gate": fused_rows,
        "tensor_experts": per_role,
        "decision": {
            "status": "screen_only",
            "rule": "Candidates near 0.30x-0.40x moved bytes need low activation-output error before runtime work.",
            "target_byte_ratio": target_ratio,
            "max_mean_rel_l2": max_mean_rel_l2,
            "passing_matvec_candidates": passing_matvec,
            "passing_fused_candidates": passing_fused,
            "advance_blockwise_lowbit": bool(passing_matvec and passing_fused),
            "runtime_overhead_note": "This tool reconstructs candidates offline. A runtime implementation would need direct compressed kernels; full reconstruction is not an acceptable runtime path.",
        },
        "reproduce_command": " ".join(os.sys.argv),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.out_md, result)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Kimi activation-output compression screen",
        "",
        "This is a dev-only offline screen using real dumped MoE activation vectors. It does not change runtime behavior.",
        "",
        f"- generated_at: `{result['generated_at']}`",
        f"- activation_csv: `{result['activation_csv']}`",
        f"- activation_bin: `{result['activation_bin']}`",
        f"- records_loaded: `{result['records_loaded']}`",
        f"- unique tensor experts: `{result['unique_tensor_experts']}`",
        "",
        "## Gate Result",
        "",
        f"- target byte ratio: `<= {result['decision']['target_byte_ratio']:.2f}x`",
        f"- max mean rel L2: `<= {result['decision']['max_mean_rel_l2']:.2f}`",
        f"- passing matvec candidates: `{len(result['decision']['passing_matvec_candidates'])}`",
        f"- passing fused candidates: `{len(result['decision']['passing_fused_candidates'])}`",
        f"- advance blockwise low-bit path: `{result['decision']['advance_blockwise_lowbit']}`",
        "",
        "## Matvec Output Error",
        "",
        "| candidate | rows | byte ratio | mean rel L2 | max rel L2 | mean abs err | max abs err |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, row in result["aggregate"].items():
        lines.append(
            f"| `{key}` | `{row['rows']}` | `{row['mean_byte_ratio']:.4f}` | "
            f"`{row['mean_rel_l2']:.6f}` | `{row['max_rel_l2']:.6f}` | "
            f"`{row['mean_abs_error']:.6g}` | `{row['max_abs_error']:.6g}` |"
        )

    lines += [
        "",
        "## Fused Up/Gate Output Error",
        "",
        "| candidate | pairs | byte ratio | mean rel L2 | max rel L2 | mean abs err | max abs err |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, row in result["fused_up_gate"].items():
        lines.append(
            f"| `{key}` | `{row['rows']}` | `{row['mean_byte_ratio']:.4f}` | "
            f"`{row['mean_rel_l2']:.6f}` | `{row['max_rel_l2']:.6f}` | "
            f"`{row['mean_abs_error']:.6g}` | `{row['max_abs_error']:.6g}` |"
        )

    lines += [
        "",
        "## Decision",
        "",
        "- This is a screen only; it does not produce a runtime pack or SOTA claim.",
        "- Full offline reconstruction is not a valid runtime path. Any accepted candidate needs direct compressed kernels.",
        "- Candidates around `0.30x-0.40x` moved bytes must show low activation-output error before runtime work.",
        "",
        "## Reproduce",
        "",
        "```bash",
        result["reproduce_command"],
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
