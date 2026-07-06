#!/usr/bin/env python3
"""Dequantize selected Kimi expert slices through ggml for D2MoE feasibility.

This is an offline smoke test for the D2MoE migration plan. It does not change
runtime behavior. The goal is to prove that the exact expert bytes selected by
the Phase 0 plan can be decoded reproducibly with this checkout's ggml build.
"""

from __future__ import annotations

import argparse
import ctypes
import csv
import hashlib
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TYPE_IDS = {
    "F32": 0,
    "F16": 1,
    "Q4_0": 2,
    "Q4_1": 3,
    "Q5_0": 6,
    "Q5_1": 7,
    "Q8_0": 8,
    "Q8_1": 9,
    "Q2_K": 10,
    "Q3_K": 11,
    "Q4_K": 12,
    "Q5_K": 13,
    "Q6_K": 14,
    "Q8_K": 15,
    "IQ2_XXS": 16,
    "IQ2_XS": 17,
    "IQ3_XXS": 18,
    "IQ1_S": 19,
    "IQ4_NL": 20,
    "IQ3_S": 21,
    "IQ2_S": 22,
    "IQ4_XS": 23,
    "BF16": 30,
    "NVFP4": 40,
}


TO_FLOAT = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64)


class GGMLTypeTraits(ctypes.Structure):
    _fields_ = [
        ("type_name", ctypes.c_char_p),
        ("blck_size", ctypes.c_int64),
        ("blck_size_interleave", ctypes.c_int64),
        ("type_size", ctypes.c_size_t),
        ("is_quantized", ctypes.c_bool),
        ("to_float", ctypes.c_void_p),
        ("from_float_ref", ctypes.c_void_p),
    ]


@dataclass(frozen=True)
class InventoryRow:
    layer: int
    kind: str
    type_name: str
    tensor_bytes: int
    expert_bytes: int
    n_experts: int
    data_offset: int
    shape: tuple[int, ...]
    tensor: str
    shard_path: str


def load_inventory(path: Path) -> dict[str, InventoryRow]:
    rows: dict[str, InventoryRow] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows[row["tensor"]] = InventoryRow(
                layer=int(row["layer"]),
                kind=row["kind"],
                type_name=row["type"],
                tensor_bytes=int(row["tensor_bytes"]),
                expert_bytes=int(row["expert_bytes"]),
                n_experts=int(row["n_experts"]),
                data_offset=int(row["data_offset"]),
                shape=tuple(int(x) for x in row["shape"].split(",")),
                tensor=row["tensor"],
                shard_path=row["shard_path"],
            )
    return rows


def load_phase0_samples(path: Path, inventory: dict[str, InventoryRow], max_per_type: int) -> list[tuple[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("plan", {}).get("selected_tensors", [])
    samples: list[tuple[str, int]] = []
    seen_types: dict[str, int] = {}

    for item in selected:
        tensor = item["tensor"]
        inv = inventory.get(tensor)
        if inv is None:
            continue
        if seen_types.get(inv.type_name, 0) >= max_per_type:
            continue
        experts = item.get("top_profile_experts") or [{"expert_idx": 0}]
        samples.append((tensor, int(experts[0]["expert_idx"])))
        seen_types[inv.type_name] = seen_types.get(inv.type_name, 0) + 1

    return samples


def load_ggml(lib_path: Path) -> ctypes.CDLL:
    lib_dir = str(lib_path.parent)
    if lib_dir not in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        os.environ["LD_LIBRARY_PATH"] = lib_dir + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    lib = ctypes.CDLL(str(lib_path))
    lib.ggml_get_type_traits.argtypes = [ctypes.c_int]
    lib.ggml_get_type_traits.restype = ctypes.POINTER(GGMLTypeTraits)
    return lib


def get_traits(lib: ctypes.CDLL, type_name: str) -> tuple[int, GGMLTypeTraits]:
    if type_name not in TYPE_IDS:
        raise ValueError(f"unsupported type name in inventory: {type_name}")
    type_id = TYPE_IDS[type_name]
    ptr = lib.ggml_get_type_traits(type_id)
    if not ptr:
        raise RuntimeError(f"ggml_get_type_traits returned null for {type_name}/{type_id}")
    traits = ptr.contents
    if not traits.to_float:
        raise RuntimeError(f"ggml type {type_name}/{type_id} does not expose to_float")
    return type_id, traits


def read_expert_bytes(row: InventoryRow, expert_idx: int) -> tuple[int, bytes]:
    if expert_idx < 0 or expert_idx >= row.n_experts:
        raise ValueError(f"expert_idx {expert_idx} outside [0, {row.n_experts}) for {row.tensor}")
    offset = row.data_offset + expert_idx * row.expert_bytes
    with open(row.shard_path, "rb") as f:
        f.seek(offset)
        data = f.read(row.expert_bytes)
    if len(data) != row.expert_bytes:
        raise RuntimeError(
            f"short read for {row.tensor} expert {expert_idx}: got {len(data)}, expected {row.expert_bytes}"
        )
    return offset, data


def dequantize(lib: ctypes.CDLL, row: InventoryRow, quant_data: bytes) -> tuple[dict[str, Any], list[float]]:
    type_id, traits = get_traits(lib, row.type_name)
    n_elements = math.prod(row.shape[:2])
    if n_elements % traits.blck_size != 0:
        raise RuntimeError(f"{row.tensor} expert elements {n_elements} not divisible by block {traits.blck_size}")
    expected_bytes = n_elements * traits.type_size // traits.blck_size
    if expected_bytes != len(quant_data):
        raise RuntimeError(
            f"{row.tensor} expert bytes mismatch: traits expect {expected_bytes}, inventory has {len(quant_data)}"
        )

    src = ctypes.create_string_buffer(quant_data)
    out = (ctypes.c_float * n_elements)()
    fn = TO_FLOAT(traits.to_float)
    t0 = time.perf_counter()
    fn(ctypes.cast(src, ctypes.c_void_p), out, n_elements)
    elapsed_s = time.perf_counter() - t0

    values = list(out)
    metadata = {
        "type_id": type_id,
        "ggml_type_name": traits.type_name.decode("utf-8") if traits.type_name else None,
        "blck_size": int(traits.blck_size),
        "blck_size_interleave": int(traits.blck_size_interleave),
        "type_size": int(traits.type_size),
        "is_quantized": bool(traits.is_quantized),
        "n_elements": int(n_elements),
        "expected_expert_bytes": int(expected_bytes),
        "dequant_elapsed_s": elapsed_s,
    }
    return metadata, values


def summarize_values(values: list[float]) -> dict[str, Any]:
    finite = [x for x in values if math.isfinite(x)]
    if not finite:
        return {"finite_count": 0, "n": len(values)}
    abs_values = [abs(x) for x in finite]
    mean = statistics.fmean(finite)
    return {
        "n": len(values),
        "finite_count": len(finite),
        "finite_ratio": len(finite) / len(values),
        "mean": mean,
        "std": statistics.pstdev(finite, mu=mean),
        "min": min(finite),
        "max": max(finite),
        "abs_mean": statistics.fmean(abs_values),
        "abs_max": max(abs_values),
        "sample_first_16": finite[:16],
    }


def run_sample(lib: ctypes.CDLL, inventory: dict[str, InventoryRow], tensor: str, expert_idx: int) -> dict[str, Any]:
    row = inventory[tensor]
    read_offset, quant_data = read_expert_bytes(row, expert_idx)
    dequant_metadata, values = dequantize(lib, row, quant_data)
    return {
        "tensor": tensor,
        "expert_idx": expert_idx,
        "layer": row.layer,
        "kind": row.kind,
        "inventory_type": row.type_name,
        "shape": row.shape,
        "shard_path": row.shard_path,
        "data_offset": row.data_offset,
        "read_offset": read_offset,
        "expert_bytes": row.expert_bytes,
        "quant_sha256_1m": hashlib.sha256(quant_data[: min(len(quant_data), 1 << 20)]).hexdigest(),
        "dequant": dequant_metadata,
        "stats": summarize_values(values),
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Kimi D2MoE dequant sample",
        "",
        f"- generated_at: `{result['generated_at']}`",
        f"- libggml_base: `{result['libggml_base']}`",
        f"- samples: `{len(result['samples'])}`",
        "",
        "| tensor | expert | type | finite | abs_mean | abs_max | dequant_s |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for sample in result["samples"]:
        stats = sample["stats"]
        dequant = sample["dequant"]
        lines.append(
            "| {tensor} | {expert} | {typ} | {finite:.6f} | {abs_mean:.6g} | {abs_max:.6g} | {elapsed:.6f} |".format(
                tensor=sample["tensor"],
                expert=sample["expert_idx"],
                typ=sample["inventory_type"],
                finite=stats.get("finite_ratio", 0.0),
                abs_mean=stats.get("abs_mean", 0.0),
                abs_max=stats.get("abs_max", 0.0),
                elapsed=dequant["dequant_elapsed_s"],
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
            "This only verifies byte addressing and ggml dequantization. It is not a runtime token-rate run and it does not make a quality claim.",
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
    parser.add_argument("--max-per-type", type=int, default=1)
    parser.add_argument("--tensor", action="append", default=[], help="Optional tensor override; use with --expert-idx.")
    parser.add_argument("--expert-idx", action="append", type=int, default=[])
    args = parser.parse_args()

    inventory = load_inventory(args.inventory)
    lib = load_ggml(args.libggml_base)
    if args.tensor:
        if len(args.expert_idx) not in (0, len(args.tensor)):
            raise SystemExit("--expert-idx must be omitted or repeated once per --tensor")
        expert_indices = args.expert_idx or [0] * len(args.tensor)
        samples = list(zip(args.tensor, expert_indices))
    else:
        samples = load_phase0_samples(args.phase0_plan, inventory, args.max_per_type)

    result = {
        "kind": "kimi_d2moe_dequant_sample",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "inventory": str(args.inventory),
        "phase0_plan": str(args.phase0_plan),
        "libggml_base": str(args.libggml_base),
        "max_per_type": args.max_per_type,
        "reproduce_command": " ".join(os.sys.argv),
        "samples": [],
    }
    for tensor, expert_idx in samples:
        if tensor not in inventory:
            raise SystemExit(f"tensor not found in inventory: {tensor}")
        result["samples"].append(run_sample(lib, inventory, tensor, expert_idx))

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.out_md, result)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
