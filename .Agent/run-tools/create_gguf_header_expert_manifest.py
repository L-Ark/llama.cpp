#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path


GGUF_VALUE_UINT8 = 0
GGUF_VALUE_INT8 = 1
GGUF_VALUE_UINT16 = 2
GGUF_VALUE_INT16 = 3
GGUF_VALUE_UINT32 = 4
GGUF_VALUE_INT32 = 5
GGUF_VALUE_FLOAT32 = 6
GGUF_VALUE_BOOL = 7
GGUF_VALUE_STRING = 8
GGUF_VALUE_ARRAY = 9
GGUF_VALUE_UINT64 = 10
GGUF_VALUE_INT64 = 11
GGUF_VALUE_FLOAT64 = 12

GGML_TYPE_NAMES = {
    0: "F32",
    1: "F16",
    8: "Q8_0",
    12: "Q4_K",
    26: "I32",
}

QK_K = 256
Q4_K_BLOCK_BYTES = 144


@dataclass
class TensorInfo:
    name: str
    dims: list[int]
    type_id: int
    offset: int


class HeaderParser:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise EOFError(f"need {n} bytes at {self.pos}, file has {len(self.data)}")
        out = self.data[self.pos:self.pos + n]
        self.pos += n
        return out

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def i64(self) -> int:
        return struct.unpack("<q", self.take(8))[0]

    def f32(self) -> float:
        return struct.unpack("<f", self.take(4))[0]

    def f64(self) -> float:
        return struct.unpack("<d", self.take(8))[0]

    def string(self) -> str:
        n = self.u64()
        return self.take(n).decode("utf-8", "replace")

    def value(self, type_id: int):
        if type_id == GGUF_VALUE_UINT8:
            return self.take(1)[0]
        if type_id == GGUF_VALUE_INT8:
            return struct.unpack("<b", self.take(1))[0]
        if type_id == GGUF_VALUE_UINT16:
            return struct.unpack("<H", self.take(2))[0]
        if type_id == GGUF_VALUE_INT16:
            return struct.unpack("<h", self.take(2))[0]
        if type_id == GGUF_VALUE_UINT32:
            return self.u32()
        if type_id == GGUF_VALUE_INT32:
            return self.i32()
        if type_id == GGUF_VALUE_FLOAT32:
            return self.f32()
        if type_id == GGUF_VALUE_BOOL:
            return bool(self.take(1)[0])
        if type_id == GGUF_VALUE_STRING:
            return self.string()
        if type_id == GGUF_VALUE_UINT64:
            return self.u64()
        if type_id == GGUF_VALUE_INT64:
            return self.i64()
        if type_id == GGUF_VALUE_FLOAT64:
            return self.f64()
        if type_id == GGUF_VALUE_ARRAY:
            elem_type = self.u32()
            n = self.u64()
            sample = []
            for i in range(n):
                v = self.value(elem_type)
                if i < 8:
                    sample.append(v)
            return {"array_type": elem_type, "len": n, "sample": sample}
        raise ValueError(f"unknown GGUF value type {type_id} at {self.pos}")

    def parse(self) -> tuple[dict[str, object], list[TensorInfo], int]:
        magic = self.take(4)
        if magic != b"GGUF":
            raise ValueError(f"not a GGUF header: {magic!r}")
        version = self.u32()
        n_tensors = self.u64()
        n_kv = self.u64()
        fields: dict[str, object] = {
            "magic": magic.decode("ascii"),
            "version": version,
            "n_tensors": n_tensors,
            "n_kv": n_kv,
        }
        for _ in range(n_kv):
            key = self.string()
            type_id = self.u32()
            fields[key] = self.value(type_id)
        tensors: list[TensorInfo] = []
        for _ in range(n_tensors):
            name = self.string()
            n_dims = self.u32()
            dims = [self.u64() for _ in range(n_dims)]
            type_id = self.u32()
            offset = self.u64()
            tensors.append(TensorInfo(name=name, dims=dims, type_id=type_id, offset=offset))
        return fields, tensors, self.pos


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def q4k_tensor_nbytes(dims: list[int]) -> int:
    if len(dims) < 2:
        raise ValueError(f"Q4_K expert tensor needs at least 2 dims, got {dims}")
    ne0 = dims[0]
    if ne0 % QK_K != 0:
        raise ValueError(f"Q4_K ne0 must be divisible by {QK_K}, got {ne0}")
    row_size = ne0 // QK_K * Q4_K_BLOCK_BYTES
    nbytes = row_size
    for dim in dims[1:]:
        nbytes *= dim
    return nbytes


def main() -> int:
    ap = argparse.ArgumentParser(description="Create a direct expert manifest from a GGUF header-only range file.")
    ap.add_argument("--header", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--summary-out", type=Path, required=True)
    ap.add_argument("--name-substring", action="append", default=["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"])
    ap.add_argument("--type", default="Q4_K")
    args = ap.parse_args()

    fields, tensors, tensor_info_end = HeaderParser(args.header.read_bytes()).parse()
    alignment = int(fields.get("general.alignment", 32))
    data_start = align_up(tensor_info_end, alignment)
    type_counts: dict[str, int] = {}
    rows: list[str] = []
    manifest_tensors = 0
    payload_bytes = 0
    expert_bytes_seen: dict[int, int] = {}

    for tensor in tensors:
        type_name = GGML_TYPE_NAMES.get(tensor.type_id, str(tensor.type_id))
        type_counts[type_name] = type_counts.get(type_name, 0) + 1
        if type_name != args.type:
            continue
        if not any(part in tensor.name for part in args.name_substring):
            continue
        if len(tensor.dims) < 3:
            raise ValueError(f"expert tensor {tensor.name} has unexpected dims {tensor.dims}")
        n_expert = tensor.dims[2]
        tensor_nbytes = q4k_tensor_nbytes(tensor.dims)
        if tensor_nbytes % n_expert != 0:
            raise ValueError(f"tensor bytes not divisible by experts: {tensor.name}")
        expert_bytes = tensor_nbytes // n_expert
        expert_bytes_seen[expert_bytes] = expert_bytes_seen.get(expert_bytes, 0) + 1
        manifest_tensors += 1
        for expert in range(n_expert):
            model_offset = data_start + tensor.offset + expert * expert_bytes
            rows.append(f"{tensor.name},{expert},{model_offset},{expert_bytes}\n")
            payload_bytes += expert_bytes

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(rows), encoding="utf-8")

    summary = {
        "header": str(args.header),
        "manifest": str(args.output),
        "alignment": alignment,
        "tensor_info_end": tensor_info_end,
        "data_start": data_start,
        "architecture": fields.get("general.architecture"),
        "block_count": fields.get("deepseek4.block_count"),
        "expert_count": fields.get("deepseek4.expert_count"),
        "expert_used_count": fields.get("deepseek4.expert_used_count"),
        "n_tensors": fields.get("n_tensors"),
        "n_kv": fields.get("n_kv"),
        "type_counts": type_counts,
        "manifest_tensors": manifest_tensors,
        "manifest_rows": len(rows),
        "payload_bytes": payload_bytes,
        "payload_gib": payload_bytes / (1024 ** 3),
        "expert_bytes_seen": expert_bytes_seen,
        "first_rows": rows[:6],
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
