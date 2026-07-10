#!/usr/bin/env python3
"""Validate the real 4Expert GGUF acquisition before load/perf tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Any

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

EXPECTED_SIZE = 164_465_760_544
EXPECTED_ARCH = "deepseek4"
EXPECTED_BLOCK_COUNT = 43
EXPECTED_EXPERT_COUNT = 256
EXPECTED_EXPERT_USED_COUNT = 4


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_text(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as exc:
        return exc.output.strip()


class StreamingGGUFParser:
    def __init__(self, fh: BinaryIO):
        self.fh = fh
        self.pos = 0

    def take(self, n: int) -> bytes:
        data = self.fh.read(n)
        if len(data) != n:
            raise EOFError(f"need {n} bytes at {self.pos}, got {len(data)}")
        self.pos += n
        return data

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
        if n > 1_000_000:
            raise ValueError(f"unreasonable GGUF string length {n} at {self.pos}")
        return self.take(n).decode("utf-8", "replace")

    def value(self, type_id: int) -> Any:
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
                value = self.value(elem_type)
                if i < 8:
                    sample.append(value)
            return {"array_type": elem_type, "len": n, "sample": sample}
        raise ValueError(f"unknown GGUF value type {type_id} at {self.pos}")

    def parse_header(self) -> dict[str, Any]:
        magic = self.take(4)
        if magic != b"GGUF":
            raise ValueError(f"not a GGUF header: {magic!r}")
        version = self.u32()
        n_tensors = self.u64()
        n_kv = self.u64()
        fields: dict[str, Any] = {
            "magic": "GGUF",
            "version": version,
            "n_tensors": n_tensors,
            "n_kv": n_kv,
        }
        for _ in range(n_kv):
            key = self.string()
            type_id = self.u32()
            fields[key] = self.value(type_id)

        type_counts: dict[str, int] = {}
        name_counts = {
            "tid2eid_plain": 0,
            "tid2eid_weight": 0,
            "q4k_expert_tensors": 0,
            "q4k_gate_exps": 0,
            "q4k_up_exps": 0,
            "q4k_down_exps": 0,
        }
        first_expert_tensors: list[dict[str, Any]] = []
        for _ in range(n_tensors):
            name = self.string()
            n_dims = self.u32()
            dims = [self.u64() for _ in range(n_dims)]
            type_id = self.u32()
            offset = self.u64()
            type_name = GGML_TYPE_NAMES.get(type_id, str(type_id))
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
            if name.endswith("ffn_gate_tid2eid"):
                name_counts["tid2eid_plain"] += 1
            if name.endswith("ffn_gate_tid2eid.weight"):
                name_counts["tid2eid_weight"] += 1
            if type_name == "Q4_K" and any(part in name for part in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")):
                name_counts["q4k_expert_tensors"] += 1
                if "ffn_gate_exps" in name:
                    name_counts["q4k_gate_exps"] += 1
                if "ffn_up_exps" in name:
                    name_counts["q4k_up_exps"] += 1
                if "ffn_down_exps" in name:
                    name_counts["q4k_down_exps"] += 1
                if len(first_expert_tensors) < 12:
                    first_expert_tensors.append({"name": name, "dims": dims, "type": type_name, "offset": offset})
        return {
            "fields": fields,
            "type_counts": type_counts,
            "name_counts": name_counts,
            "first_expert_tensors": first_expert_tensors,
            "tensor_info_end": self.pos,
        }


def sha256_file(path: Path, chunk_size: int = 32 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/root/lfz/models/DeepSeek-V4-Flash-4Expert-GGUF/ds4flash-4expert.gguf"))
    parser.add_argument("--expected-size", type=int, default=EXPECTED_SIZE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true", help="Record header/progress even when .aria2 exists; exits nonzero unless set.")
    parser.add_argument("--sha256", action="store_true", help="Compute full-file sha256. Use only after download completion.")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    sidecar = Path(str(args.model) + ".aria2")
    data: dict[str, Any] = {
        "attempt_id": "20260706-4expert-ready-validation",
        "timestamp_utc": utc_now(),
        "repo": str(repo),
        "branch": run_text(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo),
        "commit": run_text(["git", "rev-parse", "HEAD"], cwd=repo),
        "git_status_short": run_text(["git", "status", "--short"], cwd=repo),
        "model": str(args.model),
        "expected_size_bytes": args.expected_size,
        "aria2_sidecar": str(sidecar),
        "aria2_sidecar_present": sidecar.exists(),
        "download_service_active": run_text(["systemctl", "is-active", "ds4-4expert-download.service"]),
        "download_progress_tail": run_text(["bash", "-lc", "tail -n 20 /root/lfz/models/DeepSeek-V4-Flash-4Expert-GGUF/aria2-4expert.log 2>/dev/null || true"]),
        "env_required_for_next_load": {
            "LLAMA_DEEPSEEK4_TID2EID_WEIGHT_ALIAS": "1",
            "GGML_MOE_STREAM_ONE_Q4K": "1",
        },
    }

    failures: list[str] = []
    if not args.model.exists():
        failures.append("missing_model")
    else:
        stat = args.model.stat()
        data["stat_size_bytes"] = stat.st_size
        data["stat_blocks_512b"] = stat.st_blocks
        data["allocated_bytes"] = stat.st_blocks * 512
        data["size_matches_expected"] = stat.st_size == args.expected_size
        if stat.st_size != args.expected_size:
            failures.append("size_mismatch")
        if sidecar.exists():
            failures.append("aria2_sidecar_present")
        try:
            with args.model.open("rb") as fh:
                header = StreamingGGUFParser(fh).parse_header()
            data["gguf_header"] = header
            fields = header["fields"]
            names = header["name_counts"]
            if fields.get("general.architecture") != EXPECTED_ARCH:
                failures.append("unexpected_architecture")
            if fields.get("deepseek4.block_count") != EXPECTED_BLOCK_COUNT:
                failures.append("unexpected_block_count")
            if fields.get("deepseek4.expert_count") != EXPECTED_EXPERT_COUNT:
                failures.append("unexpected_expert_count")
            if fields.get("deepseek4.expert_used_count") != EXPECTED_EXPERT_USED_COUNT:
                failures.append("unexpected_expert_used_count")
            if names.get("tid2eid_weight", 0) <= 0:
                failures.append("missing_tid2eid_weight_alias_tensor")
            if names.get("q4k_expert_tensors", 0) <= 0:
                failures.append("missing_q4k_expert_tensors")
        except Exception as exc:  # noqa: BLE001 - records exact parse failure for diagnostics.
            failures.append("header_parse_failed")
            data["header_parse_error"] = repr(exc)

    if args.sha256 and args.model.exists() and not sidecar.exists() and data.get("size_matches_expected"):
        data["sha256"] = sha256_file(args.model)
    elif args.sha256:
        failures.append("sha256_requested_before_complete")

    complete = not failures
    data["complete_and_ready_for_load_validation"] = complete
    data["failures"] = failures
    if failures and args.allow_incomplete and failures == ["aria2_sidecar_present"]:
        data["incomplete_but_header_ok"] = True
    elif failures and args.allow_incomplete:
        data["incomplete_but_header_ok"] = "header_parse_failed" not in failures

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(data, indent=2, sort_keys=True))

    if complete:
        return 0
    return 0 if args.allow_incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
