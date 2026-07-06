#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import tempfile
from pathlib import Path


MAGIC = b"GGMLMOEPACKv2\0\0\0"
HEADER = struct.Struct("<16sIIQQ")
ENTRY = struct.Struct("<128siiQQqqQQ")
HEADER_SIZE = HEADER.size
DATA_START = 4096


def tensor_field(name: str) -> bytes:
    raw = name.encode("utf-8")
    if len(raw) >= 128:
        raise ValueError(f"tensor name too long: {name}")
    return raw + b"\0" * (128 - len(raw))


def payload_bytes(entry: dict) -> bytes:
    seed = entry["expert_idx"] * 17 + entry["packed_type"] * 31
    return bytes((seed + i * 13) & 0xFF for i in range(entry["nbytes"]))


def write_pack(path: Path) -> list[dict]:
    entries = [
        {
            "tensor": "blk.1.ffn_up_exps.weight",
            "expert_idx": 7,
            "packed_type": 24,
            "offset": DATA_START,
            "nbytes": 64,
            "ne00": 256,
            "ne01": 4,
            "nb01": 16,
            "reserved": 0,
        },
        {
            "tensor": "blk.1.ffn_down_exps.weight",
            "expert_idx": 11,
            "packed_type": 10,
            "offset": DATA_START + 4096,
            "nbytes": 128,
            "ne00": 4,
            "ne01": 256,
            "nb01": 32,
            "reserved": 0,
        },
    ]
    with path.open("wb") as f:
        f.write(HEADER.pack(MAGIC, 2, HEADER_SIZE, len(entries), DATA_START))
        for entry in entries:
            f.write(
                ENTRY.pack(
                    tensor_field(entry["tensor"]),
                    entry["expert_idx"],
                    entry["packed_type"],
                    entry["offset"],
                    entry["nbytes"],
                    entry["ne00"],
                    entry["ne01"],
                    entry["nb01"],
                    entry["reserved"],
                )
            )
        for entry in entries:
            payload = payload_bytes(entry)
            entry["sha256"] = hashlib.sha256(payload).hexdigest()
            f.seek(entry["offset"])
            f.write(payload)
        f.truncate(DATA_START + 8192)
    return entries


def parse_pack(path: Path) -> list[dict]:
    data = path.read_bytes()
    magic, version, header_size, n_entries, data_start = HEADER.unpack_from(data, 0)
    if magic != MAGIC:
        raise AssertionError(f"bad magic: {magic!r}")
    if version != 2:
        raise AssertionError(f"bad version: {version}")
    if header_size != HEADER_SIZE:
        raise AssertionError(f"bad header size: {header_size}")
    if data_start < header_size:
        raise AssertionError(f"bad data_start: {data_start}")
    out = []
    offset = HEADER.size
    for _ in range(n_entries):
        raw = ENTRY.unpack_from(data, offset)
        offset += ENTRY.size
        name = raw[0].split(b"\0", 1)[0].decode("utf-8")
        item = {
            "tensor": name,
            "expert_idx": raw[1],
            "packed_type": raw[2],
            "offset": raw[3],
            "nbytes": raw[4],
            "ne00": raw[5],
            "ne01": raw[6],
            "nb01": raw[7],
            "reserved": raw[8],
        }
        if not item["tensor"] or item["expert_idx"] < 0:
            raise AssertionError(f"invalid key: {item}")
        if item["packed_type"] <= 0 or item["nbytes"] <= 0:
            raise AssertionError(f"invalid packed metadata: {item}")
        if item["ne00"] <= 0 or item["ne01"] <= 0 or item["nb01"] <= 0:
            raise AssertionError(f"invalid dims: {item}")
        payload = data[item["offset"] : item["offset"] + item["nbytes"]]
        if len(payload) != item["nbytes"]:
            raise AssertionError(f"short payload: {item}")
        item["sha256"] = hashlib.sha256(payload).hexdigest()
        out.append(item)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and validate a tiny GGMLMOEPACKv2 metadata pack.")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    if args.out_dir is None:
        with tempfile.TemporaryDirectory(prefix="kimi-moepack-v2-") as tmp:
            path = Path(tmp) / "synthetic-v2.expert-pack"
            expected = write_pack(path)
            got = parse_pack(path)
    else:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        path = args.out_dir / "synthetic-v2.expert-pack"
        expected = write_pack(path)
        got = parse_pack(path)
        (args.out_dir / "synthetic-v2.json").write_text(json.dumps(got, indent=2) + "\n", encoding="utf-8")

    if got != expected:
        raise AssertionError(f"round-trip mismatch got={got!r} expected={expected!r}")
    print(f"kimi_moepack_v2_synthetic_test pass entries={len(got)} entry_size={ENTRY.size} path={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
