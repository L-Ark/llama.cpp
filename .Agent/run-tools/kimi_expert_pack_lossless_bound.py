#!/usr/bin/env python3
"""Lossless/exact-residual compression bound for Kimi expert packs.

This tool samples GGMLMOEPACKv1 entries and measures whether exact payload
bytes, or exact XOR residuals against a same-tensor base expert, are
compressible enough to be a viable byte-reduction path.

It is offline and non-destructive. It does not create packs or change runtime.
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import lzma
import os
import random
import re
import struct
import time
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any


PACK_MAGIC = b"GGMLMOEPACKv1\0\0\0"
PACK_HEADER = struct.Struct("<16sIIQQ")
PACK_ENTRY = struct.Struct("<128siIQQ")
ROLE_RE = re.compile(r"ffn_(up|gate|down)_exps")
LAYER_RE = re.compile(r"blk\.(\d+)\.")


def parse_csv_ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def role_of(tensor: str) -> str:
    match = ROLE_RE.search(tensor)
    return match.group(1) if match else "other"


def layer_of(tensor: str) -> int:
    match = LAYER_RE.search(tensor)
    return int(match.group(1)) if match else -1


def read_pack_entries(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as f:
        raw = f.read(PACK_HEADER.size)
        if len(raw) != PACK_HEADER.size:
            raise RuntimeError(f"{path}: short pack header")
        magic, version, header_size, n_entries, data_start = PACK_HEADER.unpack(raw)
        if magic != PACK_MAGIC or version != 1:
            raise RuntimeError(f"{path}: invalid pack header")
        if header_size > PACK_HEADER.size:
            f.seek(header_size)
        entries = []
        for entry_idx in range(n_entries):
            item = f.read(PACK_ENTRY.size)
            if len(item) != PACK_ENTRY.size:
                raise RuntimeError(f"{path}: short pack index at {entry_idx}")
            name_raw, expert_idx, _reserved, offset, nbytes = PACK_ENTRY.unpack(item)
            tensor = name_raw.split(b"\0", 1)[0].decode("utf-8", errors="replace")
            entries.append(
                {
                    "pack": str(path),
                    "entry_idx": entry_idx,
                    "tensor": tensor,
                    "role": role_of(tensor),
                    "layer": layer_of(tensor),
                    "expert_idx": int(expert_idx),
                    "offset": int(offset),
                    "nbytes": int(nbytes),
                    "data_start": int(data_start),
                    "pack_size": path.stat().st_size,
                }
            )
    return entries


def choose_samples(entries: list[dict[str, Any]], per_role: int, per_tensor: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_pack_role: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_pack_role[(entry["pack"], entry["role"])].append(entry)
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, int]] = set()
    packs = sorted({entry["pack"] for entry in entries})
    for pack in packs:
        for role in ("up", "gate", "down", "other"):
            role_entries = by_pack_role.get((pack, role), [])
            if not role_entries:
                continue
            by_tensor: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for entry in role_entries:
                by_tensor[entry["tensor"]].append(entry)
            tensor_names = sorted(by_tensor)
            rng.shuffle(tensor_names)
            role_selected = 0
            for tensor in tensor_names:
                tensor_entries = by_tensor[tensor][:]
                rng.shuffle(tensor_entries)
                for entry in tensor_entries[:per_tensor]:
                    key = (entry["pack"], entry["tensor"], entry["expert_idx"], entry["nbytes"])
                    if key in seen:
                        continue
                    seen.add(key)
                    selected.append(entry)
                    role_selected += 1
                    if role_selected >= per_role:
                        break
                if role_selected >= per_role:
                    break
    return selected


def read_payloads(pack_paths: list[Path], samples: list[dict[str, Any]]) -> dict[tuple[str, int], bytes]:
    by_pack: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_pack[sample["pack"]].append(sample)
    out: dict[tuple[str, int], bytes] = {}
    for pack in pack_paths:
        rows = sorted(by_pack.get(str(pack), []), key=lambda r: r["offset"])
        if not rows:
            continue
        with pack.open("rb") as f:
            for row in rows:
                f.seek(row["offset"])
                data = f.read(row["nbytes"])
                if len(data) != row["nbytes"]:
                    raise RuntimeError(f"{pack}: short payload read for entry {row['entry_idx']}")
                out[(row["pack"], row["entry_idx"])] = data
    return out


def xor_bytes(a: bytes, b: bytes) -> bytes:
    if len(a) != len(b):
        raise RuntimeError("xor length mismatch")
    return bytes(x ^ y for x, y in zip(a, b))


def compress_payload(data: bytes, codecs: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for codec in codecs:
        t0 = time.perf_counter()
        if codec == "zlib6":
            comp = zlib.compress(data, level=6)
        elif codec == "zlib9":
            comp = zlib.compress(data, level=9)
        elif codec == "bz2":
            comp = bz2.compress(data, compresslevel=9)
        elif codec == "lzma6":
            comp = lzma.compress(data, preset=6)
        else:
            raise RuntimeError(f"unknown codec: {codec}")
        elapsed = time.perf_counter() - t0
        out[codec] = {
            "compressed_bytes": len(comp),
            "ratio": len(comp) / max(len(data), 1),
            "compress_s": elapsed,
        }
    return out


class Acc:
    def __init__(self) -> None:
        self.n = 0
        self.raw = 0
        self.comp = 0
        self.sum_ratio = 0.0
        self.max_ratio = 0.0
        self.min_ratio = 1e9

    def add(self, raw: int, comp: int) -> None:
        ratio = comp / max(raw, 1)
        self.n += 1
        self.raw += raw
        self.comp += comp
        self.sum_ratio += ratio
        self.max_ratio = max(self.max_ratio, ratio)
        self.min_ratio = min(self.min_ratio, ratio)

    def row(self) -> dict[str, Any]:
        return {
            "entries": self.n,
            "raw_bytes": self.raw,
            "compressed_bytes": self.comp,
            "aggregate_ratio": self.comp / max(self.raw, 1),
            "mean_ratio": self.sum_ratio / self.n if self.n else 0.0,
            "min_ratio": self.min_ratio if self.n else 0.0,
            "max_ratio": self.max_ratio,
        }


def summarize(rows: list[dict[str, Any]], codecs: list[str], mode: str) -> dict[str, Any]:
    by_codec = {codec: Acc() for codec in codecs}
    by_role = {(role, codec): Acc() for role in ("up", "gate", "down", "other") for codec in codecs}
    for row in rows:
        raw = int(row["raw_bytes"])
        for codec in codecs:
            comp = int(row[mode][codec]["compressed_bytes"])
            by_codec[codec].add(raw, comp)
            by_role[(row["role"], codec)].add(raw, comp)
    return {
        "by_codec": {codec: acc.row() for codec, acc in by_codec.items()},
        "by_role_codec": {f"{role}:{codec}": acc.row() for (role, codec), acc in by_role.items() if acc.n},
    }


def write_md(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Kimi expert-pack lossless compression bound",
        "",
        "This is an offline non-destructive bound. It does not change runtime behavior or claim SOTA.",
        "",
        f"- generated_at: `{result['generated_at']}`",
        f"- sampled entries: `{result['sampled_entries']}`",
        f"- sampled raw bytes: `{result['sampled_raw_bytes']}`",
        f"- target ratio: `{result['target_ratio']}`",
        "",
        "## Pack Sources",
        "",
        "| pack | entries | payload GiB | sampled entries |",
        "|---|---:|---:|---:|",
    ]
    for pack in result["packs"]:
        lines.append(
            f"| `{pack['path']}` | `{pack['entries']}` | `{pack['payload_gib']:.3f}` | `{pack['sampled_entries']}` |"
        )
    for mode, title in (
        ("raw_summary", "Raw Payload Compression"),
        ("xor_summary", "Base-XOR Residual Compression"),
        ("xor_nonbase_summary", "Base-XOR Residual Compression, Excluding Base-Self Rows"),
    ):
        lines += [
            "",
            f"## {title}",
            "",
            "| codec | entries | raw MiB | compressed MiB | aggregate ratio | mean ratio | min | max |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for codec, row in sorted(result[mode]["by_codec"].items()):
            lines.append(
                f"| `{codec}` | `{row['entries']}` | `{row['raw_bytes'] / 1024**2:.2f}` | "
                f"`{row['compressed_bytes'] / 1024**2:.2f}` | `{row['aggregate_ratio']:.4f}` | "
                f"`{row['mean_ratio']:.4f}` | `{row['min_ratio']:.4f}` | `{row['max_ratio']:.4f}` |"
            )
        lines += [
            "",
            "### By Role",
            "",
            "| role | codec | entries | aggregate ratio | mean ratio |",
            "|---|---|---:|---:|---:|",
        ]
        for key, row in sorted(result[mode]["by_role_codec"].items()):
            role, codec = key.split(":", 1)
            lines.append(
                f"| `{role}` | `{codec}` | `{row['entries']}` | `{row['aggregate_ratio']:.4f}` | `{row['mean_ratio']:.4f}` |"
            )
    lines += [
        "",
        "## Best Rows",
        "",
        "| mode | codec | aggregate ratio | decision |",
        "|---|---|---:|---|",
    ]
    for mode in ("raw_summary", "xor_summary", "xor_nonbase_summary"):
        best_codec, best = min(result[mode]["by_codec"].items(), key=lambda kv: kv[1]["aggregate_ratio"])
        decision = "pass" if best["aggregate_ratio"] <= result["target_ratio"] else "fail"
        lines.append(f"| `{mode}` | `{best_codec}` | `{best['aggregate_ratio']:.4f}` | `{decision}` |")
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
    parser.add_argument("--pack", type=Path, action="append", required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--per-role", type=int, default=24)
    parser.add_argument("--per-tensor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--codecs", default="zlib6,zlib9,lzma6")
    parser.add_argument("--target-ratio", type=float, default=0.40)
    args = parser.parse_args()

    codecs = [part.strip() for part in args.codecs.split(",") if part.strip()]
    all_entries: list[dict[str, Any]] = []
    pack_summaries = []
    for pack in args.pack:
        entries = read_pack_entries(pack)
        payload = sum(entry["nbytes"] for entry in entries)
        all_entries.extend(entries)
        pack_summaries.append(
            {
                "path": str(pack),
                "entries": len(entries),
                "payload_bytes": payload,
                "payload_gib": payload / 1024**3,
                "sampled_entries": 0,
            }
        )

    samples = choose_samples(all_entries, args.per_role, args.per_tensor, args.seed)
    for pack in pack_summaries:
        pack["sampled_entries"] = sum(1 for sample in samples if sample["pack"] == pack["path"])

    payloads = read_payloads(args.pack, samples)
    by_tensor: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_tensor[(sample["pack"], sample["tensor"])].append(sample)

    base_for: dict[tuple[str, int], bytes] = {}
    base_meta: dict[tuple[str, int], dict[str, Any]] = {}
    for key, rows in by_tensor.items():
        rows = sorted(rows, key=lambda r: r["expert_idx"])
        base = rows[0]
        base_for[key] = payloads[(base["pack"], base["entry_idx"])]
        base_meta[key] = base

    rows = []
    for sample in samples:
        payload = payloads[(sample["pack"], sample["entry_idx"])]
        key = (sample["pack"], sample["tensor"])
        base = base_for[key]
        if len(base) == len(payload):
            delta = xor_bytes(payload, base)
            delta_is_self = sample["entry_idx"] == base_meta[key]["entry_idx"]
        else:
            delta = payload
            delta_is_self = False
        raw_comp = compress_payload(payload, codecs)
        xor_comp = compress_payload(delta, codecs)
        rows.append(
            {
                "pack": sample["pack"],
                "entry_idx": sample["entry_idx"],
                "tensor": sample["tensor"],
                "role": sample["role"],
                "layer": sample["layer"],
                "expert_idx": sample["expert_idx"],
                "raw_bytes": sample["nbytes"],
                "sha256_16": hashlib.sha256(payload).hexdigest()[:16],
                "base_expert_idx": base_meta[key]["expert_idx"],
                "xor_delta_is_base_self": delta_is_self,
                "raw": raw_comp,
                "xor": xor_comp,
            }
        )

    raw_summary = summarize(rows, codecs, "raw")
    xor_summary = summarize(rows, codecs, "xor")
    xor_nonbase_rows = [row for row in rows if not row["xor_delta_is_base_self"]]
    xor_nonbase_summary = summarize(xor_nonbase_rows, codecs, "xor")
    best_raw = min(raw_summary["by_codec"].values(), key=lambda r: r["aggregate_ratio"])
    best_xor_nonbase = min(xor_nonbase_summary["by_codec"].values(), key=lambda r: r["aggregate_ratio"])
    if min(best_raw["aggregate_ratio"], best_xor_nonbase["aggregate_ratio"]) <= args.target_ratio:
        decision = (
            "Proceed: at least one exact lossless representation reaches the target ratio. "
            "Next step would be decompression cost modeling before runtime work."
        )
    else:
        decision = (
            "Reject as primary: neither raw lossless compression nor exact base-XOR residual compression "
            f"approaches the {args.target_ratio:.2f} target ratio. Generic exact expert-byte compression "
            "cannot close the Kimi 5 tok/s byte gap."
        )

    result = {
        "kind": "kimi_expert_pack_lossless_bound",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "packs": pack_summaries,
        "sampled_entries": len(rows),
        "sampled_raw_bytes": sum(row["raw_bytes"] for row in rows),
        "per_role": args.per_role,
        "per_tensor": args.per_tensor,
        "seed": args.seed,
        "codecs": codecs,
        "target_ratio": args.target_ratio,
        "raw_summary": raw_summary,
        "xor_summary": xor_summary,
        "xor_nonbase_summary": xor_nonbase_summary,
        "sample_rows": rows,
        "decision": decision,
        "reproduce_command": " ".join(os.sys.argv),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_md(args.out_md, result)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    print(decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
