#!/usr/bin/env python3
"""Build a tiny selected IQ1_S/Q2_K payload pack from mradermacher multipart GGUF.

This is for Phase 5D source-offset validation. It is default-off and not used by
the current runtime.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import struct
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path


PACK_MAGIC = b"GGMLMOEPACKv2\0\0\0"
PACK_HEADER = struct.Struct("<16sIIQQ")
PACK_ENTRY = struct.Struct("<128siiQQqqQQ")
PACK_ALIGNMENT = 4096

GGUF_MAGIC = b"GGUF"
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

GGML_TYPE_Q2_K = 10
GGML_TYPE_IQ1_S = 19
TYPE_IDS = {
    "Q2_K": GGML_TYPE_Q2_K,
    "IQ1_S": GGML_TYPE_IQ1_S,
}
TYPE_BYTES_PER_BLOCK = {
    "Q2_K": 2 + 2 + 256 // 16 + 256 // 4,
    "IQ1_S": 2 + 256 // 8 + 256 // 16,
}
QK_K = 256

IQ1S_PART_URLS = [
    "https://huggingface.co/mradermacher/Kimi-K2.7-Code-i1-GGUF/resolve/main/Kimi-K2.7-Code.i1-IQ1_S.gguf.part1of5",
    "https://huggingface.co/mradermacher/Kimi-K2.7-Code-i1-GGUF/resolve/main/Kimi-K2.7-Code.i1-IQ1_S.gguf.part2of5",
    "https://huggingface.co/mradermacher/Kimi-K2.7-Code-i1-GGUF/resolve/main/Kimi-K2.7-Code.i1-IQ1_S.gguf.part3of5",
    "https://huggingface.co/mradermacher/Kimi-K2.7-Code-i1-GGUF/resolve/main/Kimi-K2.7-Code.i1-IQ1_S.gguf.part4of5",
    "https://huggingface.co/mradermacher/Kimi-K2.7-Code-i1-GGUF/resolve/main/Kimi-K2.7-Code.i1-IQ1_S.gguf.part5of5",
]
IQ1S_PART_BYTES = [41875931136, 41875931136, 41875931136, 41875931136, 36927147936]


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dims: list[int]
    type_id: int
    offset: int


class Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise EOFError(f"short GGUF header at {self.pos}, need {n}")
        out = self.data[self.pos:self.pos + n]
        self.pos += n
        return out

    def u32(self) -> int:
        return struct.unpack_from("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack_from("<Q", self.take(8))[0]

    def i32(self) -> int:
        return struct.unpack_from("<i", self.take(4))[0]

    def i64(self) -> int:
        return struct.unpack_from("<q", self.take(8))[0]

    def f32(self) -> float:
        return struct.unpack_from("<f", self.take(4))[0]

    def f64(self) -> float:
        return struct.unpack_from("<d", self.take(8))[0]

    def string(self) -> str:
        n = self.u64()
        return self.take(n).decode("utf-8", errors="replace")


def align_up(value: int, alignment: int = PACK_ALIGNMENT) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def tensor_field(name: str) -> bytes:
    raw = name.encode("utf-8")
    if len(raw) >= 128:
        raise ValueError(f"tensor name too long: {name}")
    return raw + b"\0" * (128 - len(raw))


def skip_value(r: Reader, value_type: int):
    if value_type in (GGUF_TYPE_UINT8, GGUF_TYPE_INT8, GGUF_TYPE_BOOL):
        return r.take(1)
    if value_type in (GGUF_TYPE_UINT16, GGUF_TYPE_INT16):
        return r.take(2)
    if value_type in (GGUF_TYPE_UINT32, GGUF_TYPE_INT32, GGUF_TYPE_FLOAT32):
        return r.take(4)
    if value_type in (GGUF_TYPE_UINT64, GGUF_TYPE_INT64, GGUF_TYPE_FLOAT64):
        return r.take(8)
    if value_type == GGUF_TYPE_STRING:
        return r.string()
    if value_type == GGUF_TYPE_ARRAY:
        elem_type = r.u32()
        n = r.u64()
        return [skip_value(r, elem_type) for _ in range(n)]
    raise ValueError(f"unsupported GGUF metadata type {value_type}")


def parse_gguf_header(data: bytes) -> tuple[dict[str, object], dict[str, TensorInfo], int]:
    r = Reader(data)
    if r.take(4) != GGUF_MAGIC:
        raise ValueError("not a GGUF header")
    version = r.u32()
    n_tensors = r.u64()
    n_kv = r.u64()
    metadata: dict[str, object] = {"version": version, "n_tensors": n_tensors, "n_kv": n_kv}
    for _ in range(n_kv):
        key = r.string()
        value_type = r.u32()
        metadata[key] = skip_value(r, value_type)
    tensors: dict[str, TensorInfo] = {}
    for _ in range(n_tensors):
        name = r.string()
        n_dims = r.u32()
        dims = [r.u64() for _ in range(n_dims)]
        type_id = r.u32()
        offset = r.u64()
        tensors[name] = TensorInfo(name=name, dims=dims, type_id=type_id, offset=offset)
    alignment = int(metadata.get("general.alignment", 32) or 32)
    data_start = align_up(r.pos, alignment)
    return metadata, tensors, data_start


def range_get_urllib(url: str, start: int, end_inclusive: int, timeout: int) -> bytes:
    req = urllib.request.Request(url, headers={
        "Range": f"bytes={start}-{end_inclusive}",
        "User-Agent": "kimi-iq1s-selected-payload/1.0",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def range_get_curl(url: str, start: int, end_inclusive: int, timeout: int) -> bytes:
    cmd = [
        "curl", "-L", "--fail", "--retry", "5", "--retry-delay", "1",
        "--connect-timeout", "30", "--max-time", str(timeout),
        "--silent", "--show-error",
        "--range", f"{start}-{end_inclusive}",
        "--user-agent", "kimi-iq1s-selected-payload/1.0",
        url,
    ]
    return subprocess.check_output(cmd, timeout=timeout + 30)


def range_get(url: str, start: int, end_inclusive: int, timeout: int = 180,
              backend: str = "auto") -> bytes:
    if backend == "curl":
        data = range_get_curl(url, start, end_inclusive, timeout)
    elif backend == "urllib":
        data = range_get_urllib(url, start, end_inclusive, timeout)
    elif backend == "auto":
        try:
            data = range_get_urllib(url, start, end_inclusive, timeout)
        except Exception:
            data = range_get_curl(url, start, end_inclusive, timeout)
    else:
        raise ValueError(f"unsupported range backend: {backend}")
    expected = end_inclusive - start + 1
    if len(data) != expected:
        raise RuntimeError(
            f"range length mismatch url={url} start={start} end={end_inclusive} "
            f"got={len(data)} expected={expected}")
    return data


def read_concat_range(urls: list[str], part_bytes: list[int], start: int, nbytes: int,
                      backend: str = "auto", timeout: int = 180) -> bytes:
    out = bytearray()
    prefix = 0
    end = start + nbytes
    for url, size in zip(urls, part_bytes):
        part_start = prefix
        part_end = prefix + size
        if start < part_end and end > part_start:
            local_start = max(start, part_start) - part_start
            local_end = min(end, part_end) - part_start
            out.extend(range_get(url, local_start, local_end - 1, timeout=timeout, backend=backend))
        prefix = part_end
    if len(out) != nbytes:
        raise RuntimeError(f"concat range short read start={start} nbytes={nbytes} got={len(out)}")
    return bytes(out)


def expert_layout(tensor: TensorInfo, type_name: str) -> tuple[int, int, int, int]:
    if len(tensor.dims) != 3:
        raise ValueError(f"not a 3D expert tensor: {tensor.name} dims={tensor.dims}")
    if type_name not in TYPE_IDS:
        raise ValueError(f"unsupported selected type {type_name}")
    if tensor.type_id != TYPE_IDS[type_name]:
        raise ValueError(f"type mismatch for {tensor.name}: header={tensor.type_id} expected={TYPE_IDS[type_name]}")
    ne00 = int(tensor.dims[0])
    ne01 = int(tensor.dims[1])
    if ne00 % QK_K != 0:
        raise ValueError(f"{type_name} ne00 must be multiple of {QK_K}: {tensor.name} ne00={ne00}")
    nb01 = (ne00 // QK_K) * TYPE_BYTES_PER_BLOCK[type_name]
    return ne00, ne01, nb01, nb01 * ne01


def parse_types(text: str) -> set[str]:
    return {part.strip() for part in text.split(",") if part.strip()}


def load_selected(path: Path, tensors: dict[str, TensorInfo], max_entries: int, include_types: set[str],
                  tensor_regex: str, exclude_tensor_regex: str) -> tuple[list[dict], dict[str, int]]:
    selected: list[dict] = []
    seen: set[tuple[str, int]] = set()
    tensor_re = re.compile(tensor_regex) if tensor_regex else None
    exclude_re = re.compile(exclude_tensor_regex) if exclude_tensor_regex else None
    stats = {
        "rows_seen": 0,
        "duplicate_rows": 0,
        "type_skipped": 0,
        "regex_skipped": 0,
        "exclude_regex_skipped": 0,
        "missing_tensor": 0,
        "invalid_rows": 0,
        "selected_rows": 0,
    }
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            stats["rows_seen"] += 1
            tensor = row.get("tensor", "")
            remote_type = row.get("remote_type", "")
            if remote_type not in include_types:
                stats["type_skipped"] += 1
                continue
            if tensor_re and not tensor_re.search(tensor):
                stats["regex_skipped"] += 1
                continue
            if exclude_re and exclude_re.search(tensor):
                stats["exclude_regex_skipped"] += 1
                continue
            try:
                expert_idx = int(row["expert_idx"])
                current_nbytes = int(row["current_nbytes"])
                remote_nbytes = int(row["remote_nbytes"])
            except (KeyError, ValueError):
                stats["invalid_rows"] += 1
                continue
            key = (tensor, expert_idx)
            if key in seen:
                stats["duplicate_rows"] += 1
                continue
            lower = tensors.get(tensor)
            if lower is None:
                stats["missing_tensor"] += 1
                continue
            try:
                ne00, ne01, nb01, expected_nbytes = expert_layout(lower, remote_type)
            except ValueError:
                stats["invalid_rows"] += 1
                continue
            if expert_idx < 0 or expert_idx >= int(lower.dims[2]) or expected_nbytes != remote_nbytes:
                stats["invalid_rows"] += 1
                continue
            selected.append({
                "tensor": tensor,
                "expert_idx": expert_idx,
                "kind": row.get("kind", ""),
                "layer": int(row.get("layer", "-1") or -1),
                "current_nbytes": current_nbytes,
                "remote_type": remote_type,
                "packed_type": TYPE_IDS[remote_type],
                "packed_nbytes": remote_nbytes,
                "packed_ne00": ne00,
                "packed_ne01": ne01,
                "packed_nb01": nb01,
                "concat_tensor_offset": int(lower.offset),
                "expert_payload_offset": expert_idx * remote_nbytes,
            })
            seen.add(key)
            stats["selected_rows"] += 1
            if len(selected) >= max_entries:
                break
    return selected, stats


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def contiguous_payload_groups(manifest: list[dict]) -> list[tuple[int, int, int, int]]:
    """Return groups as (first_row, end_row, source_offset, pack_offset)."""
    groups: list[tuple[int, int, int, int]] = []
    i = 0
    while i < len(manifest):
        source_offset = int(manifest[i]["source_absolute_offset"])
        pack_offset = int(manifest[i]["pack_offset"])
        j = i + 1
        prev_source_end = source_offset + int(manifest[i]["packed_nbytes"])
        prev_pack_end = pack_offset + int(manifest[i]["packed_nbytes"])
        while j < len(manifest):
            row_source = int(manifest[j]["source_absolute_offset"])
            row_pack = int(manifest[j]["pack_offset"])
            if row_source != prev_source_end or row_pack != prev_pack_end:
                break
            nbytes = int(manifest[j]["packed_nbytes"])
            prev_source_end += nbytes
            prev_pack_end += nbytes
            j += 1
        groups.append((i, j, source_offset, pack_offset))
        i = j
    return groups


def write_payloads_chunked(
        f,
        manifest: list[dict],
        range_backend: str,
        range_timeout: int,
        payload_chunk_bytes: int) -> dict:
    if payload_chunk_bytes <= 0:
        raise ValueError("--payload-chunk-mib must be positive")

    groups = contiguous_payload_groups(manifest)
    range_reads = 0
    payload_bytes = 0
    t0 = time.monotonic()
    for first, end, source_offset, pack_offset in groups:
        group_bytes = sum(int(row["packed_nbytes"]) for row in manifest[first:end])
        done = 0
        while done < group_bytes:
            nread = min(payload_chunk_bytes, group_bytes - done)
            payload = read_concat_range(
                IQ1S_PART_URLS, IQ1S_PART_BYTES, source_offset + done, nread,
                backend=range_backend, timeout=range_timeout)
            f.seek(pack_offset + done)
            f.write(payload)
            done += nread
            payload_bytes += nread
            range_reads += 1
    return {
        "payload_write_mode": "coalesced_chunked",
        "payload_groups": len(groups),
        "payload_range_reads": range_reads,
        "payload_chunk_bytes": payload_chunk_bytes,
        "payload_streamed_bytes": payload_bytes,
        "payload_write_wall_sec": time.monotonic() - t0,
    }


def write_pack(path: Path, selected: list[dict], data_start: int, metadata_only: bool,
               range_backend: str, range_timeout: int, payload_chunk_bytes: int) -> tuple[list[dict], dict]:
    pack_data_start = align_up(PACK_HEADER.size + len(selected) * PACK_ENTRY.size, PACK_ALIGNMENT)
    offset = pack_data_start
    manifest: list[dict] = []
    for item in selected:
        src_offset = data_start + int(item["concat_tensor_offset"]) + int(item["expert_payload_offset"])
        row = dict(item)
        row["source_absolute_offset"] = src_offset
        row["pack_offset"] = offset
        manifest.append(row)
        offset = align_up(offset + int(item["packed_nbytes"]), PACK_ALIGNMENT)

    payload_stats = {
        "payload_write_mode": "metadata_only" if metadata_only else "coalesced_chunked",
        "payload_groups": 0,
        "payload_range_reads": 0,
        "payload_chunk_bytes": payload_chunk_bytes,
        "payload_streamed_bytes": 0,
        "payload_write_wall_sec": 0.0,
    }
    with path.open("wb") as f:
        f.write(PACK_HEADER.pack(PACK_MAGIC, 2, PACK_HEADER.size, len(manifest), pack_data_start))
        for row in manifest:
            f.write(PACK_ENTRY.pack(
                tensor_field(str(row["tensor"])),
                int(row["expert_idx"]),
                int(row["packed_type"]),
                int(row["pack_offset"]),
                int(row["packed_nbytes"]),
                int(row["packed_ne00"]),
                int(row["packed_ne01"]),
                int(row["packed_nb01"]),
                0,
            ))
        f.truncate(pack_data_start)
        if not metadata_only:
            payload_stats = write_payloads_chunked(
                f, manifest, range_backend, range_timeout, payload_chunk_bytes)
    return manifest, payload_stats


def write_manifest(path: Path, rows: list[dict]) -> None:
    fields = [
        "tensor", "expert_idx", "layer", "kind", "remote_type", "packed_type",
        "packed_nbytes", "packed_ne00", "packed_ne01", "packed_nb01",
        "source_absolute_offset", "pack_offset", "current_nbytes",
        "concat_tensor_offset", "expert_payload_offset",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def write_markdown(path: Path, result: dict) -> None:
    lines = [
        "# Kimi selected IQ1_S/Q2_K payload pack",
        "",
        "This is a tiny source-offset/payload builder for Phase 5D. It does not change runtime behavior or claim SOTA.",
        "",
        f"- generated_at: `{result['generated_at']}`",
        f"- selected_plan_tsv: `{result['selected_plan_tsv']}`",
        f"- out_pack: `{result['out_pack']}`",
        f"- manifest_tsv: `{result['manifest_tsv']}`",
        f"- metadata_only: `{result['metadata_only']}`",
        f"- selected entries: `{result['selected_entries']}`",
        f"- payload bytes: `{result['payload_bytes']}`",
        f"- pack bytes: `{result['pack_bytes']}`",
        f"- pack sha256: `{result['pack_sha256']}`",
        f"- payload write mode: `{result['payload_write_mode']}`",
        f"- payload groups: `{result['payload_groups']}`",
        f"- payload range reads: `{result['payload_range_reads']}`",
        f"- payload chunk bytes: `{result['payload_chunk_bytes']}`",
        f"- payload write wall sec: `{result['payload_write_wall_sec']:.3f}`",
        f"- gguf data start: `{result['gguf_data_start']}`",
        "",
        "## Selected Entries",
        "",
        "| tensor | expert | type | nbytes | source offset | pack offset |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in result["selected"]:
        lines.append(
            f"| `{row['tensor']}` | {row['expert_idx']} | `{row['remote_type']}` | "
            f"{row['packed_nbytes']} | {row['source_absolute_offset']} | {row['pack_offset']} |"
        )
    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        result["reproduce_command"],
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def json_default(value):
    if isinstance(value, (bytes, bytearray)):
        return value.hex()
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a tiny selected Kimi IQ1_S/Q2_K payload pack.")
    parser.add_argument("--selected-plan-tsv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-entries", type=int, default=8)
    parser.add_argument("--include-types", default="IQ1_S,Q2_K")
    parser.add_argument("--tensor-regex", default="")
    parser.add_argument("--exclude-tensor-regex", default="")
    parser.add_argument("--header-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument(
        "--header-cache",
        type=Path,
        default=None,
        help="Optional local GGUF part1 header cache. If present, use it instead of a remote range request.")
    parser.add_argument("--range-backend", choices=("auto", "urllib", "curl"), default="auto")
    parser.add_argument("--range-timeout", type=int, default=180)
    parser.add_argument(
        "--payload-chunk-mib",
        type=int,
        default=64,
        help="Maximum payload range read size for non-metadata builds.")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()

    if args.max_entries <= 0:
        raise ValueError("--max-entries must be positive")
    if args.payload_chunk_mib <= 0:
        raise ValueError("--payload-chunk-mib must be positive")
    include_types = parse_types(args.include_types)
    invalid_types = include_types - set(TYPE_IDS)
    if invalid_types:
        raise ValueError(f"unsupported include types: {sorted(invalid_types)}")

    if args.header_cache is not None and args.header_cache.exists():
        header = args.header_cache.read_bytes()
        if len(header) < args.header_bytes:
            raise RuntimeError(
                f"header cache too small: {args.header_cache} got={len(header)} need={args.header_bytes}")
        header = header[:args.header_bytes]
    else:
        header = range_get(
            IQ1S_PART_URLS[0], 0, args.header_bytes - 1,
            timeout=args.range_timeout, backend=args.range_backend)
    metadata, tensors, data_start = parse_gguf_header(header)
    selected, stats = load_selected(
        args.selected_plan_tsv, tensors, args.max_entries, include_types,
        args.tensor_regex, args.exclude_tensor_regex)
    if not selected:
        raise RuntimeError(f"no selected entries; stats={stats}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pack_path = args.out_dir / "selected-iq1s-overlay-v2.expert-pack"
    manifest_tsv = args.out_dir / "selected-iq1s-overlay-manifest.tsv"
    manifest, payload_stats = write_pack(
        pack_path, selected, data_start, args.metadata_only,
        args.range_backend, args.range_timeout, args.payload_chunk_mib * 1024 * 1024)
    write_manifest(manifest_tsv, manifest)

    payload_bytes = sum(int(row["packed_nbytes"]) for row in manifest)
    pack_bytes = pack_path.stat().st_size
    result = {
        "kind": "kimi_iq1s_selected_payload_pack",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "selected_plan_tsv": str(args.selected_plan_tsv),
        "out_pack": str(pack_path),
        "manifest_tsv": str(manifest_tsv),
        "metadata_only": args.metadata_only,
        "max_entries": args.max_entries,
        "include_types": sorted(include_types),
        "header_bytes": args.header_bytes,
        "header_cache": str(args.header_cache) if args.header_cache is not None else "",
        "range_backend": args.range_backend,
        "range_timeout": args.range_timeout,
        "payload_chunk_mib": args.payload_chunk_mib,
        "gguf_data_start": data_start,
        "metadata": {
            "version": metadata.get("version"),
            "n_tensors": metadata.get("n_tensors"),
            "n_kv": metadata.get("n_kv"),
            "general.name": metadata.get("general.name"),
            "general.file_type": metadata.get("general.file_type"),
            "general.alignment": metadata.get("general.alignment", 32),
        },
        "stats": stats,
        "selected_entries": len(manifest),
        "payload_bytes": payload_bytes,
        "pack_bytes": pack_bytes,
        "pack_sha256": sha256_file(pack_path),
        **payload_stats,
        "selected": manifest,
        "reproduce_command": " ".join([
            ".Agent/run-tools/kimi_iq1s_selected_payload_pack.py",
            f"--selected-plan-tsv {args.selected_plan_tsv}",
            f"--out-dir {args.out_dir}",
            f"--max-entries {args.max_entries}",
            f"--include-types {args.include_types}",
        ] + ([f"--header-cache {args.header_cache}"] if args.header_cache is not None else []) +
            ([f"--range-backend {args.range_backend}"] if args.range_backend != "auto" else []) +
            ([f"--range-timeout {args.range_timeout}"] if args.range_timeout != 180 else []) +
            ([f"--payload-chunk-mib {args.payload_chunk_mib}"] if args.payload_chunk_mib != 64 else []) +
            (["--metadata-only"] if args.metadata_only else [])),
    }
    (args.out_dir / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8")
    write_markdown(args.out_dir / "report.md", result)
    print(args.out_dir / "report.md")
    print(f"selected_entries={len(manifest)}")
    print(f"payload_bytes={payload_bytes}")
    print(f"pack_bytes={pack_bytes}")
    print(f"pack_sha256={result['pack_sha256']}")
    print(f"payload_write_mode={result['payload_write_mode']}")
    print(f"payload_groups={result['payload_groups']}")
    print(f"payload_range_reads={result['payload_range_reads']}")
    print(f"payload_streamed_bytes={result['payload_streamed_bytes']}")
    print(f"payload_write_wall_sec={result['payload_write_wall_sec']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
