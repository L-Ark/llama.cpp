#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any


PACK_MAGIC = b"GGMLMOEPACKv1\0\0\0"
PACK_HEADER = struct.Struct("<16sIIQQ")
PACK_ENTRY = struct.Struct("<128siIQQ")
GGUF_MAGIC = 0x46554747
ALIGNMENT = 4096
EXPERT_RE = re.compile(r"^blk\.(?P<layer>\d+)\.ffn_(?P<kind>up|gate|down)_exps\.weight$")

GGUF_UINT8 = 0
GGUF_INT8 = 1
GGUF_UINT16 = 2
GGUF_INT16 = 3
GGUF_UINT32 = 4
GGUF_INT32 = 5
GGUF_FLOAT32 = 6
GGUF_BOOL = 7
GGUF_STRING = 8
GGUF_ARRAY = 9
GGUF_UINT64 = 10
GGUF_INT64 = 11
GGUF_FLOAT64 = 12

SCALAR_FORMATS: dict[int, str] = {
    GGUF_UINT8: "B",
    GGUF_INT8: "b",
    GGUF_UINT16: "H",
    GGUF_INT16: "h",
    GGUF_UINT32: "I",
    GGUF_INT32: "i",
    GGUF_FLOAT32: "f",
    GGUF_BOOL: "?",
    GGUF_UINT64: "Q",
    GGUF_INT64: "q",
    GGUF_FLOAT64: "d",
}


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def install_gguf_import(repo_root: Path) -> None:
    gguf_py = repo_root / "gguf-py"
    if str(gguf_py) not in sys.path:
        sys.path.insert(0, str(gguf_py))


def git_head(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def tensor_type_name(raw_type: int) -> str:
    from gguf.constants import GGMLQuantizationType

    return GGMLQuantizationType(raw_type).name


def tensor_nbytes(raw_type: int, dims: list[int]) -> int:
    from gguf.constants import GGMLQuantizationType, GGML_QUANT_SIZES

    ggml_type = GGMLQuantizationType(raw_type)
    block_size, type_size = GGML_QUANT_SIZES[ggml_type]
    n_elements = 1
    for dim in dims:
        n_elements *= dim
    if n_elements % block_size != 0:
        raise RuntimeError(f"tensor elements {n_elements} not divisible by block size {block_size}")
    return n_elements * type_size // block_size


class PrefixReader:
    def __init__(self, data: bytes, source: str) -> None:
        self.data = data
        self.source = source
        self.offset = 0

    def need(self, nbytes: int) -> None:
        if self.offset + nbytes > len(self.data):
            raise RuntimeError(
                f"{self.source}: metadata prefix too small at offset {self.offset}, "
                f"need {nbytes} more bytes, prefix={len(self.data)}"
            )

    def read(self, nbytes: int) -> bytes:
        self.need(nbytes)
        out = self.data[self.offset:self.offset + nbytes]
        self.offset += nbytes
        return out

    def read_scalar(self, type_id: int) -> Any:
        fmt = SCALAR_FORMATS.get(type_id)
        if fmt is None:
            raise RuntimeError(f"{self.source}: unsupported GGUF scalar type {type_id}")
        return struct.unpack("<" + fmt, self.read(struct.calcsize("<" + fmt)))[0]

    def read_u32(self) -> int:
        return int(self.read_scalar(GGUF_UINT32))

    def read_u64(self) -> int:
        return int(self.read_scalar(GGUF_UINT64))

    def read_str(self) -> str:
        nbytes = self.read_u64()
        return self.read(nbytes).decode("utf-8")

    def skip_value(self, type_id: int) -> None:
        if type_id == GGUF_STRING:
            nbytes = self.read_u64()
            self.read(nbytes)
            return
        if type_id == GGUF_ARRAY:
            item_type = self.read_u32()
            count = self.read_u64()
            if item_type in SCALAR_FORMATS:
                self.read(struct.calcsize("<" + SCALAR_FORMATS[item_type]) * count)
                return
            for _ in range(count):
                self.skip_value(item_type)
            return
        if type_id in SCALAR_FORMATS:
            self.read_scalar(type_id)
            return
        raise RuntimeError(f"{self.source}: unsupported GGUF value type {type_id}")

    def read_value(self, type_id: int) -> Any:
        if type_id == GGUF_STRING:
            return self.read_str()
        if type_id == GGUF_ARRAY:
            item_type = self.read_u32()
            count = self.read_u64()
            values = []
            for _ in range(count):
                values.append(self.read_value(item_type))
            return values
        if type_id in SCALAR_FORMATS:
            return self.read_scalar(type_id)
        raise RuntimeError(f"{self.source}: unsupported GGUF value type {type_id}")


def parse_gguf_prefix(data: bytes, source: str, shard_size: int, n_experts: int) -> tuple[list[dict], dict[str, Any]]:
    reader = PrefixReader(data, source)
    magic = reader.read_u32()
    if magic != GGUF_MAGIC:
        raise RuntimeError(f"{source}: invalid GGUF magic {magic:#x}")
    version = reader.read_u32()
    if version not in (2, 3):
        raise RuntimeError(f"{source}: unsupported GGUF version {version}")
    tensor_count = reader.read_u64()
    kv_count = reader.read_u64()

    alignment = 32
    kv_seen = 0
    for _ in range(kv_count):
        key = reader.read_str()
        value_type = reader.read_u32()
        if key == "general.alignment":
            alignment = int(reader.read_value(value_type))
        else:
            reader.skip_value(value_type)
        kv_seen += 1

    rows = []
    for _ in range(tensor_count):
        name = reader.read_str()
        n_dims = reader.read_u32()
        dims = [reader.read_u64() for _ in range(n_dims)]
        raw_type = reader.read_u32()
        tensor_offset = reader.read_u64()
        match = EXPERT_RE.match(name)
        if not match:
            continue
        n_bytes = tensor_nbytes(raw_type, dims)
        if n_bytes % n_experts != 0:
            raise RuntimeError(f"{source}: {name}: {n_bytes} bytes not divisible by n_experts={n_experts}")
        rows.append({
            "tensor": name,
            "layer": int(match.group("layer")),
            "kind": match.group("kind"),
            "type": tensor_type_name(raw_type),
            "shape": dims,
            "tensor_bytes": n_bytes,
            "expert_bytes": n_bytes // n_experts,
            "tensor_offset": int(tensor_offset),
            "n_experts": n_experts,
            "shard_name": source,
            "shard_size": shard_size,
        })

    data_offset = align_up(reader.offset, alignment)
    meta = {
        "source": source,
        "version": version,
        "tensor_count": tensor_count,
        "kv_count": kv_count,
        "kv_seen": kv_seen,
        "alignment": alignment,
        "metadata_bytes": reader.offset,
        "data_offset": data_offset,
        "expert_tensors": len(rows),
    }
    return rows, meta


def load_pack_entries(paths: list[Path]) -> tuple[list[dict], dict[str, Any]]:
    selected: dict[tuple[str, int], dict] = {}
    input_entries = 0
    duplicate_replacements = 0
    by_pack = []
    for source_order, path in enumerate(paths):
        count = 0
        data_bytes = 0
        with path.open("rb") as f:
            header = f.read(PACK_HEADER.size)
            if len(header) != PACK_HEADER.size:
                raise RuntimeError(f"{path}: short pack header")
            magic, version, header_size, n_entries, data_start = PACK_HEADER.unpack(header)
            if magic != PACK_MAGIC or version != 1 or header_size < PACK_HEADER.size or data_start < header_size:
                raise RuntimeError(f"{path}: invalid pack header")
            for entry_idx in range(n_entries):
                raw = f.read(PACK_ENTRY.size)
                if len(raw) != PACK_ENTRY.size:
                    raise RuntimeError(f"{path}: short pack index")
                name_raw, expert_idx, _reserved, offset, nbytes = PACK_ENTRY.unpack(raw)
                name = name_raw.split(b"\0", 1)[0].decode("utf-8")
                key = (name, int(expert_idx))
                if key in selected:
                    duplicate_replacements += 1
                selected[key] = {
                    "tensor": name,
                    "expert_idx": int(expert_idx),
                    "current_nbytes": int(nbytes),
                    "source_pack": str(path),
                    "source_order": source_order,
                    "entry_idx": entry_idx,
                    "pack_offset": int(offset),
                }
                count += 1
                input_entries += 1
                data_bytes += int(nbytes)
        by_pack.append({
            "pack": str(path),
            "entries": count,
            "indexed_data_bytes": data_bytes,
        })
    entries = sorted(selected.values(), key=lambda e: (e["tensor"], e["expert_idx"]))
    current_selected_bytes = sum(int(e["current_nbytes"]) for e in entries)
    return entries, {
        "input_entries": input_entries,
        "unique_entries": len(entries),
        "duplicate_replacements": duplicate_replacements,
        "current_selected_bytes": current_selected_bytes,
        "by_pack": by_pack,
    }


def hf_api_json(repo: str, revision: str) -> dict:
    quoted_repo = urllib.parse.quote(repo, safe="/")
    quoted_rev = urllib.parse.quote(revision, safe="")
    url = f"https://huggingface.co/api/models/{quoted_repo}/revision/{quoted_rev}?blobs=true"
    req = urllib.request.Request(url, headers={"User-Agent": "kimi-hotkey-pack-planner/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def hf_files(repo: str, revision: str, remote_prefix: str) -> list[dict]:
    data = hf_api_json(repo, revision)
    out = []
    for item in data.get("siblings", []):
        name = item.get("rfilename", "")
        if not name.startswith(remote_prefix) or not name.endswith(".gguf"):
            continue
        size = item.get("size")
        if size is None and isinstance(item.get("lfs"), dict):
            size = item["lfs"].get("size")
        if size is None:
            raise RuntimeError(f"missing size for remote file {name}")
        out.append({"rfilename": name, "size": int(size)})
    if not out:
        raise RuntimeError(f"no remote GGUF files found for prefix {remote_prefix!r} in {repo}@{revision}")
    out.sort(key=lambda x: x["rfilename"])
    return out


def hf_resolve_url(repo: str, revision: str, rfilename: str) -> str:
    quoted_repo = urllib.parse.quote(repo, safe="/")
    quoted_rev = urllib.parse.quote(revision, safe="")
    quoted_file = urllib.parse.quote(rfilename, safe="/")
    return f"https://huggingface.co/{quoted_repo}/resolve/{quoted_rev}/{quoted_file}"


def fetch_prefix(url: str, nbytes: int) -> tuple[bytes, str]:
    req = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes=0-{nbytes - 1}",
            "User-Agent": "kimi-hotkey-pack-planner/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        status = getattr(resp, "status", resp.getcode())
        content_range = resp.headers.get("Content-Range", "")
        if status != 206:
            raise RuntimeError(f"remote did not honor Range request: status={status} url={url}")
        if not content_range.startswith("bytes 0-"):
            raise RuntimeError(f"unexpected Content-Range {content_range!r} for {url}")
        data = resp.read(nbytes + 1)
    if len(data) > nbytes:
        raise RuntimeError(f"remote returned more than requested range: {len(data)} > {nbytes}")
    return data, content_range


def estimate_pack_size(entries: list[dict], nbytes_key: str) -> dict[str, int]:
    data_start = align_up(PACK_HEADER.size + len(entries) * PACK_ENTRY.size)
    offset = data_start
    payload = 0
    padding = data_start - (PACK_HEADER.size + len(entries) * PACK_ENTRY.size)
    for entry in entries:
        aligned = align_up(offset)
        padding += aligned - offset
        offset = aligned + int(entry[nbytes_key])
        payload += int(entry[nbytes_key])
    return {
        "entries": len(entries),
        "data_start": data_start,
        "payload_bytes": payload,
        "padding_bytes": padding,
        "estimated_pack_bytes": offset,
    }


def build_plan(pack_entries: list[dict], remote_rows: list[dict], n_experts: int) -> tuple[list[dict], dict[str, Any]]:
    remote_by_tensor = {}
    for row in remote_rows:
        tensor = row["tensor"]
        if tensor in remote_by_tensor:
            raise RuntimeError(f"duplicate remote tensor metadata: {tensor}")
        remote_by_tensor[tensor] = row

    planned = []
    missing = []
    for entry in pack_entries:
        tensor = entry["tensor"]
        remote = remote_by_tensor.get(tensor)
        if remote is None:
            missing.append({"tensor": tensor, "expert_idx": entry["expert_idx"]})
            continue
        if entry["expert_idx"] >= n_experts:
            raise RuntimeError(f"{tensor}: expert {entry['expert_idx']} >= n_experts {n_experts}")
        item = dict(entry)
        item.update({
            "layer": remote["layer"],
            "kind": remote["kind"],
            "remote_type": remote["type"],
            "remote_shape": remote["shape"],
            "remote_tensor_bytes": remote["tensor_bytes"],
            "remote_nbytes": remote["expert_bytes"],
            "remote_shard_name": remote["shard_name"],
            "remote_shard_size": remote["shard_size"],
            "remote_tensor_offset": remote["tensor_offset"],
        })
        planned.append(item)
    if missing:
        preview = ", ".join(f"{m['tensor']}:{m['expert_idx']}" for m in missing[:20])
        raise RuntimeError(f"missing {len(missing)} selected keys in remote metadata; first keys: {preview}")

    by_kind = defaultdict(lambda: {"entries": 0, "current_bytes": 0, "remote_bytes": 0})
    by_kind_type = defaultdict(lambda: {"entries": 0, "current_bytes": 0, "remote_bytes": 0})
    required_shards = {}
    for item in planned:
        for bucket in (by_kind[item["kind"]], by_kind_type[f"{item['kind']}/{item['remote_type']}"]):
            bucket["entries"] += 1
            bucket["current_bytes"] += int(item["current_nbytes"])
            bucket["remote_bytes"] += int(item["remote_nbytes"])
        shard_name = item["remote_shard_name"]
        required_shards[shard_name] = max(int(item["remote_shard_size"]), int(required_shards.get(shard_name, 0)))

    current_pack = estimate_pack_size(planned, "current_nbytes")
    remote_pack = estimate_pack_size(planned, "remote_nbytes")
    largest_required_shard = max(required_shards.values(), default=0)
    return planned, {
        "selected_entries": len(planned),
        "selected_tensors": len({item["tensor"] for item in planned}),
        "required_remote_shards": len(required_shards),
        "largest_required_remote_shard_bytes": largest_required_shard,
        "current_selected_pack": current_pack,
        "remote_selected_pack": remote_pack,
        "one_shard_streaming_temp_bytes": largest_required_shard + remote_pack["estimated_pack_bytes"],
        "by_kind": dict(sorted(by_kind.items())),
        "by_kind_type": dict(sorted(by_kind_type.items())),
    }


def write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "layer", "kind", "tensor", "expert_idx", "current_nbytes", "remote_nbytes",
        "remote_type", "remote_shard_name", "remote_shard_size", "remote_tensor_offset",
        "source_pack",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan a selected hot-key expert pack from remote GGUF metadata without downloading tensor payloads."
    )
    parser.add_argument("--pack", action="append", type=Path, required=True,
                        help="Input GGMLMOEPACKv1 pack index; repeat in runtime source order.")
    parser.add_argument("--hf-repo", required=True, help="HuggingFace repo id, e.g. AesSedai/Kimi-K2.7-Code-GGUF.")
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--remote-prefix", required=True, help="Remote GGUF path prefix, e.g. IQ2_XXS/.")
    parser.add_argument("--range-mib", type=int, default=64,
                        help="Maximum metadata prefix bytes to request from each shard.")
    parser.add_argument("--n-experts", type=int, default=384)
    parser.add_argument("--disk-path", type=Path, default=Path("."),
                        help="Filesystem path used to report available bytes.")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--tsv", type=Path, required=True)
    args = parser.parse_args()

    if args.range_mib <= 0:
        raise RuntimeError("--range-mib must be positive")
    if args.n_experts <= 0:
        raise RuntimeError("--n-experts must be positive")

    repo_root = Path(__file__).resolve().parents[1]
    install_gguf_import(repo_root)

    pack_entries, pack_summary = load_pack_entries(args.pack)
    range_bytes = args.range_mib * 1024 * 1024
    remote_files = hf_files(args.hf_repo, args.hf_revision, args.remote_prefix)

    remote_rows = []
    remote_meta = []
    range_reads = []
    for remote_file in remote_files:
        name = remote_file["rfilename"]
        url = hf_resolve_url(args.hf_repo, args.hf_revision, name)
        prefix, content_range = fetch_prefix(url, range_bytes)
        rows, meta = parse_gguf_prefix(prefix, name, int(remote_file["size"]), args.n_experts)
        remote_rows.extend(rows)
        remote_meta.append(meta)
        range_reads.append({
            "rfilename": name,
            "requested_bytes": range_bytes,
            "received_bytes": len(prefix),
            "content_range": content_range,
            "expert_tensors": len(rows),
            "metadata_bytes": meta["metadata_bytes"],
            "data_offset": meta["data_offset"],
            "shard_size": int(remote_file["size"]),
        })

    planned, selected_summary = build_plan(pack_entries, remote_rows, args.n_experts)
    disk = shutil.disk_usage(args.disk_path)
    direct_range_temp_upper_bound = (
        selected_summary["remote_selected_pack"]["estimated_pack_bytes"] + range_bytes
    )

    summary = {
        "repo_head": git_head(repo_root),
        "hf_repo": args.hf_repo,
        "hf_revision": args.hf_revision,
        "remote_prefix": args.remote_prefix,
        "range_mib": args.range_mib,
        "n_experts": args.n_experts,
        "packs": [str(p) for p in args.pack],
        "pack_summary": pack_summary,
        "remote_files": remote_files,
        "remote_metadata": remote_meta,
        "range_reads": range_reads,
        "selected_summary": selected_summary,
        "direct_range_temp_upper_bound_bytes": direct_range_temp_upper_bound,
        "disk_path": str(args.disk_path),
        "disk_free_bytes": disk.free,
        "fits_one_shard_streaming_current_disk": disk.free >= selected_summary["one_shard_streaming_temp_bytes"],
        "fits_direct_range_current_disk": disk.free >= direct_range_temp_upper_bound,
    }

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_tsv(args.tsv, planned)

    current_bytes = selected_summary["current_selected_pack"]["payload_bytes"]
    remote_bytes = selected_summary["remote_selected_pack"]["payload_bytes"]
    ratio = remote_bytes / current_bytes if current_bytes else 0.0
    print(f"remote_files={len(remote_files)}")
    print(f"remote_expert_tensors={len(remote_rows)}")
    print(f"selected_entries={selected_summary['selected_entries']}")
    print(f"selected_tensors={selected_summary['selected_tensors']}")
    print(f"current_selected_payload_gib={current_bytes / (1024 ** 3):.3f}")
    print(f"remote_selected_payload_gib={remote_bytes / (1024 ** 3):.3f}")
    print(f"remote_to_current_payload_ratio={ratio:.6f}")
    print(
        "remote_selected_pack_gib="
        f"{selected_summary['remote_selected_pack']['estimated_pack_bytes'] / (1024 ** 3):.3f}"
    )
    print(
        "one_shard_streaming_temp_gib="
        f"{selected_summary['one_shard_streaming_temp_bytes'] / (1024 ** 3):.3f}"
    )
    print(f"direct_range_temp_upper_bound_gib={direct_range_temp_upper_bound / (1024 ** 3):.3f}")
    print(f"disk_free_gib={disk.free / (1024 ** 3):.3f}")
    print(f"fits_one_shard_streaming_current_disk={int(summary['fits_one_shard_streaming_current_disk'])}")
    print(f"fits_direct_range_current_disk={int(summary['fits_direct_range_current_disk'])}")
    for key, item in summary["selected_summary"]["by_kind_type"].items():
        print(
            f"{key}: entries={item['entries']} "
            f"current_gib={item['current_bytes'] / (1024 ** 3):.3f} "
            f"remote_gib={item['remote_bytes'] / (1024 ** 3):.3f}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
