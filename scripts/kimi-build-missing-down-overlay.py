#!/usr/bin/env python3
import argparse
import glob
import os
import struct
import sys
from pathlib import Path


PACK_MAGIC = b"GGMLMOEPACKv1\0\0\0"
PACK_HEADER = struct.Struct("<16sIIQQ")
PACK_ENTRY = struct.Struct("<128siIQQ")
ALIGNMENT = 4096


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def parse_entry(value: str) -> tuple[str, int]:
    if ":" not in value:
        raise argparse.ArgumentTypeError(f"entry must be TENSOR:EXPERT, got {value!r}")
    tensor, expert = value.rsplit(":", 1)
    if not tensor:
        raise argparse.ArgumentTypeError(f"entry tensor is empty: {value!r}")
    try:
        expert_idx = int(expert)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"entry expert is not an int: {value!r}") from exc
    if expert_idx < 0:
        raise argparse.ArgumentTypeError(f"entry expert must be non-negative: {value!r}")
    return tensor, expert_idx


def install_gguf_import(repo_root: Path) -> None:
    gguf_py = repo_root / "gguf-py"
    if str(gguf_py) not in sys.path:
        sys.path.insert(0, str(gguf_py))


def load_pack_entries(path: Path) -> list[dict]:
    entries = []
    with path.open("rb") as f:
        header = f.read(PACK_HEADER.size)
        if len(header) != PACK_HEADER.size:
            raise RuntimeError(f"{path}: short pack header")
        magic, version, header_size, n_entries, data_start = PACK_HEADER.unpack(header)
        if magic != PACK_MAGIC or version != 1 or header_size < PACK_HEADER.size or data_start < header_size:
            raise RuntimeError(f"{path}: invalid pack header")
        for _ in range(n_entries):
            raw = f.read(PACK_ENTRY.size)
            if len(raw) != PACK_ENTRY.size:
                raise RuntimeError(f"{path}: short pack index")
            name_raw, expert_idx, _reserved, _offset, nbytes = PACK_ENTRY.unpack(raw)
            name = name_raw.split(b"\0", 1)[0].decode("utf-8")
            if _offset < data_start:
                raise RuntimeError(f"{path}: entry offset before data_start: {name}:{expert_idx}")
            entries.append({
                "name": name,
                "expert_idx": expert_idx,
                "source_path": path,
                "source_offset": _offset,
                "nbytes": nbytes,
                "source_kind": "pack",
            })
    return entries


def load_existing_pack_keys(path: Path) -> set[tuple[str, int, int]]:
    return {(e["name"], e["expert_idx"], e["nbytes"]) for e in load_pack_entries(path)}


def find_tensors(model_paths: list[Path], wanted_names: set[str]):
    from gguf.gguf_reader import GGUFReader

    found = {}
    for model_path in model_paths:
        if wanted_names.issubset(found.keys()):
            break
        reader = GGUFReader(str(model_path))
        for tensor in reader.tensors:
            if tensor.name in wanted_names and tensor.name not in found:
                found[tensor.name] = {
                    "path": model_path,
                    "data_offset": int(tensor.data_offset),
                    "n_bytes": int(tensor.n_bytes),
                    "tensor_type": tensor.tensor_type.name,
                    "shape": tuple(int(x) for x in tensor.shape.tolist()),
                }
    missing = sorted(wanted_names - set(found.keys()))
    if missing:
        raise RuntimeError("missing tensors in GGUF shards: " + ", ".join(missing))
    return found


def copy_exact(src, dst, nbytes: int, chunk_size: int) -> None:
    remaining = nbytes
    while remaining:
        chunk = min(chunk_size, remaining)
        data = src.read(chunk)
        if len(data) != chunk:
            raise RuntimeError("short read while copying tensor slice")
        dst.write(data)
        remaining -= chunk


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a GGMLMOEPACKv1 overlay from selected GGUF expert slices."
    )
    parser.add_argument("--model-glob", required=True, help="Glob for GGUF shards.")
    parser.add_argument("--out", required=True, type=Path, help="Output overlay pack.")
    parser.add_argument("--entry", action="append", type=parse_entry, required=True,
                        help="Tensor/expert key, for example blk.9.ffn_down_exps.weight:264.")
    parser.add_argument("--n-experts", type=int, default=384,
                        help="Experts per tensor used to infer each expert slice size.")
    parser.add_argument("--expert-bytes", type=int, default=0,
                        help="Optional expected bytes per expert slice.")
    parser.add_argument("--reject-pack", action="append", type=Path, default=[],
                        help="Reject keys already present in an existing pack.")
    parser.add_argument("--include-pack", action="append", type=Path, default=[],
                        help="Copy all entries from an existing pack into the output first.")
    parser.add_argument("--chunk-size", type=int, default=16 * 1024 * 1024,
                        help="Copy chunk size in bytes.")
    args = parser.parse_args()

    if args.n_experts <= 0:
        raise RuntimeError("--n-experts must be positive")
    if args.chunk_size <= 0:
        raise RuntimeError("--chunk-size must be positive")

    repo_root = Path(__file__).resolve().parents[1]
    install_gguf_import(repo_root)

    model_paths = [Path(p) for p in sorted(glob.glob(args.model_glob))]
    if not model_paths:
        raise RuntimeError(f"model glob matched no files: {args.model_glob}")

    unique_entries = sorted(set(args.entry), key=lambda item: (item[0], item[1]))
    if len(unique_entries) != len(args.entry):
        print(f"deduplicated entries: requested={len(args.entry)} unique={len(unique_entries)}", file=sys.stderr)

    wanted_names = {name for name, _expert in unique_entries}
    tensors = find_tensors(model_paths, wanted_names)

    existing_keys = set()
    for reject_pack in args.reject_pack:
        existing_keys.update(load_existing_pack_keys(reject_pack))

    planned = []
    seen_keys = set()
    for include_pack in args.include_pack:
        for item in load_pack_entries(include_pack):
            key = (item["name"], item["expert_idx"], item["nbytes"])
            if key in seen_keys:
                raise RuntimeError(f"{include_pack}: duplicate included key {key}")
            if key in existing_keys:
                raise RuntimeError(f"{include_pack}: included key is present in reject pack {key}")
            seen_keys.add(key)
            planned.append(item)

    for name, expert_idx in unique_entries:
        info = tensors[name]
        tensor_nbytes = info["n_bytes"]
        if tensor_nbytes % args.n_experts != 0:
            raise RuntimeError(f"{name}: tensor bytes {tensor_nbytes} not divisible by {args.n_experts}")
        expert_bytes = tensor_nbytes // args.n_experts
        if args.expert_bytes and expert_bytes != args.expert_bytes:
            raise RuntimeError(f"{name}: expert bytes {expert_bytes} != expected {args.expert_bytes}")
        if expert_idx >= args.n_experts:
            raise RuntimeError(f"{name}: expert {expert_idx} >= n_experts {args.n_experts}")
        if len(name.encode("utf-8")) >= 128:
            raise RuntimeError(f"{name}: tensor name is too long for pack index")
        key = (name, expert_idx, expert_bytes)
        if key in seen_keys:
            raise RuntimeError(f"{name}:{expert_idx}: key already present in included pack")
        if key in existing_keys:
            raise RuntimeError(f"{name}:{expert_idx}: key already present in reject pack")
        seen_keys.add(key)
        planned.append({
            "name": name,
            "expert_idx": expert_idx,
            "source_path": info["path"],
            "source_offset": info["data_offset"] + expert_idx * expert_bytes,
            "nbytes": expert_bytes,
            "source_kind": "gguf",
            "tensor_type": info["tensor_type"],
            "shape": info["shape"],
        })

    planned.sort(key=lambda item: (item["name"], item["expert_idx"], item["nbytes"]))

    index_bytes = len(planned) * PACK_ENTRY.size
    data_start = align_up(PACK_HEADER.size + index_bytes)
    offset = data_start
    for item in planned:
        offset = align_up(offset)
        item["pack_offset"] = offset
        offset += item["nbytes"]
    total_size = offset

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = args.out.with_suffix(args.out.suffix + ".tmp")
    with tmp_out.open("wb") as out:
        out.write(PACK_HEADER.pack(PACK_MAGIC, 1, PACK_HEADER.size, len(planned), data_start))
        for item in planned:
            name_bytes = item["name"].encode("utf-8")
            out.write(PACK_ENTRY.pack(
                name_bytes + b"\0" * (128 - len(name_bytes)),
                item["expert_idx"],
                0,
                item["pack_offset"],
                item["nbytes"],
            ))
        pad = data_start - out.tell()
        if pad < 0:
            raise RuntimeError("internal error: negative pack padding")
        out.write(b"\0" * pad)

        open_sources = {}
        try:
            for item in planned:
                pos = out.tell()
                if pos > item["pack_offset"]:
                    raise RuntimeError("internal error: pack offset overlap")
                out.write(b"\0" * (item["pack_offset"] - pos))
                source_path = item["source_path"]
                src = open_sources.get(source_path)
                if src is None:
                    src = source_path.open("rb")
                    open_sources[source_path] = src
                src.seek(item["source_offset"])
                copy_exact(src, out, item["nbytes"], args.chunk_size)
        finally:
            for src in open_sources.values():
                src.close()

    os.replace(tmp_out, args.out)

    print(f"wrote {args.out}")
    print(f"entries={len(planned)} data_start={data_start} size={total_size}")
    for item in planned:
        print(
            f"{item['name']} expert={item['expert_idx']} bytes={item['nbytes']} "
            f"pack_offset={item['pack_offset']} source={item['source_path'].name} "
            f"source_offset={item['source_offset']} source_kind={item['source_kind']} "
            f"type={item.get('tensor_type', 'PACK')} shape={item.get('shape', '-')}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
