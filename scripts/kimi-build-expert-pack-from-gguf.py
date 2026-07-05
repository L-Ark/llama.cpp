#!/usr/bin/env python3
import argparse
import glob
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


PACK_MAGIC = b"GGMLMOEPACKv1\0\0\0"
PACK_HEADER_BYTES = 16 + 4 + 4 + 8 + 8
PACK_ENTRY_BYTES = 128 + 4 + 4 + 8 + 8
ALIGNMENT = 4096
EXPERT_RE = re.compile(r"^blk\.(?P<layer>\d+)\.ffn_(?P<kind>up|gate|down)_exps\.weight$")


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


def tensor_type_name(tensor) -> str:
    tt = getattr(tensor, "tensor_type", "")
    return getattr(tt, "name", str(tt))


def tensor_shape(tensor) -> list[int]:
    shape = getattr(tensor, "shape", [])
    try:
        return [int(x) for x in shape.tolist()]
    except AttributeError:
        return [int(x) for x in shape]


def parse_model_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matched = [Path(p) for p in sorted(glob.glob(pattern))]
        if not matched:
            raise RuntimeError(f"model glob matched no files: {pattern}")
        paths.extend(matched)
    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def scan_tensors(model_paths: list[Path], include_kinds: set[str], n_experts: int) -> list[dict]:
    from gguf.gguf_reader import GGUFReader

    rows = []
    seen_names = set()
    for shard_index, model_path in enumerate(model_paths):
        reader = GGUFReader(str(model_path))
        for tensor in reader.tensors:
            match = EXPERT_RE.match(tensor.name)
            if not match:
                continue
            kind = match.group("kind")
            if kind not in include_kinds:
                continue
            if tensor.name in seen_names:
                raise RuntimeError(f"duplicate expert tensor across shards: {tensor.name}")
            seen_names.add(tensor.name)
            name_bytes = tensor.name.encode("utf-8")
            if len(name_bytes) >= 128:
                raise RuntimeError(f"tensor name too long for pack index: {tensor.name}")
            n_bytes = int(tensor.n_bytes)
            if n_bytes % n_experts != 0:
                raise RuntimeError(f"{tensor.name}: {n_bytes} bytes not divisible by n_experts={n_experts}")
            expert_bytes = n_bytes // n_experts
            rows.append({
                "tensor": tensor.name,
                "layer": int(match.group("layer")),
                "kind": kind,
                "type": tensor_type_name(tensor),
                "shape": tensor_shape(tensor),
                "shard_index": shard_index,
                "shard_path": str(model_path),
                "data_offset": int(tensor.data_offset),
                "tensor_bytes": n_bytes,
                "expert_bytes": expert_bytes,
                "n_experts": n_experts,
                "entries": n_experts,
            })
    rows.sort(key=lambda r: (r["layer"], r["kind"], r["tensor"]))
    return rows


def estimate_pack(rows: list[dict]) -> dict:
    n_entries = sum(int(r["entries"]) for r in rows)
    index_bytes = n_entries * PACK_ENTRY_BYTES
    data_start = align_up(PACK_HEADER_BYTES + index_bytes)
    offset = data_start
    data_bytes = 0
    padding_bytes = data_start - (PACK_HEADER_BYTES + index_bytes)
    for row in rows:
        for _ in range(int(row["entries"])):
            aligned = align_up(offset)
            padding_bytes += aligned - offset
            offset = aligned + int(row["expert_bytes"])
            data_bytes += int(row["expert_bytes"])
    return {
        "pack_magic": PACK_MAGIC.decode("ascii", errors="replace").rstrip("\0"),
        "alignment": ALIGNMENT,
        "header_bytes": PACK_HEADER_BYTES,
        "entry_bytes": PACK_ENTRY_BYTES,
        "n_entries": n_entries,
        "index_bytes": index_bytes,
        "data_start": data_start,
        "data_bytes": data_bytes,
        "padding_bytes": padding_bytes,
        "estimated_pack_bytes": offset,
    }


def summarize(rows: list[dict], pack: dict, model_paths: list[Path], include_kinds: set[str], n_experts: int, repo_root: Path) -> dict:
    by_kind = defaultdict(lambda: {"tensors": 0, "entries": 0, "tensor_bytes": 0, "pack_data_bytes": 0})
    by_type = defaultdict(lambda: {"tensors": 0, "entries": 0, "tensor_bytes": 0, "pack_data_bytes": 0})
    by_kind_type = defaultdict(lambda: {"tensors": 0, "entries": 0, "tensor_bytes": 0, "pack_data_bytes": 0})
    layers = defaultdict(set)
    largest_shard = 0
    total_shard_bytes = 0
    for path in model_paths:
        st = path.stat()
        largest_shard = max(largest_shard, st.st_size)
        total_shard_bytes += st.st_size
    for row in rows:
        entries = int(row["entries"])
        tensor_bytes = int(row["tensor_bytes"])
        kind = row["kind"]
        typ = row["type"]
        for target in (by_kind[kind], by_type[typ], by_kind_type[f"{kind}/{typ}"]):
            target["tensors"] += 1
            target["entries"] += entries
            target["tensor_bytes"] += tensor_bytes
            target["pack_data_bytes"] += tensor_bytes
        layers[kind].add(int(row["layer"]))
    streaming_required = largest_shard + int(pack["estimated_pack_bytes"])
    return {
        "repo_head": git_head(repo_root),
        "model_paths": [str(p) for p in model_paths],
        "model_shards": len(model_paths),
        "model_total_bytes": total_shard_bytes,
        "largest_shard_bytes": largest_shard,
        "include_kinds": sorted(include_kinds),
        "n_experts": n_experts,
        "dry_run": True,
        "expert_tensors": len(rows),
        "layers_by_kind": {k: sorted(v) for k, v in sorted(layers.items())},
        "pack": pack,
        "streaming_required_bytes": streaming_required,
        "by_kind": dict(sorted(by_kind.items())),
        "by_type": dict(sorted(by_type.items())),
        "by_kind_type": dict(sorted(by_kind_type.items())),
    }


def write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("layer\tkind\ttype\ttensor_bytes\texpert_bytes\tn_experts\tentries\tshard_index\tdata_offset\tshape\ttensor\tshard_path\n")
        for row in rows:
            f.write(
                f"{row['layer']}\t{row['kind']}\t{row['type']}\t{row['tensor_bytes']}\t"
                f"{row['expert_bytes']}\t{row['n_experts']}\t{row['entries']}\t"
                f"{row['shard_index']}\t{row['data_offset']}\t{','.join(map(str, row['shape']))}\t"
                f"{row['tensor']}\t{row['shard_path']}\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan a Kimi GGMLMOEPACKv1 expert pack from GGUF shards without copying tensor data."
    )
    parser.add_argument("--model-glob", action="append", required=True, help="Glob for GGUF shards; may be repeated.")
    parser.add_argument("--n-experts", type=int, default=384)
    parser.add_argument("--include-kind", default="up,gate,down", help="Comma-separated subset of up,gate,down.")
    parser.add_argument("--dry-run", action="store_true", help="Required in this phase; no pack data is written.")
    parser.add_argument("--json", type=Path, required=True, help="Summary JSON output path.")
    parser.add_argument("--tsv", type=Path, required=True, help="Tensor inventory TSV output path.")
    args = parser.parse_args()

    if not args.dry_run:
        raise RuntimeError("only --dry-run is implemented; pack writing requires a later plan")
    if args.n_experts <= 0:
        raise RuntimeError("--n-experts must be positive")
    include_kinds = {k.strip() for k in args.include_kind.split(",") if k.strip()}
    invalid = include_kinds - {"up", "gate", "down"}
    if invalid:
        raise RuntimeError(f"invalid --include-kind values: {sorted(invalid)}")
    if not include_kinds:
        raise RuntimeError("--include-kind selected no tensor kinds")

    repo_root = Path(__file__).resolve().parents[1]
    install_gguf_import(repo_root)
    model_paths = parse_model_paths(args.model_glob)
    rows = scan_tensors(model_paths, include_kinds, args.n_experts)
    if not rows:
        raise RuntimeError("no matching expert tensors found")
    pack = estimate_pack(rows)
    summary = summarize(rows, pack, model_paths, include_kinds, args.n_experts, repo_root)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_tsv(args.tsv, rows)

    print(f"model_shards={summary['model_shards']}")
    print(f"expert_tensors={summary['expert_tensors']}")
    print(f"entries={pack['n_entries']}")
    print(f"estimated_pack_bytes={pack['estimated_pack_bytes']}")
    print(f"estimated_pack_gib={pack['estimated_pack_bytes'] / (1024 ** 3):.3f}")
    print(f"streaming_required_gib={summary['streaming_required_bytes'] / (1024 ** 3):.3f}")
    for key, item in summary["by_kind_type"].items():
        print(f"{key}: tensors={item['tensors']} entries={item['entries']} bytes={item['tensor_bytes']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
