#!/usr/bin/env python3
import argparse
import csv
import struct
from pathlib import Path


PACK_MAGIC = b"GGMLMOEPACKv1"
PACK_HEADER = struct.Struct("<16sIIQQ")
PACK_ENTRY = struct.Struct("<128siIQQ")


def load_pack_entries(path: Path):
    with path.open("rb") as f:
        header = f.read(PACK_HEADER.size)
        if len(header) != PACK_HEADER.size:
            raise RuntimeError(f"{path}: short expert-pack header")
        magic, version, header_size, n_entries, data_start = PACK_HEADER.unpack(header)
        if not magic.startswith(PACK_MAGIC) or version != 1 or header_size < PACK_HEADER.size or data_start < header_size:
            raise RuntimeError(f"{path}: invalid expert-pack header")
        f.seek(header_size)
        entries = []
        for _ in range(n_entries):
            raw = f.read(PACK_ENTRY.size)
            if len(raw) != PACK_ENTRY.size:
                raise RuntimeError(f"{path}: short expert-pack index")
            name_raw, expert_idx, _reserved, offset, nbytes = PACK_ENTRY.unpack(raw)
            name = name_raw.split(b"\0", 1)[0].decode("utf-8")
            if offset < data_start:
                raise RuntimeError(f"{path}: entry before data_start: {name}:{expert_idx}")
            entries.append(
                {
                    "tensor": name,
                    "expert_idx": int(expert_idx),
                    "offset": int(offset),
                    "nbytes": int(nbytes),
                    "pack_order": len(entries),
                }
            )
        return entries


def load_rank_counts(path: Path | None):
    if path is None:
        return {}
    counts = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tensor = row.get("tensor") or row.get("src0_name") or row.get("name")
            expert = row.get("expert") or row.get("expert_idx") or row.get("expert_index")
            if tensor is None or expert in (None, ""):
                continue
            try:
                key = (tensor, int(expert))
            except ValueError:
                continue
            counts[key] = counts.get(key, 0) + 1
    return counts


def main():
    ap = argparse.ArgumentParser(description="Export an expert-pack direct manifest for DS4 one-direct hot-pool experiments.")
    ap.add_argument("--pack", required=True, type=Path, help="Input .expert-pack file.")
    ap.add_argument("--output", required=True, type=Path, help="Output direct manifest: tensor,expert,offset,nbytes.")
    ap.add_argument("--tensor-substr", default="", help="Only include tensors containing this substring.")
    ap.add_argument("--rank-csv", type=Path, help="Optional CSV with tensor/expert columns; occurrence count ranks hot entries first.")
    ap.add_argument("--limit", type=int, default=0, help="Maximum number of rows to write after filtering/ranking.")
    args = ap.parse_args()

    entries = load_pack_entries(args.pack)
    if args.tensor_substr:
        entries = [e for e in entries if args.tensor_substr in e["tensor"]]
    counts = load_rank_counts(args.rank_csv)
    if counts:
        entries.sort(
            key=lambda e: (
                -counts.get((e["tensor"], e["expert_idx"]), 0),
                e["pack_order"],
            )
        )
    if args.limit > 0:
        entries = entries[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        for e in entries:
            f.write(f"{e['tensor']},{e['expert_idx']},{e['offset']},{e['nbytes']}\n")

    ranked = sum(1 for e in entries if counts.get((e["tensor"], e["expert_idx"]), 0) > 0)
    total_bytes = sum(e["nbytes"] for e in entries)
    print(f"entries={len(entries)} ranked_entries={ranked} bytes={total_bytes} output={args.output}")


if __name__ == "__main__":
    main()
