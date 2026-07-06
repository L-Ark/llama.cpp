#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path


MAGIC = b"GGMLMOEPACKv2\0\0\0"
HEADER = struct.Struct("<16sIIQQ")
ENTRY = struct.Struct("<128siiQQqqQQ")
ALIGNMENT = 4096


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def tensor_field(name: str) -> bytes:
    raw = name.encode("utf-8")
    if len(raw) >= 128:
        raise ValueError(f"tensor name too long: {name}")
    return raw + b"\0" * (128 - len(raw))


def kind_from_tensor(name: str) -> str:
    if ".ffn_up_exps." in name:
        return "up"
    if ".ffn_gate_exps." in name:
        return "gate"
    if ".ffn_down_exps." in name:
        return "down"
    return "other"


@dataclass
class RouteEntry:
    tensor: str
    expert_idx: int
    kind: str
    count: int = 0
    logical_nbytes: int = 0

    @property
    def logical_total(self) -> int:
        return self.count * self.logical_nbytes


def load_routes(paths: list[Path], kinds: set[str]) -> dict[tuple[str, int], RouteEntry]:
    entries: dict[tuple[str, int], RouteEntry] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                tensor = row.get("tensor", "")
                kind = kind_from_tensor(tensor)
                if kind not in kinds:
                    continue
                expert_idx = int(row["expert_idx"])
                count = int(row["count"])
                logical_nbytes = int(row["expert_bytes"])
                key = (tensor, expert_idx)
                entry = entries.get(key)
                if entry is None:
                    entries[key] = RouteEntry(tensor, expert_idx, kind, count, logical_nbytes)
                else:
                    entry.count += count
                    if entry.logical_nbytes != logical_nbytes:
                        entry.logical_nbytes = max(entry.logical_nbytes, logical_nbytes)
    return entries


def summarize(entries: list[RouteEntry], selected: set[tuple[str, int]], packed_ratio: float) -> dict:
    total_events = 0
    covered_events = 0
    total_logical_bytes = 0
    covered_logical_bytes = 0
    covered_packed_bytes = 0
    by_kind: dict[str, dict[str, int]] = {}

    for entry in entries:
        bucket = by_kind.setdefault(
            entry.kind,
            {
                "events": 0,
                "covered_events": 0,
                "logical_bytes": 0,
                "covered_logical_bytes": 0,
                "covered_packed_bytes": 0,
            },
        )
        logical_total = entry.logical_total
        total_events += entry.count
        total_logical_bytes += logical_total
        bucket["events"] += entry.count
        bucket["logical_bytes"] += logical_total
        if (entry.tensor, entry.expert_idx) in selected:
            packed_nbytes = math.ceil(entry.logical_nbytes * packed_ratio)
            packed_total = entry.count * packed_nbytes
            covered_events += entry.count
            covered_logical_bytes += logical_total
            covered_packed_bytes += packed_total
            bucket["covered_events"] += entry.count
            bucket["covered_logical_bytes"] += logical_total
            bucket["covered_packed_bytes"] += packed_total

    def finalize(bucket: dict[str, int]) -> dict:
        logical = bucket["logical_bytes"]
        covered_logical = bucket["covered_logical_bytes"]
        hybrid = logical - covered_logical + bucket["covered_packed_bytes"]
        return {
            **bucket,
            "hybrid_bytes": hybrid,
            "event_coverage": bucket["covered_events"] / bucket["events"] if bucket["events"] else 0.0,
            "byte_coverage": covered_logical / logical if logical else 0.0,
            "hybrid_byte_ratio": hybrid / logical if logical else 0.0,
        }

    by_kind_final = {kind: finalize(bucket) for kind, bucket in sorted(by_kind.items())}
    total = finalize(
        {
            "events": total_events,
            "covered_events": covered_events,
            "logical_bytes": total_logical_bytes,
            "covered_logical_bytes": covered_logical_bytes,
            "covered_packed_bytes": covered_packed_bytes,
        }
    )
    return {"total": total, "by_kind": by_kind_final}


def write_pack(path: Path, selected: list[RouteEntry], packed_type: int, packed_ratio: float) -> list[dict]:
    header_size = HEADER.size
    data_start = align_up(header_size + len(selected) * ENTRY.size)
    offset = data_start
    out = []
    with path.open("wb") as f:
        f.write(HEADER.pack(MAGIC, 2, header_size, len(selected), data_start))
        for entry in selected:
            packed_nbytes = max(1, math.ceil(entry.logical_nbytes * packed_ratio))
            packed_nb01 = packed_nbytes
            row = {
                "tensor": entry.tensor,
                "expert_idx": entry.expert_idx,
                "kind": entry.kind,
                "count": entry.count,
                "logical_nbytes": entry.logical_nbytes,
                "packed_type": packed_type,
                "offset": offset,
                "packed_nbytes": packed_nbytes,
                "ne00": entry.logical_nbytes,
                "ne01": 1,
                "nb01": packed_nb01,
            }
            out.append(row)
            f.write(
                ENTRY.pack(
                    tensor_field(entry.tensor),
                    entry.expert_idx,
                    packed_type,
                    offset,
                    packed_nbytes,
                    row["ne00"],
                    row["ne01"],
                    row["nb01"],
                    0,
                )
            )
            offset = align_up(offset + packed_nbytes)
        f.truncate(data_start)
    return out


def write_markdown(path: Path, report: dict) -> None:
    total = report["summary"]["total"]
    lines = [
        "# Kimi v2 shadow pack from dev routes",
        "",
        f"- Output pack: `{report['out_pack']}`",
        f"- Packed type id: `{report['packed_type']}`",
        f"- Packed ratio: `{report['packed_ratio']}`",
        f"- Selected entries: `{report['selected_entries']}`",
        f"- Input profiles: `{len(report['input_profiles'])}`",
        "",
        "This is a metadata-only pack for `GGML_MOE_EXPERT_PACK_V2` shadow profiling.",
        "It does not contain real expert payload bytes and must not be used for runtime H2D.",
        "",
        "## Aggregate",
        "",
        "| events | event coverage | byte coverage | hybrid byte ratio | logical GiB | hybrid GiB |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {total['events']} | {total['event_coverage']:.4f} | {total['byte_coverage']:.4f} | "
            f"{total['hybrid_byte_ratio']:.4f} | {total['logical_bytes'] / 1024**3:.3f} | "
            f"{total['hybrid_bytes'] / 1024**3:.3f} |"
        ),
        "",
        "## By Kind",
        "",
        "| kind | events | event coverage | byte coverage | hybrid byte ratio |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for kind, bucket in report["summary"]["by_kind"].items():
        lines.append(
            f"| {kind} | {bucket['events']} | {bucket['event_coverage']:.4f} | "
            f"{bucket['byte_coverage']:.4f} | {bucket['hybrid_byte_ratio']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a metadata-only GGMLMOEPACKv2 shadow pack from dev route profiles.")
    parser.add_argument("--profile", type=Path, action="append", required=True)
    parser.add_argument("--out-pack", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--max-entries", type=int, default=4096)
    parser.add_argument("--kind", default="up,gate,down")
    parser.add_argument("--packed-type", type=int, default=24, help="Default 24 matches the local GGML_TYPE_IQ1_S id.")
    parser.add_argument("--packed-ratio", type=float, default=0.55)
    args = parser.parse_args()

    if args.max_entries <= 0:
        raise ValueError("--max-entries must be positive")
    if args.packed_ratio <= 0.0 or args.packed_ratio > 1.0:
        raise ValueError("--packed-ratio must be in (0, 1]")

    kinds = {item.strip() for item in args.kind.split(",") if item.strip()}
    entries = list(load_routes(args.profile, kinds).values())
    entries.sort(key=lambda item: (-item.logical_total, item.tensor, item.expert_idx))
    selected_entries = entries[: args.max_entries]
    selected_keys = {(entry.tensor, entry.expert_idx) for entry in selected_entries}

    args.out_pack.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)

    selected_manifest = write_pack(args.out_pack, selected_entries, args.packed_type, args.packed_ratio)
    report = {
        "input_profiles": [str(path) for path in args.profile],
        "out_pack": str(args.out_pack),
        "packed_type": args.packed_type,
        "packed_ratio": args.packed_ratio,
        "max_entries": args.max_entries,
        "kinds": sorted(kinds),
        "candidate_entries": len(entries),
        "selected_entries": len(selected_entries),
        "summary": summarize(entries, selected_keys, args.packed_ratio),
        "selected": selected_manifest,
    }
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.out_md, report)

    total = report["summary"]["total"]
    print(f"candidate_entries={len(entries)}")
    print(f"selected_entries={len(selected_entries)}")
    print(f"events={total['events']}")
    print(f"event_coverage={total['event_coverage']:.6f}")
    print(f"byte_coverage={total['byte_coverage']:.6f}")
    print(f"hybrid_byte_ratio={total['hybrid_byte_ratio']:.6f}")
    print(f"pack={args.out_pack}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
