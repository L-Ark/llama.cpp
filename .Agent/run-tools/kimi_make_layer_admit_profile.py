#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


LAYER_RE = re.compile(r"^blk\.(\d+)\.ffn_(up|gate|down)_exps\.weight$")


def parse_layer(tensor: str) -> int | None:
    m = LAYER_RE.match(tensor)
    return int(m.group(1)) if m else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a Kimi cache admission TSV for selected MoE layers.")
    ap.add_argument("--route-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-layer", type=int, required=True)
    ap.add_argument("--min-layer", type=int, default=0)
    ap.add_argument("--format", choices=("admit_tsv", "profile_csv"), default="admit_tsv")
    args = ap.parse_args()

    counts: Counter[tuple[str, int]] = Counter()
    expert_bytes_by_key: dict[tuple[str, int], int] = {}
    for path in sorted(args.route_root.glob("*/route-trace.csv")):
        with path.open(newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                tensor = row["tensor"]
                layer = parse_layer(tensor)
                if layer is None or layer < args.min_layer or layer > args.max_layer:
                    continue
                key = (tensor, int(row["expert_idx"]))
                counts[key] += 1
                expert_bytes_by_key[key] = int(row["expert_bytes"])
    if not counts:
        raise RuntimeError(f"no route entries found for layers {args.min_layer}..{args.max_layer}")

    rows = sorted(
        ((tensor, expert, count) for (tensor, expert), count in counts.items()),
        key=lambda r: (-r[2], r[0], r[1]),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "admit_tsv":
        with args.out.open("w", encoding="utf-8", newline="") as f:
            for tensor, expert, count in rows:
                f.write(f"{tensor}\t{expert}\t{count}\n")
    else:
        cumulative = 0
        with args.out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["rank", "count", "expert_bytes", "cumulative_bytes", "tensor_base", "expert_idx", "tensor"],
            )
            writer.writeheader()
            for rank, (tensor, expert, count) in enumerate(rows, start=1):
                expert_bytes = expert_bytes_by_key[(tensor, expert)]
                cumulative += expert_bytes
                writer.writerow({
                    "rank": rank,
                    "count": count,
                    "expert_bytes": expert_bytes,
                    "cumulative_bytes": cumulative,
                    "tensor_base": "0x0",
                    "expert_idx": expert,
                    "tensor": tensor,
                })
    summary = {
        "kind": "kimi_layer_admit_profile",
        "format": args.format,
        "route_root": str(args.route_root),
        "min_layer": args.min_layer,
        "max_layer": args.max_layer,
        "entries": len(rows),
        "total_route_hits_in_source": sum(counts.values()),
        "out": str(args.out),
    }
    args.out.with_suffix(args.out.suffix + ".json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
