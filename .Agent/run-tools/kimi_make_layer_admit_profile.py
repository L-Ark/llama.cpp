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
    args = ap.parse_args()

    counts: Counter[tuple[str, int]] = Counter()
    for path in sorted(args.route_root.glob("*/route-trace.csv")):
        with path.open(newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                tensor = row["tensor"]
                layer = parse_layer(tensor)
                if layer is None or layer < args.min_layer or layer > args.max_layer:
                    continue
                counts[(tensor, int(row["expert_idx"]))] += 1
    if not counts:
        raise RuntimeError(f"no route entries found for layers {args.min_layer}..{args.max_layer}")

    rows = sorted(
        ((tensor, expert, count) for (tensor, expert), count in counts.items()),
        key=lambda r: (-r[2], r[0], r[1]),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        for tensor, expert, count in rows:
            f.write(f"{tensor}\t{expert}\t{count}\n")
    summary = {
        "kind": "kimi_layer_admit_profile",
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
