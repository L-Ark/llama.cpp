#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import statistics
from pathlib import Path


LAYER_RE = re.compile(r"blk\.(\d+)\.ffn_(up|down|gate)_exps")


def bucket_layer(layer: int | None) -> str:
    if layer is None:
        return "other"
    if layer <= 2:
        return "0-2"
    if layer <= 9:
        return "3-9"
    if layer <= 19:
        return "10-19"
    if layer <= 29:
        return "20-29"
    return "30-39"


def add(bucket: dict, key, ms: float) -> None:
    item = bucket.setdefault(key, {"rows": 0, "ms": 0.0})
    item["rows"] += 1
    item["ms"] += ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path)
    ap.add_argument("-o", "--output", type=Path)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    by_kind: dict[str, dict[str, float | int]] = {}
    by_band: dict[str, dict[str, float | int]] = {}
    by_tensor: dict[str, dict[str, float | int]] = {}
    by_thread: dict[int, dict[str, float | int]] = {}
    rows = 0
    sum_chunk_ms = 0.0

    with args.trace.open(newline="") as f:
        for row in csv.DictReader(f):
            rows += 1
            ms = float(row["ms"])
            sum_chunk_ms += ms
            match = LAYER_RE.search(row["tensor"])
            layer = int(match.group(1)) if match else None
            kind = match.group(2) if match else "other"
            add(by_kind, kind, ms)
            add(by_band, bucket_layer(layer), ms)
            add(by_tensor, row["tensor"], ms)
            add(by_thread, int(row["ith"]), ms)

    thread_sums = [float(v["ms"]) for v in by_thread.values()]
    out = {
        "trace": str(args.trace),
        "trace_rows": rows,
        "sum_chunk_ms": sum_chunk_ms,
        "max_thread_ms": max(thread_sums) if thread_sums else 0.0,
        "median_thread_ms": statistics.median(thread_sums) if thread_sums else 0.0,
        "by_kind": dict(sorted(by_kind.items(), key=lambda kv: -float(kv[1]["ms"]))),
        "by_band": dict(sorted(by_band.items())),
        "top_tensors": [
            {"tensor": key, **value}
            for key, value in sorted(by_tensor.items(), key=lambda kv: -float(kv[1]["ms"]))[: args.top]
        ],
        "top_threads": [
            {"thread": key, **value}
            for key, value in sorted(by_thread.items(), key=lambda kv: -float(kv[1]["ms"]))[: args.top]
        ],
    }

    text = json.dumps(out, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
