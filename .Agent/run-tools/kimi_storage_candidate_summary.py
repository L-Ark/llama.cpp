#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


GIB = 1024 ** 3
MIB = 1024 ** 2


def f(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, 0) or 0)
    except ValueError:
        return 0.0


def i(row: dict[str, str], key: str) -> int:
    try:
        return int(float(row.get(key, 0) or 0))
    except ValueError:
        return 0


def layer_of(tensor: str) -> int:
    parts = tensor.split(".")
    if len(parts) > 1 and parts[0] == "blk":
        try:
            return int(parts[1])
        except ValueError:
            return -1
    return -1


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as csv_file:
        return list(csv.DictReader(csv_file))


def read_metrics(run: Path) -> dict[str, Any]:
    path = run / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def summarize_upgate(run: Path) -> list[dict[str, Any]]:
    rows = read_csv(run / "up-gate-profile.csv")
    agg: dict[tuple[int, str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        if row.get("mode") != "decode":
            continue
        layer = layer_of(row.get("up_tensor", ""))
        key = (layer, row.get("up_type", ""), row.get("gate_type", ""))
        values = agg[key]
        values["calls"] += 1
        for field in (
            "wall_ms",
            "stage_ms",
            "up_wait_ms",
            "gate_wait_ms",
            "up_compute_ms",
            "gate_compute_ms",
            "up_cache_hits",
            "up_cache_misses",
            "gate_cache_hits",
            "gate_cache_misses",
            "up_stage_jobs",
            "gate_stage_jobs",
        ):
            values[field] += f(row, field)
    out = []
    for (layer, up_type, gate_type), values in agg.items():
        hits = values["up_cache_hits"] + values["gate_cache_hits"]
        misses = values["up_cache_misses"] + values["gate_cache_misses"]
        out.append(
            {
                "layer": layer,
                "role": "upgate",
                "type_pair": f"{up_type}/{gate_type}",
                "calls": int(values["calls"]),
                "wall_ms": values["wall_ms"],
                "stage_ms": values["stage_ms"],
                "wait_ms": values["up_wait_ms"] + values["gate_wait_ms"],
                "up_wait_ms": values["up_wait_ms"],
                "gate_wait_ms": values["gate_wait_ms"],
                "compute_ms": values["up_compute_ms"] + values["gate_compute_ms"],
                "misses": int(misses),
                "hits": int(hits),
                "hit_rate": hits / max(hits + misses, 1),
                "stage_jobs": int(values["up_stage_jobs"] + values["gate_stage_jobs"]),
            }
        )
    out.sort(key=lambda row: row["wall_ms"], reverse=True)
    return out


def summarize_down(run: Path) -> list[dict[str, Any]]:
    rows = read_csv(run / "down-batch-profile.csv")
    agg: dict[tuple[int, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        if i(row, "n_active") > 8:
            continue
        layer = layer_of(row.get("tensor", ""))
        key = (layer, row.get("src0_type", ""))
        values = agg[key]
        values["calls"] += 1
        for field in ("wall_ms", "stage_ms", "kernel_ms", "cache_hits", "cache_misses", "staged_jobs"):
            values[field] += f(row, field)
    out = []
    for (layer, src0_type), values in agg.items():
        hits = values["cache_hits"]
        misses = values["cache_misses"]
        out.append(
            {
                "layer": layer,
                "role": "down",
                "type_pair": src0_type,
                "calls": int(values["calls"]),
                "wall_ms": values["wall_ms"],
                "stage_ms": values["stage_ms"],
                "wait_ms": values["stage_ms"],
                "kernel_ms": values["kernel_ms"],
                "misses": int(misses),
                "hits": int(hits),
                "hit_rate": hits / max(hits + misses, 1),
                "stage_jobs": int(values["staged_jobs"]),
            }
        )
    out.sort(key=lambda row: row["wall_ms"], reverse=True)
    return out


def summarize_copy(run: Path) -> list[dict[str, Any]]:
    rows = read_csv(run / "copy-profile.csv")
    agg: dict[tuple[str, int, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        tensor = row.get("tensor", "")
        if ".ffn_up_exps." in tensor:
            role = "up"
        elif ".ffn_gate_exps." in tensor:
            role = "gate"
        elif ".ffn_down_exps." in tensor:
            role = "down"
        else:
            role = "other"
        layer = layer_of(tensor)
        key = (row.get("op", ""), layer, role)
        values = agg[key]
        values["calls"] += 1
        for field in ("bytes", "io_wait_ms", "h2d_ms", "wall_ms", "enqueue_ms", "slot_wait_ms"):
            values[field] += f(row, field)
    out = []
    for (op, layer, role), values in agg.items():
        out.append(
            {
                "op": op,
                "layer": layer,
                "role": role,
                "calls": int(values["calls"]),
                "gib": values["bytes"] / GIB,
                "io_wait_ms": values["io_wait_ms"],
                "h2d_ms": values["h2d_ms"],
                "wall_ms": values["wall_ms"],
                "enqueue_ms": values["enqueue_ms"],
                "slot_wait_ms": values["slot_wait_ms"],
            }
        )
    out.sort(key=lambda row: row["wall_ms"], reverse=True)
    return out


def read_residency(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as csv_file:
        rows = list(csv.DictReader(csv_file))
    rows.sort(key=lambda row: int(row.get("cached_bytes", 0) or 0), reverse=True)
    return rows


def table(lines: list[str], headers: list[str], rows: list[list[str]]) -> None:
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")


def write_report(run: Path, out: Path, residency: Path | None) -> None:
    metrics = read_metrics(run)
    upgate = summarize_upgate(run)
    down = summarize_down(run)
    copy = summarize_copy(run)
    cache_rows = read_residency(residency)

    lines: list[str] = [
        "# Kimi Storage Candidate Summary",
        "",
        f"- run: `{run}`",
        f"- prompt: `{metrics.get('prompt_id', run.name)}`",
        f"- quality: `{metrics.get('quality', '')}`",
        f"- TTFT: `{metrics.get('ttft_ms', '')} ms`",
        f"- decode: `{metrics.get('decode_ms', '')} ms / {metrics.get('decode_runs', '')}`",
        f"- token rate: `{metrics.get('token_rate', '')} tok/s`",
        f"- memory peak: `{metrics.get('memory.peak', '')}`",
        "",
        "## Top Decode Up/Gate Layer Buckets",
        "",
    ]
    table(
        lines,
        ["layer", "types", "calls", "wall ms", "wait ms", "compute ms", "misses", "hit rate", "stage jobs"],
        [
            [
                str(row["layer"]),
                row["type_pair"],
                str(row["calls"]),
                f"{row['wall_ms']:.1f}",
                f"{row['wait_ms']:.1f}",
                f"{row['compute_ms']:.1f}",
                str(row["misses"]),
                f"{100.0 * row['hit_rate']:.1f}%",
                str(row["stage_jobs"]),
            ]
            for row in upgate[:20]
        ],
    )
    lines.extend(["", "## Top Decode Down Layer Buckets", ""])
    table(
        lines,
        ["layer", "type", "calls", "wall ms", "stage ms", "kernel ms", "misses", "hit rate", "stage jobs"],
        [
            [
                str(row["layer"]),
                row["type_pair"],
                str(row["calls"]),
                f"{row['wall_ms']:.1f}",
                f"{row['stage_ms']:.1f}",
                f"{row['kernel_ms']:.1f}",
                str(row["misses"]),
                f"{100.0 * row['hit_rate']:.1f}%",
                str(row["stage_jobs"]),
            ]
            for row in down[:20]
        ],
    )
    lines.extend(["", "## Top Copy Wall Layer/Role Buckets", ""])
    table(
        lines,
        ["op", "layer", "role", "calls", "GiB", "io wait ms", "H2D ms", "wall ms"],
        [
            [
                row["op"],
                str(row["layer"]),
                row["role"],
                str(row["calls"]),
                f"{row['gib']:.2f}",
                f"{row['io_wait_ms']:.1f}",
                f"{row['h2d_ms']:.1f}",
                f"{row['wall_ms']:.1f}",
            ]
            for row in copy[:30]
        ],
    )
    if cache_rows:
        lines.extend(["", "## Page Cache Residency After Run", ""])
        by_label: dict[str, int] = defaultdict(int)
        for row in cache_rows:
            by_label[row["label"]] += int(row["cached_bytes"])
        for label, cached in sorted(by_label.items(), key=lambda item: -item[1]):
            lines.append(f"- {label}: `{cached / GIB:.3f} GiB`")
        lines.append("")
        table(
            lines,
            ["cached GiB", "cached %", "label", "file"],
            [
                [
                    f"{int(row['cached_bytes']) / GIB:.3f}",
                    row["cached_pct"],
                    row["label"],
                    Path(row["path"]).name,
                ]
                for row in cache_rows[:16]
            ],
        )

    lines.extend(
        [
            "",
            "## Candidate Direction",
            "",
            "- Up/gate remains the first RAM/VRAM storage target when its wait bucket is larger than down and compute is small.",
            "- Prefer candidates whose layer/role appears in both top decode wait and top copy wall lists.",
            "- The page-cache residency table identifies reclaimable file-cache bytes, but a RAM slab is acceptable only if paired A/B shows lower endpoint decode time without TTFT/refault regressions.",
            "",
        ]
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize Kimi storage candidates from profile CSVs.")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--file-cache-residency", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    write_report(args.run_dir, args.out, args.file_cache_residency)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
