#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


KV_RE = re.compile(r"([A-Za-z0-9_]+)=([0-9.]+%?)")


def parse_kv(text: object) -> dict[str, float | str]:
    out: dict[str, float | str] = {}
    for key, value in KV_RE.findall(str(text or "")):
        if value.endswith("%"):
            out[key] = float(value[:-1]) / 100.0
        elif "." in value:
            out[key] = float(value)
        else:
            out[key] = int(value)
    return out


def gib(nbytes: float) -> float:
    return nbytes / 1024**3


def load_one(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    expert = parse_kv(data.get("expert_pack_0", ""))
    iouring = parse_kv(data.get("expert_pack_iouring_0", ""))
    upgate = parse_kv(data.get("vram_upgate_0", ""))
    down = parse_kv(data.get("vram_down_0", ""))
    overlap = parse_kv(data.get("current_down_overlap_0", ""))

    decode_s = float(data.get("decode_ms", 0.0)) / 1000.0
    iouring_bytes = float(expert.get("iouring_bytes", 0))
    iouring_reads = float(expert.get("iouring_reads", 0))
    direct_reads = float(expert.get("direct_reads", 0))
    total_reads = iouring_reads + direct_reads
    wait_us = float(expert.get("iouring_wait_us", 0))
    submit_us = float(expert.get("iouring_submit_us", 0))

    return {
        "prompt_id": data.get("prompt_id", path.parent.name),
        "token_rate": float(data.get("token_rate", 0.0)),
        "decode_runs": int(data.get("decode_runs", 0)),
        "decode_s": decode_s,
        "ttft_ms": float(data.get("ttft_ms", 0.0)),
        "memory_peak_gib": gib(float(data.get("memory.peak", 0))),
        "quality": data.get("quality", ""),
        "iouring_gib": gib(iouring_bytes),
        "iouring_gib_s": gib(iouring_bytes) / decode_s if decode_s > 0 else 0.0,
        "iouring_reads": int(iouring_reads),
        "direct_reads": int(direct_reads),
        "iouring_read_ratio": iouring_reads / total_reads if total_reads else 0.0,
        "direct_read_ratio": direct_reads / total_reads if total_reads else 0.0,
        "iouring_wait_s": wait_us / 1_000_000.0,
        "iouring_wait_per_read_ms": wait_us / 1000.0 / iouring_reads if iouring_reads else 0.0,
        "iouring_submit_per_read_us": submit_us / iouring_reads if iouring_reads else 0.0,
        "iouring_inflight_avg": float(iouring.get("inflight_avg", 0.0)),
        "iouring_inflight_max": int(iouring.get("inflight_max", 0)),
        "iouring_batches": int(iouring.get("batches", 0)),
        "iouring_batch_1": int(iouring.get("1", 0)) if "1" in iouring else 0,
        "upgate_hit_rate": float(upgate.get("hit_rate", 0.0)),
        "down_hit_rate": float(down.get("hit_rate", 0.0)),
        "overlap_planned_jobs": int(overlap.get("planned_jobs", 0)),
        "overlap_completed_jobs": int(overlap.get("completed_jobs", 0)),
        "overlap_missing_pack": int(overlap.get("missing_pack", 0)),
        "overlap_max_jobs": int(overlap.get("max_jobs", 0)),
        "metrics_path": str(path),
    }


def summarize(rows: list[dict]) -> dict:
    decode_s = sum(row["decode_s"] for row in rows)
    decode_runs = sum(row["decode_runs"] for row in rows)
    iouring_gib = sum(row["iouring_gib"] for row in rows)
    iouring_reads = sum(row["iouring_reads"] for row in rows)
    direct_reads = sum(row["direct_reads"] for row in rows)
    total_reads = iouring_reads + direct_reads
    wait_s = sum(row["iouring_wait_s"] for row in rows)
    weighted_inflight_num = sum(row["iouring_inflight_avg"] * row["iouring_reads"] for row in rows)
    weighted_upgate_hit = sum(row["upgate_hit_rate"] * row["decode_runs"] for row in rows)
    weighted_down_hit = sum(row["down_hit_rate"] * row["decode_runs"] for row in rows)
    return {
        "prompts": len(rows),
        "decode_runs": decode_runs,
        "decode_s": decode_s,
        "mean_token_rate_weighted": decode_runs / decode_s if decode_s > 0 else 0.0,
        "iouring_gib": iouring_gib,
        "iouring_gib_s": iouring_gib / decode_s if decode_s > 0 else 0.0,
        "iouring_reads": iouring_reads,
        "direct_reads": direct_reads,
        "iouring_read_ratio": iouring_reads / total_reads if total_reads else 0.0,
        "direct_read_ratio": direct_reads / total_reads if total_reads else 0.0,
        "iouring_wait_s": wait_s,
        "iouring_wait_decode_fraction": wait_s / decode_s if decode_s > 0 else 0.0,
        "iouring_wait_per_read_ms": wait_s * 1000.0 / iouring_reads if iouring_reads else 0.0,
        "iouring_inflight_avg_weighted": weighted_inflight_num / iouring_reads if iouring_reads else 0.0,
        "upgate_hit_rate_weighted": weighted_upgate_hit / decode_runs if decode_runs else 0.0,
        "down_hit_rate_weighted": weighted_down_hit / decode_runs if decode_runs else 0.0,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "prompt_id",
        "token_rate",
        "decode_runs",
        "decode_s",
        "iouring_gib",
        "iouring_gib_s",
        "iouring_reads",
        "direct_reads",
        "iouring_read_ratio",
        "direct_read_ratio",
        "iouring_wait_s",
        "iouring_wait_per_read_ms",
        "iouring_inflight_avg",
        "iouring_inflight_max",
        "upgate_hit_rate",
        "down_hit_rate",
        "overlap_planned_jobs",
        "overlap_completed_jobs",
        "memory_peak_gib",
        "quality",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})


def write_md(path: Path, rows: list[dict], total: dict, peak_gib_s: float) -> None:
    lines = [
        "# Kimi IO queue metrics summary",
        "",
        f"- Prompts: `{total['prompts']}`",
        f"- Weighted token rate: `{total['mean_token_rate_weighted']:.3f} tok/s`",
        f"- Aggregate iouring throughput: `{total['iouring_gib_s']:.3f} GiB/s`",
        f"- Pure IO peak reference: `{peak_gib_s:.3f} GiB/s`",
        f"- Peak utilization: `{(total['iouring_gib_s'] / peak_gib_s if peak_gib_s else 0.0):.3f}`",
        f"- Direct read ratio: `{total['direct_read_ratio']:.3f}`",
        f"- Weighted iouring inflight avg: `{total['iouring_inflight_avg_weighted']:.3f}`",
        f"- Iouring wait/decode fraction: `{total['iouring_wait_decode_fraction']:.3f}`",
        "",
        "| prompt | tok/s | iouring GiB/s | iouring read ratio | direct reads | inflight avg | upgate hit | down hit |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['prompt_id']} | {row['token_rate']:.3f} | {row['iouring_gib_s']:.3f} | "
            f"{row['iouring_read_ratio']:.3f} | {row['direct_reads']} | "
            f"{row['iouring_inflight_avg']:.3f} | {row['upgate_hit_rate']:.3f} | {row['down_hit_rate']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Kimi IO queue/backend metrics from metrics.json files.")
    parser.add_argument("--metrics", type=Path, action="append", required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--peak-gib-s", type=float, default=10.3)
    args = parser.parse_args()

    rows = [load_one(path) for path in args.metrics]
    rows.sort(key=lambda row: row["prompt_id"])
    total = summarize(rows)
    report = {
        "metrics": [str(path) for path in args.metrics],
        "peak_gib_s": args.peak_gib_s,
        "summary": total,
        "rows": rows,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.out_csv, rows)
    write_md(args.out_md, rows, total, args.peak_gib_s)

    print(f"prompts={total['prompts']}")
    print(f"weighted_token_rate={total['mean_token_rate_weighted']:.6f}")
    print(f"iouring_gib_s={total['iouring_gib_s']:.6f}")
    print(f"peak_utilization={total['iouring_gib_s'] / args.peak_gib_s:.6f}")
    print(f"iouring_read_ratio={total['iouring_read_ratio']:.6f}")
    print(f"direct_read_ratio={total['direct_read_ratio']:.6f}")
    print(f"iouring_inflight_avg={total['iouring_inflight_avg_weighted']:.6f}")
    print(f"iouring_wait_decode_fraction={total['iouring_wait_decode_fraction']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
