#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def parse_metrics(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        out[key] = value
        if re.fullmatch(r"-?\d+", value):
            out[key + "__int"] = int(value)
        else:
            try:
                out[key + "__float"] = float(value)
            except ValueError:
                pass
    return out


def extract_int(text: str, name: str) -> int:
    m = re.search(rf"\b{re.escape(name)}=(\d+)", text)
    return int(m.group(1)) if m else 0


def extract_float_ms(text: str, name: str) -> float:
    m = re.search(rf"\b{re.escape(name)}=([0-9]+(?:\.[0-9]+)?)\s*ms", text)
    if m:
        return float(m.group(1))
    m = re.search(rf"\b{re.escape(name)}=([0-9]+(?:\.[0-9]+)?)", text)
    return float(m.group(1)) if m else 0.0


def metric_float(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
    if key + "__float" in metrics:
        return float(metrics[key + "__float"])
    if key + "__int" in metrics:
        return float(metrics[key + "__int"])
    return default


def metric_int(metrics: dict[str, Any], key: str, default: int = 0) -> int:
    if key + "__int" in metrics:
        return int(metrics[key + "__int"])
    if key + "__float" in metrics:
        return int(metrics[key + "__float"])
    return default


@dataclass
class Bucket:
    calls: int = 0
    active: int = 0
    cache_hits: int = 0
    accepted: int = 0
    full_cover_calls: int = 0
    not_allowed: int = 0
    reject_no_manifest: int = 0
    reject_no_entry: int = 0
    reject_unsupported: int = 0
    reject_manifest_mismatch: int = 0
    reject_shape_mismatch: int = 0
    reject_not_smaller: int = 0
    logical_bytes: int = 0
    accepted_bytes: int = 0
    partial_saved_bytes: int = 0
    full_saved_bytes: int = 0
    logical_miss_bytes: float = 0.0
    partial_miss_saved_bytes: float = 0.0
    full_miss_saved_bytes: float = 0.0

    def add_row(self, row: dict[str, str]) -> None:
        n_active = int(row.get("n_active") or 0)
        cache_hits = int(row.get("cache_hits") or 0)
        accepted = int(row.get("accepted") or 0)
        full_cover = int(row.get("full_cover") or 0)
        logical = int(row.get("logical_total_bytes") or 0)
        accepted_bytes = int(row.get("accepted_bytes") or 0)
        saved = int(row.get("accepted_saved_bytes") or 0)
        miss = max(0, n_active - cache_hits)
        miss_frac = (miss / n_active) if n_active > 0 else 0.0

        self.calls += 1
        self.active += n_active
        self.cache_hits += cache_hits
        self.accepted += accepted
        self.full_cover_calls += full_cover
        self.not_allowed += int(row.get("reject_not_allowed") or 0)
        self.reject_no_manifest += int(row.get("reject_no_manifest") or 0)
        self.reject_no_entry += int(row.get("reject_no_entry") or 0)
        self.reject_unsupported += int(row.get("reject_unsupported") or 0)
        self.reject_manifest_mismatch += int(row.get("reject_manifest_mismatch") or 0)
        self.reject_shape_mismatch += int(row.get("reject_shape_mismatch") or 0)
        self.reject_not_smaller += int(row.get("reject_not_smaller") or 0)
        self.logical_bytes += logical
        self.accepted_bytes += accepted_bytes
        self.partial_saved_bytes += saved
        if full_cover:
            self.full_saved_bytes += saved
        self.logical_miss_bytes += logical * miss_frac
        self.partial_miss_saved_bytes += saved * miss_frac
        if full_cover:
            self.full_miss_saved_bytes += saved * miss_frac

    def as_dict(self) -> dict[str, Any]:
        active = self.active or 1
        calls = self.calls or 1
        logical = self.logical_bytes or 1
        logical_miss = self.logical_miss_bytes or 1.0
        return {
            "calls": self.calls,
            "active": self.active,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / active,
            "accepted": self.accepted,
            "accepted_rate": self.accepted / active,
            "full_cover_calls": self.full_cover_calls,
            "full_cover_call_rate": self.full_cover_calls / calls,
            "reject_not_allowed": self.not_allowed,
            "reject_no_manifest": self.reject_no_manifest,
            "reject_no_entry": self.reject_no_entry,
            "reject_unsupported": self.reject_unsupported,
            "reject_manifest_mismatch": self.reject_manifest_mismatch,
            "reject_shape_mismatch": self.reject_shape_mismatch,
            "reject_not_smaller": self.reject_not_smaller,
            "logical_bytes": self.logical_bytes,
            "accepted_bytes": self.accepted_bytes,
            "partial_saved_bytes": self.partial_saved_bytes,
            "partial_saved_ratio_all": self.partial_saved_bytes / logical,
            "full_saved_bytes": self.full_saved_bytes,
            "full_saved_ratio_all": self.full_saved_bytes / logical,
            "logical_miss_bytes": self.logical_miss_bytes,
            "partial_miss_saved_bytes": self.partial_miss_saved_bytes,
            "partial_miss_saved_ratio": self.partial_miss_saved_bytes / logical_miss,
            "full_miss_saved_bytes": self.full_miss_saved_bytes,
            "full_miss_saved_ratio": self.full_miss_saved_bytes / logical_miss,
        }


def collect_profile(preflight_csv: Path) -> tuple[Bucket, dict[str, Bucket]]:
    total = Bucket()
    by_phase_role: dict[str, Bucket] = defaultdict(Bucket)
    with preflight_csv.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            key = f"{row.get('phase', '')}:{row.get('role', '')}"
            total.add_row(row)
            by_phase_role[key].add_row(row)
    return total, by_phase_role


def measured_transfer(metrics: dict[str, Any]) -> dict[str, Any]:
    expert_text = str(metrics.get("expert_pack_0", ""))
    iouring_bytes = extract_int(expert_text, "iouring_bytes")
    iouring_wait_ms = extract_int(expert_text, "iouring_wait_us") / 1000.0
    direct_reads = extract_int(expert_text, "direct_reads")

    h2d_ms = 0.0
    slot_wait_ms = 0.0
    staging_copies = 0
    for key, value in metrics.items():
        if not key.startswith("pinned_staging_") or not isinstance(value, str):
            continue
        if "copies=" not in value:
            continue
        staging_copies += extract_int(value, "copies")
        h2d_ms += extract_float_ms(value, "h2d")
        slot_wait_ms += extract_float_ms(value, "slot_wait")

    return {
        "iouring_bytes": iouring_bytes,
        "iouring_wait_ms": iouring_wait_ms,
        "direct_reads": direct_reads,
        "pinned_h2d_ms": h2d_ms,
        "pinned_slot_wait_ms": slot_wait_ms,
        "staging_copies": staging_copies,
    }


def ceiling(current_decode_ms: float, decode_runs: int, pool_ms: float, miss_saved_ratio: float) -> dict[str, float]:
    saved_ms = max(0.0, pool_ms * max(0.0, min(1.0, miss_saved_ratio)))
    new_decode_ms = max(1e-6, current_decode_ms - saved_ms)
    return {
        "pool_ms": pool_ms,
        "saved_ms": saved_ms,
        "saved_ms_per_token": saved_ms / max(1, decode_runs),
        "ceiling_decode_ms": new_decode_ms,
        "ceiling_ms_per_token": new_decode_ms / max(1, decode_runs),
        "ceiling_tok_s": decode_runs / (new_decode_ms / 1000.0),
    }


def analyze_run(run: Path) -> dict[str, Any]:
    metrics_path = run / "metrics.txt"
    preflight_path = run / "v2-override-preflight.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    if not preflight_path.exists():
        raise FileNotFoundError(preflight_path)

    metrics = parse_metrics(metrics_path)
    total, by_phase_role = collect_profile(preflight_path)
    total_d = total.as_dict()
    transfer = measured_transfer(metrics)

    decode_ms = metric_float(metrics, "decode_ms")
    decode_runs = metric_int(metrics, "decode_runs")
    token_rate = metric_float(metrics, "token_rate")
    ttft_ms = metric_float(metrics, "ttft_ms")
    memory_peak = metric_int(metrics, "memory.peak")

    full_ratio = float(total_d["full_miss_saved_ratio"])
    partial_ratio = float(total_d["partial_miss_saved_ratio"])
    io_pool = float(transfer["iouring_wait_ms"])
    io_h2d_pool = io_pool + float(transfer["pinned_h2d_ms"])
    io_h2d_slot_pool = io_h2d_pool + float(transfer["pinned_slot_wait_ms"])

    return {
        "run": str(run),
        "prompt_id": metrics.get("prompt_id", ""),
        "prompt": metrics.get("prompt", ""),
        "quality": metrics.get("quality", ""),
        "quality_reason": metrics.get("quality_reason", ""),
        "output": metrics.get("output", ""),
        "token_rate": token_rate,
        "ttft_ms": ttft_ms,
        "decode_ms": decode_ms,
        "decode_runs": decode_runs,
        "ms_per_token": decode_ms / max(1, decode_runs),
        "memory_peak": memory_peak,
        "transfer": transfer,
        "total": total_d,
        "by_phase_role": {k: v.as_dict() for k, v in sorted(by_phase_role.items())},
        "ceilings": {
            "full_io_wait_only": ceiling(decode_ms, decode_runs, io_pool, full_ratio),
            "partial_io_wait_only": ceiling(decode_ms, decode_runs, io_pool, partial_ratio),
            "full_io_h2d": ceiling(decode_ms, decode_runs, io_h2d_pool, full_ratio),
            "partial_io_h2d": ceiling(decode_ms, decode_runs, io_h2d_pool, partial_ratio),
            "full_io_h2d_slot": ceiling(decode_ms, decode_runs, io_h2d_slot_pool, full_ratio),
            "partial_io_h2d_slot": ceiling(decode_ms, decode_runs, io_h2d_slot_pool, partial_ratio),
        },
    }


def gib(value: float) -> float:
    return value / (1024.0 ** 3)


def pct(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def write_md(path: Path, results: list[dict[str, Any]]) -> None:
    lines: list[str] = [
        "# Kimi v2 partial split bound",
        "",
        "This is an offline bound from `v2-override-preflight.csv` and `metrics.txt`.",
        "It does not change runtime behavior and does not claim SOTA.",
        "",
        "## Summary",
        "",
        "| run | quality | tok/s | ms/tok | accepted | full-cover calls | partial saved | full saved | partial miss ratio | full miss ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        t = r["total"]
        run_name = Path(r["run"]).name
        lines.append(
            f"| `{run_name}` | {r['quality']} | {r['token_rate']:.2f} | {r['ms_per_token']:.2f} | "
            f"{pct(t['accepted_rate'])} | {pct(t['full_cover_call_rate'])} | "
            f"{gib(t['partial_saved_bytes']):.3f} GiB | {gib(t['full_saved_bytes']):.3f} GiB | "
            f"{pct(t['partial_miss_saved_ratio'])} | {pct(t['full_miss_saved_ratio'])} |"
        )

    lines += [
        "",
        "## Token-Rate Ceilings",
        "",
        "Ceilings assume saved bytes reduce the named measured pool linearly. They exclude new split-dispatch overhead, extra kernels, quality risk, and payload materialization cost.",
        "",
        "| run | mode | pool ms | saved ms | saved ms/tok | ceiling tok/s |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        run_name = Path(r["run"]).name
        for mode in [
            "full_io_wait_only",
            "partial_io_wait_only",
            "full_io_h2d",
            "partial_io_h2d",
            "full_io_h2d_slot",
            "partial_io_h2d_slot",
        ]:
            c = r["ceilings"][mode]
            lines.append(
                f"| `{run_name}` | `{mode}` | {c['pool_ms']:.2f} | {c['saved_ms']:.2f} | "
                f"{c['saved_ms_per_token']:.2f} | {c['ceiling_tok_s']:.2f} |"
            )

    lines += [
        "",
        "## By Phase/Role",
        "",
        "| run | phase:role | accepted | full-cover | partial saved GiB | full saved GiB | partial miss ratio | full miss ratio | not allowlisted |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        run_name = Path(r["run"]).name
        for key, b in r["by_phase_role"].items():
            lines.append(
                f"| `{run_name}` | `{key}` | {pct(b['accepted_rate'])} | "
                f"{pct(b['full_cover_call_rate'])} | {gib(b['partial_saved_bytes']):.3f} | "
                f"{gib(b['full_saved_bytes']):.3f} | {pct(b['partial_miss_saved_ratio'])} | "
                f"{pct(b['full_miss_saved_ratio'])} | {b['reject_not_allowed']} |"
            )

    lines += [
        "",
        "## Interpretation Rules",
        "",
        "- `full-*` corresponds to a simpler dispatch that only uses v2 when every active expert in the call is covered.",
        "- `partial-*` corresponds to a split dispatch that can use v2 for covered experts and current weights for uncovered experts in the same call.",
        "- `miss ratio` is weighted by current VRAM miss fraction per call; it is a conservative transfer-only estimate.",
        "- If held-out `partial_io_h2d` cannot plausibly exceed the target after overhead, do not implement split dispatch.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Bound Kimi v2 partial-covered split dispatch from preflight CSVs.")
    parser.add_argument("--run", type=Path, action="append", required=True, help="Run directory containing metrics.txt and v2-override-preflight.csv")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    results = [analyze_run(run) for run in args.run]
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({"runs": results}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_md(args.out_md, results)
    print(args.out_md)
    for r in results:
        t = r["total"]
        c = r["ceilings"]["partial_io_h2d"]
        print(
            f"{Path(r['run']).name}: accepted={t['accepted_rate']:.4f} "
            f"full={t['full_cover_call_rate']:.4f} partial_miss_saved={t['partial_miss_saved_ratio']:.4f} "
            f"partial_io_h2d_ceiling={c['ceiling_tok_s']:.3f} tok/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
