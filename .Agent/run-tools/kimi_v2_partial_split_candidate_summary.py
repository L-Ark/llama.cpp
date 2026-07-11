#!/usr/bin/env python3
"""Summarize Kimi v2 partial-split planner output.

The planner records one row per real runtime up/gate/down call. This tool turns
that raw CSV into dispatch-candidate groups: high coverage, low segmentation,
and enough miss-weighted byte saving to justify a guarded implementation.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


def to_int(value: str | None, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(value)


def to_float(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def gib(value: int | float) -> float:
    return float(value) / float(1 << 30)


@dataclass
class Metrics:
    token_rate: float | None = None
    decode_ms: float | None = None
    decode_runs: int | None = None
    ttft_ms: float | None = None
    ram_peak: int | None = None

    @property
    def decode_ms_per_token(self) -> float | None:
        if self.decode_ms is None or not self.decode_runs:
            return None
        return self.decode_ms / self.decode_runs


@dataclass
class Agg:
    name: tuple[str, ...]
    calls: int = 0
    active: int = 0
    covered: int = 0
    cache_hits: int = 0
    covered_cache_hits: int = 0
    full_calls: int = 0
    no_cover_calls: int = 0
    partial_calls: int = 0
    covered_runs: int = 0
    uncovered_runs: int = 0
    transitions: int = 0
    extra_groups: int = 0
    logical_total_bytes: int = 0
    logical_miss_bytes: int = 0
    covered_saved_bytes: int = 0
    covered_miss_saved_bytes: int = 0
    reject_not_allowed: int = 0
    reject_other: int = 0

    def add(self, row: dict[str, str]) -> None:
        active = to_int(row.get("n_active"))
        covered = to_int(row.get("covered"))
        self.calls += 1
        self.active += active
        self.covered += covered
        self.cache_hits += to_int(row.get("cache_hits"))
        self.covered_cache_hits += to_int(row.get("covered_cache_hits"))
        self.full_calls += to_int(row.get("full_cover"))
        self.no_cover_calls += to_int(row.get("no_cover"))
        if 0 < covered < active:
            self.partial_calls += 1
        self.covered_runs += to_int(row.get("covered_runs"))
        self.uncovered_runs += to_int(row.get("uncovered_runs"))
        self.transitions += to_int(row.get("transitions"))
        self.extra_groups += to_int(row.get("extra_compute_groups"))
        self.logical_total_bytes += to_int(row.get("logical_total_bytes"))
        self.logical_miss_bytes += to_int(row.get("logical_miss_bytes"))
        self.covered_saved_bytes += to_int(row.get("covered_saved_bytes"))
        self.covered_miss_saved_bytes += to_int(row.get("covered_miss_saved_bytes"))
        self.reject_not_allowed += to_int(row.get("reject_not_allowed"))
        self.reject_other += sum(
            to_int(row.get(key))
            for key in [
                "reject_no_manifest",
                "reject_no_entry",
                "reject_unsupported",
                "reject_manifest_mismatch",
                "reject_shape_mismatch",
                "reject_not_smaller",
            ]
        )

    @property
    def coverage(self) -> float:
        return self.covered / self.active if self.active else 0.0

    @property
    def full_call_ratio(self) -> float:
        return self.full_calls / self.calls if self.calls else 0.0

    @property
    def cache_hit_ratio(self) -> float:
        return self.cache_hits / self.active if self.active else 0.0

    @property
    def avg_transitions(self) -> float:
        return self.transitions / self.calls if self.calls else 0.0

    @property
    def avg_extra_groups(self) -> float:
        return self.extra_groups / self.calls if self.calls else 0.0

    @property
    def miss_saved_ratio(self) -> float:
        if self.logical_miss_bytes <= 0:
            return 0.0
        return self.covered_miss_saved_bytes / self.logical_miss_bytes

    @property
    def saved_ratio(self) -> float:
        if self.logical_total_bytes <= 0:
            return 0.0
        return self.covered_saved_bytes / self.logical_total_bytes

    @property
    def miss_saved_gib(self) -> float:
        return gib(self.covered_miss_saved_bytes)

    def estimate_saved_ms(self, io_gib_s: float, h2d_gib_s: float, include_h2d: bool) -> float:
        saved = self.miss_saved_gib / io_gib_s * 1000.0 if io_gib_s > 0 else 0.0
        if include_h2d and h2d_gib_s > 0:
            saved += self.miss_saved_gib / h2d_gib_s * 1000.0
        return saved


def load_metrics(path: Path | None) -> Metrics:
    out = Metrics()
    if not path or not path.exists():
        return out
    values: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    out.token_rate = to_float(values.get("token_rate"), math.nan)
    if math.isnan(out.token_rate):
        out.token_rate = None
    out.decode_ms = to_float(values.get("decode_ms"), math.nan)
    if math.isnan(out.decode_ms):
        out.decode_ms = None
    runs = values.get("decode_runs")
    out.decode_runs = int(runs) if runs and runs.isdigit() else None
    out.ttft_ms = to_float(values.get("ttft_ms"), math.nan)
    if math.isnan(out.ttft_ms):
        out.ttft_ms = None
    peak = values.get("memory.peak")
    out.ram_peak = int(peak) if peak and peak.isdigit() else None
    return out


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def aggregate(rows: Iterable[dict[str, str]], key_fn: Callable[[dict[str, str]], tuple[str, ...]]) -> list[Agg]:
    by: dict[tuple[str, ...], Agg] = {}
    for row in rows:
        key = key_fn(row)
        if key not in by:
            by[key] = Agg(key)
        by[key].add(row)
    return list(by.values())


def aggregate_one(name: tuple[str, ...], rows: Iterable[dict[str, str]]) -> Agg:
    agg = Agg(name)
    for row in rows:
        agg.add(row)
    return agg


def needed_saved_ms(metrics: Metrics, target_tps: float) -> float | None:
    ms_per_token = metrics.decode_ms_per_token
    if ms_per_token is None or metrics.decode_runs is None:
        return None
    target_ms = 1000.0 / target_tps
    return max(0.0, (ms_per_token - target_ms) * metrics.decode_runs)


def overhead_budget_ms_per_group(
        agg: Agg,
        metrics: Metrics,
        target_tps: float,
        io_gib_s: float,
        h2d_gib_s: float,
        include_h2d: bool) -> float | None:
    need = needed_saved_ms(metrics, target_tps)
    if need is None or agg.extra_groups <= 0:
        return None
    saved = agg.estimate_saved_ms(io_gib_s, h2d_gib_s, include_h2d)
    return (saved - need) / agg.extra_groups


def agg_row(
        agg: Agg,
        metrics: Metrics,
        target_tps: float,
        io_gib_s: float,
        h2d_gib_s: float,
        include_h2d: bool) -> dict[str, str]:
    saved_ms = agg.estimate_saved_ms(io_gib_s, h2d_gib_s, include_h2d)
    budget = overhead_budget_ms_per_group(
        agg, metrics, target_tps, io_gib_s, h2d_gib_s, include_h2d)
    return {
        "key": "/".join(agg.name),
        "calls": str(agg.calls),
        "active": str(agg.active),
        "covered": str(agg.covered),
        "coverage_pct": f"{agg.coverage * 100.0:.3f}",
        "full_calls": str(agg.full_calls),
        "partial_calls": str(agg.partial_calls),
        "no_cover_calls": str(agg.no_cover_calls),
        "cache_hit_pct": f"{agg.cache_hit_ratio * 100.0:.3f}",
        "avg_transitions": f"{agg.avg_transitions:.3f}",
        "extra_groups": str(agg.extra_groups),
        "miss_saved_gib": f"{agg.miss_saved_gib:.6f}",
        "miss_saved_ratio_pct": f"{agg.miss_saved_ratio * 100.0:.3f}",
        "estimated_saved_ms": f"{saved_ms:.3f}",
        "overhead_budget_ms_per_extra_group": "" if budget is None else f"{budget:.6f}",
        "reject_not_allowed": str(agg.reject_not_allowed),
        "reject_other": str(agg.reject_other),
    }


def write_tsv(path: Path, groups: list[Agg], metrics: Metrics, args: argparse.Namespace) -> None:
    fields = [
        "key",
        "calls",
        "active",
        "covered",
        "coverage_pct",
        "full_calls",
        "partial_calls",
        "no_cover_calls",
        "cache_hit_pct",
        "avg_transitions",
        "extra_groups",
        "miss_saved_gib",
        "miss_saved_ratio_pct",
        "estimated_saved_ms",
        "overhead_budget_ms_per_extra_group",
        "reject_not_allowed",
        "reject_other",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for agg in groups:
            writer.writerow(agg_row(
                agg, metrics, args.target_tps, args.io_gib_s, args.h2d_gib_s, args.include_h2d))


def md_table(groups: list[Agg], metrics: Metrics, args: argparse.Namespace, limit: int) -> list[str]:
    lines = [
        "| key | calls | cov | full | partial | avg_trans | extra | miss_saved GiB | est_saved ms | overhead budget ms/group |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for agg in groups[:limit]:
        saved_ms = agg.estimate_saved_ms(args.io_gib_s, args.h2d_gib_s, args.include_h2d)
        budget = overhead_budget_ms_per_group(
            agg, metrics, args.target_tps, args.io_gib_s, args.h2d_gib_s, args.include_h2d)
        budget_s = "" if budget is None else f"{budget:.3f}"
        lines.append(
            f"| {'/'.join(agg.name)} | {agg.calls} | {agg.coverage*100:.1f}% | "
            f"{agg.full_calls} | {agg.partial_calls} | {agg.avg_transitions:.2f} | "
            f"{agg.extra_groups} | {agg.miss_saved_gib:.3f} | {saved_ms:.1f} | {budget_s} |"
        )
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--planner-csv", required=True, type=Path)
    ap.add_argument("--metrics", type=Path)
    ap.add_argument("--out-md", required=True, type=Path)
    ap.add_argument("--out-tensor-tsv", required=True, type=Path)
    ap.add_argument("--target-tps", type=float, default=2.0)
    ap.add_argument("--io-gib-s", type=float, default=10.4)
    ap.add_argument("--h2d-gib-s", type=float, default=24.0)
    ap.add_argument("--include-h2d", action="store_true")
    ap.add_argument("--min-coverage", type=float, default=0.85)
    ap.add_argument("--max-avg-transitions", type=float, default=1.5)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    rows = read_rows(args.planner_csv)
    if not rows:
        raise SystemExit(f"no planner rows in {args.planner_csv}")
    metrics = load_metrics(args.metrics)

    overall = aggregate(rows, lambda _r: ("overall",))[0]
    by_phase_role = aggregate(rows, lambda r: (r["phase"], r["role"]))
    by_tensor = aggregate(rows, lambda r: (r["phase"], r["role"], r["tensor"]))
    by_tensor.sort(key=lambda a: (-a.covered_miss_saved_bytes, a.avg_transitions, -a.coverage))

    decode_rows = [r for r in rows if r["phase"].startswith("decode")]
    decode_upgate_rows = [r for r in rows if r["phase"] == "decode_upgate"]
    decode_down_rows = [r for r in rows if r["phase"] == "decode_down"]
    full_cover_decode_rows = [r for r in decode_rows if to_int(r.get("full_cover")) == 1]
    partial_decode_rows = [
        r for r in decode_rows
        if 0 < to_int(r.get("covered")) < to_int(r.get("n_active"))
    ]
    portfolios = [
        aggregate_one(("decode_all",), decode_rows),
        aggregate_one(("decode_upgate_all",), decode_upgate_rows),
        aggregate_one(("decode_down_all",), decode_down_rows),
        aggregate_one(("decode_full_cover_only",), full_cover_decode_rows),
        aggregate_one(("decode_partial_only",), partial_decode_rows),
    ]

    decode_upgate_candidates = [
        agg for agg in by_tensor
        if len(agg.name) >= 2
        and agg.name[0] == "decode_upgate"
        and agg.name[1] in {"up", "gate"}
        and agg.coverage >= args.min_coverage
        and agg.avg_transitions <= args.max_avg_transitions
        and agg.covered_miss_saved_bytes > 0
    ]
    decode_upgate_candidates.sort(
        key=lambda a: (-(a.covered_miss_saved_bytes / max(1, a.extra_groups)), -a.coverage, a.avg_transitions))

    first_wave = Agg(("first_wave_decode_upgate",))
    for agg in decode_upgate_candidates:
        first_wave.calls += agg.calls
        first_wave.active += agg.active
        first_wave.covered += agg.covered
        first_wave.cache_hits += agg.cache_hits
        first_wave.covered_cache_hits += agg.covered_cache_hits
        first_wave.full_calls += agg.full_calls
        first_wave.no_cover_calls += agg.no_cover_calls
        first_wave.partial_calls += agg.partial_calls
        first_wave.covered_runs += agg.covered_runs
        first_wave.uncovered_runs += agg.uncovered_runs
        first_wave.transitions += agg.transitions
        first_wave.extra_groups += agg.extra_groups
        first_wave.logical_total_bytes += agg.logical_total_bytes
        first_wave.logical_miss_bytes += agg.logical_miss_bytes
        first_wave.covered_saved_bytes += agg.covered_saved_bytes
        first_wave.covered_miss_saved_bytes += agg.covered_miss_saved_bytes
        first_wave.reject_not_allowed += agg.reject_not_allowed
        first_wave.reject_other += agg.reject_other

    write_tsv(args.out_tensor_tsv, by_tensor, metrics, args)

    need = needed_saved_ms(metrics, args.target_tps)
    current_ms = metrics.decode_ms_per_token
    mode = "io+h2d" if args.include_h2d else "io-only"

    lines: list[str] = []
    lines.append("# Kimi v2 partial split candidate summary")
    lines.append("")
    lines.append(f"Planner CSV: `{args.planner_csv}`")
    if args.metrics:
        lines.append(f"Metrics: `{args.metrics}`")
    lines.append(f"Estimate mode: `{mode}`, io_gib_s={args.io_gib_s}, h2d_gib_s={args.h2d_gib_s}")
    lines.append(f"Target: `{args.target_tps:.3f} tok/s`")
    if current_ms is not None:
        lines.append(f"Current decode: `{current_ms:.3f} ms/token` over `{metrics.decode_runs}` tokens")
    if need is not None:
        lines.append(f"Saved time needed for target: `{need:.3f} ms` total")
    if metrics.ram_peak is not None:
        lines.append(f"RAM peak from run: `{metrics.ram_peak}` bytes")
    lines.append("")
    lines.append("## Overall")
    lines.extend(md_table([overall], metrics, args, 1))
    lines.append("")
    lines.append("## By Phase/Role")
    by_phase_role.sort(key=lambda a: (a.name, ))
    lines.extend(md_table(by_phase_role, metrics, args, len(by_phase_role)))
    lines.append("")
    lines.append("## Decode Portfolios")
    lines.extend(md_table(portfolios, metrics, args, len(portfolios)))
    lines.append("")
    lines.append("## First-Wave Decode Up/Gate Candidates")
    lines.append("")
    lines.append(
        f"Filter: phase=`decode_upgate`, role in `up,gate`, coverage >= `{args.min_coverage:.3f}`, "
        f"avg_transitions <= `{args.max_avg_transitions:.3f}`."
    )
    lines.append(f"Candidate tensors: `{len(decode_upgate_candidates)}`")
    if decode_upgate_candidates:
        lines.extend(md_table([first_wave], metrics, args, 1))
        lines.append("")
        lines.extend(md_table(decode_upgate_candidates, metrics, args, args.top))
    else:
        lines.append("")
        lines.append("No first-wave candidates matched the filter.")
    lines.append("")
    lines.append("## Top Tensors By Miss-Weighted Saving")
    lines.extend(md_table(by_tensor, metrics, args, args.top))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- `miss_saved GiB` is the primary critical-path byte-reduction estimate because rows already in VRAM cache "
        "do not represent exposed SSD misses."
    )
    lines.append(
        "- `avg_trans` and `extra` are the dispatch-risk indicators. High values mean partial dispatch will add more "
        "split/scatter or second-kernel overhead."
    )
    lines.append(
        "- A dispatch wave should start with high-coverage decode up/gate tensors whose overhead budget per extra "
        "group stays positive under the selected bandwidth model."
    )
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out_md}")
    print(f"wrote {args.out_tensor_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
