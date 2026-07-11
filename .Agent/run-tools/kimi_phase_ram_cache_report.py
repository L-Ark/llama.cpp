#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
from dataclasses import dataclass
from typing import Any


KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^ \n]+)")
GIB = 1024 ** 3
MIB = 1024 ** 2


def inum(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return default


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def fmt(value: float) -> str:
    return f"{value:.3f}"


def parse_kv_line(line: str) -> dict[str, str]:
    return {key: value for key, value in KV_RE.findall(line or "")}


def read_json(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {}


def read_stat(path: pathlib.Path) -> dict[str, int]:
    out: dict[str, int] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[parts[0]] = inum(parts[1])
    return out


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def discover_runs(paths: list[pathlib.Path]) -> list[pathlib.Path]:
    runs: list[pathlib.Path] = []
    seen: set[pathlib.Path] = set()
    for path in paths:
        if path.is_file() and path.name == "metrics.json":
            run = path.parent
            if run not in seen:
                seen.add(run)
                runs.append(run)
            continue
        if path.is_dir() and (path / "metrics.json").exists():
            if path not in seen:
                seen.add(path)
                runs.append(path)
            continue
        if path.is_dir():
            for metrics in sorted(path.glob("*/metrics.json")):
                run = metrics.parent
                if run not in seen:
                    seen.add(run)
                    runs.append(run)
    return runs


def phase_records(metrics: dict[str, Any]) -> list[dict[str, str]]:
    records = []
    for key, value in metrics.items():
        if key.startswith("kimi_phase_"):
            rec = parse_kv_line(str(value))
            rec["_kind"] = "kimi"
            rec["_key"] = key
            records.append(rec)
    records.sort(key=lambda r: inum(r["_key"].split("_")[-1]))
    return records


def moe_records(metrics: dict[str, Any]) -> list[dict[str, str]]:
    records = []
    for key, value in metrics.items():
        if key.startswith("moe_phase_"):
            rec = parse_kv_line(str(value))
            rec["_kind"] = "moe"
            rec["_key"] = key
            records.append(rec)
    records.sort(key=lambda r: inum(r["_key"].split("_")[-1]))
    return records


def find_phase(phases: list[dict[str, str]], label: str) -> dict[str, str]:
    for rec in phases:
        if rec.get("label") == label:
            return rec
    return {}


def find_moe(metrics: dict[str, Any], label: str, selector: str | None = None, value: str | None = None) -> dict[str, str]:
    for rec in moe_records(metrics):
        if rec.get("label") != label:
            continue
        if selector is None:
            if "pack_hits" in rec:
                return rec
            continue
        if rec.get(selector) == value:
            return rec
    return {}


def delta(after: dict[str, str], before: dict[str, str], key: str) -> int:
    return inum(after.get(key)) - inum(before.get(key))


def maybe_phase_delta(phases: list[dict[str, str]], start: str, end: str, key: str) -> int:
    return delta(find_phase(phases, end), find_phase(phases, start), key)


def parse_pack_line(metrics: dict[str, Any]) -> dict[str, int]:
    line = str(metrics.get("expert_pack_0", ""))
    return {key: inum(value) for key, value in KV_RE.findall(line)}


def parse_fallback_line(metrics: dict[str, Any]) -> dict[str, int]:
    for value in metrics.values():
        text = str(value)
        if "fallback_gguf=" in text and "enabled=" in text:
            return {key: inum(val) for key, val in KV_RE.findall(text)}
    return {}


def read_memory_samples(run: pathlib.Path) -> dict[str, Any]:
    path = run / "memory-samples.tsv"
    if not path.exists():
        return {}
    rows = []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rows.append(row)
    if not rows:
        return {}

    def max_int(field: str) -> int:
        return max((inum(r.get(field)) for r in rows), default=0)

    peak = max(rows, key=lambda r: inum(r.get("memory_current")))
    final = rows[-1]
    return {
        "samples": len(rows),
        "sample_peak_memory_current": inum(peak.get("memory_current")),
        "sample_peak_file": inum(peak.get("file")),
        "sample_peak_active_file": inum(peak.get("active_file")),
        "sample_peak_inactive_file": inum(peak.get("inactive_file")),
        "sample_peak_file_mapped": inum(peak.get("file_mapped")),
        "sample_peak_pgmajfault": inum(peak.get("pgmajfault")),
        "sample_peak_refault_file": inum(peak.get("workingset_refault_file")),
        "sample_max_file_mapped": max_int("file_mapped"),
        "sample_final_file_mapped": inum(final.get("file_mapped")),
    }


def summarize_run(run: pathlib.Path) -> dict[str, Any]:
    metrics = read_json(run / "metrics.json")
    stat = read_stat(run / "memory.stat.final.txt")
    phases = phase_records(metrics)
    pack = parse_pack_line(metrics)
    fallback = parse_fallback_line(metrics)

    before = find_phase(phases, "before_prompt_eval")
    after_prompt = find_phase(phases, "after_prompt_eval")
    after_gen = find_phase(phases, "after_generation")
    prompt_pack = find_moe(metrics, "after_prompt_eval")
    decode_pack = find_moe(metrics, "after_generation")
    prompt_down = find_moe(metrics, "after_prompt_eval", "cache", "down")
    decode_down = find_moe(metrics, "after_generation", "cache", "down")
    prompt_upgate = find_moe(metrics, "after_prompt_eval", "cache", "upgate")
    decode_upgate = find_moe(metrics, "after_generation", "cache", "upgate")

    final_file = stat.get("file", inum(metrics.get("file")))
    final_file_mapped = stat.get("file_mapped", 0)
    final_active = stat.get("active_file", inum(metrics.get("active_file")))
    final_inactive = stat.get("inactive_file", inum(metrics.get("inactive_file")))
    final_dirty = stat.get("file_dirty", 0) + stat.get("file_writeback", 0)
    final_unmapped_clean = max(final_file - final_file_mapped - final_dirty, 0)

    decode_iouring_bytes = inum(decode_pack.get("delta_iouring_bytes"))
    decode_iouring_wait_us = inum(decode_pack.get("delta_iouring_wait_us"))
    prompt_iouring_bytes = inum(prompt_pack.get("delta_iouring_bytes"))
    prompt_iouring_wait_us = inum(prompt_pack.get("delta_iouring_wait_us"))
    decode_file_delta = maybe_phase_delta(phases, "after_prompt_eval", "after_generation", "file")
    decode_active_delta = maybe_phase_delta(phases, "after_prompt_eval", "after_generation", "active_file")
    decode_refault_delta = maybe_phase_delta(phases, "after_prompt_eval", "after_generation", "workingset_refault_file")
    decode_pgmaj_delta = maybe_phase_delta(phases, "after_prompt_eval", "after_generation", "pgmajfault")

    row: dict[str, Any] = {
        "run": str(run),
        "prompt_id": metrics.get("prompt_id", run.name),
        "quality": metrics.get("quality", ""),
        "phase_data_available": bool(before and after_prompt and after_gen),
        "token_rate": fnum(metrics.get("token_rate")),
        "ttft_ms": fnum(metrics.get("ttft_ms")),
        "decode_ms": fnum(metrics.get("decode_ms")),
        "decode_runs": inum(metrics.get("decode_runs")),
        "memory_peak": inum(metrics.get("memory.peak", metrics.get("memory_peak"))),
        "memory_current_final": inum(metrics.get("memory.current.final", metrics.get("memory_current_final"))),
        "file_before_prompt": inum(before.get("file")),
        "file_after_prompt": inum(after_prompt.get("file")),
        "file_after_generation": inum(after_gen.get("file")),
        "file_final": final_file,
        "file_mapped_final": final_file_mapped,
        "file_dirty_writeback_final": final_dirty,
        "file_unmapped_clean_final": final_unmapped_clean,
        "active_file_final": final_active,
        "inactive_file_final": final_inactive,
        "prompt_file_delta": maybe_phase_delta(phases, "before_prompt_eval", "after_prompt_eval", "file"),
        "decode_file_delta": decode_file_delta,
        "prompt_active_file_delta": maybe_phase_delta(phases, "before_prompt_eval", "after_prompt_eval", "active_file"),
        "decode_active_file_delta": decode_active_delta,
        "prompt_pgmajfault_delta": maybe_phase_delta(phases, "before_prompt_eval", "after_prompt_eval", "pgmajfault"),
        "decode_pgmajfault_delta": decode_pgmaj_delta,
        "prompt_refault_file_delta": maybe_phase_delta(phases, "before_prompt_eval", "after_prompt_eval", "workingset_refault_file"),
        "decode_refault_file_delta": decode_refault_delta,
        "pgscan_direct_final": stat.get("pgscan_direct", inum(after_gen.get("pgscan_direct"))),
        "pgsteal_direct_final": stat.get("pgsteal_direct", inum(after_gen.get("pgsteal_direct"))),
        "pack_iouring_bytes_total": pack.get("iouring_bytes", 0),
        "pack_iouring_wait_ms_total": pack.get("iouring_wait_us", 0) / 1000.0,
        "prompt_iouring_bytes": prompt_iouring_bytes,
        "prompt_iouring_wait_ms": prompt_iouring_wait_us / 1000.0,
        "decode_iouring_bytes": decode_iouring_bytes,
        "decode_iouring_wait_ms": decode_iouring_wait_us / 1000.0,
        "decode_iouring_reads": inum(decode_pack.get("delta_iouring_reads")),
        "decode_iouring_batches": inum(decode_pack.get("delta_iouring_batches")),
        "decode_direct_reads": inum(decode_pack.get("delta_direct_reads")),
        "decode_read_failures": inum(decode_pack.get("delta_read_failures")),
        "decode_pack_misses": inum(decode_pack.get("delta_pack_misses")),
        "fallback_gguf": fallback.get("fallback_gguf", 0),
        "fallback_pack_mmap_hits": fallback.get("hits", 0),
        "fallback_pack_mmap_misses": fallback.get("misses", 0),
        "decode_down_hits": inum(decode_down.get("delta_hits")),
        "decode_down_misses": inum(decode_down.get("delta_misses")),
        "decode_upgate_hits": inum(decode_upgate.get("delta_hits")),
        "decode_upgate_misses": inum(decode_upgate.get("delta_misses")),
        "prompt_down_misses": inum(prompt_down.get("delta_misses")),
        "prompt_upgate_misses": inum(prompt_upgate.get("delta_misses")),
    }
    row.update(read_memory_samples(run))
    row["decode_file_delta_per_iouring_gib"] = (
        decode_file_delta / max(decode_iouring_bytes / GIB, 1e-9)
        if decode_iouring_bytes else 0.0
    )
    row["decode_file_cache_is_main_expert_source"] = (
        decode_iouring_bytes == 0 and decode_file_delta > 256 * MIB
    )
    row["normal_decode_low_refault"] = (
        row["phase_data_available"]
        and decode_iouring_bytes > 0
        and decode_refault_delta == 0
        and stat.get("workingset_refault_file", 0) == 0
        and stat.get("pgscan_direct", 0) == 0
        and stat.get("pgsteal_direct", 0) == 0
    )
    return row


def build_report(rows: list[dict[str, Any]], out: pathlib.Path) -> None:
    lines = [
        "# Kimi Phase 5C RAM/page-cache baseline",
        "",
        "This report is diagnostic only. It does not claim a SOTA result or change runtime behavior.",
        "",
        "## Run Summary",
        "",
        "| prompt | phase | tok/s | TTFT ms | RAM peak GiB | final file GiB | final file_mapped MiB | clean unmapped file GiB | decode iouring GiB | decode file delta MiB | decode refault | direct reads | fallback GGUF |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['prompt_id']}` | {'yes' if r['phase_data_available'] else 'no'} | "
            f"{float(r['token_rate']):.2f} | {float(r['ttft_ms']):.1f} | "
            f"{r['memory_peak'] / GIB:.2f} | {r['file_final'] / GIB:.2f} | "
            f"{r['file_mapped_final'] / MIB:.3f} | {r['file_unmapped_clean_final'] / GIB:.2f} | "
            f"{r['decode_iouring_bytes'] / GIB:.1f} | {r['decode_file_delta'] / MIB:.1f} | "
            f"{r['decode_refault_file_delta']} | {r['decode_direct_reads']} | {r['fallback_gguf']} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
    ])

    normal = [r for r in rows if r["normal_decode_low_refault"] and r["decode_iouring_bytes"] > 0]
    if normal:
        avg_decode_gib = sum(r["decode_iouring_bytes"] for r in normal) / len(normal) / GIB
        avg_file_delta_mib = sum(r["decode_file_delta"] for r in normal) / len(normal) / MIB
        avg_unmapped_gib = sum(r["file_unmapped_clean_final"] for r in normal) / len(normal) / GIB
        lines.extend([
            f"- Normal low-refault runs: `{len(normal)}`.",
            f"- Average decode expert-pack traffic in those runs: `{avg_decode_gib:.1f} GiB`.",
            f"- Average decode file-cache growth in those runs: `{avg_file_delta_mib:.1f} MiB`.",
            f"- Average final clean unmapped file cache in those runs: `{avg_unmapped_gib:.2f} GiB`.",
            "- This supports the Phase 5C assumption that normal decode is not primarily served by Linux page cache; it is still served by expert-pack io_uring/H2D.",
            "- The clean unmapped file cache is a plausible replacement pool for controlled RAM expert residency, but eviction safety must be verified by paired cold-start refault/TTFT counters.",
        ])
    else:
        lines.append("- No normal low-refault run was found in the selected inputs.")

    pressure = [r for r in rows if r.get("pgscan_direct_final", 0) > 0 or r.get("decode_refault_file_delta", 0) > 0]
    if pressure:
        lines.append("- Memory-pressure/refault runs are present and must not be used as RAM-tier success evidence without paired TTFT analysis.")
    lines.extend([
        "",
        "## Next Gate",
        "",
        "- Build RAM-slab candidates from dev-only foreground IO/wait profiles.",
        "- Start with pageable, batchable layer/role or pack-contiguous slabs.",
        "- Accept a RAM tier only if it reduces demand wait without increasing TTFT, refaults, direct reclaim, or mixed SSD/RAM batch fragmentation.",
        "",
    ])
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize Kimi phase RAM/page-cache state for Phase 5C planning.")
    ap.add_argument("--run", action="append", type=pathlib.Path, default=[])
    ap.add_argument("--root", action="append", type=pathlib.Path, default=[])
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = ap.parse_args()

    runs = discover_runs(args.run + args.root)
    if not runs:
        raise SystemExit("No run directories with metrics.json found")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [summarize_run(run) for run in runs]
    fields = sorted({key for row in rows for key in row})
    write_csv(args.out_dir / "ram_cache_summary.csv", rows, fields)
    (args.out_dir / "ram_cache_summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    build_report(rows, args.out_dir / "report.md")


if __name__ == "__main__":
    main()
