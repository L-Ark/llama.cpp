#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


GIB = 1024 ** 3


def load_metrics(run: Path) -> dict:
    return json.loads((run / "metrics.json").read_text(encoding="utf-8", errors="replace"))


def stat_value(run: Path, key: str) -> int:
    path = run / "memory.stat.final.txt"
    if not path.exists():
        return 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == key:
            return int(parts[1])
    return 0


def parse_kv(line: str) -> dict[str, str]:
    return {key: value for key, value in re.findall(r"([A-Za-z0-9_]+)=([^ \n]+)", line or "")}


def ram_line(run: Path) -> str:
    stderr = run / "stderr.txt"
    if not stderr.exists():
        return ""
    matches = re.findall(r"\[moe_stream_batch\] RAM tier: hits=.*", stderr.read_text(encoding="utf-8", errors="replace"))
    return matches[-1] if matches else ""


def blk1_upgate(run: Path) -> dict[str, float]:
    total = {"wall_ms": 0.0, "up_wait_ms": 0.0, "gate_wait_ms": 0.0, "calls": 0.0}
    path = run / "up-gate-profile.csv"
    if not path.exists():
        return total
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            if row.get("mode") != "decode":
                continue
            if row.get("up_tensor") != "blk.1.ffn_up_exps.weight":
                continue
            total["calls"] += 1
            for key in ("wall_ms", "up_wait_ms", "gate_wait_ms"):
                total[key] += float(row.get(key) or 0)
    return total


def summarize_pair(prompt: str, n_label: str, control: Path, candidate: Path) -> dict:
    c = load_metrics(control)
    x = load_metrics(candidate)
    c_ram = parse_kv(ram_line(control))
    x_ram = parse_kv(ram_line(candidate))
    c_blk = blk1_upgate(control)
    x_blk = blk1_upgate(candidate)
    c_rate = float(c.get("token_rate", 0) or 0)
    x_rate = float(x.get("token_rate", 0) or 0)
    c_decode = float(c.get("decode_ms", 0) or 0)
    x_decode = float(x.get("decode_ms", 0) or 0)
    c_ttft = float(c.get("ttft_ms", 0) or 0)
    x_ttft = float(x.get("ttft_ms", 0) or 0)
    ram_hits = int(x_ram.get("hits", "0") or 0)
    ram_total = int(x_ram.get("total", "0") or 0)
    return {
        "prompt": prompt,
        "n": n_label,
        "control_quality": c.get("quality", ""),
        "candidate_quality": x.get("quality", ""),
        "control_tok_s": c_rate,
        "candidate_tok_s": x_rate,
        "delta_tok_s": x_rate - c_rate,
        "control_decode_ms": c_decode,
        "candidate_decode_ms": x_decode,
        "delta_decode_ms": x_decode - c_decode,
        "control_ttft_ms": c_ttft,
        "candidate_ttft_ms": x_ttft,
        "ttft_ratio": x_ttft / max(c_ttft, 1e-9),
        "control_peak": int(c.get("memory.peak", 0) or 0),
        "candidate_peak": int(x.get("memory.peak", 0) or 0),
        "control_refault": stat_value(control, "workingset_refault_file"),
        "candidate_refault": stat_value(candidate, "workingset_refault_file"),
        "control_pgscan_direct": stat_value(control, "pgscan_direct"),
        "candidate_pgscan_direct": stat_value(candidate, "pgscan_direct"),
        "control_pgsteal_direct": stat_value(control, "pgsteal_direct"),
        "candidate_pgsteal_direct": stat_value(candidate, "pgsteal_direct"),
        "ram_hits": ram_hits,
        "ram_total": ram_total,
        "ram_hit_rate": ram_hits / max(ram_total, 1),
        "ram_h2d_bytes": int(x_ram.get("h2d_bytes", "0") or 0),
        "control_blk1_wall_ms": c_blk["wall_ms"],
        "candidate_blk1_wall_ms": x_blk["wall_ms"],
        "delta_blk1_wall_ms": x_blk["wall_ms"] - c_blk["wall_ms"],
        "control_blk1_wait_ms": c_blk["up_wait_ms"] + c_blk["gate_wait_ms"],
        "candidate_blk1_wait_ms": x_blk["up_wait_ms"] + x_blk["gate_wait_ms"],
        "delta_blk1_wait_ms": (x_blk["up_wait_ms"] + x_blk["gate_wait_ms"]) - (c_blk["up_wait_ms"] + c_blk["gate_wait_ms"]),
    }


def write_report(out_dir: Path, rows: list[dict], run_roots: dict[str, str]) -> dict:
    summary = {
        "rows": rows,
        "valid_prompt_count": len(rows),
        "mean_control_tok_s": sum(r["control_tok_s"] for r in rows) / len(rows),
        "mean_candidate_tok_s": sum(r["candidate_tok_s"] for r in rows) / len(rows),
        "mean_delta_tok_s": sum(r["delta_tok_s"] for r in rows) / len(rows),
        "mean_control_decode_ms": sum(r["control_decode_ms"] for r in rows) / len(rows),
        "mean_candidate_decode_ms": sum(r["candidate_decode_ms"] for r in rows) / len(rows),
        "mean_delta_decode_ms": sum(r["delta_decode_ms"] for r in rows) / len(rows),
        "max_ttft_ratio": max(r["ttft_ratio"] for r in rows),
        "max_candidate_peak": max(r["candidate_peak"] for r in rows),
        "max_candidate_refault": max(r["candidate_refault"] for r in rows),
        "max_candidate_pgscan_direct": max(r["candidate_pgscan_direct"] for r in rows),
        "max_candidate_pgsteal_direct": max(r["candidate_pgsteal_direct"] for r in rows),
        "total_ram_hits": sum(r["ram_hits"] for r in rows),
        "total_ram_total": sum(r["ram_total"] for r in rows),
        "total_ram_h2d_gib": sum(r["ram_h2d_bytes"] for r in rows) / GIB,
        "total_blk1_wall_delta_ms": sum(r["delta_blk1_wall_ms"] for r in rows),
        "total_blk1_wait_delta_ms": sum(r["delta_blk1_wait_ms"] for r in rows),
        "run_roots": run_roots,
    }
    summary["total_ram_hit_rate"] = summary["total_ram_hits"] / max(summary["total_ram_total"], 1)
    summary["decision"] = "reject" if summary["mean_candidate_tok_s"] <= summary["mean_control_tok_s"] else "candidate"

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Candidate C compact RAM slab A/B",
        "",
        f"- decision: `{summary['decision']}`",
        f"- mean control token rate: `{summary['mean_control_tok_s']:.3f}`",
        f"- mean candidate token rate: `{summary['mean_candidate_tok_s']:.3f}`",
        f"- mean delta token rate: `{summary['mean_delta_tok_s']:.3f}`",
        f"- mean decode delta: `{summary['mean_delta_decode_ms']:.2f} ms`",
        f"- max TTFT ratio: `{summary['max_ttft_ratio']:.3f}`",
        f"- max candidate RAM peak: `{summary['max_candidate_peak']}` bytes (`{summary['max_candidate_peak'] / GIB:.2f} GiB`)",
        f"- max candidate refault: `{summary['max_candidate_refault']}`",
        f"- max candidate pgscan/pgsteal direct: `{summary['max_candidate_pgscan_direct']} / {summary['max_candidate_pgsteal_direct']}`",
        f"- RAM tier hits: `{summary['total_ram_hits']} / {summary['total_ram_total']}` = `{100 * summary['total_ram_hit_rate']:.2f}%`",
        f"- RAM tier H2D: `{summary['total_ram_h2d_gib']:.2f} GiB`",
        f"- total blk1 wall delta: `{summary['total_blk1_wall_delta_ms']:.2f} ms`",
        f"- total blk1 wait delta: `{summary['total_blk1_wait_delta_ms']:.2f} ms`",
        "",
        "| prompt | N | quality | tok/s control | tok/s cand | delta | TTFT ratio | peak cand GiB | refault cand | RAM hit % | decode delta ms | blk1 wall delta ms |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['prompt']}` | {r['n']} | {r['control_quality']}->{r['candidate_quality']} | "
            f"{r['control_tok_s']:.2f} | {r['candidate_tok_s']:.2f} | {r['delta_tok_s']:.2f} | "
            f"{r['ttft_ratio']:.3f} | {r['candidate_peak'] / GIB:.2f} | {r['candidate_refault']} | "
            f"{100 * r['ram_hit_rate']:.2f}% | {r['delta_decode_ms']:.1f} | {r['delta_blk1_wall_ms']:.1f} |"
        )
    lines.extend(["", "## Run Roots", ""])
    for key, value in run_roots.items():
        lines.append(f"- {key}: `{value}`")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize Kimi RAM candidate A/B runs.")
    ap.add_argument("--control-root", type=Path, required=True)
    ap.add_argument("--candidate-root", type=Path, required=True)
    ap.add_argument("--control-linear", type=Path, required=True)
    ap.add_argument("--candidate-linear", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--valid-n32-prompts", nargs="+", required=True)
    args = ap.parse_args()

    rows = [
        summarize_pair(prompt, "N32", args.control_root / prompt, args.candidate_root / prompt)
        for prompt in args.valid_n32_prompts
    ]
    rows.append(summarize_pair("dev_linear_equation", "N48", args.control_linear, args.candidate_linear))
    summary = write_report(args.out_dir, rows, {
        "control N32": str(args.control_root),
        "candidate N32": str(args.candidate_root),
        "control linear N48": str(args.control_linear),
        "candidate linear N48": str(args.candidate_linear),
    })
    print(args.out_dir / "summary.md")
    print(f"decision={summary['decision']} mean_control={summary['mean_control_tok_s']:.3f} mean_candidate={summary['mean_candidate_tok_s']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
