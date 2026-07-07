#!/usr/bin/env python3
"""Audit Kimi SOTA candidates for cold-start and warmed-IO artifacts.

The script compares two or three sweep roots that contain per-prompt
`metrics.json` files. It is intentionally conservative: a run that improves
token rate while moving the same bytes with the same VRAM hit rates is flagged
as "wait-only" and cannot be accepted as a model/runtime optimization without a
stronger cold-device reproduction protocol.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any


PACK_INT_RE = re.compile(r"([a-zA-Z_]+)=(\d+)")
HIT_RATE_RE = re.compile(r"hit_rate=([0-9.]+)%")


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def inum(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return default


def parse_pack(line: str) -> dict[str, int]:
    return {key: int(value) for key, value in PACK_INT_RE.findall(line or "")}


def parse_hit(line: str) -> float:
    match = HIT_RATE_RE.search(line or "")
    return float(match.group(1)) if match else 0.0


def load_metrics(root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("*/metrics.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        prompt_id = str(row.get("prompt_id") or path.parent.name)
        out[prompt_id] = row
    if not out:
        raise SystemExit(f"no metrics.json files found under {root}")
    return out


def compact(row: dict[str, Any]) -> dict[str, Any]:
    pack = parse_pack(str(row.get("expert_pack_0", "")))
    return {
        "quality": row.get("quality", ""),
        "token_rate": fnum(row.get("token_rate")),
        "ttft_ms": fnum(row.get("ttft_ms")),
        "decode_ms": fnum(row.get("decode_ms")),
        "decode_runs": inum(row.get("decode_runs")),
        "memory_peak": inum(row.get("memory.peak", row.get("memory_peak"))),
        "direct_reads": pack.get("direct_reads", 0),
        "iouring_reads": pack.get("iouring_reads", 0),
        "iouring_bytes": pack.get("iouring_bytes", 0),
        "iouring_wait_ms": pack.get("iouring_wait_us", 0) / 1000.0,
        "iouring_submit_ms": pack.get("iouring_submit_us", 0) / 1000.0,
        "upgate_hit_pct": parse_hit(str(row.get("vram_upgate_0", ""))),
        "down_hit_pct": parse_hit(str(row.get("vram_down_0", ""))),
        "output": row.get("output", ""),
    }


def pct_delta(new: float, old: float) -> float:
    return (new / old - 1.0) * 100.0 if old else 0.0


def row_compare(prompt_id: str, base: dict[str, Any], cand: dict[str, Any]) -> dict[str, Any]:
    b = compact(base)
    c = compact(cand)
    bytes_delta = pct_delta(c["iouring_bytes"], b["iouring_bytes"])
    wait_delta = pct_delta(c["iouring_wait_ms"], b["iouring_wait_ms"])
    up_hit_delta = c["upgate_hit_pct"] - b["upgate_hit_pct"]
    down_hit_delta = c["down_hit_pct"] - b["down_hit_pct"]
    return {
        "prompt_id": prompt_id,
        "base": b,
        "candidate": c,
        "token_rate_delta_pct": pct_delta(c["token_rate"], b["token_rate"]),
        "ttft_delta_pct": pct_delta(c["ttft_ms"], b["ttft_ms"]),
        "decode_delta_pct": pct_delta(c["decode_ms"], b["decode_ms"]),
        "bytes_delta_pct": bytes_delta,
        "wait_delta_pct": wait_delta,
        "upgate_hit_delta_pp": up_hit_delta,
        "down_hit_delta_pp": down_hit_delta,
        "same_bytes_and_hits": (
            abs(bytes_delta) <= 1.0
            and abs(up_hit_delta) <= 0.2
            and abs(down_hit_delta) <= 0.2
        ),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    base_rates = [r["base"]["token_rate"] for r in rows]
    cand_rates = [r["candidate"]["token_rate"] for r in rows]
    ttft_deltas = [r["ttft_delta_pct"] for r in rows]
    suspicious = [
        r for r in rows
        if r["same_bytes_and_hits"]
        and r["token_rate_delta_pct"] > 5.0
        and r["wait_delta_pct"] < -20.0
    ]
    common_pass = all(
        r["candidate"]["quality"] == "pass"
        and r["candidate"]["memory_peak"] < 16_000_000_000
        and r["candidate"]["direct_reads"] == 0
        for r in rows
    )
    ttft_pass = all(delta <= 20.0 for delta in ttft_deltas)
    rate_pass = statistics.mean(cand_rates) > statistics.mean(base_rates) and min(cand_rates) >= min(base_rates)
    return {
        "base_mean_tps": statistics.mean(base_rates),
        "candidate_mean_tps": statistics.mean(cand_rates),
        "base_min_tps": min(base_rates),
        "candidate_min_tps": min(cand_rates),
        "mean_tps_delta_pct": pct_delta(statistics.mean(cand_rates), statistics.mean(base_rates)),
        "max_ttft_delta_pct": max(ttft_deltas),
        "common_gate_pass": common_pass,
        "ttft_gate_pass": ttft_pass,
        "rate_gate_pass": rate_pass,
        "suspicious_wait_only_prompts": [r["prompt_id"] for r in suspicious],
        "wait_only_suspicion": bool(suspicious),
        "acceptance_safe": common_pass and ttft_pass and rate_pass and not suspicious,
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    baseline = load_metrics(args.baseline_root)
    candidate = load_metrics(args.candidate_root)
    ids = sorted(set(baseline) & set(candidate))
    if not ids:
        raise SystemExit("baseline and candidate have no prompt ids in common")
    missing_baseline = sorted(set(candidate) - set(baseline))
    missing_candidate = sorted(set(baseline) - set(candidate))
    rows = [row_compare(prompt_id, baseline[prompt_id], candidate[prompt_id]) for prompt_id in ids]
    payload = {
        "kind": "kimi_cold_start_sota_audit",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "baseline_root": str(args.baseline_root),
        "candidate_root": str(args.candidate_root),
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "prompt_ids": ids,
        "missing_from_baseline": missing_baseline,
        "missing_from_candidate": missing_candidate,
        "comparison": rows,
        "summary": summarize(rows),
        "reproduce_command": " ".join(args.argv),
    }
    if args.reference_root:
        reference = load_metrics(args.reference_root)
        ref_ids = sorted(set(reference) & set(candidate))
        payload["reference_root"] = str(args.reference_root)
        payload["reference_label"] = args.reference_label
        payload["reference_comparison"] = [
            row_compare(prompt_id, reference[prompt_id], candidate[prompt_id])
            for prompt_id in ref_ids
        ]
        payload["reference_summary"] = summarize(payload["reference_comparison"])
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "# Kimi cold-start SOTA audit",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- baseline: `{payload['baseline_label']}` -> `{payload['baseline_root']}`",
        f"- candidate: `{payload['candidate_label']}` -> `{payload['candidate_root']}`",
        f"- prompts compared: `{len(payload['prompt_ids'])}`",
        "",
        "## Summary",
        "",
        f"- baseline mean/min tok/s: `{s['base_mean_tps']:.4f}` / `{s['base_min_tps']:.4f}`",
        f"- candidate mean/min tok/s: `{s['candidate_mean_tps']:.4f}` / `{s['candidate_min_tps']:.4f}`",
        f"- mean token-rate delta: `{s['mean_tps_delta_pct']:.2f}%`",
        f"- max TTFT delta: `{s['max_ttft_delta_pct']:.2f}%`",
        f"- common quality/RAM/direct-read gate: `{s['common_gate_pass']}`",
        f"- TTFT gate: `{s['ttft_gate_pass']}`",
        f"- token-rate gate: `{s['rate_gate_pass']}`",
        f"- wait-only suspicion: `{s['wait_only_suspicion']}`",
        f"- acceptance safe: `{s['acceptance_safe']}`",
        "",
    ]
    if s["suspicious_wait_only_prompts"]:
        lines.extend([
            "Suspicious prompts:",
            "",
            *[f"- `{prompt}`" for prompt in s["suspicious_wait_only_prompts"]],
            "",
        ])
    lines.extend([
        "## Prompt Comparison",
        "",
        "| prompt | quality | base tok/s | cand tok/s | tok/s delta | TTFT delta | bytes delta | wait delta | up hit delta | down hit delta | same bytes/hits | direct reads |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in payload["comparison"]:
        c = row["candidate"]
        lines.append(
            f"| `{row['prompt_id']}` | {c['quality']} | "
            f"{row['base']['token_rate']:.3f} | {c['token_rate']:.3f} | "
            f"{row['token_rate_delta_pct']:.2f}% | {row['ttft_delta_pct']:.2f}% | "
            f"{row['bytes_delta_pct']:.2f}% | {row['wait_delta_pct']:.2f}% | "
            f"{row['upgate_hit_delta_pp']:.2f} pp | {row['down_hit_delta_pp']:.2f} pp | "
            f"{row['same_bytes_and_hits']} | {c['direct_reads']} |"
        )

    if "reference_summary" in payload:
        rs = payload["reference_summary"]
        lines.extend([
            "",
            "## Reference Comparison",
            "",
            f"- reference: `{payload['reference_label']}` -> `{payload['reference_root']}`",
            f"- candidate mean/min tok/s vs reference: `{rs['candidate_mean_tps']:.4f}` / `{rs['candidate_min_tps']:.4f}`",
            f"- mean token-rate delta vs reference: `{rs['mean_tps_delta_pct']:.2f}%`",
            f"- max TTFT delta vs reference: `{rs['max_ttft_delta_pct']:.2f}%`",
            f"- wait-only suspicion vs reference: `{rs['wait_only_suspicion']}`",
            f"- acceptance safe vs reference: `{rs['acceptance_safe']}`",
            "",
        ])

    lines.extend([
        "## Interpretation Rule",
        "",
        "- A candidate that improves token rate only because `iouring_wait` drops while moved bytes and hit rates stay unchanged is not accepted as a runtime/model optimization.",
        "- Such a result must be reproduced under a stronger cold-device protocol before it can replace the accepted SOTA.",
        "- This audit does not replace semantic quality review; it only checks metric consistency and cold-start risk.",
        "",
        "## Reproduce",
        "",
        "```bash",
        payload["reproduce_command"],
        "```",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit Kimi SOTA candidates for warmed-IO artifacts.")
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--reference-label", default="reference")
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    args = parser.parse_args(argv)
    args.argv = ["kimi_cold_start_sota_audit.py"] + (argv if argv is not None else sys.argv[1:])
    payload = build_payload(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.out_md, payload)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
