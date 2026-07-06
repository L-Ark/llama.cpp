#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from kimi_make_v2_shadow_pack_from_routes import load_routes, summarize  # noqa: E402


def parse_int_list(text: str) -> list[int]:
    out = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"non-positive size: {value}")
        out.append(value)
    return sorted(set(out))


def parse_float_list(text: str) -> list[float]:
    out = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value <= 0.0 or value > 1.0:
            raise ValueError(f"ratio must be in (0, 1]: {value}")
        out.append(value)
    return sorted(set(out))


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "packed_ratio",
        "max_entries",
        "selected_entries",
        "events",
        "event_coverage",
        "byte_coverage",
        "hybrid_byte_ratio",
        "logical_gib",
        "hybrid_gib",
        "ideal_transfer_only_tps",
        "target_met_transfer_only",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# Kimi v2 low-byte hotset sweep",
        "",
        f"- Input profiles: `{len(report['input_profiles'])}`",
        f"- Candidate entries: `{report['candidate_entries']}`",
        f"- Baseline tok/s: `{report['baseline_tps']}`",
        f"- Target tok/s: `{report['target_tps']}`",
        f"- Required ideal byte ratio: `{report['required_ideal_byte_ratio']:.4f}`",
        "",
        "This is a static dev-route upper-bound sweep. It does not use held-out test prompts, real low-byte payloads, or runtime H2D.",
        "",
        "| packed ratio | max entries | selected | event cov | byte cov | hybrid byte ratio | ideal transfer-only tok/s | target met? |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['packed_ratio']:.3f} | {row['max_entries']} | {row['selected_entries']} | "
            f"{row['event_coverage']:.4f} | {row['byte_coverage']:.4f} | "
            f"{row['hybrid_byte_ratio']:.4f} | {row['ideal_transfer_only_tps']:.3f} | "
            f"{'yes' if row['target_met_transfer_only'] else 'no'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep selected v2 low-byte hotset sizes from dev route profiles.")
    parser.add_argument("--profile", type=Path, action="append", required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--sizes", default="512,1024,2048,4096,8192,16384,32768,56896")
    parser.add_argument("--ratios", default="0.55,0.35,0.276")
    parser.add_argument("--kind", default="up,gate,down")
    parser.add_argument("--baseline-tps", type=float, default=1.385)
    parser.add_argument("--target-tps", type=float, default=5.0)
    args = parser.parse_args()

    kinds = {item.strip() for item in args.kind.split(",") if item.strip()}
    sizes = parse_int_list(args.sizes)
    ratios = parse_float_list(args.ratios)
    entries = list(load_routes(args.profile, kinds).values())
    entries.sort(key=lambda item: (-item.logical_total, item.tensor, item.expert_idx))
    required_ratio = args.baseline_tps / args.target_tps if args.target_tps > 0 else 0.0

    rows = []
    for ratio in ratios:
        for size in sizes:
            selected_entries = entries[: min(size, len(entries))]
            selected = {(entry.tensor, entry.expert_idx) for entry in selected_entries}
            summary = summarize(entries, selected, ratio)["total"]
            hybrid_ratio = summary["hybrid_byte_ratio"]
            ideal_tps = args.baseline_tps / hybrid_ratio if hybrid_ratio > 0 else 0.0
            rows.append(
                {
                    "packed_ratio": ratio,
                    "max_entries": size,
                    "selected_entries": len(selected_entries),
                    "events": summary["events"],
                    "event_coverage": summary["event_coverage"],
                    "byte_coverage": summary["byte_coverage"],
                    "hybrid_byte_ratio": hybrid_ratio,
                    "logical_gib": summary["logical_bytes"] / 1024**3,
                    "hybrid_gib": summary["hybrid_bytes"] / 1024**3,
                    "ideal_transfer_only_tps": ideal_tps,
                    "target_met_transfer_only": ideal_tps >= args.target_tps,
                }
            )

    report = {
        "input_profiles": [str(path) for path in args.profile],
        "kinds": sorted(kinds),
        "candidate_entries": len(entries),
        "sizes": sizes,
        "ratios": ratios,
        "baseline_tps": args.baseline_tps,
        "target_tps": args.target_tps,
        "required_ideal_byte_ratio": required_ratio,
        "rows": rows,
    }

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_csv, rows)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.out_md, report)

    best = min(rows, key=lambda row: row["hybrid_byte_ratio"]) if rows else None
    print(f"candidate_entries={len(entries)}")
    print(f"rows={len(rows)}")
    if best:
        print(f"best_packed_ratio={best['packed_ratio']:.6f}")
        print(f"best_selected_entries={best['selected_entries']}")
        print(f"best_hybrid_byte_ratio={best['hybrid_byte_ratio']:.6f}")
        print(f"best_ideal_transfer_only_tps={best['ideal_transfer_only_tps']:.6f}")
        print(f"best_target_met_transfer_only={int(best['target_met_transfer_only'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
