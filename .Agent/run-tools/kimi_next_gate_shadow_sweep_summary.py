#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path


def run_analyzer(analyzer: Path, run_dir: Path, expert_bytes: int):
    csv_path = run_dir / "next-gate-shadow.csv"
    if not csv_path.exists():
        return None
    json_out = run_dir / "next-gate-shadow-summary.json"
    md_out = run_dir / "next-gate-shadow-report.md"
    subprocess.run(
        [
            "python3",
            str(analyzer),
            str(csv_path),
            "--expert-bytes",
            str(expert_bytes),
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ],
        check=True,
    )
    return json.loads(json_out.read_text(encoding="utf-8"))


def load_metrics(run_dir: Path):
    path = run_dir / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def summarize_root(root: Path, analyzer: Path, expert_bytes: int, label: str):
    rows = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        data = run_analyzer(analyzer, run_dir, expert_bytes)
        if data is None:
            rows.append({"prompt": run_dir.name, "missing_csv": True})
            continue
        metrics = load_metrics(run_dir)
        s = data["summary"]
        rows.append({
            "prompt": run_dir.name,
            "quality": metrics.get("quality"),
            "token_rate": metrics.get("token_rate"),
            "ttft_ms": metrics.get("ttft_ms"),
            "decode_ms": metrics.get("decode_ms"),
            "decode_runs": metrics.get("decode_runs"),
            "memory_peak": metrics.get("memory.peak"),
            "output": metrics.get("output"),
            "pairs": s["pairs"],
            "actual_experts": s["actual_experts"],
            "hit_experts": s["hit_experts"],
            "predicted_experts": s["predicted_experts"],
            "false_experts": s["false_experts"],
            "false_prefetch_bytes": s["false_prefetch_bytes"],
            "expert_recall": s["expert_recall"],
            "expert_precision": s["expert_precision"],
            "byte_recall": s["byte_recall"],
            "false_over_actual_bytes": s["false_over_actual_bytes"],
            "record_us_avg": s["record_us_avg"],
            "record_us_max": s["record_us_max"],
        })

    valid = [r for r in rows if not r.get("missing_csv")]
    actual = sum(r["actual_experts"] for r in valid)
    hit = sum(r["hit_experts"] for r in valid)
    predicted = sum(r["predicted_experts"] for r in valid)
    false = sum(r["false_experts"] for r in valid)

    aggregate = {
        "label": label,
        "root": str(root),
        "prompt_count": len(rows),
        "valid_prompt_count": len(valid),
        "missing_csv": [r["prompt"] for r in rows if r.get("missing_csv")],
        "quality_failures": [
            r["prompt"] for r in valid if r.get("quality") != "pass"
        ],
        "pairs": sum(r["pairs"] for r in valid),
        "expert_recall": hit / actual if actual else 0.0,
        "expert_precision": hit / predicted if predicted else 0.0,
        "byte_recall": hit / actual if actual else 0.0,
        "false_over_actual_bytes": false / actual if actual else 0.0,
        "false_prefetch_bytes_units": sum(r["false_prefetch_bytes"] for r in valid),
        "record_us_max": max((r["record_us_max"] for r in valid), default=0),
        "record_us_avg_prompt_mean": (
            sum(r["record_us_avg"] for r in valid) / len(valid) if valid else 0.0
        ),
        "avg_token_rate": (
            sum(float(r.get("token_rate") or 0.0) for r in valid) / len(valid)
            if valid else 0.0
        ),
        "max_memory_peak": max(
            (int(r.get("memory_peak") or 0) for r in valid), default=0
        ),
    }
    return {"aggregate": aggregate, "prompts": rows}


def write_markdown(results, out_path: Path, prompt_file: str, n_predict: int):
    lines = [
        "# Kimi next-gate shadow sweep summary",
        "",
        f"- Prompt file: `{prompt_file}`",
        "- Mode: dev-only unless explicitly stated by the caller",
        f"- N: `{n_predict}`",
        "",
        "| label | prompts | recall | precision | false/actual | quality failures | max record us | avg tok/s | max memory bytes |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for result in results:
        a = result["aggregate"]
        lines.append(
            "| {label} | {valid_prompt_count}/{prompt_count} | {recall:.2f}% | "
            "{precision:.2f}% | {false:.4f} | {quality} | {record} | "
            "{tok:.3f} | {mem} |".format(
                label=a["label"],
                valid_prompt_count=a["valid_prompt_count"],
                prompt_count=a["prompt_count"],
                recall=a["expert_recall"] * 100.0,
                precision=a["expert_precision"] * 100.0,
                false=a["false_over_actual_bytes"],
                quality=",".join(a["quality_failures"]) or "none",
                record=a["record_us_max"],
                tok=a["avg_token_rate"],
                mem=a["max_memory_peak"],
            )
        )

    for result in results:
        a = result["aggregate"]
        lines += [
            "",
            f"## {a['label']}",
            "",
            f"- Root: `{a['root']}`",
            f"- Missing CSV: `{a['missing_csv']}`",
            f"- Matched pairs: `{a['pairs']}`",
            f"- False-prefetch byte units: `{a['false_prefetch_bytes_units']}`",
            "",
            "| prompt | quality | tok/s | pairs | recall | precision | false/actual | record avg us | output |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for row in result["prompts"]:
            if row.get("missing_csv"):
                lines.append(f"| {row['prompt']} | missing_csv | | | | | | | |")
                continue
            output = str(row.get("output") or "").replace("|", "\\|")[:100]
            lines.append(
                "| {prompt} | {quality} | {tok:.2f} | {pairs} | {recall:.2f}% | "
                "{precision:.2f}% | {false:.4f} | {record:.1f} | {output} |".format(
                    prompt=row["prompt"],
                    quality=row.get("quality"),
                    tok=float(row.get("token_rate") or 0.0),
                    pairs=row["pairs"],
                    recall=row["expert_recall"] * 100.0,
                    precision=row["expert_precision"] * 100.0,
                    false=row["false_over_actual_bytes"],
                    record=row["record_us_avg"],
                    output=output,
                )
            )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Summarize one or more Kimi next-gate shadow sweep roots.")
    ap.add_argument("--root", action="append", type=Path, required=True,
                    help="Sweep root containing one subdirectory per prompt. May be repeated.")
    ap.add_argument("--label", action="append", required=True,
                    help="Label for the corresponding --root. May be repeated.")
    ap.add_argument("--analyzer", type=Path,
                    default=Path(".Agent/run-tools/kimi_next_gate_shadow_analyze.py"))
    ap.add_argument("--expert-bytes", type=int, default=1)
    ap.add_argument("--prompt-file", default="")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--json-out", type=Path, required=True)
    ap.add_argument("--md-out", type=Path, required=True)
    args = ap.parse_args()

    if len(args.root) != len(args.label):
        raise SystemExit("--root and --label counts must match")

    results = [
        summarize_root(root, args.analyzer, args.expert_bytes, label)
        for root, label in zip(args.root, args.label)
    ]
    args.json_out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(results, args.md_out, args.prompt_file, args.n)


if __name__ == "__main__":
    main()
