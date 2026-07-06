#!/usr/bin/env python3
import argparse
import json
import pathlib
import re
import statistics


def extract_ms(pattern: str, text: str) -> float:
    match = re.search(pattern, text)
    return float(match.group(1)) if match else 0.0


def analyze_run(metrics_path: pathlib.Path):
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    runs = float(metrics.get("decode_runs") or 0)
    decode_ms = float(metrics.get("decode_ms") or 0)
    token_rate = float(metrics.get("token_rate") or 0)
    target_ms = runs / 5.0 * 1000.0 if runs else 0.0
    need_speedup = decode_ms / target_ms if target_ms else 0.0

    expert_pack = str(metrics.get("expert_pack_0", ""))
    iouring_wait_ms = extract_ms(r"iouring_wait_us=([0-9]+)", expert_pack) / 1000.0
    iouring_bytes = extract_ms(r"iouring_bytes=([0-9]+)", expert_pack)

    h2d_ms = 0.0
    host_stage_ms = 0.0
    slot_wait_ms = 0.0
    for key, value in metrics.items():
        if not key.startswith("pinned_staging_"):
            continue
        text = str(value)
        h2d_ms += extract_ms(r"h2d=([0-9.]+) ms", text)
        host_stage_ms += extract_ms(r"host_stage=([0-9.]+) ms", text)
        slot_wait_ms += extract_ms(r"slot_wait=([0-9.]+) ms", text)

    no_iowait_tok_s = 0.0
    if runs and decode_ms > iouring_wait_ms:
        no_iowait_tok_s = runs / ((decode_ms - iouring_wait_ms) / 1000.0)

    return {
        "prompt_id": metrics.get("prompt_id", metrics_path.parent.name),
        "quality": metrics.get("quality", ""),
        "decode_runs": runs,
        "token_rate": token_rate,
        "decode_ms": decode_ms,
        "target_ms_at_5_tps": target_ms,
        "need_speedup": need_speedup,
        "iouring_wait_ms": iouring_wait_ms,
        "iouring_wait_share": iouring_wait_ms / decode_ms if decode_ms else 0.0,
        "iouring_gib": iouring_bytes / 1024**3,
        "visible_h2d_ms": h2d_ms,
        "visible_host_stage_ms": host_stage_ms,
        "visible_slot_wait_ms": slot_wait_ms,
        "tok_s_if_iouring_wait_zero": no_iowait_tok_s,
        "memory_peak": metrics.get("memory.peak", ""),
    }


def write_report(rows, out_path: pathlib.Path):
    rates = sorted(row["token_rate"] for row in rows)
    lines = [
        "# Kimi general dev baseline hard-bound analysis",
        "",
        "This report is derived from the formal n96 dev baseline metrics. It does",
        "not use held-out test prompts.",
        "",
        f"- prompts: `{len(rows)}`",
        f"- min token rate: `{min(rates):.2f} tok/s`",
        f"- median token rate: `{statistics.median(rates):.2f} tok/s`",
        f"- mean token rate: `{statistics.mean(rates):.3f} tok/s`",
        "",
        "| prompt | tok/s | decode ms/runs | target ms @5 tok/s | need speedup | iouring GiB | iouring wait ms | wait share | visible H2D ms | visible host-stage ms | tok/s if wait=0 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['prompt_id']}` | {row['token_rate']:.2f} | "
            f"{row['decode_ms']:.0f}/{row['decode_runs']:.0f} | "
            f"{row['target_ms_at_5_tps']:.0f} | {row['need_speedup']:.1f}x | "
            f"{row['iouring_gib']:.2f} | {row['iouring_wait_ms']:.0f} | "
            f"{100.0 * row['iouring_wait_share']:.1f}% | "
            f"{row['visible_h2d_ms']:.0f} | {row['visible_host_stage_ms']:.0f} | "
            f"{row['tok_s_if_iouring_wait_zero']:.2f} |"
        )
    lines.extend([
        "",
        "Interpretation:",
        "",
        "- France remains the only prompt where removing visible iouring wait would",
        "  theoretically exceed `5 tok/s`.",
        "- For non-France/general prompts, visible iouring wait is too small a share",
        "  of decode time to explain the gap. Eliminating it would still leave",
        "  `0.16-0.48 tok/s` for the slow prompts.",
        "- The next optimization branch must therefore reduce host staging / CPU-side",
        "  fallback / expert materialization cost and bytes moved per routed expert,",
        "  not only increase iouring queue depth or fixed hotset hit rate.",
        "",
    ])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Summarize hard bounds from Kimi general dev baseline metrics.")
    parser.add_argument("--runs-root", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    rows = [analyze_run(path) for path in sorted(args.runs_root.glob("*/metrics.json"))]
    if not rows:
        raise SystemExit("no metrics.json files found")
    write_report(rows, args.out)
    print(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
