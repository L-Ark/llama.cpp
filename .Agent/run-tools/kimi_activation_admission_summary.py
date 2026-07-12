#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path


def main() -> int:
    out = Path(".Agent/runs/20260712-activation-aware-next-admission")
    files = sorted(out.glob("dev_*-aw-mse-smoke48.json"))
    if not files:
        raise SystemExit(f"no smoke json files under {out}")

    rows = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        prompt = path.name.replace("-aw-mse-smoke48.json", "")
        matvec = [
            {"prompt": prompt, "kind": "matvec", "candidate": candidate, **row}
            for candidate, row in data["aggregate"].items()
        ]
        fused = [
            {"prompt": prompt, "kind": "fused", "candidate": candidate, **row}
            for candidate, row in data["fused_up_gate"].items()
        ]
        for role in ("down", "gate", "up"):
            role_rows = [row for row in matvec if row["candidate"].startswith(role + ":")]
            under_040 = [row for row in role_rows if row["mean_byte_ratio"] <= 0.40]
            under_050 = [row for row in role_rows if row["mean_byte_ratio"] <= 0.50]
            rows.append(
                {
                    "prompt": prompt,
                    "role": role,
                    "best_under_040": min(under_040, key=lambda row: row["mean_rel_l2"], default=None),
                    "best_under_050": min(under_050, key=lambda row: row["mean_rel_l2"], default=None),
                }
            )
        for role, role_rows in (("fused_up_gate", fused),):
            under_040 = [row for row in role_rows if row["mean_byte_ratio"] <= 0.40]
            under_050 = [row for row in role_rows if row["mean_byte_ratio"] <= 0.50]
            rows.append(
                {
                    "prompt": prompt,
                    "role": role,
                    "best_under_040": min(under_040, key=lambda row: row["mean_rel_l2"], default=None),
                    "best_under_050": min(under_050, key=lambda row: row["mean_rel_l2"], default=None),
                }
            )

    summary = []
    for role in ("down", "gate", "up", "fused_up_gate"):
        role_rows = [row for row in rows if row["role"] == role]
        for target, key in (("0.40x", "best_under_040"), ("0.50x", "best_under_050")):
            values = [row[key]["mean_rel_l2"] for row in role_rows if row[key]]
            ratios = [row[key]["mean_byte_ratio"] for row in role_rows if row[key]]
            summary.append(
                {
                    "role": role,
                    "target": target,
                    "prompts": len(values),
                    "mean_best_rel_l2": statistics.mean(values) if values else None,
                    "max_best_rel_l2": max(values) if values else None,
                    "mean_ratio": statistics.mean(ratios) if ratios else None,
                    "pass_all": bool(values) and max(values) <= 0.10,
                }
            )

    result = {
        "kind": "kimi_activation_aware_next_admission_summary",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scope": "dev-only non-destructive smoke; no runtime change; no SOTA claim",
        "prompts": [path.name.replace("-aw-mse-smoke48.json", "") for path in files],
        "inputs": [str(path) for path in files],
        "screen_config": "bits=1,2 blocks=256 scale=aw_mse keep_input=0.05,0.10 max_records=48",
        "gate": {
            "target_byte_ratio_primary": 0.40,
            "target_byte_ratio_2tps_warning": 0.50,
            "target_mean_rel_l2": 0.10,
        },
        "rows": rows,
        "summary": summary,
        "decision": (
            "reject activation-aware AW-MSE/input-correction as next primary runtime path: "
            "no role passes rel-L2<=0.10 at <=0.40x or <=0.50x across dev prompts; "
            "fused up/gate remains far above the quality gate."
        ),
    }

    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Kimi activation-aware next admission",
        "",
        f"generated_at: `{result['generated_at']}`",
        "",
        result["scope"],
        "",
        f"screen config: `{result['screen_config']}`",
        "",
        "## Gate",
        "",
        "- primary compressed-runtime target: `<=0.40x` moved bytes and `mean rel-L2 <=0.10`",
        "- short-term 2 tok/s warning target: `<=0.50x`; still requires `mean rel-L2 <=0.10` before runtime work",
        "- prompts: `" + "`, `".join(result["prompts"]) + "`",
        "",
        "## Summary",
        "",
        "| role | byte target | prompts | mean best rel-L2 | max best rel-L2 | mean ratio | pass all |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in summary:
        lines.append(
            f"| `{item['role']}` | `{item['target']}` | `{item['prompts']}` | "
            f"`{item['mean_best_rel_l2']:.6f}` | `{item['max_best_rel_l2']:.6f}` | "
            f"`{item['mean_ratio']:.4f}` | `{item['pass_all']}` |"
        )

    lines += [
        "",
        "## Per-Prompt Best Candidates",
        "",
        "| prompt | role | target | candidate | ratio | mean rel-L2 | max rel-L2 |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        for target, key in (("0.40x", "best_under_040"), ("0.50x", "best_under_050")):
            best = row[key]
            if best is None:
                lines.append(f"| `{row['prompt']}` | `{row['role']}` | `{target}` | none | - | - | - |")
            else:
                lines.append(
                    f"| `{row['prompt']}` | `{row['role']}` | `{target}` | `{best['candidate']}` | "
                    f"`{best['mean_byte_ratio']:.4f}` | `{best['mean_rel_l2']:.6f}` | `{best['max_rel_l2']:.6f}` |"
                )

    lines += [
        "",
        "## Decision",
        "",
        result["decision"],
        "",
        "This reinforces earlier rejections of exact input-channel keep, partial exact contribution, "
        "output subspace, joint intermediate keep/scalar, and cluster-base residual candidates. "
        "The next runtime implementation should not be an activation-aware AW-MSE/input-correction compressed expert path.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "cd /root/lfz/llama.cpp-vendor-kimi",
        "OUT=.Agent/runs/20260712-activation-aware-next-admission",
        "for P in dev_japan_factual dev_mixed_summary dev_python_reverse; do",
        "  ROOT=/root/lfz/runs/vendor-kimi-token-rate/20260708-gp88-callstride-activation-corpus/$P",
        "  timeout 600 python3 .Agent/run-tools/kimi_activation_output_compression_screen.py \\",
        "    --activation-csv \"$ROOT/act/activations.csv\" \\",
        "    --activation-bin \"$ROOT/act/activations.f32\" \\",
        "    --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv \\",
        "    --libggml-base build-cuda-batch/bin/libggml-base.so \\",
        "    --out-json \"$OUT/$P-aw-mse-smoke48.json\" \\",
        "    --out-md \"$OUT/$P-aw-mse-smoke48.md\" \\",
        "    --bits 1,2 --blocks 256 --scale-modes aw_mse \\",
        "    --keep-input-fracs 0.05,0.10 --max-records 48 --torch-threads 8",
        "done",
        "python3 .Agent/run-tools/kimi_activation_admission_summary.py",
        "```",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(out / "summary.json")
    print(out / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
