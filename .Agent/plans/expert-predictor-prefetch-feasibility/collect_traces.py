#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


TASK_DIR = Path(__file__).resolve().parent
REPO = TASK_DIR.parents[2]
RUNNER = REPO / ".Agent/plans/5090-theoretical-token-rate/run_5090_matrix.py"


def load_prompts(path: Path) -> dict[str, list[dict[str, str]]]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_prompt(args: argparse.Namespace, split: str, item: dict[str, str]) -> dict[str, object]:
    trace_dir = TASK_DIR / "traces" / split
    profile_dir = TASK_DIR / "replay" / "profiles" / split
    run_dir = TASK_DIR / "runs" / "collect" / split
    trace_dir.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    label = item["id"]
    trace_path = trace_dir / f"{label}.route.csv"
    profile_path = profile_dir / f"{label}.profile.csv"
    cmd = [
        sys.executable,
        str(RUNNER),
        "--run-dir", str(run_dir),
        "--label", label,
        "--profile", "top8",
        "--memory-max", "2G",
        "--memory-swap-max", "0",
        "--ram-tier-mib", "0",
        "--vram-cache-mib", str(args.vram_cache_mib),
        "--vram-upgate-pct", str(args.vram_upgate_pct),
        "--profile-reserve-pct", str(args.profile_reserve_pct),
        "--backend", args.backend,
        "--predict-tokens", str(args.predict_tokens),
        "--route-trace-out", str(trace_path),
        "--route-profile-out", str(profile_path),
        "--prompt", item["prompt"],
    ]
    if args.iouring_sort_offset:
        cmd.append("--iouring-sort-offset")
    if args.iouring_sqpoll:
        cmd.append("--iouring-sqpoll")
    if args.cuda_disable_graphs:
        cmd.append("--cuda-disable-graphs")
    if args.cache_auto_clamp:
        cmd.extend(["--cache-auto-clamp", "--cache-safety-mib", str(args.cache_safety_mib)])

    proc = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True)
    summary_path = run_dir / label / "summary.json"
    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "id": label,
        "split": split,
        "prompt": item["prompt"],
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "trace": str(trace_path),
        "profile": str(profile_path),
        "summary": str(summary_path),
        "eval_tokens_per_s": summary.get("eval_tokens_per_s"),
        "prompt_eval_tokens_per_s": summary.get("prompt_eval_tokens_per_s"),
        "total_ms": summary.get("total_ms"),
        "eval_ms": summary.get("eval_ms"),
        "vram_cache_hit_rate_pct": summary.get("vram_cache_hit_rate_pct"),
        "expert_pack_direct_reads": summary.get("expert_pack_direct_reads"),
        "expert_pack_iouring_reads": summary.get("expert_pack_iouring_reads"),
        "expert_pack_read_failures": summary.get("expert_pack_read_failures"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=TASK_DIR / "prompts.json")
    parser.add_argument("--splits", nargs="+", choices=["train", "test"], default=["train", "test"])
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-test", type=int)
    parser.add_argument("--predict-tokens", type=int, default=8)
    parser.add_argument("--backend", choices=["direct", "iouring"], default="iouring")
    parser.add_argument("--vram-cache-mib", type=int, default=12288)
    parser.add_argument("--vram-upgate-pct", type=int, default=60)
    parser.add_argument("--profile-reserve-pct", type=int, default=10)
    parser.add_argument("--iouring-sort-offset", action="store_true")
    parser.add_argument("--iouring-sqpoll", action="store_true")
    parser.add_argument("--cuda-disable-graphs", action="store_true")
    parser.add_argument("--cache-auto-clamp", action="store_true")
    parser.add_argument("--cache-safety-mib", type=int, default=256)
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    records: list[dict[str, object]] = []
    for split in args.splits:
        items = prompts[split]
        limit = args.limit_train if split == "train" else args.limit_test
        if limit is not None:
            items = items[:limit]
        for item in items:
            record = run_prompt(args, split, item)
            records.append(record)
            print(json.dumps(record, ensure_ascii=False, sort_keys=True))
            if record["returncode"] != 0:
                break

    notes_dir = TASK_DIR / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = notes_dir / "trace_collection_manifest.json"
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    manifest = {
        "config": config,
        "records": records,
        "ok": all(row["returncode"] == 0 for row in records),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if manifest["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
