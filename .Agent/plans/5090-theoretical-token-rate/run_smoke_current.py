#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import run_5090_matrix


REPO = Path(__file__).resolve().parents[3]
BIN = REPO / "build-cuda/bin/llama-cli"
MODEL = REPO / "models/GLM-5.1-UD-IQ3_XXS/GLM-5.1-UD-IQ3_XXS-00001-of-00007.gguf"


def load_samples(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def first_choice_letter(text: str) -> str:
    if "答案：" in text:
        text = text.rsplit("答案：", 1)[1]
    elif "Answer:" in text:
        text = text.rsplit("Answer:", 1)[1]
    match = re.search(r"\b([ABCD])\b", text.strip(), re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return text.strip()[:1].upper()


def env_args(args: argparse.Namespace, run_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        profile=args.profile,
        cache_policy="lfu_lru",
        cache_profile_after=12624,
        slots=args.slots,
        prefetch_down=False,
        prefetch_down_depth=8,
        ram_tier_mib=0,
        ram_tier_skip=0,
        vram_cache_mib=args.vram_cache_mib,
        vram_upgate_pct=args.vram_upgate_pct,
        profile_reserve_pct=10,
        backend="iouring",
        depth=args.depth,
        io_bytes=2097152,
        stage_pinned=False,
        up_gate_stage_split=False,
        iouring_sort_offset=True,
        iouring_sqpoll=False,
        gpu_handoff=True,
        prompt_dynamic_experts=True,
        prompt_startup_preload=True,
        prompt_ttft_markers=True,
        eager_cublas=True,
        decline_debug=False,
        cache_auto_clamp=True,
        cache_safety_mib=args.cache_safety_mib,
        profile_preload_evict=False,
        route_profile_out=None,
        route_trace_out=None,
        ttft_trace_out=None,
        trace_prefetch=None,
        trace_prefetch_window=96,
        trace_prefetch_max_loads=8,
        run_dir=run_dir,
    )


def run_one(args: argparse.Namespace, sample: dict[str, Any], index: int, run_dir: Path) -> dict[str, Any]:
    stdout_path = run_dir / f"{args.label}.{index:02d}.{sample['id']}.stdout.txt"
    stderr_path = run_dir / f"{args.label}.{index:02d}.{sample['id']}.stderr.txt"
    env = run_5090_matrix.env_list(env_args(args, run_dir), run_dir)
    cmd = [
        *env,
        str(BIN),
        "-m", str(MODEL),
        "-c", str(args.ctx),
        "-ngl", "79",
        "-fa", "on",
        "-mla", "3",
        "-cmoe",
        "--defer-experts",
        "--no-warmup",
        "-b", str(args.batch),
        "-n", str(args.predict_tokens),
        "--temp", "0",
        "--top-p", "1.0",
        "--min-p", "0.0",
        "--repeat-last-n", "512",
        "--repeat-penalty", "1.20",
        "--presence-penalty", "0.1",
        "--frequency-penalty", "0.2",
        "-t", "16",
        "-tb", "24",
        "--seed", "42",
        "-p", str(sample["prompt"]),
    ]
    if args.ser:
        cmd.extend(["--smart-expert-reduction", args.ser])

    shell = (
        f"cd {shlex.quote(str(REPO))} && exec {shlex.join(cmd)} "
        f">{shlex.quote(str(stdout_path))} 2>{shlex.quote(str(stderr_path))}"
    )
    systemd_cmd = [
        "systemd-run", "--user", "--wait", "--collect", "--quiet",
        "-p", f"WorkingDirectory={REPO}",
        "-p", "MemoryMax=2G",
        "-p", "MemorySwapMax=0",
        "bash", "-lc", shell,
    ]
    start = time.perf_counter()
    proc = subprocess.run(systemd_cmd, cwd=REPO, text=True, capture_output=True)
    wall_s = time.perf_counter() - start
    stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
    scored_output = first_choice_letter(stdout_text)
    record: dict[str, Any] = {
        "sample_id": sample["id"],
        "expected": sample["expected"],
        "scored_output": scored_output,
        "correct": scored_output == sample["expected"],
        "returncode": proc.returncode,
        "wall_s": wall_s,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "command": shlex.join(systemd_cmd),
        "ser": args.ser,
        "systemd_stdout": proc.stdout,
        "systemd_stderr": proc.stderr,
    }
    record.update(run_5090_matrix.parse_stderr(stderr_text))
    return record


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    agg: dict[str, Any] = {
        "wall_s": sum(float(row.get("wall_s") or 0.0) for row in records),
        "direct_reads": sum(int(row.get("expert_pack_direct_reads") or 0) for row in records),
        "iouring_reads": sum(int(row.get("expert_pack_iouring_reads") or 0) for row in records),
        "read_failures": sum(int(row.get("expert_pack_read_failures") or 0) for row in records),
        "total_ms": sum(float(row.get("total_ms") or 0.0) for row in records),
        "prompt_eval_tokens": sum(int(row.get("prompt_eval_tokens") or 0) for row in records),
        "prompt_eval_ms": sum(float(row.get("prompt_eval_ms") or 0.0) for row in records),
        "eval_tokens": sum(int(row.get("eval_tokens") or 0) for row in records),
        "eval_ms": sum(float(row.get("eval_ms") or 0.0) for row in records),
        "iouring_wait_us": sum(int(row.get("expert_pack_iouring_wait_us") or 0) for row in records),
        "iouring_submit_us": sum(int(row.get("expert_pack_iouring_submit_us") or 0) for row in records),
        "ram_hit_rate_pct": 0.0,
    }
    if agg["prompt_eval_ms"]:
        agg["prompt_eval_tokens_per_s"] = 1000.0 * agg["prompt_eval_tokens"] / agg["prompt_eval_ms"]
    if agg["eval_ms"]:
        agg["eval_tokens_per_s"] = 1000.0 * agg["eval_tokens"] / agg["eval_ms"]
    if records:
        agg["vram_cache_hit_rate_pct_avg"] = sum(
            float(row.get("vram_cache_hit_rate_pct") or 0.0) for row in records
        ) / len(records)
    return agg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--ser")
    parser.add_argument("--profile", default="top8")
    parser.add_argument("--vram-cache-mib", type=int, default=20480)
    parser.add_argument("--vram-upgate-pct", type=int, default=65)
    parser.add_argument("--cache-safety-mib", type=int, default=256)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--slots", type=int, default=16)
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--predict-tokens", type=int, default=4)
    args = parser.parse_args()

    run_dir = (args.run_dir / args.label).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.dataset)
    records = [run_one(args, sample, index, run_dir) for index, sample in enumerate(samples, start=1)]
    predictions = [
        {
            "id": row["sample_id"],
            "output": row["scored_output"],
            "expected": row["expected"],
            "correct": row["correct"],
        }
        for row in records
    ]
    correct = sum(1 for row in records if row["correct"])
    summary = {
        "label": args.label,
        "ser": args.ser,
        "dataset": str(args.dataset),
        "profile": args.profile,
        "strict_2gb": True,
        "ram_tier_mib": 0,
        "backend": "iouring",
        "iouring_sort_offset": True,
        "vram_cache_mib": args.vram_cache_mib,
        "vram_upgate_pct": args.vram_upgate_pct,
        "cache_safety_mib": args.cache_safety_mib,
        "correct": correct,
        "scored": len(records),
        "accuracy": correct / len(records) if records else None,
        "predictions": predictions,
        "samples": records,
        "aggregate": aggregate(records),
    }
    summary_path = run_dir / "summary.json"
    pred_path = run_dir / "predictions.jsonl"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pred_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if all(row.get("returncode") == 0 for row in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
