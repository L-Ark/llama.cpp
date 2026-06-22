#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
MODEL = REPO / "models/GLM-5.1-UD-IQ3_XXS/GLM-5.1-UD-IQ3_XXS-00001-of-00007.gguf"
PACK = REPO / "models/GLM-5.1-UD-IQ3_XXS/glm51-iq3xxs.expert-pack"
PROFILE = REPO / "presets/moe/groundtruth/wici-glm51-interactive-n84-expert-top8.runtime.csv"
BIN = REPO / "build-cuda/bin/llama-cli"

TIMING_RE = re.compile(
    r"llama_print_timings:\s+(?P<name>load|prompt eval|eval|total) time =\s+"
    r"(?P<ms>[0-9.]+) ms(?: /(?:\s+(?P<tokens>[0-9]+) (?:tokens|runs))?"
    r".*?(?P<tps>[0-9.]+|-nan|inf) tokens per second)?"
)
PACK_RE = re.compile(
    r"\[moe_stream_batch\] expert pack: hits=(?P<hits>\d+) misses=(?P<misses>\d+) "
    r"read_failures=(?P<read_failures>\d+) direct_reads=(?P<direct_reads>\d+) "
    r"direct_fallbacks=(?P<direct_fallbacks>\d+)"
)
VCACHE_RE = re.compile(
    r"\[moe_stream_batch\] VRAM cache: hits=(?P<hits>\d+) misses=(?P<misses>\d+) "
    r"preloads=(?P<preloads>\d+) pinned=(?P<pinned>\d+) hit_rate=(?P<hit_rate>[0-9.]+)%"
)


def load_samples(dataset: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]


def first_choice_letter(text: str) -> str:
    if "答案：" in text:
        text = text.rsplit("答案：", 1)[1]
    elif "Answer:" in text:
        text = text.rsplit("Answer:", 1)[1]
    match = re.search(r"\b([ABCD])\b", text.strip(), re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return text.strip()[:1].upper()


def add_metrics(record: dict[str, Any], stderr_text: str) -> None:
    for match in TIMING_RE.finditer(stderr_text):
        key = match.group("name").replace(" ", "_")
        record[f"{key}_ms"] = float(match.group("ms"))
        if match.group("tokens"):
            record[f"{key}_tokens"] = int(match.group("tokens"))
        tps = match.group("tps")
        if tps and tps not in {"-nan", "inf"}:
            record[f"{key}_tokens_per_s"] = float(tps)
    pack_matches = list(PACK_RE.finditer(stderr_text))
    if pack_matches:
        for key, value in pack_matches[-1].groupdict().items():
            record[f"expert_pack_{key}"] = int(value)
    cache_matches = list(VCACHE_RE.finditer(stderr_text))
    if cache_matches:
        cache = cache_matches[-1]
        record["vram_cache_hits"] = int(cache.group("hits"))
        record["vram_cache_misses"] = int(cache.group("misses"))
        record["vram_cache_hit_rate_pct"] = float(cache.group("hit_rate"))
    record["ram_hit_rate_pct"] = 0.0


def inner_command(args: argparse.Namespace, sample: dict[str, Any], run_dir: Path, stdout_path: Path, stderr_path: Path) -> list[str]:
    trace_path = run_dir / f"{args.label}.{sample['id']}.route.csv" if args.route_trace else None
    env = [
        "env",
        f"LLAMA_CACHE={run_dir / 'llama-cache'}",
        "MLA=3",
        "CPU_MOE=1",
        "IGNORE_EOS=1",
        "GGML_CUDA_NO_PINNED=1",
        "GGML_MOE_STREAM=1",
        "GGML_MOE_STREAM_BATCH_ONLY=1",
        "GGML_MOE_STREAM_DEFER=1",
        "GGML_MOE_STREAM_CPU_OPS=1",
        "GGML_MOE_PARALLEL_EXPERTS=1",
        "GGML_MOE_STREAM_FUSED_UP_GATE=1",
        "GGML_MOE_VRAM_CACHE_SPLIT=1",
        "GGML_MOE_STREAM_ONE_CACHE_MIB=0",
        "GGML_MOE_VRAM_CACHE_POLICY=lfu_lru",
        "GGML_MOE_STAGE_PINNED_SLOTS=16",
        "GGML_MOE_PREFETCH_DOWN=0",
        "GGML_MOE_PREFETCH_DOWN_DEPTH=8",
        f"GGML_MOE_EXPERT_PACK={PACK}",
        "GGML_MOE_RAM_TIER_MIB=0",
        "GGML_MOE_RAM_TIER_SKIP=0",
        "GGML_MOE_VRAM_CACHE_MIB=12288",
        "GGML_MOE_VRAM_CACHE_UPGATE_PCT=60",
        f"GGML_MOE_VRAM_PROFILE={PROFILE}",
        "GGML_MOE_VRAM_PROFILE_PROTECT=1",
        "GGML_MOE_VRAM_PROFILE_RESERVE_PCT=10",
        "GGML_MOE_IO_BACKEND=direct",
        "GGML_MOE_STAGE_PINNED=0",
    ]
    if trace_path is not None:
        env.append(f"GGML_MOE_ROUTE_TRACE_OUT={trace_path}")
    cmd = [
        *env,
        str(BIN),
        "-m", str(MODEL),
        "-c", "2048",
        "-ngl", "79",
        "-fa", "on",
        "-mla", "3",
        "-cmoe",
        "--defer-experts",
        "--no-warmup",
        "-b", "2048",
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
    return cmd


def run_one(args: argparse.Namespace, sample: dict[str, Any], index: int, run_dir: Path) -> dict[str, Any]:
    stdout_path = run_dir / f"{args.label}.{index:02d}.{sample['id']}.stdout.txt"
    stderr_path = run_dir / f"{args.label}.{index:02d}.{sample['id']}.stderr.txt"
    cmd = inner_command(args, sample, run_dir, stdout_path, stderr_path)
    shell = f"cd {shlex.quote(str(REPO))} && exec {shlex.join(cmd)} >{shlex.quote(str(stdout_path))} 2>{shlex.quote(str(stderr_path))}"
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
    stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.is_file() else ""
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.is_file() else ""
    scored_output = first_choice_letter(stdout_text)
    record: dict[str, Any] = {
        "sample_id": sample["id"],
        "expected": sample["expected"],
        "scored_output": scored_output,
        "correct": scored_output == sample["expected"],
        "returncode": proc.returncode,
        "systemd_stdout": proc.stdout,
        "systemd_stderr": proc.stderr,
        "wall_s": wall_s,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "command": shlex.join(systemd_cmd),
        "ser": args.ser,
    }
    add_metrics(record, stderr_text)
    trace_path = run_dir / f"{args.label}.{sample['id']}.route.csv"
    if trace_path.exists():
        record["route_trace"] = str(trace_path)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--ser")
    parser.add_argument("--predict-tokens", type=int, default=4)
    parser.add_argument("--route-trace", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.dataset)
    records = [run_one(args, sample, index, run_dir) for index, sample in enumerate(samples, start=1)]
    predictions = [
        {"id": row["sample_id"], "output": row["scored_output"], "expected": row["expected"], "correct": row["correct"]}
        for row in records
    ]
    correct = sum(1 for row in records if row["correct"])
    summary: dict[str, Any] = {
        "label": args.label,
        "ser": args.ser,
        "dataset": str(args.dataset),
        "correct": correct,
        "scored": len(records),
        "accuracy": correct / len(records) if records else None,
        "predictions": predictions,
        "samples": records,
        "aggregate": {
            "wall_s": sum(float(row.get("wall_s") or 0.0) for row in records),
            "direct_reads": sum(int(row.get("expert_pack_direct_reads") or 0) for row in records),
            "read_failures": sum(int(row.get("expert_pack_read_failures") or 0) for row in records),
            "total_ms": sum(float(row.get("total_ms") or 0.0) for row in records),
            "prompt_eval_tokens": sum(int(row.get("prompt_eval_tokens") or 0) for row in records),
            "prompt_eval_ms": sum(float(row.get("prompt_eval_ms") or 0.0) for row in records),
            "eval_tokens": sum(int(row.get("eval_tokens") or 0) for row in records),
            "eval_ms": sum(float(row.get("eval_ms") or 0.0) for row in records),
            "ram_hit_rate_pct": 0.0,
        },
    }
    agg = summary["aggregate"]
    if agg["prompt_eval_ms"]:
        agg["prompt_eval_tokens_per_s"] = 1000.0 * agg["prompt_eval_tokens"] / agg["prompt_eval_ms"]
    if agg["eval_ms"]:
        agg["eval_tokens_per_s"] = 1000.0 * agg["eval_tokens"] / agg["eval_ms"]
    if records:
        agg["vram_cache_hit_rate_pct_avg"] = sum(float(row.get("vram_cache_hit_rate_pct") or 0.0) for row in records) / len(records)
    summary_path = run_dir / f"{args.label}.summary.json"
    pred_path = run_dir / f"{args.label}.predictions.jsonl"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pred_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if all(row.get("returncode") == 0 for row in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
