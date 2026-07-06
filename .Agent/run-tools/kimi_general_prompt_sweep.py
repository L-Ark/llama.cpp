#!/usr/bin/env python3
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time


def load_prompts(path: pathlib.Path):
    prompts = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("id", "prompt", "quality_keywords"):
            if key not in row:
                raise SystemExit(f"{path}:{lineno}: missing {key}")
        prompts.append(row)
    return prompts


def write_sweep_metadata(out_root: pathlib.Path, prompt_file: pathlib.Path, prompts, args):
    out_root.mkdir(parents=True, exist_ok=True)
    meta = {
        "prompt_file": str(prompt_file),
        "prompt_count": len(prompts),
        "mode": args.mode,
        "n": args.n,
        "profile": args.profile,
        "memory_max": args.memory_max,
        "memory_swap_max": 0,
        "cold_start": "per prompt: repro script sync + drop_caches inside each systemd cgroup",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_root / "sweep.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (out_root / "prompts.jsonl").open("w", encoding="utf-8") as f:
        for row in prompts:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run_one(repo: pathlib.Path, out_root: pathlib.Path, row, args):
    run_dir = out_root / row["id"]
    cmd = [
        "systemd-run",
        "--wait",
        "--collect",
        "--same-dir",
        "-p",
        f"MemoryMax={args.memory_max}",
        "-p",
        "MemorySwapMax=0",
        "-p",
        "IOAccounting=yes",
        "-p",
        "IOWeight=10000",
        "-p",
        "CPUWeight=10000",
        "-p",
        "Nice=-10",
        "-p",
        "IOSchedulingClass=realtime",
        "-p",
        "IOSchedulingPriority=0",
        "env",
        f"RUN={run_dir}",
        f"N={args.n}",
        f"VRAM_MIB={args.vram_mib}",
        f"THREADS={args.threads}",
        f"PINNED_SLOTS={args.pinned_slots}",
        f"UPGATE_PCT={args.upgate_pct}",
        f"PROFILE={1 if args.profile else 0}",
        f"PROMPT_ID={row['id']}",
        f"PROMPT_USER_TEXT={row['prompt']}",
        f"QUALITY_KEYWORDS={row['quality_keywords']}",
        ".Agent/run-tools/kimi-general-prompt-repro.sh",
    ]
    print(f"=== {row['id']} ===", flush=True)
    print(" ".join(subprocess.list2cmdline([part]) for part in cmd), flush=True)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out_root / f"{row['id']}.systemd.txt").write_text(proc.stdout, encoding="utf-8", errors="replace")
    if proc.returncode != 0 and not args.keep_going:
        raise SystemExit(proc.returncode)
    return proc.returncode


def summarize(out_root: pathlib.Path):
    rows = []
    for metrics_path in sorted(out_root.glob("*/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(metrics)
    if not rows:
        return
    summary = out_root / "summary.md"
    with summary.open("w", encoding="utf-8") as f:
        f.write("# Kimi general prompt dev baseline\n\n")
        f.write("| prompt | quality | tok/s | TTFT ms | decode ms/runs | memory peak GiB | output |\n")
        f.write("|---|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            mem = row.get("memory.peak")
            mem_gib = ""
            if isinstance(mem, int):
                mem_gib = f"{mem / 1024**3:.2f}"
            output = str(row.get("output", "")).replace("|", "\\|")[:180]
            f.write(
                f"| {row.get('prompt_id','')} | {row.get('quality','')} | {row.get('token_rate','')} | "
                f"{row.get('ttft_ms','')} | {row.get('decode_ms','')}/{row.get('decode_runs','')} | "
                f"{mem_gib} | {output} |\n"
            )
    print(summary)


def main():
    parser = argparse.ArgumentParser(description="Run Kimi general-prompt cold-start sweep from a JSONL prompt file.")
    parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path("/root/lfz/llama.cpp-vendor-kimi"))
    parser.add_argument("--prompt-file", type=pathlib.Path, required=True)
    parser.add_argument("--out-root", type=pathlib.Path, required=True)
    parser.add_argument("--mode", choices=("dev", "test"), default="dev")
    parser.add_argument("--n", type=int, default=96)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--memory-max", default="15900000000")
    parser.add_argument("--vram-mib", default="15000")
    parser.add_argument("--threads", default="32")
    parser.add_argument("--pinned-slots", default="12")
    parser.add_argument("--upgate-pct", default="62")
    args = parser.parse_args()

    if args.mode == "dev" and "test" in args.prompt_file.name:
        raise SystemExit("refusing to use a test prompt file in dev mode")
    if args.mode == "test":
        print("WARNING: held-out test mode; use only after candidate freeze", file=sys.stderr)

    prompts = load_prompts(args.prompt_file)
    if args.max_prompts:
        prompts = prompts[: args.max_prompts]
    write_sweep_metadata(args.out_root, args.prompt_file, prompts, args)

    failures = 0
    for row in prompts:
        failures += 1 if run_one(args.repo, args.out_root, row, args) != 0 else 0
    summarize(args.out_root)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
