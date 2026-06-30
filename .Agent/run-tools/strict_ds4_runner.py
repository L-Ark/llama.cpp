#!/usr/bin/env python3
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


REPO = Path("/root/lfz/vendor/llama.cpp-deepseek-v4")
RUN_ROOT = REPO / ".Agent/runs/20260630-ds4-token-rate"
MODEL = Path("/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf")
PROMPT = "Question: Please introduce France in a short paragraph.\n\nAnswer:"


def run_text(cmd, check=False):
    p = subprocess.run(cmd, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if check and p.returncode != 0:
        raise RuntimeError(p.stdout)
    return p.stdout


def parse_env(items):
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--env expects NAME=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        out[k] = v
    return out


def build_env(extra):
    env = {}
    for k, v in os.environ.items():
        if k.startswith(("CUDA_", "GGML_", "LLAMA_", "DS4_", "IK_LLAMA_")):
            env[k] = v
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("GGML_CUDA_NO_PINNED", "1")
    env.setdefault("GGML_MOE_VRAM_CACHE_MIB", "16384")
    env.setdefault("LLAMA_MMAP_LOW_RAM", "1")
    env.update(extra)
    return env


def build_command(n_predict, ngl):
    return [
        "build-a101-nvcc/bin/llama-completion",
        "-m", str(MODEL),
        "-p", PROMPT,
        "-ngl", str(ngl),
        "-c", "512",
        "-n", str(n_predict),
        "--ignore-eos",
        "--temp", "0",
        "--top-p", "1.0",
        "--top-k", "1",
        "--seed", "1",
        "--no-warmup",
        "--no-display-prompt",
        "-no-cnv",
        "-ub", "1",
        "-t", "24",
        "-tb", "24",
    ]


def read_int(path):
    try:
        return int(Path(path).read_text().strip())
    except Exception:
        return None


def read_stat(path):
    stat = {}
    try:
        for line in Path(path).read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                try:
                    stat[parts[0]] = int(parts[1])
                except ValueError:
                    pass
    except Exception:
        pass
    return stat


def gpu_sample():
    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,utilization.memory",
        "--format=csv,noheader,nounits",
    ]
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    rows = []
    if p.returncode == 0:
        for line in p.stdout.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 6:
                rows.append({
                    "timestamp": parts[0],
                    "index": parts[1],
                    "memory_used_mib": to_int(parts[2]),
                    "memory_total_mib": to_int(parts[3]),
                    "util_gpu_percent": to_int(parts[4]),
                    "util_memory_percent": to_int(parts[5]),
                })
    return rows


def to_int(s):
    try:
        return int(s)
    except Exception:
        return None


def find_cgroup(unit):
    candidates = [
        Path("/sys/fs/cgroup/system.slice") / f"{unit}.service",
        Path("/sys/fs/cgroup") / "system.slice" / f"{unit}.service",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def parse_timings(stderr):
    timings = {}
    patterns = {
        "prompt_eval": r"prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*tokens.*?([0-9.]+)\s*tokens per second",
        "eval": r"eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*(?:runs|tokens).*?([0-9.]+)\s*tokens per second",
        "total": r"total time\s*=\s*([0-9.]+)\s*ms",
    }
    for name, pat in patterns.items():
        m = re.search(pat, stderr, re.S)
        if m:
            if name == "total":
                timings[name] = {"ms": float(m.group(1))}
            else:
                timings[name] = {
                    "ms": float(m.group(1)),
                    "tokens": int(m.group(2)),
                    "tokens_per_second": float(m.group(3)),
                }
    return timings


def judge_output(text, n_predict):
    stripped = text.strip()
    fail_reasons = []
    if len(stripped) < 40:
        fail_reasons.append("too short")
    low = stripped.lower()
    if not any(word in low for word in ["france", "french", "paris", "europe"]):
        fail_reasons.append("does not mention expected France context")
    if "\ufffd" in stripped:
        fail_reasons.append("contains replacement character")
    if re.search(r"([^\w\s])\1{5,}", stripped):
        fail_reasons.append("repeated punctuation artifact")
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", stripped):
        fail_reasons.append("control characters")
    non_ascii = sum(1 for ch in stripped if ord(ch) > 127)
    if stripped and non_ascii / max(len(stripped), 1) > 0.10:
        fail_reasons.append("high non-ascii ratio for English prompt")
    tail = stripped[-120:]
    if re.search(r"(\b\w{1,2}\b[\s,;:.!?]*){12,}$", tail):
        fail_reasons.append("tail looks fragmented")
    status = "fail" if fail_reasons else "pass"
    return {"status": status, "reason": "; ".join(fail_reasons) if fail_reasons else f"coherent France paragraph; n_predict={n_predict}", "tail": tail}


def write_run_script(path, env, cmd, stdout_path, stderr_path, exit_path):
    exports = "\n".join(f"export {shlex.quote(k)}={shlex.quote(v)}" for k, v in sorted(env.items()))
    cmdline = " ".join(shlex.quote(x) for x in cmd)
    text = f"""#!/usr/bin/env bash
set -uo pipefail
cd {shlex.quote(str(REPO))}
{exports}
stdbuf -o0 -e0 {cmdline} > {shlex.quote(str(stdout_path))} 2> {shlex.quote(str(stderr_path))}
rc=$?
printf '%s\\n' "$rc" > {shlex.quote(str(exit_path))}
exit "$rc"
"""
    path.write_text(text)
    path.chmod(0o755)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default=None)
    ap.add_argument("--n-predict", type=int, default=96)
    ap.add_argument("--ngl", type=int, default=8)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    label = args.label or time.strftime("run-%Y%m%d-%H%M%S")
    run_dir = RUN_ROOT / label
    env = build_env(parse_env(args.env))
    cmd = build_command(args.n_predict, args.ngl)
    unit = re.sub(r"[^A-Za-z0-9_.-]", "-", f"ds4bench-{label}-{int(time.time())}")[:63]

    if args.dry_run:
        print(f"run_dir={run_dir}")
        print("systemd-run --wait --collect --unit", unit, "-p MemoryMax=14G -p MemorySwapMax=0 ...")
        print("monitors: memory.current memory.peak memory.stat nvidia-smi")
        print("env:")
        for k, v in sorted(env.items()):
            print(f"  {k}={v}")
        print("command:")
        print("  " + " ".join(shlex.quote(x) for x in cmd))
        return 0

    if run_dir.exists():
        raise SystemExit(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"
    exit_path = run_dir / "exit_code.txt"
    script_path = run_dir / "run-service.sh"
    memory_jsonl = run_dir / "memory_samples.jsonl"
    gpu_jsonl = run_dir / "gpu_samples.jsonl"

    git_sha = run_text(["git", "rev-parse", "HEAD"], check=True).strip()
    git_status = run_text(["git", "status", "--short", "--branch"], check=True)
    command_meta = {
        "repo": str(REPO),
        "git_sha": git_sha,
        "git_status": git_status,
        "unit": unit,
        "memory_max": "14G",
        "memory_swap_max": "0",
        "n_predict": args.n_predict,
        "ngl": args.ngl,
        "prompt": PROMPT,
        "env": env,
        "command": cmd,
    }
    (run_dir / "command.json").write_text(json.dumps(command_meta, indent=2, sort_keys=True) + "\n")
    (run_dir / "env.txt").write_text("\n".join(f"{k}={v}" for k, v in sorted(env.items())) + "\n")
    (run_dir / "git-status.txt").write_text(git_status)
    write_run_script(script_path, env, cmd, stdout_path, stderr_path, exit_path)

    systemd_cmd = [
        "systemd-run",
        "--wait",
        "--collect",
        "--unit", unit,
        "-p", "MemoryMax=14G",
        "-p", "MemorySwapMax=0",
        "/bin/bash",
        str(script_path),
    ]
    start = time.time()
    p = subprocess.Popen(systemd_cmd, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    first_stdout_time = None
    max_current = 0
    max_peak = 0
    peak_stat = {}
    last_stat = {}
    gpu_peak = 0
    samples = 0

    with memory_jsonl.open("w") as mf, gpu_jsonl.open("w") as gf:
        while True:
            now = time.time()
            cg = find_cgroup(unit)
            if cg:
                cur = read_int(cg / "memory.current")
                peak = read_int(cg / "memory.peak")
                stat = read_stat(cg / "memory.stat")
                rec = {"time": now, "elapsed_s": now - start, "cgroup": str(cg), "memory_current": cur, "memory_peak": peak, "memory_stat": stat}
                mf.write(json.dumps(rec, sort_keys=True) + "\n")
                mf.flush()
                samples += 1
                if cur is not None:
                    max_current = max(max_current, cur)
                if peak is not None and peak >= max_peak:
                    max_peak = peak
                    peak_stat = stat
                last_stat = stat
            if first_stdout_time is None and stdout_path.exists() and stdout_path.stat().st_size > 0:
                first_stdout_time = now
            for row in gpu_sample():
                row["time"] = now
                row["elapsed_s"] = now - start
                gf.write(json.dumps(row, sort_keys=True) + "\n")
                if row.get("memory_used_mib") is not None:
                    gpu_peak = max(gpu_peak, row["memory_used_mib"])
            gf.flush()
            if p.poll() is not None:
                # one final post-exit sample if the cgroup is still visible
                cg = find_cgroup(unit)
                if cg:
                    cur = read_int(cg / "memory.current")
                    peak = read_int(cg / "memory.peak")
                    stat = read_stat(cg / "memory.stat")
                    mf.write(json.dumps({"time": time.time(), "elapsed_s": time.time() - start, "cgroup": str(cg), "memory_current": cur, "memory_peak": peak, "memory_stat": stat}, sort_keys=True) + "\n")
                    if peak is not None and peak >= max_peak:
                        max_peak = peak
                        peak_stat = stat
                    last_stat = stat
                break
            time.sleep(0.5)

    systemd_output, _ = p.communicate()
    end = time.time()
    (run_dir / "systemd-run.log").write_text(systemd_output or "")
    stdout = stdout_path.read_text(errors="replace") if stdout_path.exists() else ""
    stderr = stderr_path.read_text(errors="replace") if stderr_path.exists() else ""
    timings = parse_timings(stderr)
    quality = judge_output(stdout, args.n_predict)
    exit_code = read_int(exit_path)
    summary = {
        **command_meta,
        "run_dir": str(run_dir),
        "exit_code": exit_code,
        "systemd_run_returncode": p.returncode,
        "wall_time_s": end - start,
        "ttft_s": None if first_stdout_time is None else first_stdout_time - start,
        "timings": timings,
        "prompt_eval_tokens_per_second": timings.get("prompt_eval", {}).get("tokens_per_second"),
        "decode_tokens_per_second": timings.get("eval", {}).get("tokens_per_second"),
        "memory_peak_bytes": max_peak or max_current,
        "memory_current_peak_bytes": max_current,
        "memory_peak_stat": peak_stat,
        "memory_last_stat": last_stat,
        "memory_sample_count": samples,
        "gpu_memory_peak_mib": gpu_peak,
        "output": stdout,
        "stderr_tail": stderr[-4000:],
        "output_quality": quality,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    md = [
        f"# DS4 strict run: {label}",
        "",
        f"- run_dir: `{run_dir}`",
        f"- git_sha: `{git_sha}`",
        f"- n_predict: `{args.n_predict}`",
        f"- exit_code: `{exit_code}`",
        f"- wall_time_s: `{summary['wall_time_s']:.3f}`",
        f"- ttft_s: `{summary['ttft_s']}`",
        f"- prompt_eval_tok_s: `{summary['prompt_eval_tokens_per_second']}`",
        f"- decode_tok_s: `{summary['decode_tokens_per_second']}`",
        f"- memory_peak_bytes: `{summary['memory_peak_bytes']}`",
        f"- gpu_memory_peak_mib: `{gpu_peak}`",
        f"- output_quality: `{quality['status']}` - {quality['reason']}",
        "",
        "## Command",
        "",
        "```bash",
        " ".join(shlex.quote(x) for x in cmd),
        "```",
        "",
        "## Output",
        "",
        "```text",
        stdout.strip(),
        "```",
    ]
    (run_dir / "summary.md").write_text("\n".join(md) + "\n")
    print(json.dumps({
        "run_dir": str(run_dir),
        "exit_code": exit_code,
        "decode_tokens_per_second": summary["decode_tokens_per_second"],
        "ttft_s": summary["ttft_s"],
        "memory_peak_bytes": summary["memory_peak_bytes"],
        "gpu_memory_peak_mib": gpu_peak,
        "output_quality": quality["status"],
    }, indent=2, sort_keys=True))
    return 0 if exit_code == 0 and p.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
