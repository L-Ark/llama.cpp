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


def parse_kv(items):
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"expected KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        if not key:
            raise SystemExit(f"empty env key in {item!r}")
        out[key] = value
    return out


def write_case_script(path, repo, binary, model, prompt, tensor_filter, extra_env):
    env = {
        "CUDA_VISIBLE_DEVICES": "0",
        "GGML_CUDA_DISABLE_GRAPHS": "1",
        "GGML_MOE_STREAM": "1",
        "GGML_MOE_STREAM_DONTNEED": "1",
        "GGML_MOE_STREAM_DOWN_BATCH": "1",
        "GGML_MOE_VRAM_CACHE_GB": "2",
    }
    env.update(extra_env)

    exports = "\n".join(f"export {key}={shlex.quote(value)}" for key, value in sorted(env.items()))
    cmd = [
        str(binary),
        "-m", str(model),
        "-p", prompt,
        "-c", "256",
        "-b", "16",
        "-ub", "16",
        "-t", "20",
        "-tb", "20",
        "-ngl", "all",
        "--fit", "on",
        "-fa", "auto",
        "--n-cpu-moe", "40",
        "--defer-experts",
        "--tensor-filter", tensor_filter,
    ]
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"cd {shlex.quote(str(path.parent))}\n"
        f"{exports}\n"
        "env | sort > environment.txt\n"
        + " ".join(shlex.quote(x) for x in ["/usr/bin/time", "-v", *cmd])
        + " > stdout.txt 2> stderr.txt\n"
    )
    path.chmod(0o755)


def run_case(case_dir, unit, memory_max_bytes):
    subprocess.run(["sync"], check=False)
    Path("/proc/sys/vm/drop_caches").write_text("3\n")
    script = case_dir / "run_case.sh"
    cmd = [
        "systemd-run",
        "--wait",
        "--collect",
        "--unit", unit,
        "-p", f"MemoryMax={memory_max_bytes}",
        "-p", "MemorySwapMax=0",
        "-p", "TasksMax=infinity",
        "bash", str(script),
    ]
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (case_dir / "systemd.txt").write_text(result.stdout)
    (case_dir / "systemd.returncode").write_text(str(result.returncode) + "\n")
    return result.returncode


def parse_debug_sums(paths):
    records = []
    current = None
    tensor_re = re.compile(r"common_debug_cb_eval:\s+(.+?)\s+=\s+\(")
    sum_re = re.compile(r"\bsum = ([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")

    for path in paths:
        current = None
        for line in path.read_text(errors="replace").splitlines():
            m = tensor_re.search(line)
            if m:
                current = {
                    "name": m.group(1).strip(),
                    "line": line.strip(),
                    "sums": [],
                }
                records.append(current)
                continue
            m = sum_re.search(line)
            if m and current is not None:
                current["sums"].append(float(m.group(1)))

    collapsed = {}
    for rec in records:
        if not rec["sums"]:
            continue
        # Most tensors have one i3 slice; keep a list to catch shape surprises.
        collapsed[rec["name"]] = {
            "sums": rec["sums"],
            "line": rec["line"],
        }
    return collapsed


def parse_timev(stderr_path):
    text = stderr_path.read_text(errors="replace")
    def last(pattern):
        values = re.findall(pattern, text)
        return values[-1] if values else None
    def last_int(pattern):
        value = last(pattern)
        return int(value) if value is not None else None
    elapsed = None
    for line in text.splitlines():
        if "Elapsed (wall clock) time" in line:
            elapsed = line.rsplit(": ", 1)[-1].strip()
    return {
        "elapsed": elapsed,
        "max_rss_kb": last_int(r"Maximum resident set size \(kbytes\):\s*([0-9]+)"),
        "major_page_faults": last_int(r"Major \(requiring I/O\) page faults:\s*([0-9]+)"),
        "exit_status": last_int(r"Exit status:\s*([0-9]+)"),
    }


def compare_records(explicit, candidate, atol):
    names = sorted(set(explicit) | set(candidate))
    missing_explicit = [name for name in names if name not in explicit]
    missing_candidate = [name for name in names if name not in candidate]
    diffs = []
    max_abs = 0.0
    for name in names:
        if name not in explicit or name not in candidate:
            continue
        a = explicit[name]["sums"]
        b = candidate[name]["sums"]
        n = min(len(a), len(b))
        if len(a) != len(b):
            diffs.append({"name": name, "reason": "sum_count_mismatch", "explicit_count": len(a), "candidate_count": len(b)})
            continue
        local = [abs(a[i] - b[i]) for i in range(n)]
        local_max = max(local) if local else 0.0
        max_abs = max(max_abs, local_max)
        if local_max > atol:
            diffs.append({
                "name": name,
                "explicit_sums": a,
                "candidate_sums": b,
                "max_abs": local_max,
            })
    return {
        "record_names": len(names),
        "explicit_records": len(explicit),
        "candidate_records": len(candidate),
        "missing_explicit": missing_explicit[:50],
        "missing_candidate": missing_candidate[:50],
        "num_missing_explicit": len(missing_explicit),
        "num_missing_candidate": len(missing_candidate),
        "max_abs_sum_diff": max_abs,
        "num_diffs_over_atol": len(diffs),
        "diffs_over_atol": diffs[:50],
        "atol": atol,
        "pass": not missing_explicit and not missing_candidate and not diffs,
    }


def main():
    parser = argparse.ArgumentParser(description="Run a DS4 gate/up/act tensor-sum parity probe with llama-debug.")
    parser.add_argument("--repo", default="/root/lfz/vendor/llama.cpp-deepseek-v4")
    parser.add_argument("--binary", default=None)
    parser.add_argument("--model", default="/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf")
    parser.add_argument("--out-root", default="/root/lfz/runs/vendor-ds4-16gb")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--prompt", default="Please introduce France in a short paragraph.")
    parser.add_argument("--tensor-filter", default=r".*ffn_moe_(gate_clamped|up_clamped|swiglu).*")
    parser.add_argument("--candidate-env", action="append", default=[], help="candidate-only KEY=VALUE env; repeatable")
    parser.add_argument("--common-env", action="append", default=[], help="extra common KEY=VALUE env; repeatable")
    parser.add_argument("--memory-max-bytes", type=int, default=16000000000)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--min-records", type=int, default=1)
    args = parser.parse_args()

    repo = Path(args.repo)
    binary = Path(args.binary) if args.binary else repo / "build-ds4-moe-stream/bin/llama-debug"
    model = Path(args.model)
    if not binary.exists():
        raise SystemExit(f"missing binary: {binary}")
    if not model.exists():
        raise SystemExit(f"missing model: {model}")

    run_name = args.run_name or time.strftime("%Y%m%dT%H%M%SZ-ds4-act-parity-probe", time.gmtime())
    run_dir = Path(args.out_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "prompt.txt").write_text(args.prompt)

    common_env = parse_kv(args.common_env)
    candidate_env = parse_kv(args.candidate_env)

    subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, text=True, stdout=(run_dir / "source_commit.txt").open("w"))
    subprocess.run(["git", "-C", str(repo), "log", "-1", "--oneline"], check=True, text=True, stdout=(run_dir / "source_commit_oneline.txt").open("w"))
    subprocess.run(["sha256sum", str(binary)], check=True, text=True, stdout=(run_dir / "llama-debug.sha256").open("w"))
    (run_dir / "model.stat").write_text(subprocess.check_output(["stat", "-c", "path=%n size=%s mtime=%y", str(model)], text=True))

    cases = {
        "explicit": common_env,
        "candidate": {**common_env, **candidate_env},
    }
    returncodes = {}
    for name, env in cases.items():
        case_dir = run_dir / name
        case_dir.mkdir()
        write_case_script(case_dir / "run_case.sh", repo, binary, model, args.prompt, args.tensor_filter, env)
        returncodes[name] = run_case(case_dir, f"ds4-act-parity-{name}-{int(time.time())}", args.memory_max_bytes)

    explicit = parse_debug_sums([run_dir / "explicit" / "stdout.txt", run_dir / "explicit" / "stderr.txt"])
    candidate = parse_debug_sums([run_dir / "candidate" / "stdout.txt", run_dir / "candidate" / "stderr.txt"])
    comparison = compare_records(explicit, candidate, args.atol)
    enough_records = len(explicit) >= args.min_records and len(candidate) >= args.min_records
    summary = {
        "artifact_id": "ds4-act-level-parity-probe",
        "run_dir": str(run_dir),
        "repo": str(repo),
        "source_commit": (run_dir / "source_commit.txt").read_text().strip(),
        "source_commit_oneline": (run_dir / "source_commit_oneline.txt").read_text().strip(),
        "binary_sha256": (run_dir / "llama-debug.sha256").read_text().strip(),
        "model_stat": (run_dir / "model.stat").read_text().strip(),
        "prompt": args.prompt,
        "tensor_filter": args.tensor_filter,
        "candidate_env": candidate_env,
        "common_env_extra": common_env,
        "memory_max_bytes": args.memory_max_bytes,
        "memory_swap_max": 0,
        "drop_caches_before_each_case": True,
        "min_records": args.min_records,
        "returncodes": returncodes,
        "runtime": {
            "explicit": parse_timev(run_dir / "explicit" / "stderr.txt"),
            "candidate": parse_timev(run_dir / "candidate" / "stderr.txt"),
        },
        "comparison": comparison,
        "status": "pass" if enough_records and comparison["pass"] and all(v == 0 for v in returncodes.values()) else "fail",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
