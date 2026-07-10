#!/usr/bin/env python3
import argparse
import csv
import json
import pathlib
import re
from collections import defaultdict


def fnum(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def inum(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_metrics(run_dir):
    metrics = {}
    path = run_dir / "metrics.json"
    if path.exists():
        try:
            metrics.update(json.loads(path.read_text(errors="replace")))
        except json.JSONDecodeError:
            pass
    txt = run_dir / "metrics.txt"
    if txt.exists():
        for line in txt.read_text(errors="replace").splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            metrics.setdefault(key.strip(), value.strip())
    return metrics


def parse_kv_numbers(text):
    parsed = {}
    if not isinstance(text, str):
        return parsed
    for key, value in re.findall(r"([A-Za-z0-9_]+)=([0-9.]+)", text):
        parsed[key] = fnum(value)
    return parsed


def layer_from_tensor(name):
    match = re.search(r"blk\.(\d+)\.", name or "")
    return int(match.group(1)) if match else None


def role_from_tensor(name):
    if "ffn_up_exps" in name:
        return "up"
    if "ffn_gate_exps" in name:
        return "gate"
    if "ffn_down_exps" in name:
        return "down"
    return "other"


def add_total(acc, key, value):
    acc[key] += value


def read_csv_rows(path):
    if not path.exists():
        return []
    with path.open(newline="", errors="replace") as f:
        return list(csv.DictReader(f))


def parse_cpu_moe_phase_profile(run_dir):
    result = {}
    path = run_dir / "stderr.txt"
    if not path.exists():
        return result
    text = path.read_text(errors="replace")
    for line in text.splitlines():
        if "[kimi_cpu_moe_profile]" not in line:
            continue
        match = re.search(r"\[kimi_cpu_moe_profile\]\s+([A-Za-z0-9_]+)\s+", line)
        if not match:
            continue
        op = match.group(1)
        if op == "up_gate":
            prefix = "cpu_moe_upgate"
        elif op == "down":
            prefix = "cpu_moe_down"
        else:
            prefix = f"cpu_moe_{op}"
        kv = parse_kv_numbers(line)
        decode_calls = kv.get("decode_calls", 0.0)
        prompt_calls = kv.get("prompt_calls", 0.0)
        result[f"{prefix}_decode_calls"] = decode_calls
        result[f"{prefix}_decode_total_ms"] = decode_calls * kv.get("decode_total", 0.0)
        result[f"{prefix}_decode_cuda_ms"] = decode_calls * kv.get("decode_cuda", 0.0)
        result[f"{prefix}_decode_cuda_batch_ms"] = decode_calls * kv.get("decode_cuda_batch", 0.0)
        result[f"{prefix}_decode_cuda_single_ms"] = decode_calls * kv.get("decode_cuda_single", 0.0)
        result[f"{prefix}_decode_fallback_ms"] = decode_calls * kv.get("decode_fallback", 0.0)
        result[f"{prefix}_prompt_calls"] = prompt_calls
        result[f"{prefix}_prompt_total_ms"] = prompt_calls * kv.get("prompt_total", 0.0)
        result[f"{prefix}_prompt_cuda_ms"] = prompt_calls * kv.get("prompt_cuda", 0.0)
    result["cpu_moe_decode_total_ms"] = (
        result.get("cpu_moe_upgate_decode_total_ms", 0.0) +
        result.get("cpu_moe_down_decode_total_ms", 0.0)
    )
    result["cpu_moe_decode_cuda_ms"] = (
        result.get("cpu_moe_upgate_decode_cuda_ms", 0.0) +
        result.get("cpu_moe_down_decode_cuda_ms", 0.0)
    )
    result["cpu_moe_prompt_total_ms"] = (
        result.get("cpu_moe_upgate_prompt_total_ms", 0.0) +
        result.get("cpu_moe_down_prompt_total_ms", 0.0)
    )
    return result


def summarize_run(run_dir, decode_down_max_active):
    metrics = read_metrics(run_dir)
    decode_runs = inum(metrics.get("decode_runs"))
    decode_ms = fnum(metrics.get("decode_ms"))
    token_rate = fnum(metrics.get("token_rate"))
    out = {
        "prompt_id": metrics.get("prompt_id", run_dir.name),
        "run_dir": str(run_dir),
        "quality": metrics.get("quality", "unknown"),
        "ttft_ms": fnum(metrics.get("ttft_ms")),
        "decode_runs": decode_runs,
        "decode_ms": decode_ms,
        "decode_ms_per_token": decode_ms / decode_runs if decode_runs else 0.0,
        "token_rate": token_rate,
        "upgate_wall_ms": 0.0,
        "down_wall_ms": 0.0,
        "down_stage_ms": 0.0,
        "decode_fallback_ms": 0.0,
        "prompt_fallback_ms": 0.0,
        "missing": [],
    }
    cpu_moe = parse_cpu_moe_phase_profile(run_dir)
    out.update(cpu_moe)
    out["residual_after_cpu_moe_ms"] = (
        max(decode_ms - cpu_moe.get("cpu_moe_decode_total_ms", 0.0), 0.0)
        if cpu_moe.get("cpu_moe_decode_total_ms", 0.0) > 0 else 0.0
    )
    expert_pack = parse_kv_numbers(metrics.get("expert_pack_0", ""))
    out["direct_reads"] = int(expert_pack.get("direct_reads", 0))
    out["iouring_reads"] = int(expert_pack.get("iouring_reads", 0))
    out["iouring_bytes"] = int(expert_pack.get("iouring_bytes", 0))
    out["iouring_wait_ms"] = expert_pack.get("iouring_wait_us", 0) / 1000.0
    total_reads = out["direct_reads"] + out["iouring_reads"]
    out["direct_read_ratio"] = out["direct_reads"] / total_reads if total_reads else 0.0
    out["iouring_read_ratio"] = out["iouring_reads"] / total_reads if total_reads else 0.0

    up_by_type = defaultdict(float)
    up_by_layer = defaultdict(float)
    up_stage_by_type = defaultdict(float)
    up_compute_by_type = defaultdict(float)
    up_jobs_by_type = defaultdict(int)
    if (run_dir / "up-gate-profile.csv").exists():
        for row in read_csv_rows(run_dir / "up-gate-profile.csv"):
            if row.get("mode") != "decode":
                continue
            key = f"up={row.get('up_type')}/gate={row.get('gate_type')}"
            wall = fnum(row.get("wall_ms"))
            out["upgate_wall_ms"] += wall
            add_total(up_by_type, key, wall)
            add_total(up_stage_by_type, key, fnum(row.get("stage_ms")))
            add_total(up_compute_by_type, key, fnum(row.get("up_ms")) + fnum(row.get("gate_ms")))
            up_jobs_by_type[key] += inum(row.get("up_stage_jobs")) + inum(row.get("gate_stage_jobs"))
            layer = layer_from_tensor(row.get("up_tensor", ""))
            if layer is not None:
                add_total(up_by_layer, f"blk.{layer}", wall)
    else:
        out["missing"].append("up-gate-profile.csv")

    down_by_type = defaultdict(float)
    down_stage_by_type = defaultdict(float)
    down_kernel_by_type = defaultdict(float)
    down_by_layer = defaultdict(float)
    down_jobs_by_type = defaultdict(int)
    if (run_dir / "down-batch-profile.csv").exists():
        for row in read_csv_rows(run_dir / "down-batch-profile.csv"):
            # down-batch-profile.csv does not carry a mode column. Decode rows
            # have one routed top-k set, while prompt rows batch many tokens and
            # have larger n_active. Keep this threshold configurable for models
            # whose routed top-k differs from Kimi's current 8.
            if inum(row.get("n_active")) > decode_down_max_active:
                continue
            key = f"type={row.get('src0_type')}"
            wall = fnum(row.get("wall_ms"))
            stage = fnum(row.get("stage_ms"))
            out["down_wall_ms"] += wall
            out["down_stage_ms"] += stage
            add_total(down_by_type, key, wall)
            add_total(down_stage_by_type, key, stage)
            add_total(down_kernel_by_type, key, fnum(row.get("kernel_ms")))
            down_jobs_by_type[key] += inum(row.get("staged_jobs"))
            layer = layer_from_tensor(row.get("tensor", ""))
            if layer is not None:
                add_total(down_by_layer, f"blk.{layer}", wall)
    else:
        out["missing"].append("down-batch-profile.csv")

    fallback_by_type = defaultdict(float)
    fallback_by_tensor = defaultdict(float)
    fallback_by_role = defaultdict(float)
    if (run_dir / "fallback-profile.csv").exists():
        for row in read_csv_rows(run_dir / "fallback-profile.csv"):
            ms = fnum(row.get("fallback_us")) / 1000.0
            phase = row.get("phase")
            if phase == "decode":
                out["decode_fallback_ms"] += ms
                key = f"type={row.get('src0_type')}"
                add_total(fallback_by_type, key, ms)
                tensor = row.get("tensor", "")
                add_total(fallback_by_tensor, f"{tensor},type={row.get('src0_type')}", ms)
                add_total(fallback_by_role, role_from_tensor(tensor), ms)
            elif phase == "prompt":
                out["prompt_fallback_ms"] += ms
    else:
        out["missing"].append("fallback-profile.csv")

    out["up_by_type"] = dict(up_by_type)
    out["up_stage_by_type"] = dict(up_stage_by_type)
    out["up_compute_by_type"] = dict(up_compute_by_type)
    out["up_jobs_by_type"] = dict(up_jobs_by_type)
    out["up_by_layer"] = dict(up_by_layer)
    out["down_by_type"] = dict(down_by_type)
    out["down_stage_by_type"] = dict(down_stage_by_type)
    out["down_kernel_by_type"] = dict(down_kernel_by_type)
    out["down_jobs_by_type"] = dict(down_jobs_by_type)
    out["down_by_layer"] = dict(down_by_layer)
    out["fallback_by_type"] = dict(fallback_by_type)
    out["fallback_by_tensor"] = dict(fallback_by_tensor)
    out["fallback_by_role"] = dict(fallback_by_role)
    explained = out["upgate_wall_ms"] + out["down_wall_ms"] + out["decode_fallback_ms"]
    out["residual_ms"] = max(out["decode_ms"] - explained, 0.0)
    return out


def merge_dict_totals(runs, key):
    acc = defaultdict(float)
    for run in runs:
        for item, value in run.get(key, {}).items():
            acc[item] += value
    return dict(acc)


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value):
    return f"{value:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        default=pathlib.Path(".Agent/runs/20260706-kimi-general-dev-baseline-n96-profile"),
    )
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument(
        "--decode-down-max-active",
        type=int,
        default=8,
        help="Treat down-batch rows with n_active above this value as prompt/prefill rows.",
    )
    args = parser.parse_args()

    run_dirs = sorted(
        p for p in args.input.iterdir()
        if p.is_dir() and p.name != "entropy" and (p / "metrics.txt").exists()
    )
    runs = [summarize_run(p, args.decode_down_max_active) for p in run_dirs]
    total_decode_runs = sum(r["decode_runs"] for r in runs)
    total_decode_ms = sum(r["decode_ms"] for r in runs)
    base_ms_per_token = total_decode_ms / total_decode_runs if total_decode_runs else 0.0

    aggregate = {
        "input": str(args.input),
        "run_count": len(runs),
        "total_decode_runs": total_decode_runs,
        "total_decode_ms": total_decode_ms,
        "weighted_decode_ms_per_token": base_ms_per_token,
        "weighted_token_rate": 1000.0 / base_ms_per_token if base_ms_per_token else 0.0,
        "runs": runs,
        "total_direct_reads": sum(r["direct_reads"] for r in runs),
        "total_iouring_reads": sum(r["iouring_reads"] for r in runs),
        "total_iouring_bytes": sum(r["iouring_bytes"] for r in runs),
        "total_iouring_wait_ms": sum(r["iouring_wait_ms"] for r in runs),
    }
    total_reads = aggregate["total_direct_reads"] + aggregate["total_iouring_reads"]
    aggregate["direct_read_ratio"] = aggregate["total_direct_reads"] / total_reads if total_reads else 0.0
    aggregate["iouring_read_ratio"] = aggregate["total_iouring_reads"] / total_reads if total_reads else 0.0
    aggregate["iouring_gib_s"] = (
        aggregate["total_iouring_bytes"] / 1024 / 1024 / 1024 / (total_decode_ms / 1000.0)
        if total_decode_ms else 0.0
    )
    for key in [
        "up_by_type",
        "up_stage_by_type",
        "up_compute_by_type",
        "up_jobs_by_type",
        "up_by_layer",
        "down_by_type",
        "down_stage_by_type",
        "down_kernel_by_type",
        "down_jobs_by_type",
        "down_by_layer",
        "fallback_by_type",
        "fallback_by_tensor",
        "fallback_by_role",
    ]:
        aggregate[key] = merge_dict_totals(runs, key)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")

    per_prompt_rows = []
    for r in runs:
        denom = r["decode_runs"] or 1
        per_prompt_rows.append({
            "prompt_id": r["prompt_id"],
            "quality": r["quality"],
            "decode_runs": r["decode_runs"],
            "token_rate": fmt(r["token_rate"]),
            "decode_ms_per_token": fmt(r["decode_ms_per_token"]),
            "ttft_ms": fmt(r["ttft_ms"]),
            "upgate_ms_per_token": fmt(r["upgate_wall_ms"] / denom),
            "down_ms_per_token": fmt(r["down_wall_ms"] / denom),
            "down_stage_ms_per_token": fmt(r["down_stage_ms"] / denom),
            "decode_fallback_ms_per_token": fmt(r["decode_fallback_ms"] / denom),
            "residual_ms_per_token": fmt(r["residual_ms"] / denom),
            "cpu_moe_upgate_ms_per_token": fmt(r.get("cpu_moe_upgate_decode_total_ms", 0.0) / denom),
            "cpu_moe_down_ms_per_token": fmt(r.get("cpu_moe_down_decode_total_ms", 0.0) / denom),
            "cpu_moe_residual_ms_per_token": fmt(r.get("residual_after_cpu_moe_ms", 0.0) / denom),
            "direct_read_ratio": fmt(r["direct_read_ratio"]),
            "iouring_gib_s": fmt(r["iouring_bytes"] / 1024 / 1024 / 1024 / (r["decode_ms"] / 1000.0) if r["decode_ms"] else 0.0),
            "missing": ";".join(r["missing"]),
        })
    write_csv(args.out / "per_prompt.csv", per_prompt_rows, list(per_prompt_rows[0].keys()) if per_prompt_rows else [])

    component_rows = []
    components = {
        "residual_unattributed": sum(r["residual_ms"] for r in runs),
        "upgate_wall": sum(r["upgate_wall_ms"] for r in runs),
        "down_wall": sum(r["down_wall_ms"] for r in runs),
        "down_stage": sum(r["down_stage_ms"] for r in runs),
        "decode_fallback": sum(r["decode_fallback_ms"] for r in runs),
        "iouring_wait_reported": sum(r["iouring_wait_ms"] for r in runs),
    }
    cpu_moe_components = {
        "cpu_moe_upgate_total": sum(r.get("cpu_moe_upgate_decode_total_ms", 0.0) for r in runs),
        "cpu_moe_down_total": sum(r.get("cpu_moe_down_decode_total_ms", 0.0) for r in runs),
        "cpu_moe_upgate_cuda": sum(r.get("cpu_moe_upgate_decode_cuda_ms", 0.0) for r in runs),
        "cpu_moe_down_cuda": sum(r.get("cpu_moe_down_decode_cuda_ms", 0.0) for r in runs),
        "residual_after_cpu_moe": sum(r.get("residual_after_cpu_moe_ms", 0.0) for r in runs),
    }
    if any(total > 0 for total in cpu_moe_components.values()):
        components.update(cpu_moe_components)
    for name, total_ms in sorted(components.items(), key=lambda kv: kv[1], reverse=True):
        mspt = total_ms / total_decode_runs if total_decode_runs else 0.0
        remaining = max(base_ms_per_token - mspt, 0.001)
        component_rows.append({
            "component": name,
            "total_ms": fmt(total_ms),
            "ms_per_token": fmt(mspt),
            "share_of_decode": fmt(total_ms / total_decode_ms if total_decode_ms else 0.0),
            "ideal_token_rate_without_component": fmt(1000.0 / remaining),
        })
    write_csv(args.out / "components.csv", component_rows, list(component_rows[0].keys()) if component_rows else [])

    detail_rows = []
    detail_specs = [
        ("upgate_type", "up_by_type"),
        ("upgate_layer", "up_by_layer"),
        ("down_type", "down_by_type"),
        ("down_layer", "down_by_layer"),
        ("fallback_type", "fallback_by_type"),
        ("fallback_tensor", "fallback_by_tensor"),
        ("fallback_role", "fallback_by_role"),
    ]
    for group, key in detail_specs:
        for item, total_ms in aggregate[key].items():
            mspt = total_ms / total_decode_runs if total_decode_runs else 0.0
            detail_rows.append({
                "group": group,
                "item": item,
                "total_ms": fmt(total_ms),
                "ms_per_token": fmt(mspt),
            })
    detail_rows.sort(key=lambda r: float(r["total_ms"]), reverse=True)
    write_csv(args.out / "details.csv", detail_rows, ["group", "item", "total_ms", "ms_per_token"])

    top_details = detail_rows[:20]
    lines = [
        "# Kimi Dev Decode Bottleneck Summary",
        "",
        f"Input: `{args.input}`",
        f"Decode down row filter: `n_active <= {args.decode_down_max_active}`",
        "",
        f"Runs: `{len(runs)}`",
        f"Total decode runs: `{total_decode_runs}`",
        f"Weighted decode ms/token: `{fmt(base_ms_per_token)}`",
        f"Weighted token rate: `{fmt(1000.0 / base_ms_per_token if base_ms_per_token else 0.0)} tok/s`",
        f"Direct read ratio: `{fmt(aggregate['direct_read_ratio'])}`",
        f"Iouring read ratio: `{fmt(aggregate['iouring_read_ratio'])}`",
        f"Iouring throughput over decode wall: `{fmt(aggregate['iouring_gib_s'])} GiB/s`",
        "",
        "## Per Prompt",
        "",
        "| Prompt | Quality | tok/s | decode ms/token | TTFT ms | up/gate CSV ms/token | down CSV ms/token | residual CSV ms/token | CPU MoE upgate ms/token | CPU MoE down ms/token | CPU MoE residual ms/token | direct read ratio | iouring GiB/s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in per_prompt_rows:
        lines.append(
            f"| `{row['prompt_id']}` | {row['quality']} | {row['token_rate']} | {row['decode_ms_per_token']} | "
            f"{row['ttft_ms']} | {row['upgate_ms_per_token']} | {row['down_ms_per_token']} | "
            f"{row['residual_ms_per_token']} | {row['cpu_moe_upgate_ms_per_token']} | "
            f"{row['cpu_moe_down_ms_per_token']} | {row['cpu_moe_residual_ms_per_token']} | "
            f"{row['direct_read_ratio']} | {row['iouring_gib_s']} |"
        )
    lines += [
        "",
        "## Components",
        "",
        "| Component | total ms | ms/token | decode share | ideal tok/s if removed |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in component_rows:
        lines.append(
            f"| `{row['component']}` | {row['total_ms']} | {row['ms_per_token']} | "
            f"{row['share_of_decode']} | {row['ideal_token_rate_without_component']} |"
        )
    lines += [
        "",
        "## Top Detail Rows",
        "",
        "| Group | Item | total ms | ms/token |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in top_details:
        lines.append(f"| `{row['group']}` | `{row['item']}` | {row['total_ms']} | {row['ms_per_token']} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "- `iouring_wait_reported` can exceed decode wall share because wait",
        "  counters are collected across overlapping worker paths. Use it as an",
        "  exposed-pressure signal, not as an additive component.",
        "- `down_wall`, `upgate_wall`, and `residual_unattributed` are closer to",
        "  additive decode-wall buckets, but they are still profiling estimates.",
        "- When `cpu_moe_*` rows are present, they come from the CPU/defer MoE",
        "  op entry profile and are broader than the CSV buckets. Prefer",
        "  `residual_after_cpu_moe` over `residual_unattributed` for deciding",
        "  whether the next bottleneck is outside MoE.",
        "- `down-batch-profile.csv` has no mode column; this report treats rows",
        "  with `n_active` above the configured threshold as prompt/prefill rows",
        "  and excludes them from decode down totals.",
        "- Any accepted optimization must name the component and detail rows it is",
        "  expected to reduce, compute a best-case bound from this report, and",
        "  then verify the reduction with paired cold-start runs.",
        "",
        "This report does not change runtime behavior and makes no SOTA claim.",
        "",
    ]
    (args.out / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
