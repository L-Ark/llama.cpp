#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


def parse_metrics(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(errors="replace").splitlines():
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        out[key] = val
        try:
            out[key + "__float"] = float(val)
        except ValueError:
            pass
    text = path.read_text(errors="replace")
    m = re.search(r"^expert_pack_0=(.*)$", text, re.M)
    if m:
        fields = {}
        for part in m.group(1).split():
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            fields[k] = v
        for k in ("iouring_reads", "iouring_bytes", "iouring_wait_us", "direct_reads"):
            try:
                out[k + "__float"] = float(fields[k])
            except (KeyError, ValueError):
                pass
    m = re.search(r"expert_pack_iouring_0=.*?inflight_avg=([0-9.]+).*?inflight_max=(\d+)", text)
    if m:
        out["inflight_avg__float"] = float(m.group(1))
        out["inflight_max__float"] = float(m.group(2))
    for role in ("down", "upgate"):
        m = re.search(rf"vram_{role}_0=slots=(\d+) slot=([0-9.]+) MiB hits=(\d+) misses=(\d+).*?hit_rate=([0-9.]+)%", text)
        if m:
            out[f"{role}_slots__float"] = float(m.group(1))
            out[f"{role}_slot_mib__float"] = float(m.group(2))
            out[f"{role}_hits__float"] = float(m.group(3))
            out[f"{role}_misses__float"] = float(m.group(4))
            out[f"{role}_hit_rate__float"] = float(m.group(5))
    return out


def parse_stderr(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    text = path.read_text(errors="replace")
    m = re.search(r"prompt eval time =\s*([0-9.]+) ms /", text)
    if m:
        out["prompt_eval_ms"] = float(m.group(1))
    m = re.search(r"eval time =\s*([0-9.]+) ms /\s*(\d+) runs", text)
    if m:
        out["stderr_decode_ms"] = float(m.group(1))
        out["stderr_decode_runs"] = float(m.group(2))
    out["q4_candidate_active"] = "Q4_0 down batch candidate active" in text
    out["q4_decodeonly_decline_seen"] = "q4_prompt_multirow_disabled" in text or "multirow_not_supported" in text
    return out


def tensor_role(name: str) -> str:
    if "ffn_up_exps" in name:
        return "up"
    if "ffn_gate_exps" in name:
        return "gate"
    if "ffn_down_exps" in name:
        return "down"
    return "other"


def parse_fallback(path: Path) -> dict:
    result = {
        "fallback_total_ms": 0.0,
        "fallback_prompt_ms": 0.0,
        "fallback_decode_ms": 0.0,
        "fallback_by_phase_type": {},
        "fallback_by_phase_role": {},
    }
    if not path.exists():
        return result
    with path.open(newline="", errors="replace") as f:
        for row in csv.DictReader(f):
            calls = float(row.get("calls") or 0)
            fallback_us = float(row.get("fallback_us") or 0)
            ms = fallback_us / 1000.0
            phase = row.get("phase") or "unknown"
            typ = row.get("src0_type") or "unknown"
            role = tensor_role(row.get("tensor") or "")
            result["fallback_total_ms"] += ms
            result[f"fallback_{phase}_ms"] = result.get(f"fallback_{phase}_ms", 0.0) + ms
            key = f"{phase},type={typ}"
            result["fallback_by_phase_type"][key] = result["fallback_by_phase_type"].get(key, 0.0) + ms
            rkey = f"{phase},{role}"
            result["fallback_by_phase_role"][rkey] = result["fallback_by_phase_role"].get(rkey, 0.0) + ms
    return result


def sum_csv(path: Path, cols: list[str]) -> dict:
    out = {c: 0.0 for c in cols}
    out["rows"] = 0.0
    if not path.exists():
        return out
    with path.open(newline="", errors="replace") as f:
        for row in csv.DictReader(f):
            out["rows"] += 1
            for c in cols:
                try:
                    out[c] += float(row.get(c) or 0)
                except ValueError:
                    pass
    return out


def parse_trace(path: Path) -> dict:
    out = {"trace_rows": 0, "trace_last_t_ms": 0.0, "trace_by_op_ms": {}}
    if not path.exists():
        return out
    prev_t = None
    prev_op = None
    with path.open(newline="", errors="replace") as f:
        for row in csv.DictReader(f):
            try:
                t = float(row.get("t_ms") or 0)
            except ValueError:
                continue
            op = row.get("op") or "unknown"
            out["trace_rows"] += 1
            out["trace_last_t_ms"] = max(out["trace_last_t_ms"], t)
            if prev_t is not None and prev_op is not None:
                out["trace_by_op_ms"][prev_op] = out["trace_by_op_ms"].get(prev_op, 0.0) + max(0.0, t - prev_t)
            prev_t = t
            prev_op = op
    return out


def load_run(run_dir: Path) -> dict:
    metrics = parse_metrics(run_dir / "metrics.txt")
    stderr = parse_stderr(run_dir / "stderr.txt")
    fallback = parse_fallback(run_dir / "fallback-profile.csv")
    upgate = sum_csv(run_dir / "up-gate-profile.csv", [
        "wall_ms", "stage_ms", "up_ms", "gate_ms", "up_wait_ms", "gate_wait_ms",
        "kernel_ms", "up_cache_misses", "gate_cache_misses", "up_stage_jobs", "gate_stage_jobs",
    ])
    down = sum_csv(run_dir / "down-batch-profile.csv", [
        "wall_ms", "stage_ms", "kernel_ms", "cache_misses", "staged_jobs",
    ])
    trace = parse_trace(run_dir / "ttft-trace.csv")
    prompt_id = metrics.get("prompt_id") or run_dir.name.split("-q4", 1)[0]
    return {
        "dir": str(run_dir),
        "prompt_id": prompt_id,
        "metrics": metrics,
        "stderr": stderr,
        "fallback": fallback,
        "upgate": upgate,
        "down": down,
        "trace": trace,
    }


def load_root(root: Path) -> dict:
    runs = {}
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / "metrics.txt").exists():
            r = load_run(d)
            runs[r["prompt_id"]] = r
    return runs


def f(run, key, default=0.0):
    return float(run["metrics"].get(key + "__float", default))


def sf(run, key, default=0.0):
    return float(run["stderr"].get(key, default))


def row_for(base, cand):
    prompt_id = base["prompt_id"]
    b_ttft = f(base, "ttft_ms")
    c_ttft = f(cand, "ttft_ms")
    b_decode = f(base, "decode_ms")
    c_decode = f(cand, "decode_ms")
    b_rate = f(base, "token_rate")
    c_rate = f(cand, "token_rate")
    b_iow = f(base, "iouring_wait_us") / 1000.0
    c_iow = f(cand, "iouring_wait_us") / 1000.0
    b_iob = f(base, "iouring_bytes") / (1024 ** 3)
    c_iob = f(cand, "iouring_bytes") / (1024 ** 3)
    prompt_type_delta = {}
    for k in set(base["fallback"]["fallback_by_phase_type"]) | set(cand["fallback"]["fallback_by_phase_type"]):
        if k.startswith("prompt,"):
            prompt_type_delta[k] = cand["fallback"]["fallback_by_phase_type"].get(k, 0.0) - base["fallback"]["fallback_by_phase_type"].get(k, 0.0)
    prompt_role_delta = {}
    for k in set(base["fallback"]["fallback_by_phase_role"]) | set(cand["fallback"]["fallback_by_phase_role"]):
        if k.startswith("prompt,"):
            prompt_role_delta[k] = cand["fallback"]["fallback_by_phase_role"].get(k, 0.0) - base["fallback"]["fallback_by_phase_role"].get(k, 0.0)
    return {
        "prompt_id": prompt_id,
        "base_ttft_ms": b_ttft,
        "cand_ttft_ms": c_ttft,
        "ttft_delta_ms": c_ttft - b_ttft,
        "ttft_ratio": c_ttft / b_ttft if b_ttft else None,
        "base_prompt_eval_ms": sf(base, "prompt_eval_ms", b_ttft),
        "cand_prompt_eval_ms": sf(cand, "prompt_eval_ms", c_ttft),
        "base_decode_ms": b_decode,
        "cand_decode_ms": c_decode,
        "decode_delta_ms": c_decode - b_decode,
        "base_token_rate": b_rate,
        "cand_token_rate": c_rate,
        "token_rate_delta": c_rate - b_rate,
        "base_iouring_wait_ms": b_iow,
        "cand_iouring_wait_ms": c_iow,
        "iouring_wait_delta_ms": c_iow - b_iow,
        "base_iouring_gib": b_iob,
        "cand_iouring_gib": c_iob,
        "iouring_gib_delta": c_iob - b_iob,
        "base_upgate_hit": f(base, "upgate_hit_rate"),
        "cand_upgate_hit": f(cand, "upgate_hit_rate"),
        "base_down_hit": f(base, "down_hit_rate"),
        "cand_down_hit": f(cand, "down_hit_rate"),
        "base_down_slot_mib": f(base, "down_slot_mib"),
        "cand_down_slot_mib": f(cand, "down_slot_mib"),
        "base_prompt_fallback_ms": base["fallback"].get("fallback_prompt_ms", 0.0),
        "cand_prompt_fallback_ms": cand["fallback"].get("fallback_prompt_ms", 0.0),
        "prompt_fallback_delta_ms": cand["fallback"].get("fallback_prompt_ms", 0.0) - base["fallback"].get("fallback_prompt_ms", 0.0),
        "base_decode_fallback_ms": base["fallback"].get("fallback_decode_ms", 0.0),
        "cand_decode_fallback_ms": cand["fallback"].get("fallback_decode_ms", 0.0),
        "decode_fallback_delta_ms": cand["fallback"].get("fallback_decode_ms", 0.0) - base["fallback"].get("fallback_decode_ms", 0.0),
        "base_upgate_wall_ms": base["upgate"].get("wall_ms", 0.0),
        "cand_upgate_wall_ms": cand["upgate"].get("wall_ms", 0.0),
        "upgate_wall_delta_ms": cand["upgate"].get("wall_ms", 0.0) - base["upgate"].get("wall_ms", 0.0),
        "base_down_wall_ms": base["down"].get("wall_ms", 0.0),
        "cand_down_wall_ms": cand["down"].get("wall_ms", 0.0),
        "down_wall_delta_ms": cand["down"].get("wall_ms", 0.0) - base["down"].get("wall_ms", 0.0),
        "prompt_fallback_type_delta": prompt_type_delta,
        "prompt_fallback_role_delta": prompt_role_delta,
    }


def fmt(x, nd=1):
    if x is None:
        return ""
    return f"{x:.{nd}f}"


def write_md(path: Path, title: str, rows_by_label: dict, detail: dict):
    lines = [f"# {title}", ""]
    for label, rows in rows_by_label.items():
        lines += [f"## {label}", ""]
        if not rows:
            lines += ["No comparable rows.", ""]
            continue
        mean = lambda k: sum(r[k] for r in rows) / len(rows)
        lines += [
            f"- prompts compared: `{len(rows)}`",
            f"- mean token rate: `{mean('base_token_rate'):.3f} -> {mean('cand_token_rate'):.3f} tok/s`",
            f"- mean TTFT: `{mean('base_ttft_ms'):.1f} -> {mean('cand_ttft_ms'):.1f} ms`",
            f"- max TTFT ratio: `{max(r['ttft_ratio'] for r in rows):.3f}`",
            f"- mean decode: `{mean('base_decode_ms'):.1f} -> {mean('cand_decode_ms'):.1f} ms`",
            f"- mean iouring wait: `{mean('base_iouring_wait_ms'):.1f} -> {mean('cand_iouring_wait_ms'):.1f} ms`",
            f"- mean iouring bytes: `{mean('base_iouring_gib'):.1f} -> {mean('cand_iouring_gib'):.1f} GiB`",
            f"- mean prompt fallback: `{mean('base_prompt_fallback_ms'):.1f} -> {mean('cand_prompt_fallback_ms'):.1f} ms`",
            f"- mean decode fallback: `{mean('base_decode_fallback_ms'):.1f} -> {mean('cand_decode_fallback_ms'):.1f} ms`",
            "",
        ]
        lines += [
            "| prompt | tok/s | TTFT ratio | TTFT delta ms | decode delta ms | IO wait delta ms | IO GiB delta | prompt fallback delta ms | decode fallback delta ms | upgate hit | down hit | down slot MiB |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in rows:
            lines.append(
                f"| `{r['prompt_id']}` | {r['base_token_rate']:.2f}->{r['cand_token_rate']:.2f} | "
                f"{r['ttft_ratio']:.3f} | {r['ttft_delta_ms']:.1f} | {r['decode_delta_ms']:.1f} | "
                f"{r['iouring_wait_delta_ms']:.1f} | {r['iouring_gib_delta']:.1f} | "
                f"{r['prompt_fallback_delta_ms']:.1f} | {r['decode_fallback_delta_ms']:.1f} | "
                f"{r['base_upgate_hit']:.1f}->{r['cand_upgate_hit']:.1f}% | "
                f"{r['base_down_hit']:.1f}->{r['cand_down_hit']:.1f}% | "
                f"{r['base_down_slot_mib']:.2f}->{r['cand_down_slot_mib']:.2f} |"
            )
        lines += ["", "### Interpretation", ""]
        ttft_bad = [r for r in rows if r["ttft_ratio"] and r["ttft_ratio"] > 1.20]
        if ttft_bad:
            lines.append(f"- TTFT gate fails on `{len(ttft_bad)}/{len(rows)}` comparable prompts.")
        else:
            lines.append("- TTFT gate passes on comparable prompts.")
        if mean("iouring_wait_delta_ms") < 0 and mean("ttft_delta_ms") > 0:
            lines.append("- Mean io_uring wait is lower while TTFT is higher, so the TTFT regression is not explained by decode IO wait.")
        if abs(mean("prompt_fallback_delta_ms")) < 0.05 * max(1.0, mean("ttft_delta_ms")):
            lines.append("- Prompt fallback delta is too small to explain the TTFT increase.")
        if mean("iouring_gib_delta") > 0:
            lines.append("- Candidate moves more expert bytes, mainly from Q4 down slot/cache changes, but the measured IO wait does not increase in the same direction.")
        lines.append("")
        for dict_key, title in (
            ("prompt_fallback_role_delta", "Prompt fallback role deltas"),
            ("prompt_fallback_type_delta", "Prompt fallback type deltas"),
        ):
            totals = {}
            for r in rows:
                for k, v in r[dict_key].items():
                    totals[k] = totals.get(k, 0.0) + v
            if totals:
                lines += [f"### {title}", "", "| key | mean delta ms |", "|---|---:|"]
                for k, v in sorted(totals.items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]:
                    lines.append(f"| `{k}` | {v / len(rows):.1f} |")
                lines.append("")
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", action="append", required=True, help="label=path")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    baseline = load_root(Path(args.baseline))
    rows_by_label = {}
    detail = {"baseline": args.baseline, "candidates": {}}
    for item in args.candidate:
        label, root = item.split("=", 1)
        cand_runs = load_root(Path(root))
        detail["candidates"][label] = root
        rows = []
        for pid, base in baseline.items():
            cand = cand_runs.get(pid)
            if cand is not None:
                rows.append(row_for(base, cand))
        rows_by_label[label] = rows
    out = {"rows": rows_by_label, "detail": detail}
    Path(args.out_json).write_text(json.dumps(out, indent=2, sort_keys=True))
    write_md(Path(args.out_md), "GP112 Q4 TTFT Source Differential", rows_by_label, detail)


if __name__ == "__main__":
    main()
