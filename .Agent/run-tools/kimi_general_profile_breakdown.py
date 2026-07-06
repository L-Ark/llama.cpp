#!/usr/bin/env python3
import argparse
import csv
import json
import pathlib
import re
from collections import Counter, defaultdict


ROLE_RE = re.compile(r"ffn_(up|gate|down)_exps")
LAYER_RE = re.compile(r"blk\.(\d+)\.")
NAME_PROFILE_RE = re.compile(
    r"\[kimi_cpu_moe_name_profile\] top(?P<rank>\d+) name=(?P<name>\S+) "
    r"calls=(?P<calls>\d+) total=(?P<total>[0-9.]+) ms/call "
    r"fallback_t0=(?P<fallback>[0-9.]+) ms/call .*?"
    r"decode_calls=(?P<decode_calls>\d+) decode_total=(?P<decode_total>[0-9.]+) ms/call "
    r"decode_fallback=(?P<decode_fallback>[0-9.]+) ms/call "
    r"prompt_calls=(?P<prompt_calls>\d+) prompt_total=(?P<prompt_total>[0-9.]+) ms/call "
    r"prompt_fallback=(?P<prompt_fallback>[0-9.]+) ms/call"
)
ELIG_RE = re.compile(
    r"\[kimi_cpu_moe_eligibility_profile\] top(?P<rank>\d+) name=(?P<name>\S+) "
    r"src0_type=(?P<src0_type>\d+) eligible=(?P<eligible>\d+) .*?"
    r"unsupported=(?P<unsupported>\d+) .*?decode_eligible=(?P<decode_eligible>\d+) "
    r"decode_unsupported=(?P<decode_unsupported>\d+) prompt_eligible=(?P<prompt_eligible>\d+) "
    r"prompt_unsupported=(?P<prompt_unsupported>\d+)"
)


def role_of(tensor: str) -> str:
    match = ROLE_RE.search(tensor)
    return match.group(1) if match else "other"


def layer_of(tensor: str) -> int:
    match = LAYER_RE.search(tensor)
    return int(match.group(1)) if match else -1


def read_csv(path: pathlib.Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def f(row, key):
    try:
        return float(row.get(key, 0) or 0)
    except ValueError:
        return 0.0


def i(row, key):
    try:
        return int(float(row.get(key, 0) or 0))
    except ValueError:
        return 0


def analyze_fallback(run_dir: pathlib.Path):
    rows = read_csv(run_dir / "fallback-profile.csv")
    by_phase_type_role = defaultdict(lambda: {"fallback_us": 0, "calls": 0, "gguf_calls": 0, "pack_mmap_calls": 0, "bytes": 0})
    top_tensors = Counter()
    for row in rows:
        role = role_of(row.get("tensor", ""))
        key = (row.get("phase", ""), row.get("src0_type", ""), role)
        calls = i(row, "calls")
        fallback_us = i(row, "fallback_us")
        by_phase_type_role[key]["fallback_us"] += fallback_us
        by_phase_type_role[key]["calls"] += calls
        by_phase_type_role[key]["gguf_calls"] += i(row, "gguf_calls")
        by_phase_type_role[key]["pack_mmap_calls"] += i(row, "pack_mmap_calls")
        by_phase_type_role[key]["bytes"] += i(row, "expert_bytes") * calls
        top_tensors[row.get("tensor", "")] += fallback_us
    return by_phase_type_role, top_tensors


def analyze_down(run_dir: pathlib.Path):
    rows = read_csv(run_dir / "down-batch-profile.csv")
    totals = defaultdict(lambda: {"calls": 0, "wall_ms": 0.0, "stage_ms": 0.0, "kernel_ms": 0.0, "staged_jobs": 0, "misses": 0, "hits": 0})
    for row in rows:
        key = (row.get("src0_type", ""), role_of(row.get("tensor", "")))
        totals[key]["calls"] += 1
        totals[key]["wall_ms"] += f(row, "wall_ms")
        totals[key]["stage_ms"] += f(row, "stage_ms")
        totals[key]["kernel_ms"] += f(row, "kernel_ms")
        totals[key]["staged_jobs"] += i(row, "staged_jobs")
        totals[key]["misses"] += i(row, "cache_misses")
        totals[key]["hits"] += i(row, "cache_hits")
    return totals


def analyze_upgate(run_dir: pathlib.Path):
    rows = read_csv(run_dir / "up-gate-profile.csv")
    totals = defaultdict(lambda: {"calls": 0, "wall_ms": 0.0, "stage_ms": 0.0, "kernel_ms": 0.0, "up_ms": 0.0, "gate_ms": 0.0, "up_wait_ms": 0.0, "gate_wait_ms": 0.0, "misses": 0, "hits": 0})
    for row in rows:
        key = (f"up{row.get('up_type','')}_gate{row.get('gate_type','')}", "upgate")
        totals[key]["calls"] += 1
        totals[key]["wall_ms"] += f(row, "wall_ms")
        totals[key]["stage_ms"] += f(row, "stage_ms")
        totals[key]["kernel_ms"] += f(row, "kernel_ms")
        totals[key]["up_ms"] += f(row, "up_ms")
        totals[key]["gate_ms"] += f(row, "gate_ms")
        totals[key]["up_wait_ms"] += f(row, "up_wait_ms")
        totals[key]["gate_wait_ms"] += f(row, "gate_wait_ms")
        totals[key]["misses"] += i(row, "up_cache_misses") + i(row, "gate_cache_misses")
        totals[key]["hits"] += i(row, "up_cache_hits") + i(row, "gate_cache_hits")
    return totals


def analyze_stderr(run_dir: pathlib.Path):
    text = (run_dir / "stderr.txt").read_text(encoding="utf-8", errors="replace") if (run_dir / "stderr.txt").exists() else ""
    name_rows = []
    elig_by_name = {}
    for match in NAME_PROFILE_RE.finditer(text):
        gd = match.groupdict()
        name_rows.append({
            "rank": int(gd["rank"]),
            "name": gd["name"],
            "role": role_of(gd["name"]),
            "layer": layer_of(gd["name"]),
            "calls": int(gd["calls"]),
            "total_ms_per_call": float(gd["total"]),
            "fallback_ms_per_call": float(gd["fallback"]),
            "decode_calls": int(gd["decode_calls"]),
            "decode_total_ms_per_call": float(gd["decode_total"]),
            "decode_fallback_ms_per_call": float(gd["decode_fallback"]),
            "prompt_calls": int(gd["prompt_calls"]),
            "prompt_total_ms_per_call": float(gd["prompt_total"]),
            "prompt_fallback_ms_per_call": float(gd["prompt_fallback"]),
        })
    for match in ELIG_RE.finditer(text):
        gd = match.groupdict()
        elig_by_name[gd["name"]] = {k: int(v) if v.isdigit() else v for k, v in gd.items() if k != "name"}
    decline_reasons = Counter(re.findall(r"down batch declined: reason=([^ ]+)", text))
    return name_rows, elig_by_name, decline_reasons


def metrics(run_dir: pathlib.Path):
    path = run_dir / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_report(runs_root: pathlib.Path, out: pathlib.Path):
    prompt_dirs = sorted(p for p in runs_root.iterdir() if p.is_dir() and (p / "metrics.json").exists())
    lines = [
        "# Kimi general dev profile breakdown",
        "",
        "This report uses dev baseline profile artifacts only. It does not use held-out test prompts.",
        "",
        "## Prompt Summary",
        "",
        "| prompt | tok/s | decode ms | down wall ms | upgate wall ms | fallback ms | top decline reason |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    all_declines = Counter()
    all_fallback = defaultdict(lambda: {"fallback_us": 0, "calls": 0, "gguf_calls": 0, "pack_mmap_calls": 0, "bytes": 0})
    all_name_rows = []
    all_down = defaultdict(lambda: {"calls": 0, "wall_ms": 0.0, "stage_ms": 0.0, "kernel_ms": 0.0, "staged_jobs": 0, "misses": 0, "hits": 0})
    all_upgate = defaultdict(lambda: {"calls": 0, "wall_ms": 0.0, "stage_ms": 0.0, "kernel_ms": 0.0, "up_ms": 0.0, "gate_ms": 0.0, "up_wait_ms": 0.0, "gate_wait_ms": 0.0, "misses": 0, "hits": 0})
    for run_dir in prompt_dirs:
        m = metrics(run_dir)
        fb, _ = analyze_fallback(run_dir)
        down = analyze_down(run_dir)
        upgate = analyze_upgate(run_dir)
        name_rows, elig, declines = analyze_stderr(run_dir)
        all_declines.update(declines)
        all_name_rows.extend((m.get("prompt_id", run_dir.name), row, elig.get(row["name"], {})) for row in name_rows)
        fallback_ms = sum(v["fallback_us"] for v in fb.values()) / 1000.0
        for k, v in fb.items():
            for field in v:
                all_fallback[k][field] += v[field]
        for k, v in down.items():
            for field in v:
                all_down[k][field] += v[field]
        for k, v in upgate.items():
            for field in v:
                all_upgate[k][field] += v[field]
        down_wall = sum(v["wall_ms"] for v in down.values())
        upgate_wall = sum(v["wall_ms"] for v in upgate.values())
        top_decline = declines.most_common(1)[0] if declines else ("", 0)
        lines.append(
            f"| `{m.get('prompt_id', run_dir.name)}` | {float(m.get('token_rate', 0)):.2f} | "
            f"{float(m.get('decode_ms', 0)):.0f} | {down_wall:.0f} | {upgate_wall:.0f} | "
            f"{fallback_ms:.0f} | {top_decline[0]}:{top_decline[1]} |"
        )

    lines.extend([
        "",
        "## Down Batch By Type",
        "",
        "| src0_type | role | calls | wall ms | stage ms | kernel ms | staged jobs | hit rate |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for (src0_type, role), v in sorted(all_down.items(), key=lambda kv: -kv[1]["wall_ms"]):
        hits = v["hits"]
        misses = v["misses"]
        hit_rate = 100.0 * hits / max(hits + misses, 1)
        lines.append(
            f"| {src0_type} | {role} | {v['calls']} | {v['wall_ms']:.0f} | "
            f"{v['stage_ms']:.0f} | {v['kernel_ms']:.0f} | {v['staged_jobs']} | {hit_rate:.1f}% |"
        )

    lines.extend([
        "",
        "## Up/Gate Batch By Type Pair",
        "",
        "| type pair | calls | wall ms | stage ms | kernel ms | up ms | gate ms | up wait ms | gate wait ms | hit rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for (type_pair, role), v in sorted(all_upgate.items(), key=lambda kv: -kv[1]["wall_ms"]):
        hits = v["hits"]
        misses = v["misses"]
        hit_rate = 100.0 * hits / max(hits + misses, 1)
        lines.append(
            f"| {type_pair} | {v['calls']} | {v['wall_ms']:.0f} | {v['stage_ms']:.0f} | "
            f"{v['kernel_ms']:.0f} | {v['up_ms']:.0f} | {v['gate_ms']:.0f} | "
            f"{v['up_wait_ms']:.0f} | {v['gate_wait_ms']:.0f} | {hit_rate:.1f}% |"
        )

    lines.extend([
        "",
        "## Fallback By Phase / Type / Role",
        "",
        "| phase | src0_type | role | fallback ms | calls | GGUF calls | pack mmap calls | bytes GiB |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ])
    for (phase, src0_type, role), v in sorted(all_fallback.items(), key=lambda kv: -kv[1]["fallback_us"])[:40]:
        lines.append(
            f"| {phase} | {src0_type} | {role} | {v['fallback_us'] / 1000.0:.0f} | "
            f"{v['calls']} | {v['gguf_calls']} | {v['pack_mmap_calls']} | {v['bytes'] / 1024**3:.2f} |"
        )

    lines.extend([
        "",
        "## Top CPU/MOE Name Profile Rows",
        "",
        "| prompt | layer | role | name | total ms/call | decode ms/call | decode fallback ms/call | src0_type | unsupported |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|",
    ])
    dedup_name_rows = {}
    for prompt_id, row, elig in all_name_rows:
        key = (prompt_id, row["name"])
        prev = dedup_name_rows.get(key)
        score = row["decode_total_ms_per_call"] * max(row["decode_calls"], 1)
        if prev is None or score > prev[3]:
            dedup_name_rows[key] = (prompt_id, row, elig, score)
    top_name_rows = sorted(
        (value[:3] for value in dedup_name_rows.values()),
        key=lambda item: item[1]["decode_total_ms_per_call"] * max(item[1]["decode_calls"], 1),
        reverse=True,
    )[:80]
    for prompt_id, row, elig in top_name_rows:
        lines.append(
            f"| `{prompt_id}` | {row['layer']} | {row['role']} | `{row['name']}` | "
            f"{row['total_ms_per_call']:.3f} | {row['decode_total_ms_per_call']:.3f} | "
            f"{row['decode_fallback_ms_per_call']:.3f} | {elig.get('src0_type', '')} | {elig.get('unsupported', '')} |"
        )

    lines.extend([
        "",
        "## Down Batch Decline Reasons",
        "",
        "| reason | count |",
        "|---|---:|",
    ])
    for reason, count in all_declines.most_common():
        lines.append(f"| {reason} | {count} |")

    lines.extend([
        "",
        "Interpretation:",
        "",
        "- The slow general prompts are not explained by iouring wait alone.",
        "- Down batch profile and name profile should be used to identify whether",
        "  the dominant cost is eligible GPU down batch wall time, unsupported CPU",
        "  fallback types, or prompt/prefill fallback materialization.",
        "- Any source optimization must first name the rows it is expected to reduce",
        "  and compute the best-case bound from this report.",
        "",
    ])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Break down Kimi general dev baseline profile artifacts.")
    parser.add_argument("--runs-root", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    write_report(args.runs_root, args.out)
    print(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
