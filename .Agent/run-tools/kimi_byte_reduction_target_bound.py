#!/usr/bin/env python3
import argparse
import collections
import csv
import json
import pathlib
import re
import statistics


EXPERT_PACK_RE = re.compile(r"iouring_bytes=(\d+)")
LAYER_RE = re.compile(r"blk\.(\d+)\.ffn_(up|gate|down)_exps\.weight")
HIT_RATE_RE = re.compile(r"hit_rate=([0-9.]+)%")
CACHE_LINE_RE = re.compile(r"slot=([0-9.]+) MiB hits=(\d+) misses=(\d+)")


def gib(nbytes):
    return nbytes / 1024**3


def pct(value):
    return 100.0 * value


def load_metrics(run_dir):
    path = run_dir / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def parse_iouring_bytes(metrics):
    text = str(metrics.get("expert_pack_0", ""))
    match = EXPERT_PACK_RE.search(text)
    return int(match.group(1)) if match else 0


def tensor_bytes_from_route_profile(run_dir):
    out = {}
    path = run_dir / "route-profile.csv"
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            tensor = row.get("tensor")
            if not tensor:
                continue
            try:
                out[tensor] = int(row["expert_bytes"])
            except (KeyError, ValueError):
                continue
    return out


def add_tensor_miss(rows, tensor, misses, tensor_bytes, prompt_id):
    if misses <= 0:
        return
    nbytes = tensor_bytes.get(tensor, 0)
    match = LAYER_RE.search(tensor)
    if match:
        layer = int(match.group(1))
        role = match.group(2)
    else:
        layer = -1
        role = "unknown"
    rows.append({
        "prompt_id": prompt_id,
        "tensor": tensor,
        "layer": layer,
        "role": role,
        "misses": misses,
        "bytes": misses * nbytes,
        "expert_bytes": nbytes,
        "estimated": False,
    })


def parse_hit_rate(metrics, group):
    key = "vram_down_0" if group == "down" else "vram_upgate_0"
    match = HIT_RATE_RE.search(str(metrics.get(key, "")))
    return float(match.group(1)) / 100.0 if match else 0.0


def parse_cache_line(metrics, key):
    match = CACHE_LINE_RE.search(str(metrics.get(key, "")))
    if not match:
        return {"slot_mib": 0.0, "hits": 0, "misses": 0}
    return {
        "slot_mib": float(match.group(1)),
        "hits": int(match.group(2)),
        "misses": int(match.group(3)),
    }


def mean_or_min(vals):
    if not vals:
        return 0.0
    return statistics.mean(vals)


def estimate_all_hit_moe_floor_ms_per_token(run_dir, decode_runs):
    """Estimate the MoE wall floor if active expert bytes were already in VRAM.

    This is still an optimistic bound. It uses all-hit rows where available and
    the minimum observed row wall as a fallback when a prompt has no all-hit
    sample for a path.
    """
    up_rows = 0
    up_hit_wall = []
    up_min_wall = []
    up_source = "all_hit_rows"
    up_path = run_dir / "up-gate-profile.csv"
    if up_path.exists():
        with up_path.open(newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                if row.get("mode") != "decode":
                    continue
                up_rows += 1
                wall = float(row.get("wall_ms", 0.0) or 0.0)
                up_min_wall.append(wall)
                misses = int(float(row.get("up_cache_misses", 0) or 0)) + int(float(row.get("gate_cache_misses", 0) or 0))
                if misses == 0:
                    up_hit_wall.append(wall)
    if not up_hit_wall and up_min_wall:
        up_hit_wall = [min(up_min_wall)]
        up_source = "min_row_fallback"

    down_rows = 0
    down_hit_wall = []
    down_min_wall = []
    down_source = "all_hit_rows"
    down_path = run_dir / "down-batch-profile.csv"
    if down_path.exists():
        with down_path.open(newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                down_rows += 1
                wall = float(row.get("wall_ms", 0.0) or 0.0)
                down_min_wall.append(wall)
                misses = int(float(row.get("cache_misses", 0) or 0))
                if misses == 0:
                    down_hit_wall.append(wall)
    if not down_hit_wall and down_min_wall:
        down_hit_wall = [min(down_min_wall)]
        down_source = "min_row_fallback"

    up_mean = mean_or_min(up_hit_wall)
    down_mean = mean_or_min(down_hit_wall)
    floor = ((up_rows * up_mean) + (down_rows * down_mean)) / decode_runs if decode_runs else 0.0
    return {
        "all_hit_moe_floor_ms_per_token": floor,
        "up_rows": up_rows,
        "down_rows": down_rows,
        "up_hit_mean_ms": up_mean,
        "down_hit_mean_ms": down_mean,
        "up_source": up_source,
        "down_source": down_source,
    }


def load_trace_estimated_misses(run_dir, prompt_id, metrics):
    path = run_dir / "route-trace.csv"
    if not path.exists():
        return []
    tensor_totals = collections.defaultdict(lambda: {"bytes": 0, "events": 0})
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            tensor = row.get("tensor", "")
            if not tensor:
                continue
            try:
                nbytes = int(row["expert_bytes"])
            except (KeyError, ValueError):
                continue
            tensor_totals[tensor]["bytes"] += nbytes
            tensor_totals[tensor]["events"] += 1

    out = []
    hit_rates = {
        "up": parse_hit_rate(metrics, "upgate"),
        "gate": parse_hit_rate(metrics, "upgate"),
        "down": parse_hit_rate(metrics, "down"),
    }
    for tensor, vals in tensor_totals.items():
        match = LAYER_RE.search(tensor)
        if match:
            layer = int(match.group(1))
            role = match.group(2)
        else:
            layer = -1
            role = "unknown"
        miss_frac = max(0.0, min(1.0, 1.0 - hit_rates.get(role, 0.0)))
        out.append({
            "prompt_id": prompt_id,
            "tensor": tensor,
            "layer": layer,
            "role": role,
            "misses": int(round(vals["events"] * miss_frac)),
            "bytes": int(round(vals["bytes"] * miss_frac)),
            "expert_bytes": int(round(vals["bytes"] / vals["events"])) if vals["events"] else 0,
            "estimated": True,
        })
    return out


def load_profile_misses(run_dir, prompt_id):
    tensor_bytes = tensor_bytes_from_route_profile(run_dir)
    rows = []

    down = run_dir / "down-batch-profile.csv"
    if down.exists():
        with down.open(newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                try:
                    misses = int(row.get("cache_misses", 0))
                except ValueError:
                    misses = 0
                add_tensor_miss(rows, row.get("tensor", ""), misses, tensor_bytes, prompt_id)

    upgate = run_dir / "up-gate-profile.csv"
    if upgate.exists():
        with upgate.open(newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                try:
                    up_misses = int(row.get("up_cache_misses", 0))
                    gate_misses = int(row.get("gate_cache_misses", 0))
                except ValueError:
                    up_misses = gate_misses = 0
                add_tensor_miss(rows, row.get("up_tensor", ""), up_misses, tensor_bytes, prompt_id)
                add_tensor_miss(rows, row.get("gate_tensor", ""), gate_misses, tensor_bytes, prompt_id)
    if rows:
        return rows
    metrics = load_metrics(run_dir) or {}
    return load_trace_estimated_misses(run_dir, prompt_id, metrics)


def summarize_prompt(run_dir, target_tps, peak_gib_s):
    metrics = load_metrics(run_dir)
    if metrics is None:
        return None, []
    prompt_id = metrics.get("prompt_id", run_dir.name)
    decode_runs = int(metrics.get("decode_runs", 0))
    decode_s = float(metrics.get("decode_ms", 0.0)) / 1000.0
    token_rate = float(metrics.get("token_rate", 0.0))
    iouring_bytes = parse_iouring_bytes(metrics)
    moved_gib_per_token = gib(iouring_bytes) / decode_runs if decode_runs else 0.0
    current_gib_s = gib(iouring_bytes) / decode_s if decode_s > 0 else 0.0
    target_decode_s = decode_runs / target_tps if target_tps > 0 else 0.0
    ratio_at_peak = (target_decode_s * peak_gib_s) / gib(iouring_bytes) if iouring_bytes else 0.0
    ratio_at_current = (target_decode_s * current_gib_s) / gib(iouring_bytes) if iouring_bytes else 0.0
    floor = estimate_all_hit_moe_floor_ms_per_token(run_dir, decode_runs)
    target_ms_per_token = 1000.0 / target_tps if target_tps > 0 else 0.0
    io_budget_ms_per_token_after_floor = max(0.0, target_ms_per_token - floor["all_hit_moe_floor_ms_per_token"])
    budget_gib_per_token_at_peak_after_floor = io_budget_ms_per_token_after_floor / 1000.0 * peak_gib_s
    budget_gib_per_token_at_current_after_floor = io_budget_ms_per_token_after_floor / 1000.0 * current_gib_s
    ratio_at_peak_after_floor = budget_gib_per_token_at_peak_after_floor / moved_gib_per_token if moved_gib_per_token else 0.0
    ratio_at_current_after_floor = budget_gib_per_token_at_current_after_floor / moved_gib_per_token if moved_gib_per_token else 0.0
    up_cache = parse_cache_line(metrics, "vram_upgate_0")
    down_cache = parse_cache_line(metrics, "vram_down_0")
    active_gib_per_token = (
        ((up_cache["hits"] + up_cache["misses"]) * up_cache["slot_mib"]) +
        ((down_cache["hits"] + down_cache["misses"]) * down_cache["slot_mib"])
    ) / 1024.0 / decode_runs if decode_runs else 0.0
    row = {
        "prompt_id": prompt_id,
        "decode_runs": decode_runs,
        "token_rate": token_rate,
        "decode_s": decode_s,
        "target_decode_s": target_decode_s,
        "iouring_gib": gib(iouring_bytes),
        "moved_gib_per_token": moved_gib_per_token,
        "active_gib_per_token": active_gib_per_token,
        "current_effective_gib_s": current_gib_s,
        "required_byte_ratio_at_peak": min(ratio_at_peak, 1.0),
        "required_byte_ratio_at_current": min(ratio_at_current, 1.0),
        "required_reduction_at_peak_pct": pct(max(0.0, 1.0 - min(ratio_at_peak, 1.0))),
        "required_reduction_at_current_pct": pct(max(0.0, 1.0 - min(ratio_at_current, 1.0))),
        "all_hit_moe_floor_ms_per_token": floor["all_hit_moe_floor_ms_per_token"],
        "io_budget_ms_per_token_after_floor": io_budget_ms_per_token_after_floor,
        "budget_gib_per_token_at_peak_after_floor": budget_gib_per_token_at_peak_after_floor,
        "budget_gib_per_token_at_current_after_floor": budget_gib_per_token_at_current_after_floor,
        "required_byte_ratio_at_peak_after_floor": min(ratio_at_peak_after_floor, 1.0),
        "required_byte_ratio_at_current_after_floor": min(ratio_at_current_after_floor, 1.0),
        "required_reduction_at_peak_after_floor_pct": pct(max(0.0, 1.0 - min(ratio_at_peak_after_floor, 1.0))),
        "required_reduction_at_current_after_floor_pct": pct(max(0.0, 1.0 - min(ratio_at_current_after_floor, 1.0))),
        "floor_detail": floor,
        "quality": metrics.get("quality", ""),
        "memory_peak": int(metrics.get("memory.peak", metrics.get("memory_peak", 0)) or 0),
    }
    return row, load_profile_misses(run_dir, prompt_id)


def aggregate_misses(miss_rows, key_fields):
    counts = collections.defaultdict(lambda: {"bytes": 0, "misses": 0, "estimated_rows": 0, "rows": 0})
    for row in miss_rows:
        key = tuple(row[field] for field in key_fields)
        counts[key]["bytes"] += row["bytes"]
        counts[key]["misses"] += row["misses"]
        counts[key]["rows"] += 1
        counts[key]["estimated_rows"] += 1 if row.get("estimated") else 0
    out = []
    for key, vals in counts.items():
        item = dict(zip(key_fields, key))
        item.update(vals)
        out.append(item)
    return sorted(out, key=lambda r: r["bytes"], reverse=True)


def write_report(out, prompt_rows, miss_rows, target_tps, peak_gib_s, evidence_scope):
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "target_tps": target_tps,
        "peak_gib_s": peak_gib_s,
        "evidence_scope": evidence_scope,
        "prompts": prompt_rows,
        "miss_by_role": aggregate_misses(miss_rows, ["role"]),
        "miss_by_layer_role": aggregate_misses(miss_rows, ["layer", "role"]),
        "miss_by_tensor": aggregate_misses(miss_rows, ["tensor", "layer", "role"]),
    }
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Kimi byte-reduction target bound",
        "",
        f"Evidence scope: `{evidence_scope}`.",
        "",
        "This is a bound report only. It must not be used to select prompt-specific experts or tune held-out prompts.",
        "",
        f"- target token rate: `{target_tps:.2f} tok/s`",
        f"- optimistic sustained movement bandwidth: `{peak_gib_s:.2f} GiB/s`",
        "",
        "## Prompt Bound",
        "",
        "| prompt | tok/s | decode tok | decode s | moved GiB | moved GiB/token | active GiB/token | eff GiB/s | transfer-only ratio @ peak | floor ms/token | IO budget GiB/token @ peak | ratio @ peak after floor | reduction @ peak after floor | ratio @ current eff after floor |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in prompt_rows:
        lines.append(
            f"| `{row['prompt_id']}` | {row['token_rate']:.2f} | {row['decode_runs']} | "
            f"{row['decode_s']:.2f} | {row['iouring_gib']:.2f} | "
            f"{row['moved_gib_per_token']:.2f} | {row['active_gib_per_token']:.2f} | "
            f"{row['current_effective_gib_s']:.2f} | "
            f"{row['required_byte_ratio_at_peak']:.2f}x | "
            f"{row['all_hit_moe_floor_ms_per_token']:.1f} | "
            f"{row['budget_gib_per_token_at_peak_after_floor']:.2f} | "
            f"{row['required_byte_ratio_at_peak_after_floor']:.2f}x | "
            f"{row['required_reduction_at_peak_after_floor_pct']:.1f}% | "
            f"{row['required_byte_ratio_at_current_after_floor']:.2f}x |"
        )

    if prompt_rows:
        moved = [row["moved_gib_per_token"] for row in prompt_rows]
        active = [row["active_gib_per_token"] for row in prompt_rows]
        floor_vals = [row["all_hit_moe_floor_ms_per_token"] for row in prompt_rows]
        ratios = [row["required_byte_ratio_at_peak_after_floor"] for row in prompt_rows]
        lines.extend([
            "",
            "## Aggregate Bound",
            "",
            f"- median moved bytes: `{statistics.median(moved):.2f} GiB/token`; mean `{statistics.mean(moved):.2f} GiB/token`.",
            f"- median active expert footprint before cache: `{statistics.median(active):.2f} GiB/token`; mean `{statistics.mean(active):.2f} GiB/token`.",
            f"- median all-hit MoE floor estimate: `{statistics.median(floor_vals):.1f} ms/token`; mean `{statistics.mean(floor_vals):.1f} ms/token`.",
            f"- median required byte ratio at `{peak_gib_s:.2f} GiB/s` after floor: `{statistics.median(ratios):.2f}x`; worst `{min(ratios):.2f}x`.",
        ])

    lines.extend([
        "",
        "## Miss Bytes By Role",
        "",
        "| role | miss GiB | misses | share | estimated rows |",
        "|---|---:|---:|---:|---:|",
    ])
    total_miss = sum(row["bytes"] for row in miss_rows)
    for row in payload["miss_by_role"]:
        lines.append(
            f"| {row['role']} | {gib(row['bytes']):.2f} | {row['misses']} | "
            f"{pct(row['bytes'] / total_miss) if total_miss else 0.0:.1f}% | "
            f"{row['estimated_rows']}/{row['rows']} |"
        )

    lines.extend([
        "",
        "## Top Layer/Role Miss Bytes",
        "",
        "| layer | role | miss GiB | misses | share | estimated rows |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    for row in payload["miss_by_layer_role"][:30]:
        lines.append(
            f"| {row['layer']} | {row['role']} | {gib(row['bytes']):.2f} | {row['misses']} | "
            f"{pct(row['bytes'] / total_miss) if total_miss else 0.0:.1f}% | "
            f"{row['estimated_rows']}/{row['rows']} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- The peak-bandwidth ratio is an optimistic lower bound: it assumes expert movement is the only remaining bottleneck",
        "  and that runtime can sustain the pure IO bench bandwidth during decode.",
        "- The after-floor ratio is stricter: it reserves time for an optimistic all-hit MoE wall estimate before assigning",
        "  the remaining token budget to expert movement.",
        "- If the required byte ratio is far below 1.0, queue tuning and prediction cannot be enough; expert bytes must shrink.",
        "- When down/upgate miss profiles are absent, role/layer miss bytes are estimated from route trace bytes",
        "  scaled by the prompt-level VRAM cache hit rates in metrics.json.",
        "- A viable representation change must preserve quality first, then reduce moved expert bytes by the required ratio on dev before held-out testing.",
        "",
    ])
    out.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Compute Kimi expert byte-reduction target from dev profile runs.")
    parser.add_argument("--runs-root", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--target-tps", type=float, default=5.0)
    parser.add_argument("--peak-gib-s", type=float, default=10.4)
    parser.add_argument("--evidence-scope", default="dev")
    args = parser.parse_args()

    prompt_rows = []
    miss_rows = []
    for run_dir in sorted(p for p in args.runs_root.iterdir() if p.is_dir()):
        row, misses = summarize_prompt(run_dir, args.target_tps, args.peak_gib_s)
        if row is None:
            continue
        prompt_rows.append(row)
        miss_rows.extend(misses)
    if not prompt_rows:
        raise SystemExit("no prompt metrics found")
    write_report(args.out, prompt_rows, miss_rows, args.target_tps, args.peak_gib_s, args.evidence_scope)
    print(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
