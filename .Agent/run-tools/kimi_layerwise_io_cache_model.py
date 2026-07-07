#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TENSOR_RE = re.compile(r"blk\.(\d+)\.ffn_(up|gate|down)_exps\.weight")


def gib(value: int | float) -> float:
    return float(value) / 1024**3


def pct(num: int | float, den: int | float) -> float:
    return 100.0 * float(num) / float(den) if den else 0.0


def parse_int_field(text: str, name: str, default: int = 0) -> int:
    match = re.search(rf"{re.escape(name)}=(\d+)", str(text))
    return int(match.group(1)) if match else default


def parse_float_field(text: str, name: str, default: float = 0.0) -> float:
    match = re.search(rf"{re.escape(name)}=([0-9.]+)", str(text))
    return float(match.group(1)) if match else default


def parse_batch_hist(text: str) -> dict[str, int]:
    match = re.search(r"batch_hist=([^\s]+)", str(text))
    if not match:
        return {}
    out: dict[str, int] = {}
    for item in match.group(1).split(","):
        if ":" not in item:
            continue
        key, value = item.split(":", 1)
        out[key] = int(value)
    return out


def tensor_parts(tensor: str) -> tuple[int, str]:
    match = TENSOR_RE.search(tensor)
    if not match:
        return -1, "other"
    layer = int(match.group(1))
    role = match.group(2)
    kind = "upgate" if role in {"up", "gate"} else "down"
    return layer, kind


def role_of(tensor: str) -> str:
    match = TENSOR_RE.search(tensor)
    return match.group(2) if match else "other"


@dataclass(frozen=True)
class Key:
    tensor: str
    expert_idx: int
    expert_bytes: int

    @property
    def layer_kind(self) -> tuple[int, str]:
        return tensor_parts(self.tensor)


@dataclass
class Call:
    prompt: str
    tensor: str
    layer: int
    kind: str
    expert_bytes: int
    experts: set[int]


def iter_run_dirs(root: Path) -> list[Path]:
    return sorted(
        path for path in root.iterdir()
        if path.is_dir() and (path / "route-profile.csv").exists() and (path / "route-trace.csv").exists()
    )


def load_metrics(run_dir: Path) -> dict:
    path = run_dir / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_profile(run_dir: Path) -> list[dict]:
    rows = []
    with (run_dir / "route-profile.csv").open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            tensor = row.get("tensor", "")
            if not tensor:
                continue
            layer, kind = tensor_parts(tensor)
            try:
                count = int(row.get("count", "0") or 0)
                expert_idx = int(row.get("expert_idx", "0") or 0)
                expert_bytes = int(row.get("expert_bytes", "0") or 0)
            except ValueError:
                continue
            if count <= 0 or expert_bytes <= 0 or kind == "other":
                continue
            key = Key(tensor, expert_idx, expert_bytes)
            rows.append({
                "prompt": run_dir.name,
                "key": key,
                "tensor": tensor,
                "role": role_of(tensor),
                "layer": layer,
                "kind": kind,
                "expert_idx": expert_idx,
                "expert_bytes": expert_bytes,
                "count": count,
                "traffic_bytes": count * expert_bytes,
            })
    return rows


def load_calls(run_dir: Path) -> list[Call]:
    calls: list[Call] = []
    current_tensor = ""
    current_experts: set[int] = set()
    current_bytes = 0

    def flush() -> None:
        nonlocal current_tensor, current_experts, current_bytes
        if not current_tensor:
            return
        layer, kind = tensor_parts(current_tensor)
        if kind != "other":
            calls.append(Call(run_dir.name, current_tensor, layer, kind, current_bytes, set(current_experts)))
        current_tensor = ""
        current_experts = set()
        current_bytes = 0

    with (run_dir / "route-trace.csv").open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            tensor = row.get("tensor", "")
            if not tensor:
                continue
            if tensor != current_tensor:
                flush()
                current_tensor = tensor
            current_experts.add(int(row["expert_idx"]))
            current_bytes = int(row["expert_bytes"])
    flush()
    return calls


def fixed_hotset(counts: collections.Counter[Key], slots: int) -> set[Key]:
    ranked = sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0].tensor, item[0].expert_idx, item[0].expert_bytes),
    )
    return {key for key, _ in ranked[:slots]}


def summarize_metrics(prompt_metrics: dict[str, dict]) -> dict:
    prompts = []
    token_rates = []
    ttfts = []
    memory_peaks = []
    decode_ms = []
    total_iouring_bytes = 0
    total_iouring_wait_us = 0
    total_iouring_reads = 0
    total_hits_down = total_misses_down = 0
    total_hits_upgate = total_misses_upgate = 0
    batch_hist = collections.Counter()

    for prompt, metrics in sorted(prompt_metrics.items()):
        prompts.append(prompt)
        token_rates.append(float(metrics.get("token_rate", 0.0) or 0.0))
        ttfts.append(float(metrics.get("ttft_ms", 0.0) or 0.0))
        memory_peaks.append(int(metrics.get("memory.peak", 0) or 0))
        decode_ms.append(float(metrics.get("decode_ms", 0.0) or 0.0))
        expert_pack = str(metrics.get("expert_pack_0", ""))
        total_iouring_bytes += parse_int_field(expert_pack, "iouring_bytes")
        total_iouring_wait_us += parse_int_field(expert_pack, "iouring_wait_us")
        total_iouring_reads += parse_int_field(expert_pack, "iouring_reads")
        total_hits_down += parse_int_field(metrics.get("vram_down_0", ""), "hits")
        total_misses_down += parse_int_field(metrics.get("vram_down_0", ""), "misses")
        total_hits_upgate += parse_int_field(metrics.get("vram_upgate_0", ""), "hits")
        total_misses_upgate += parse_int_field(metrics.get("vram_upgate_0", ""), "misses")
        batch_hist.update(parse_batch_hist(metrics.get("expert_pack_iouring_0", "")))

    return {
        "prompt_count": len(prompts),
        "prompts": prompts,
        "token_rate_avg": sum(token_rates) / len(token_rates) if token_rates else 0.0,
        "token_rate_min": min(token_rates) if token_rates else 0.0,
        "token_rate_max": max(token_rates) if token_rates else 0.0,
        "ttft_ms_avg": sum(ttfts) / len(ttfts) if ttfts else 0.0,
        "ttft_ms_max": max(ttfts) if ttfts else 0.0,
        "memory_peak_max": max(memory_peaks) if memory_peaks else 0,
        "decode_s_total": sum(decode_ms) / 1000.0,
        "iouring_bytes_gib": gib(total_iouring_bytes),
        "iouring_wait_s": total_iouring_wait_us / 1_000_000.0,
        "iouring_reads": total_iouring_reads,
        "iouring_observed_gibs": gib(total_iouring_bytes) / max(total_iouring_wait_us / 1_000_000.0, 1e-9),
        "vram_down_hit_rate": pct(total_hits_down, total_hits_down + total_misses_down),
        "vram_upgate_hit_rate": pct(total_hits_upgate, total_hits_upgate + total_misses_upgate),
        "expert_pack_batch_hist": dict(sorted(batch_hist.items())),
    }


def build_traffic_models(profile_rows: list[dict], group_slots: dict[str, int]) -> dict:
    counts_by_group: dict[str, collections.Counter[Key]] = {
        "down": collections.Counter(),
        "upgate": collections.Counter(),
    }
    prompt_traffic_by_key: dict[Key, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    layer_kind_total: dict[tuple[int, str], dict] = collections.defaultdict(lambda: {
        "events": 0,
        "traffic_bytes": 0,
        "unique_keys": set(),
        "prompt_set": set(),
        "role_traffic": collections.Counter(),
    })

    for row in profile_rows:
        key = row["key"]
        kind = row["kind"]
        counts_by_group[kind][key] += int(row["traffic_bytes"])
        prompt_traffic_by_key[key][row["prompt"]] += int(row["traffic_bytes"])
        lk = (row["layer"], kind)
        layer_kind_total[lk]["events"] += int(row["count"])
        layer_kind_total[lk]["traffic_bytes"] += int(row["traffic_bytes"])
        layer_kind_total[lk]["unique_keys"].add(key)
        layer_kind_total[lk]["prompt_set"].add(row["prompt"])
        layer_kind_total[lk]["role_traffic"][row["role"]] += int(row["traffic_bytes"])

    hotsets = {
        group: fixed_hotset(counts, group_slots[group])
        for group, counts in counts_by_group.items()
    }

    current_hot_rows = []
    next_gain_rows = []
    layer_rows = []
    for lk, rec in sorted(layer_kind_total.items()):
        layer, kind = lk
        traffic = int(rec["traffic_bytes"])
        keys = set(rec["unique_keys"])
        hot_keys = keys & hotsets[kind]
        hit_traffic = sum(counts_by_group[kind][key] for key in hot_keys)
        miss_traffic = traffic - hit_traffic
        sorted_hot = sorted(
            hot_keys,
            key=lambda key: (counts_by_group[kind][key], key.tensor, key.expert_idx, key.expert_bytes),
        )
        sorted_miss = sorted(
            keys - hotsets[kind],
            key=lambda key: (-counts_by_group[kind][key], key.tensor, key.expert_idx, key.expert_bytes),
        )
        removable64 = sorted_hot[:64]
        add64 = sorted_miss[:64]
        removable64_traffic = sum(counts_by_group[kind][key] for key in removable64)
        add64_traffic = sum(counts_by_group[kind][key] for key in add64)
        add64_stability = (
            sum(len(prompt_traffic_by_key[key]) for key in add64) / len(add64)
            if add64 else 0.0
        )
        hot64_stability = (
            sum(len(prompt_traffic_by_key[key]) for key in removable64) / len(removable64)
            if removable64 else 0.0
        )
        row = {
            "layer": layer,
            "kind": kind,
            "events": int(rec["events"]),
            "traffic_bytes": traffic,
            "traffic_gib": gib(traffic),
            "unique_keys": len(keys),
            "prompt_count": len(rec["prompt_set"]),
            "proxy_hot_keys": len(hot_keys),
            "proxy_hit_traffic_bytes": hit_traffic,
            "proxy_miss_traffic_bytes": miss_traffic,
            "proxy_hit_traffic_pct": pct(hit_traffic, traffic),
            "next64_gain_bytes": add64_traffic,
            "next64_gain_gib": gib(add64_traffic),
            "next64_gain_per_slot_mib": add64_traffic / max(len(add64), 1) / 1024**2,
            "next64_avg_prompt_coverage": add64_stability,
            "remove64_cost_bytes": removable64_traffic,
            "remove64_cost_gib": gib(removable64_traffic),
            "remove64_cost_per_slot_mib": removable64_traffic / max(len(removable64), 1) / 1024**2,
            "remove64_avg_prompt_coverage": hot64_stability,
            "role_traffic": dict(rec["role_traffic"]),
        }
        layer_rows.append(row)
        if add64:
            next_gain_rows.append(row)
        if removable64:
            current_hot_rows.append(row)

    return {
        "counts_by_group": counts_by_group,
        "hotsets": hotsets,
        "prompt_traffic_by_key": prompt_traffic_by_key,
        "layer_rows": layer_rows,
        "cache_increase_candidates": sorted(
            next_gain_rows,
            key=lambda row: (row["next64_gain_bytes"], row["next64_avg_prompt_coverage"]),
            reverse=True,
        )[:20],
        "cache_decrease_candidates": sorted(
            current_hot_rows,
            key=lambda row: (row["remove64_cost_bytes"], -row["proxy_hit_traffic_pct"]),
        )[:20],
    }


def build_swap_plans(
    counts_by_group: dict[str, collections.Counter[Key]],
    hotsets: dict[str, set[Key]],
    target_free_mib: list[int],
    throughput_gibs: float,
    decode_s_total: float,
) -> list[dict]:
    hot_keys: set[Key] = set()
    all_keys: set[Key] = set()
    traffic_by_key: dict[Key, int] = {}
    for group, counts in counts_by_group.items():
        hot_keys.update(hotsets[group])
        all_keys.update(counts)
        for key, traffic in counts.items():
            traffic_by_key[key] = traffic

    cold_keys = all_keys - hot_keys

    def density(key: Key) -> float:
        return traffic_by_key[key] / max(key.expert_bytes, 1)

    removable = sorted(
        hot_keys,
        key=lambda key: (density(key), key.tensor, key.expert_idx, key.expert_bytes),
    )
    addable = sorted(
        cold_keys,
        key=lambda key: (-density(key), key.tensor, key.expert_idx, key.expert_bytes),
    )
    plans = []
    for mib in target_free_mib:
        target_bytes = mib * 1024**2
        removed: list[Key] = []
        removed_bytes = 0
        removed_traffic = 0
        for key in removable:
            if removed_bytes >= target_bytes:
                break
            removed.append(key)
            removed_bytes += key.expert_bytes
            removed_traffic += traffic_by_key[key]

        added: list[Key] = []
        added_bytes = 0
        added_traffic = 0
        for key in addable:
            if added_bytes + key.expert_bytes > removed_bytes:
                continue
            added.append(key)
            added_bytes += key.expert_bytes
            added_traffic += traffic_by_key[key]
            if added_bytes >= removed_bytes * 0.98:
                break

        net_bytes = added_traffic - removed_traffic
        estimated_saved_s = gib(net_bytes) / max(throughput_gibs, 1e-9)
        estimated_decode_improvement_pct = pct(estimated_saved_s, decode_s_total)
        remove_by_kind = collections.Counter(tensor_parts(key.tensor)[1] for key in removed)
        add_by_kind = collections.Counter(tensor_parts(key.tensor)[1] for key in added)
        plans.append({
            "freed_mib_target": mib,
            "removed_keys": len(removed),
            "removed_mib": removed_bytes / 1024**2,
            "removed_traffic_gib": gib(removed_traffic),
            "added_keys": len(added),
            "added_mib": added_bytes / 1024**2,
            "added_traffic_gib": gib(added_traffic),
            "net_traffic_reduction_gib": gib(net_bytes),
            "estimated_saved_s": estimated_saved_s,
            "estimated_decode_improvement_pct": estimated_decode_improvement_pct,
            "remove_by_kind": dict(remove_by_kind),
            "add_by_kind": dict(add_by_kind),
        })
    return plans


def train_priors(calls: Iterable[Call]) -> dict[str, list[int]]:
    counts: dict[str, collections.Counter[int]] = collections.defaultdict(collections.Counter)
    for call in calls:
        for expert in call.experts:
            counts[call.tensor][expert] += 1
    return {tensor: [expert for expert, _ in counter.most_common()] for tensor, counter in counts.items()}


def evaluate_overfetch(calls_by_prompt: dict[str, list[Call]], top_k_values: list[int]) -> tuple[list[dict], list[dict]]:
    by_layer_topk: dict[tuple[int, str, int], dict] = collections.defaultdict(lambda: {
        "calls": 0,
        "actual_events": 0,
        "predicted_events": 0,
        "hit_events": 0,
        "actual_bytes": 0,
        "predicted_bytes": 0,
        "hit_bytes": 0,
    })

    for heldout_prompt, heldout_calls in sorted(calls_by_prompt.items()):
        train_calls = [call for prompt, calls in calls_by_prompt.items() if prompt != heldout_prompt for call in calls]
        priors = train_priors(train_calls)
        for call in heldout_calls:
            actual_events = len(call.experts)
            for top_k in top_k_values:
                prior = priors.get(call.tensor, [])
                predicted = set(prior[:top_k])
                hit_events = len(call.experts & predicted)
                bucket = by_layer_topk[(call.layer, call.kind, top_k)]
                bucket["calls"] += 1
                bucket["actual_events"] += actual_events
                bucket["predicted_events"] += len(predicted)
                bucket["hit_events"] += hit_events
                bucket["actual_bytes"] += actual_events * call.expert_bytes
                bucket["predicted_bytes"] += len(predicted) * call.expert_bytes
                bucket["hit_bytes"] += hit_events * call.expert_bytes

    rows = []
    for (layer, kind, top_k), bucket in sorted(by_layer_topk.items()):
        actual_bytes = bucket["actual_bytes"]
        predicted_bytes = bucket["predicted_bytes"]
        hit_bytes = bucket["hit_bytes"]
        false_bytes = max(predicted_bytes - hit_bytes, 0)
        row = {
            "layer": layer,
            "kind": kind,
            "top_k": top_k,
            "calls": bucket["calls"],
            "actual_events": bucket["actual_events"],
            "actual_bytes": actual_bytes,
            "actual_gib": gib(actual_bytes),
            "byte_recall": hit_bytes / actual_bytes if actual_bytes else 0.0,
            "precision": bucket["hit_events"] / bucket["predicted_events"] if bucket["predicted_events"] else 0.0,
            "predicted_to_actual_byte_ratio": predicted_bytes / actual_bytes if actual_bytes else 0.0,
            "false_to_actual_byte_ratio": false_bytes / actual_bytes if actual_bytes else 0.0,
            "extra_to_actual_byte_ratio": max(predicted_bytes - actual_bytes, 0) / actual_bytes if actual_bytes else 0.0,
        }
        rows.append(row)

    candidates = [
        row for row in rows
        if row["actual_gib"] >= 1.0
        and row["byte_recall"] >= 0.35
        and row["false_to_actual_byte_ratio"] <= 0.25
    ]
    candidates = sorted(candidates, key=lambda row: (row["byte_recall"], row["actual_bytes"]), reverse=True)[:20]
    return rows, candidates


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def md_table(rows: list[dict], columns: list[tuple[str, str]], limit: int = 10) -> list[str]:
    lines = []
    lines.append("| " + " | ".join(title for title, _ in columns) + " |")
    lines.append("| " + " | ".join("---:" if key.endswith(("_gib", "_pct", "_ratio", "_mib", "_coverage")) or key in {"layer", "top_k", "calls", "unique_keys"} else "---" for _, key in columns) + " |")
    for row in rows[:limit]:
        values = []
        for _, key in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_markdown(path: Path, report: dict) -> None:
    metrics = report["metrics_summary"]
    lines = [
        "# GP63 layer-wise IO/cache model",
        "",
        f"Timestamp: `{report['timestamp']}`.",
        "",
        "Scope: dev-only Phase 1 analysis for GP63. Held-out test prompts were not used.",
        "",
        "## Inputs",
        "",
        f"- Command: `{report['command']}`",
        f"- Runs root: `{report['runs_root']}`",
        f"- Prompts: `{metrics['prompt_count']}`",
        f"- Average token rate: `{metrics['token_rate_avg']:.3f} tok/s`",
        f"- Token-rate range: `{metrics['token_rate_min']:.3f}-{metrics['token_rate_max']:.3f} tok/s`",
        f"- Max TTFT: `{metrics['ttft_ms_max']:.2f} ms`",
        f"- Max memory peak: `{metrics['memory_peak_max']}` bytes",
        f"- Aggregate expert-pack iouring bytes: `{metrics['iouring_bytes_gib']:.2f} GiB`",
        f"- Aggregate expert-pack iouring wait: `{metrics['iouring_wait_s']:.2f} s`",
        f"- Observed bytes/wait throughput: `{metrics['iouring_observed_gibs']:.2f} GiB/s`",
        f"- Aggregate down VRAM hit rate: `{metrics['vram_down_hit_rate']:.2f}%`",
        f"- Aggregate up/gate VRAM hit rate: `{metrics['vram_upgate_hit_rate']:.2f}%`",
        "",
        "## Interpretation",
        "",
        "- This is a demand-side and trace-side model, not a runtime SOTA claim.",
        "- The current profiles expose aggregate VRAM cache hits by group, but not exact per-layer hit/miss events.",
        "- Per-layer cache rows therefore use a fixed prompt-agnostic LFU proxy with the current slot counts.",
        "- Overfetch rows use leave-one-prompt-out static priors and are gated by false-byte ratio.",
        "",
        "## Cache-Increase Candidates",
        "",
    ]
    lines.extend(md_table(report["cache_increase_candidates"], [
        ("layer", "layer"),
        ("kind", "kind"),
        ("traffic GiB", "traffic_gib"),
        ("proxy hit %", "proxy_hit_traffic_pct"),
        ("next64 GiB", "next64_gain_gib"),
        ("MiB/slot", "next64_gain_per_slot_mib"),
        ("avg prompts", "next64_avg_prompt_coverage"),
    ]))
    lines.extend(["", "## Cache-Decrease Candidates", ""])
    lines.extend(md_table(report["cache_decrease_candidates"], [
        ("layer", "layer"),
        ("kind", "kind"),
        ("traffic GiB", "traffic_gib"),
        ("proxy hit %", "proxy_hit_traffic_pct"),
        ("remove64 GiB", "remove64_cost_gib"),
        ("MiB/slot", "remove64_cost_per_slot_mib"),
        ("avg prompts", "remove64_avg_prompt_coverage"),
    ]))
    lines.extend(["", "## Bounded Overfetch Candidates", ""])
    if report["overfetch_candidates"]:
        lines.extend(md_table(report["overfetch_candidates"], [
            ("layer", "layer"),
            ("kind", "kind"),
            ("top K", "top_k"),
            ("actual GiB", "actual_gib"),
            ("byte recall", "byte_recall"),
            ("precision", "precision"),
            ("false/actual", "false_to_actual_byte_ratio"),
        ]))
    else:
        lines.append("No leave-one-prompt-out static-prior overfetch setting passed the GP63 false-byte gate.")
    lines.extend(["", "## Same-Budget Cache Swap Bound", ""])
    lines.extend(md_table(report["swap_plans"], [
        ("free MiB", "freed_mib_target"),
        ("removed", "removed_keys"),
        ("added", "added_keys"),
        ("net GiB", "net_traffic_reduction_gib"),
        ("saved s", "estimated_saved_s"),
        ("decode %", "estimated_decode_improvement_pct"),
    ]))
    lines.extend(["", "## Do-Not-Touch Layers", ""])
    lines.extend(md_table(report["do_not_touch_layers"], [
        ("layer", "layer"),
        ("kind", "kind"),
        ("traffic GiB", "traffic_gib"),
        ("proxy hit %", "proxy_hit_traffic_pct"),
        ("next64 GiB", "next64_gain_gib"),
        ("remove64 GiB", "remove64_cost_gib"),
    ]))
    lines.extend([
        "",
        "## Phase 2 Gate",
        "",
        f"- Best cache-increase next64 gain: `{report['best_next64_gain_gib']:.3f} GiB`.",
        f"- Best accepted overfetch candidate count: `{len(report['overfetch_candidates'])}`.",
        f"- Best same-budget cache-swap decode improvement estimate: `{report['best_swap_decode_improvement_pct']:.3f}%`.",
        f"- Phase 2 recommendation: `{report['phase2_recommendation']}`.",
        "",
        "Required next action:",
        "",
        report["next_action"],
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build GP63 layer-wise IO/cache model from Kimi dev profiles.")
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-k", default="8,10,12,16")
    parser.add_argument("--timestamp", default="2026-07-07T12:45:00+08:00")
    args = parser.parse_args()

    run_dirs = iter_run_dirs(args.runs_root)
    if not run_dirs:
        raise RuntimeError(f"no prompt run dirs found under {args.runs_root}")

    prompt_metrics = {run_dir.name: load_metrics(run_dir) for run_dir in run_dirs}
    metrics_summary = summarize_metrics(prompt_metrics)
    profile_rows = [row for run_dir in run_dirs for row in load_profile(run_dir)]
    calls_by_prompt = {run_dir.name: load_calls(run_dir) for run_dir in run_dirs}

    first_metrics = next(iter(prompt_metrics.values()))
    group_slots = {
        "down": parse_int_field(first_metrics.get("vram_down_0", ""), "slots", 766),
        "upgate": parse_int_field(first_metrics.get("vram_upgate_0", ""), "slots", 1735),
    }
    traffic = build_traffic_models(profile_rows, group_slots)
    swap_plans = build_swap_plans(
        traffic["counts_by_group"],
        traffic["hotsets"],
        target_free_mib=[256, 512, 1024, 2048, 4096],
        throughput_gibs=metrics_summary["iouring_observed_gibs"],
        decode_s_total=metrics_summary["decode_s_total"],
    )
    top_k_values = sorted({int(item.strip()) for item in args.top_k.split(",") if item.strip()})
    overfetch_rows, overfetch_candidates = evaluate_overfetch(calls_by_prompt, top_k_values)

    layer_rows = traffic["layer_rows"]
    do_not_touch = sorted(
        [
            row for row in layer_rows
            if row["proxy_hit_traffic_pct"] >= 80.0 and row["next64_gain_gib"] < 0.5
        ],
        key=lambda row: (row["traffic_bytes"], row["proxy_hit_traffic_pct"]),
        reverse=True,
    )[:20]

    best_next64 = traffic["cache_increase_candidates"][0]["next64_gain_gib"] if traffic["cache_increase_candidates"] else 0.0
    best_swap_improvement = max((row["estimated_decode_improvement_pct"] for row in swap_plans), default=0.0)
    if overfetch_candidates:
        phase2 = "run bounded-overfetch proxy for passing candidates before runtime source changes"
        next_action = (
            "Proceed to Phase 2 offline/proxy overfetch validation only for the listed passing candidates; "
            "do not enable global static-prior overfetch."
        )
    elif best_swap_improvement >= 10.0:
        phase2 = "run cache-rebalance proxy; overfetch remains rejected"
        next_action = (
            "Proceed to a no-source cache-rebalance proxy/simulation using the ranked next64 and remove64 tables. "
            "Do not implement runtime overfetch because no static-prior setting passed the false-byte gate."
        )
    else:
        phase2 = "do not implement runtime change from this model; Phase 2 gate not met"
        next_action = (
            "Record GP63 Phase 1 as insufficient for runtime implementation. The next plan should target a stronger "
            "predictor or a lower-byte expert representation rather than simple layer-wise cache/overfetch."
        )

    report = {
        "timestamp": args.timestamp,
        "command": " ".join(sys.argv),
        "runs_root": str(args.runs_root),
        "group_slots": group_slots,
        "metrics_summary": metrics_summary,
        "layer_rows": layer_rows,
        "cache_increase_candidates": traffic["cache_increase_candidates"],
        "cache_decrease_candidates": traffic["cache_decrease_candidates"],
        "overfetch_rows": overfetch_rows,
        "overfetch_candidates": overfetch_candidates,
        "swap_plans": swap_plans,
        "do_not_touch_layers": do_not_touch,
        "best_next64_gain_gib": best_next64,
        "best_swap_decode_improvement_pct": best_swap_improvement,
        "phase2_recommendation": phase2,
        "next_action": next_action,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "layerwise-io-cache-model.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.out_dir / "layerwise-cache-candidates.csv", layer_rows, [
        "layer",
        "kind",
        "events",
        "traffic_gib",
        "unique_keys",
        "prompt_count",
        "proxy_hot_keys",
        "proxy_hit_traffic_pct",
        "next64_gain_gib",
        "next64_gain_per_slot_mib",
        "next64_avg_prompt_coverage",
        "remove64_cost_gib",
        "remove64_cost_per_slot_mib",
        "remove64_avg_prompt_coverage",
    ])
    write_csv(args.out_dir / "overfetch-static-prior-by-layer.csv", overfetch_rows, [
        "layer",
        "kind",
        "top_k",
        "calls",
        "actual_events",
        "actual_gib",
        "byte_recall",
        "precision",
        "predicted_to_actual_byte_ratio",
        "false_to_actual_byte_ratio",
        "extra_to_actual_byte_ratio",
    ])
    write_csv(args.out_dir / "same-budget-cache-swap-bound.csv", swap_plans, [
        "freed_mib_target",
        "removed_keys",
        "removed_mib",
        "removed_traffic_gib",
        "added_keys",
        "added_mib",
        "added_traffic_gib",
        "net_traffic_reduction_gib",
        "estimated_saved_s",
        "estimated_decode_improvement_pct",
        "remove_by_kind",
        "add_by_kind",
    ])
    write_markdown(args.out_dir / "report.md", report)
    print(args.out_dir / "report.md")
    print(f"prompts={metrics_summary['prompt_count']}")
    print(f"best_next64_gain_gib={best_next64:.3f}")
    print(f"best_swap_decode_improvement_pct={best_swap_improvement:.3f}")
    print(f"overfetch_candidates={len(overfetch_candidates)}")
    print(f"phase2={phase2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
