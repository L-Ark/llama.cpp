#!/usr/bin/env python3
import argparse
import collections
import csv
import json
import pathlib
import re
from collections import Counter, defaultdict, deque


ROLE_RE = re.compile(r"ffn_(up|gate|down)_exps")


def role_group(tensor: str) -> str:
    match = ROLE_RE.search(tensor)
    if not match:
        return "other"
    role = match.group(1)
    return "down" if role == "down" else "upgate"


def parse_slots(metrics: dict, group: str) -> int:
    key = "vram_down_0" if group == "down" else "vram_upgate_0"
    match = re.search(r"slots=(\d+)", str(metrics.get(key, "")))
    return int(match.group(1)) if match else (766 if group == "down" else 1735)


def parse_current_hit(metrics: dict, group: str) -> float:
    key = "vram_down_0" if group == "down" else "vram_upgate_0"
    match = re.search(r"hit_rate=([0-9.]+)%", str(metrics.get(key, "")))
    return float(match.group(1)) if match else 0.0


def load_prompt(run_dir: pathlib.Path):
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    events = {"upgate": [], "down": []}
    bytes_by_key = {}
    with (run_dir / "route-trace.csv").open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            group = role_group(row["tensor"])
            if group not in events:
                continue
            key = (row["tensor"], int(row["expert_idx"]))
            events[group].append(key)
            bytes_by_key[key] = int(row["expert_bytes"])
    return metrics, events, bytes_by_key


def static_lfu_set(counter: Counter, cap: int):
    return {key for key, _ in counter.most_common(cap)}


def replay_static(seq, hotset, bytes_by_key):
    hits = misses = miss_bytes = 0
    for key in seq:
        if key in hotset:
            hits += 1
        else:
            misses += 1
            miss_bytes += bytes_by_key[key]
    return hits, misses, miss_bytes


def belady(seq, cap, bytes_by_key):
    future = defaultdict(deque)
    for idx, key in enumerate(seq):
        future[key].append(idx)
    cache = set()
    hits = misses = miss_bytes = 0
    never = 10**18
    for key in seq:
        future[key].popleft()
        if key in cache:
            hits += 1
            continue
        misses += 1
        miss_bytes += bytes_by_key[key]
        if len(cache) >= cap and cache:
            victim = max(cache, key=lambda k: future[k][0] if future[k] else never)
            cache.remove(victim)
        if cap > 0:
            cache.add(key)
    return hits, misses, miss_bytes


def pct(hits, misses):
    return 100.0 * hits / max(hits + misses, 1)


def write_report(rows, out: pathlib.Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Kimi dev route-cache oracle bound",
        "",
        "This is an offline bound from dev route traces. It does not use held-out test prompts.",
        "",
        "| prompt | group | slots | events | unique | current hit | global LFU hit | prompt LFU hit | Belady hit | current miss GiB | global miss GiB | prompt miss GiB | Belady miss GiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['prompt_id']}` | {r['group']} | {r['slots']} | {r['events']} | {r['unique']} | "
            f"{r['current_hit_pct']:.1f}% | {r['global_lfu_hit_pct']:.1f}% | "
            f"{r['prompt_lfu_hit_pct']:.1f}% | {r['belady_hit_pct']:.1f}% | "
            f"{r['current_miss_gib']:.2f} | {r['global_lfu_miss_gib']:.2f} | "
            f"{r['prompt_lfu_miss_gib']:.2f} | {r['belady_miss_gib']:.2f} |"
        )
    lines.extend([
        "",
        "Interpretation:",
        "",
        "- `global LFU` is the best fixed dev-wide frequency hotset with the current slot counts.",
        "- `prompt LFU` is an oracle upper bound for a prompt-adaptive static hotset.",
        "- `Belady` is an offline replacement upper bound for the same trace and capacity.",
        "- If prompt LFU/Belady materially beat global LFU on slow prompts, fixed",
        "  prompt-agnostic cache is insufficient and runtime-adaptive or byte-reduction",
        "  work should be prioritized.",
        "",
    ])
    out.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Offline cache upper-bound simulation for Kimi dev route traces.")
    parser.add_argument("--runs-root", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    prompts = []
    global_counts = {"upgate": Counter(), "down": Counter()}
    global_bytes = {}
    for run_dir in sorted(p for p in args.runs_root.iterdir() if p.is_dir() and (p / "metrics.json").exists()):
        metrics, events, bytes_by_key = load_prompt(run_dir)
        prompts.append((run_dir, metrics, events, bytes_by_key))
        global_bytes.update(bytes_by_key)
        for group in ("upgate", "down"):
            global_counts[group].update(events[group])

    global_hot = {}
    for group in ("upgate", "down"):
        # Use the first prompt's slot count; current runtime uses the same slot
        # count for every prompt in this baseline.
        slots = parse_slots(prompts[0][1], group) if prompts else 0
        global_hot[group] = static_lfu_set(global_counts[group], slots)

    rows = []
    for run_dir, metrics, events, bytes_by_key in prompts:
        prompt_id = metrics.get("prompt_id", run_dir.name)
        for group in ("upgate", "down"):
            seq = events[group]
            slots = parse_slots(metrics, group)
            prompt_hot = static_lfu_set(Counter(seq), slots)
            g_hits, g_misses, g_miss_bytes = replay_static(seq, global_hot[group], bytes_by_key)
            p_hits, p_misses, p_miss_bytes = replay_static(seq, prompt_hot, bytes_by_key)
            b_hits, b_misses, b_miss_bytes = belady(seq, slots, bytes_by_key)
            current_hit = parse_current_hit(metrics, group)
            total_bytes = sum(bytes_by_key[key] for key in seq)
            current_miss_gib = total_bytes * (1.0 - current_hit / 100.0) / 1024**3
            rows.append({
                "prompt_id": prompt_id,
                "group": group,
                "slots": slots,
                "events": len(seq),
                "unique": len(set(seq)),
                "current_hit_pct": current_hit,
                "global_lfu_hit_pct": pct(g_hits, g_misses),
                "prompt_lfu_hit_pct": pct(p_hits, p_misses),
                "belady_hit_pct": pct(b_hits, b_misses),
                "current_miss_gib": current_miss_gib,
                "global_lfu_miss_gib": g_miss_bytes / 1024**3,
                "prompt_lfu_miss_gib": p_miss_bytes / 1024**3,
                "belady_miss_gib": b_miss_bytes / 1024**3,
            })
    write_report(rows, args.out)
    print(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
