#!/usr/bin/env python3
import argparse
import collections
import csv
import json
import math
import pathlib
import re


ROLE_RE = re.compile(r"blk\.(\d+)\.ffn_(up|gate|down)_exps\.weight")


def parse_trace(path: pathlib.Path):
    rows = []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            m = ROLE_RE.search(row["tensor"])
            if not m:
                continue
            rows.append({
                "layer": int(m.group(1)),
                "role": m.group(2),
                "expert": int(row["expert_idx"]),
                "bytes": int(row["expert_bytes"]),
                "tensor": row["tensor"],
            })
    return rows


def entropy(counts):
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for count in counts.values():
        p = count / total
        h -= p * math.log2(p)
    return h


def load_run(run_dir: pathlib.Path):
    metrics = {}
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    trace_path = run_dir / "route-trace.csv"
    if not trace_path.exists():
        return metrics, []
    return metrics, parse_trace(trace_path)


def summarize_prompt(run_dir: pathlib.Path, metrics, rows):
    by_lr = collections.defaultdict(collections.Counter)
    bytes_by_lr = collections.defaultdict(int)
    for row in rows:
        key = (row["layer"], row["role"])
        by_lr[key][row["expert"]] += 1
        bytes_by_lr[key] += row["bytes"]
    out = []
    for (layer, role), counts in sorted(by_lr.items()):
        total = sum(counts.values())
        h = entropy(counts)
        max_h = math.log2(max(len(counts), 1))
        top1 = counts.most_common(1)[0][1] / total if total else 0.0
        out.append({
            "prompt_id": metrics.get("prompt_id", run_dir.name),
            "layer": layer,
            "role": role,
            "events": total,
            "unique_experts": len(counts),
            "entropy_bits": h,
            "normalized_entropy": h / max_h if max_h > 0 else 0.0,
            "top1_share": top1,
            "routed_gib": bytes_by_lr[(layer, role)] / 1024**3,
            "top8": [expert for expert, _ in counts.most_common(8)],
        })
    return out


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def cross_prompt(rows):
    grouped = collections.defaultdict(dict)
    for row in rows:
        grouped[(row["layer"], row["role"])][row["prompt_id"]] = row["top8"]
    out = []
    for (layer, role), by_prompt in sorted(grouped.items()):
        prompts = sorted(by_prompt)
        vals = []
        for i, left in enumerate(prompts):
            for right in prompts[i + 1:]:
                vals.append(jaccard(by_prompt[left], by_prompt[right]))
        if not vals:
            continue
        out.append({
            "layer": layer,
            "role": role,
            "prompt_count": len(prompts),
            "top8_jaccard_mean": sum(vals) / len(vals),
            "top8_jaccard_min": min(vals),
            "events_min": min(r["events"] for r in rows if r["layer"] == layer and r["role"] == role),
            "routed_gib_sum": sum(r["routed_gib"] for r in rows if r["layer"] == layer and r["role"] == role),
            "normalized_entropy_mean": sum(r["normalized_entropy"] for r in rows if r["layer"] == layer and r["role"] == role) / len([r for r in rows if r["layer"] == layer and r["role"] == role]),
        })
    return out


def classify(row):
    if row["top8_jaccard_mean"] >= 0.50 and row["normalized_entropy_mean"] <= 0.70:
        return "stable_low_entropy"
    if row["top8_jaccard_mean"] >= 0.25:
        return "prompt_local_or_partial_overlap"
    return "high_entropy_or_low_overlap"


def write_outputs(out_dir: pathlib.Path, prompt_rows, overlap_rows):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prompt-layer-role.json").write_text(json.dumps(prompt_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for row in overlap_rows:
        row["class"] = classify(row)
    (out_dir / "cross-prompt-overlap.json").write_text(json.dumps(overlap_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    ranked = sorted(overlap_rows, key=lambda r: (r["class"], -r["routed_gib_sum"], r["top8_jaccard_mean"]))
    with (out_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# Kimi route entropy analysis\n\n")
        f.write("| layer | role | class | top8 J mean | top8 J min | entropy mean | routed GiB sum |\n")
        f.write("|---:|---|---|---:|---:|---:|---:|\n")
        for row in ranked[:120]:
            f.write(
                f"| {row['layer']} | {row['role']} | {row['class']} | "
                f"{row['top8_jaccard_mean']:.3f} | {row['top8_jaccard_min']:.3f} | "
                f"{row['normalized_entropy_mean']:.3f} | {row['routed_gib_sum']:.2f} |\n"
            )


def main():
    parser = argparse.ArgumentParser(description="Analyze Kimi route trace entropy and cross-prompt expert overlap.")
    parser.add_argument("--runs-root", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    prompt_rows = []
    loaded = 0
    for run_dir in sorted(p for p in args.runs_root.iterdir() if p.is_dir()):
        metrics, rows = load_run(run_dir)
        if not rows:
            continue
        loaded += 1
        prompt_rows.extend(summarize_prompt(run_dir, metrics, rows))
    if loaded < 2:
        raise SystemExit(f"need at least two profiled prompt runs with route-trace.csv, found {loaded}")
    overlap_rows = cross_prompt(prompt_rows)
    write_outputs(args.out_dir, prompt_rows, overlap_rows)
    print(args.out_dir / "summary.md")


if __name__ == "__main__":
    raise SystemExit(main())
