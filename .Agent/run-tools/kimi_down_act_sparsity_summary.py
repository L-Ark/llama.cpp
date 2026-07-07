#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_list(text, cast):
    if not text:
        return []
    return [cast(x) for x in text.split(";") if x != ""]


class Acc:
    def __init__(self, n_thresh):
        self.rows = 0
        self.blocks = 0
        self.values = 0
        self.zero_values = 0
        self.l1 = 0.0
        self.l2_sq = 0.0
        self.max_abs = 0.0
        self.low = [0] * n_thresh

    def add(self, row, lows):
        blocks = int(row["blocks"])
        ne00 = int(row["ne00"])
        l2 = float(row["l2"])
        max_abs = float(row["max_abs"])
        self.rows += 1
        self.blocks += blocks
        self.values += ne00
        self.zero_values += int(row["exact_zero_values"])
        self.l1 += float(row["l1"])
        self.l2_sq += l2 * l2
        self.max_abs = max(self.max_abs, max_abs)
        for i, v in enumerate(lows):
            self.low[i] += v

    def ratios(self):
        if self.blocks <= 0:
            return [0.0 for _ in self.low]
        return [v / self.blocks for v in self.low]


def read_profiles(paths):
    rows = []
    thresholds = None
    for path in paths:
        with Path(path).open(newline="") as f:
            for row in csv.DictReader(f):
                row_thresholds = parse_list(row["thresholds"], float)
                if thresholds is None:
                    thresholds = row_thresholds
                elif thresholds != row_thresholds:
                    raise SystemExit(f"threshold mismatch in {path}: {row_thresholds} != {thresholds}")
                rows.append(row)
    return thresholds or [], rows


def write_csv(path, thresholds, items):
    with Path(path).open("w", newline="") as f:
        fieldnames = [
            "scope", "key", "rows", "blocks", "values", "zero_value_ratio",
            "mean_l1_per_value", "rms", "max_abs",
        ] + [f"low_ratio_le_{t:g}" for t in thresholds]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for scope, key, acc in items:
            out = {
                "scope": scope,
                "key": key,
                "rows": acc.rows,
                "blocks": acc.blocks,
                "values": acc.values,
                "zero_value_ratio": acc.zero_values / acc.values if acc.values else 0.0,
                "mean_l1_per_value": acc.l1 / acc.values if acc.values else 0.0,
                "rms": (acc.l2_sq / acc.values) ** 0.5 if acc.values else 0.0,
                "max_abs": acc.max_abs,
            }
            for t, ratio in zip(thresholds, acc.ratios()):
                out[f"low_ratio_le_{t:g}"] = ratio
            w.writerow(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profiles", nargs="+", help="down activation sparsity CSV files")
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-csv")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    thresholds, rows = read_profiles(args.profiles)
    if not rows:
        raise SystemExit("no rows")

    n_thresh = len(thresholds)
    global_acc = Acc(n_thresh)
    by_layer = defaultdict(lambda: Acc(n_thresh))
    by_expert = defaultdict(lambda: Acc(n_thresh))
    for row in rows:
        lows = parse_list(row["low_blocks"], int)
        if len(lows) != n_thresh:
            raise SystemExit(f"low_blocks length mismatch: {row}")
        global_acc.add(row, lows)
        layer = row["layer"]
        tensor = row["tensor"]
        expert = row["expert_idx"]
        by_layer[layer].add(row, lows)
        by_expert[(layer, tensor, expert)].add(row, lows)

    items = [("global", "all", global_acc)]
    items += [("layer", layer, acc) for layer, acc in sorted(by_layer.items(), key=lambda kv: int(kv[0]))]
    items += [
        ("layer_expert", f"{layer}:{expert}:{tensor}", acc)
        for (layer, tensor, expert), acc in sorted(by_expert.items(), key=lambda kv: (int(kv[0][0]), int(kv[0][2])))
    ]
    if args.out_csv:
        write_csv(args.out_csv, thresholds, items)

    lines = [
        "# Kimi Down Activation Block Sparsity Summary",
        "",
        f"profiles: `{len(args.profiles)}`",
        f"rows: `{len(rows)}`",
        f"thresholds: `{';'.join(f'{t:g}' for t in thresholds)}`",
        "",
        "## Global",
        "",
        "| threshold | low block ratio | low blocks | total blocks |",
        "|---:|---:|---:|---:|",
    ]
    for t, low, ratio in zip(thresholds, global_acc.low, global_acc.ratios()):
        lines.append(f"| `{t:g}` | `{ratio:.6f}` | `{low}` | `{global_acc.blocks}` |")

    lines += [
        "",
        "## Layer Ratios",
        "",
        "| layer | rows | blocks | " + " | ".join(f"<= {t:g}" for t in thresholds) + " |",
        "|---:|---:|---:|" + "|".join("---:" for _ in thresholds) + "|",
    ]
    for layer, acc in sorted(by_layer.items(), key=lambda kv: int(kv[0])):
        ratios = " | ".join(f"`{r:.4f}`" for r in acc.ratios())
        lines.append(f"| `{layer}` | `{acc.rows}` | `{acc.blocks}` | {ratios} |")

    for i, t in enumerate(thresholds):
        lines += [
            "",
            f"## Top Layer+Expert At <= {t:g}",
            "",
            "| layer | expert | rows | blocks | low ratio | tensor |",
            "|---:|---:|---:|---:|---:|---|",
        ]
        ranked = sorted(
            by_expert.items(),
            key=lambda kv: (kv[1].low[i] / kv[1].blocks if kv[1].blocks else 0.0, kv[1].blocks),
            reverse=True,
        )
        for (layer, tensor, expert), acc in ranked[: args.top]:
            ratio = acc.low[i] / acc.blocks if acc.blocks else 0.0
            lines.append(f"| `{layer}` | `{expert}` | `{acc.rows}` | `{acc.blocks}` | `{ratio:.4f}` | `{tensor}` |")

    Path(args.out_md).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
