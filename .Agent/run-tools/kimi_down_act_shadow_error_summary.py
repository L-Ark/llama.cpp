#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path


class Acc:
    def __init__(self):
        self.rows = 0
        self.values = 0
        self.input_values = 0
        self.skipped_values = 0
        self.blocks = 0
        self.skipped_blocks = 0
        self.l2_abs_sum = 0.0
        self.l2_rel_sum = 0.0
        self.l2_rel_max = 0.0
        self.max_abs_err = 0.0
        self.mean_abs_err_sum = 0.0
        self.exact_l2_sum = 0.0

    def add(self, row):
        ne01 = int(row["ne01"])
        ne00 = int(row["ne00"])
        self.rows += 1
        self.values += ne01
        self.input_values += ne00
        self.skipped_values += int(row["skipped_values"])
        self.blocks += int(row["total_blocks"])
        self.skipped_blocks += int(row["skipped_blocks"])
        l2_abs = float(row["l2_abs"])
        l2_rel = float(row["l2_rel"])
        max_abs_err = float(row["max_abs_err"])
        mean_abs_err = float(row["mean_abs_err"])
        exact_l2 = float(row["exact_l2"])
        self.l2_abs_sum += l2_abs
        self.l2_rel_sum += l2_rel
        self.l2_rel_max = max(self.l2_rel_max, l2_rel)
        self.max_abs_err = max(self.max_abs_err, max_abs_err)
        self.mean_abs_err_sum += mean_abs_err
        self.exact_l2_sum += exact_l2

    def row(self, scope, key, threshold):
        return {
            "scope": scope,
            "key": key,
            "threshold": threshold,
            "rows": self.rows,
            "skip_block_ratio": self.skipped_blocks / self.blocks if self.blocks else 0.0,
            "skip_value_ratio": self.skipped_values / self.input_values if self.input_values else 0.0,
            "mean_l2_abs": self.l2_abs_sum / self.rows if self.rows else 0.0,
            "mean_l2_rel": self.l2_rel_sum / self.rows if self.rows else 0.0,
            "max_l2_rel": self.l2_rel_max,
            "max_abs_err": self.max_abs_err,
            "mean_abs_err": self.mean_abs_err_sum / self.rows if self.rows else 0.0,
            "mean_exact_l2": self.exact_l2_sum / self.rows if self.rows else 0.0,
        }


def read_rows(paths):
    rows = []
    for path in paths:
        with Path(path).open(newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profiles", nargs="+")
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-csv")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    rows = read_rows(args.profiles)
    if not rows:
        raise SystemExit("no rows")

    global_acc = defaultdict(Acc)
    by_layer = defaultdict(Acc)
    by_expert = defaultdict(Acc)
    for row in rows:
        th = row["threshold"]
        layer = row["layer"]
        expert_key = f'{row["layer"]}:{row["expert_idx"]}:{row["tensor"]}'
        global_acc[th].add(row)
        by_layer[(th, layer)].add(row)
        by_expert[(th, expert_key)].add(row)

    out_rows = []
    for th, acc in sorted(global_acc.items(), key=lambda kv: float(kv[0])):
        out_rows.append(acc.row("global", "all", th))
    for (th, layer), acc in sorted(by_layer.items(), key=lambda kv: (float(kv[0][0]), int(kv[0][1]))):
        out_rows.append(acc.row("layer", layer, th))
    for (th, key), acc in sorted(by_expert.items(), key=lambda kv: (float(kv[0][0]), kv[0][1])):
        out_rows.append(acc.row("layer_expert", key, th))

    if args.out_csv:
        with Path(args.out_csv).open("w", newline="") as f:
            fields = list(out_rows[0].keys())
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(out_rows)

    lines = [
        "# Kimi Down Activation Shadow Error Summary",
        "",
        f"profiles: `{len(args.profiles)}`",
        f"rows: `{len(rows)}`",
        "",
        "## Global",
        "",
        "| threshold | rows | skip blocks | mean l2 rel | max l2 rel | mean abs err | max abs err |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for th, acc in sorted(global_acc.items(), key=lambda kv: float(kv[0])):
        r = acc.row("global", "all", th)
        lines.append(
            f"| `{float(th):g}` | `{r['rows']}` | `{r['skip_block_ratio']:.6f}` | "
            f"`{r['mean_l2_rel']:.6f}` | `{r['max_l2_rel']:.6f}` | "
            f"`{r['mean_abs_err']:.6g}` | `{r['max_abs_err']:.6g}` |"
        )

    for th in sorted(global_acc, key=float):
        lines += [
            "",
            f"## Layer Summary At {float(th):g}",
            "",
            "| layer | rows | skip blocks | mean l2 rel | max l2 rel | mean abs err | max abs err |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        layer_items = [
            (layer, acc) for (t, layer), acc in by_layer.items() if t == th
        ]
        for layer, acc in sorted(layer_items, key=lambda kv: int(kv[0])):
            r = acc.row("layer", layer, th)
            lines.append(
                f"| `{layer}` | `{r['rows']}` | `{r['skip_block_ratio']:.4f}` | "
                f"`{r['mean_l2_rel']:.4f}` | `{r['max_l2_rel']:.4f}` | "
                f"`{r['mean_abs_err']:.6g}` | `{r['max_abs_err']:.6g}` |"
            )

        lines += [
            "",
            f"## Worst Layer+Expert At {float(th):g}",
            "",
            "| key | rows | skip blocks | mean l2 rel | max l2 rel | max abs err |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        expert_items = [
            (key, acc) for (t, key), acc in by_expert.items() if t == th
        ]
        ranked = sorted(
            expert_items,
            key=lambda kv: (kv[1].l2_rel_sum / kv[1].rows if kv[1].rows else 0.0, kv[1].l2_rel_max),
            reverse=True,
        )
        for key, acc in ranked[: args.top]:
            r = acc.row("layer_expert", key, th)
            lines.append(
                f"| `{key}` | `{r['rows']}` | `{r['skip_block_ratio']:.4f}` | "
                f"`{r['mean_l2_rel']:.4f}` | `{r['max_l2_rel']:.4f}` | `{r['max_abs_err']:.6g}` |"
            )

    Path(args.out_md).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
