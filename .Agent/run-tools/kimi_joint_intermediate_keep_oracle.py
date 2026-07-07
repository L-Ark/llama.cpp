#!/usr/bin/env python3
"""Oracle for keeping only top fused-upgate intermediate dimensions.

This is a dev-only offline screen. It uses dumped down activations as the exact
fused up/gate intermediate vector h and computes W_down @ h from only the
largest-magnitude entries of h. The byte ratio is optimistic: it assumes a
future layout can read matching up rows, gate rows, and down columns.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_residual_helper() -> Any:
    helper_path = Path(__file__).with_name("kimi_d2moe_residual_rank.py")
    spec = importlib.util.spec_from_file_location("kimi_d2moe_residual_rank", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper module from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


R = load_residual_helper()
H = R.H


def parse_floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def load_rows(path: Path, max_records: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            if row.get("role") != "down":
                continue
            for key in (
                "record_id",
                "call",
                "type",
                "expert_idx",
                "active_slot",
                "dst_id",
                "token_id",
                "ne00",
                "ne01",
                "expert_bytes",
                "offset_bytes",
                "nbytes",
            ):
                row[key] = int(row[key])
            rows.append(row)
            if max_records > 0 and len(rows) >= max_records:
                break
    if not rows:
        raise RuntimeError(f"no down activation rows in {path}")
    return rows


def load_vector(torch: Any, bin_path: Path, row: dict[str, Any]) -> Any:
    with bin_path.open("rb") as f:
        f.seek(row["offset_bytes"])
        raw = f.read(row["nbytes"])
    expected = row["ne00"] * 4
    if len(raw) != expected:
        raise RuntimeError(f"short activation read for record {row['record_id']}: {len(raw)} != {expected}")
    return torch.frombuffer(bytearray(raw), dtype=torch.float32).clone()


def layer_name(tensor: str) -> str:
    parts = tensor.split(".")
    if len(parts) >= 2 and parts[0] == "blk":
        return f"{parts[0]}.{parts[1]}"
    return tensor


def down_matvec(torch: Any, matrix: Any, vec: Any, row: dict[str, Any]) -> Any:
    if matrix.shape[1] == row["ne00"] and matrix.shape[0] == row["ne01"]:
        return torch.mv(matrix, vec)
    if matrix.shape[0] == row["ne00"] and matrix.shape[1] == row["ne01"]:
        return torch.mv(matrix.t().contiguous(), vec)
    raise RuntimeError(
        f"shape mismatch for {row['tensor']} expert {row['expert_idx']}: "
        f"matrix={tuple(matrix.shape)} activation ne00/ne01={row['ne00']}/{row['ne01']}"
    )


def down_partial_matvec(torch: Any, matrix: Any, vec: Any, row: dict[str, Any], idx: Any) -> Any:
    if matrix.shape[1] == row["ne00"] and matrix.shape[0] == row["ne01"]:
        return torch.mv(matrix[:, idx].contiguous(), vec[idx])
    if matrix.shape[0] == row["ne00"] and matrix.shape[1] == row["ne01"]:
        return torch.mv(matrix[idx, :].t().contiguous(), vec[idx])
    raise RuntimeError(
        f"shape mismatch for partial down matvec: matrix={tuple(matrix.shape)} "
        f"activation ne00/ne01={row['ne00']}/{row['ne01']}"
    )


def rel_l2(torch: Any, exact: Any, cand: Any) -> float:
    return float(torch.linalg.vector_norm(cand - exact).item()) / max(float(torch.linalg.vector_norm(exact).item()), 1e-30)


class Acc:
    def __init__(self) -> None:
        self.n = 0
        self.sum_rel = 0.0
        self.max_rel = 0.0
        self.sum_keep = 0.0

    def add(self, rel: float, keep: float) -> None:
        self.n += 1
        self.sum_rel += rel
        self.max_rel = max(self.max_rel, rel)
        self.sum_keep += keep

    def row(self) -> dict[str, Any]:
        return {
            "rows": self.n,
            "mean_rel_l2": self.sum_rel / self.n if self.n else 0.0,
            "max_rel_l2": self.max_rel,
            "mean_byte_ratio": self.sum_keep / self.n if self.n else 0.0,
        }


def analyze_prompt(torch: Any, root: Path, inventory: dict[str, Any], lib: Any, keep_fracs: list[float], max_records: int):
    rows = load_rows(root / "act" / "activations.csv", max_records)
    bin_path = root / "act" / "activations.f32"
    by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[(row["tensor"], row["expert_idx"])].append(row)

    exact_outputs: dict[int, Any] = {}
    cand_outputs: dict[tuple[int, float], Any] = {}
    row_acc: dict[str, Acc] = defaultdict(Acc)

    for (tensor, expert_idx), key_rows in sorted(by_key.items()):
        if tensor not in inventory:
            raise RuntimeError(f"tensor not in inventory: {tensor}")
        matrix, _ = R.dequant_to_torch(lib, inventory[tensor], expert_idx, torch)
        matrix = matrix.to(torch.float32)
        for row in key_rows:
            vec = load_vector(torch, bin_path, row)
            exact = down_matvec(torch, matrix, vec, row)
            exact_outputs[row["record_id"]] = exact
            for keep_frac in keep_fracs:
                keep = max(1, min(row["ne00"], int(round(row["ne00"] * keep_frac))))
                idx = torch.topk(vec.abs(), keep, largest=True, sorted=False).indices
                cand = down_partial_matvec(torch, matrix, vec, row, idx)
                cand_outputs[(row["record_id"], keep_frac)] = cand
                row_acc[f"{keep_frac:g}"].add(rel_l2(torch, exact, cand), keep_frac)
        del matrix

    group_acc: dict[str, Acc] = defaultdict(Acc)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["mode"], row["call"], layer_name(row["tensor"]), row["tensor"], row["token_id"])
        groups[key].append(row)
    for group_rows in groups.values():
        group_rows.sort(key=lambda r: r["active_slot"])
        exact_sum = torch.stack([exact_outputs[row["record_id"]] for row in group_rows], dim=0).sum(dim=0)
        for keep_frac in keep_fracs:
            cand_sum = torch.stack([cand_outputs[(row["record_id"], keep_frac)] for row in group_rows], dim=0).sum(dim=0)
            group_acc[f"{keep_frac:g}"].add(rel_l2(torch, exact_sum, cand_sum), keep_frac)

    return {
        "prompt_id": root.name,
        "records": len(rows),
        "unique_tensor_experts": len(by_key),
        "groups": len(groups),
        "row_summary": {key: acc.row() for key, acc in sorted(row_acc.items(), key=lambda kv: float(kv[0]))},
        "group_summary": {key: acc.row() for key, acc in sorted(group_acc.items(), key=lambda kv: float(kv[0]))},
    }


def write_md(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Kimi joint intermediate-dimension keep oracle",
        "",
        "This is a dev-only offline oracle. It does not change runtime behavior or claim SOTA.",
        "",
        f"- generated_at: `{result['generated_at']}`",
        f"- prompts: `{len(result['prompts'])}`",
        f"- keep fractions: `{result['keep_fracs']}`",
        "",
        "## Aggregate",
        "",
        "| keep | byte ratio | row mean rel L2 | row max rel L2 | group mean rel L2 | group max rel L2 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for keep in result["keep_fracs"]:
        key = f"{keep:g}"
        row = result["aggregate"]["row"][key]
        group = result["aggregate"]["group"][key]
        lines.append(
            f"| `{key}` | `{row['mean_byte_ratio']:.4f}` | `{row['mean_rel_l2']:.6f}` | "
            f"`{row['max_rel_l2']:.6f}` | `{group['mean_rel_l2']:.6f}` | `{group['max_rel_l2']:.6f}` |"
        )
    lines += [
        "",
        "## Decision",
        "",
        result["decision"],
        "",
        "## Prompt Summaries",
        "",
    ]
    for prompt in result["prompts"]:
        lines += [
            f"### {prompt['prompt_id']}",
            "",
            f"- records: `{prompt['records']}`",
            f"- groups: `{prompt['groups']}`",
            "",
            "| keep | row mean rel L2 | group mean rel L2 |",
            "|---:|---:|---:|",
        ]
        for keep in result["keep_fracs"]:
            key = f"{keep:g}"
            lines.append(
                f"| `{key}` | `{prompt['row_summary'][key]['mean_rel_l2']:.6f}` | "
                f"`{prompt['group_summary'][key]['mean_rel_l2']:.6f}` |"
            )
        lines.append("")
    lines += [
        "## Reproduce",
        "",
        "```bash",
        result["reproduce_command"],
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-root", type=Path, action="append", required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--libggml-base", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--keep-fracs", default="0.05,0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--max-records-per-prompt", type=int, default=512)
    parser.add_argument("--target-keep-frac", type=float, default=0.4)
    parser.add_argument("--target-group-rel-l2", type=float, default=0.10)
    parser.add_argument("--torch-threads", type=int, default=8)
    args = parser.parse_args()

    import torch

    torch.set_num_threads(args.torch_threads)
    keep_fracs = parse_floats(args.keep_fracs)
    inventory = H.load_inventory(args.inventory)
    lib = H.load_ggml(args.libggml_base)

    prompts = [
        analyze_prompt(torch, root, inventory, lib, keep_fracs, args.max_records_per_prompt)
        for root in args.prompt_root
    ]

    aggregate = {"row": {}, "group": {}}
    for keep in keep_fracs:
        key = f"{keep:g}"
        row_acc = Acc()
        group_acc = Acc()
        for prompt in prompts:
            row = prompt["row_summary"][key]
            group = prompt["group_summary"][key]
            row_acc.n += row["rows"]
            row_acc.sum_rel += row["mean_rel_l2"] * row["rows"]
            row_acc.max_rel = max(row_acc.max_rel, row["max_rel_l2"])
            row_acc.sum_keep += row["mean_byte_ratio"] * row["rows"]
            group_acc.n += group["rows"]
            group_acc.sum_rel += group["mean_rel_l2"] * group["rows"]
            group_acc.max_rel = max(group_acc.max_rel, group["max_rel_l2"])
            group_acc.sum_keep += group["mean_byte_ratio"] * group["rows"]
        aggregate["row"][key] = row_acc.row()
        aggregate["group"][key] = group_acc.row()

    target_key = f"{args.target_keep_frac:g}"
    target_group = aggregate["group"].get(target_key)
    if target_group and target_group["mean_rel_l2"] <= args.target_group_rel_l2:
        decision = (
            "The joint intermediate-dimension keep oracle passes the dev group-error gate at the target byte ratio. "
            "Do not use held-out yet; next design a concrete sliced up/gate/down storage and runtime cost model."
        )
    else:
        got = target_group["mean_rel_l2"] if target_group else float("nan")
        decision = (
            f"Reject joint intermediate-dimension partial reads as the next primary runtime path: "
            f"at keep <= {args.target_keep_frac:g}, grouped down-output mean rel L2 is {got:.6f}, "
            f"above the {args.target_group_rel_l2:g} gate."
        )

    result = {
        "kind": "kimi_joint_intermediate_keep_oracle",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "keep_fracs": keep_fracs,
        "target_keep_frac": args.target_keep_frac,
        "target_group_rel_l2": args.target_group_rel_l2,
        "prompts": prompts,
        "aggregate": aggregate,
        "decision": decision,
        "reproduce_command": " ".join(os.sys.argv),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_md(args.out_md, result)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
