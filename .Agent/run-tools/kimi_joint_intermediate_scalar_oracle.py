#!/usr/bin/env python3
"""Scalar correction oracle for top intermediate-dimension partial down output.

This builds on GP95. It computes W_down @ h from only the largest entries of h,
then tests whether an optimal scalar can rescue the missing dropped-dimension
contribution. The scalar uses the exact output, so this is an optimistic bound,
not a deployable runtime method.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_gp95_helper() -> Any:
    helper_path = Path(__file__).with_name("kimi_joint_intermediate_keep_oracle.py")
    spec = importlib.util.spec_from_file_location("kimi_joint_intermediate_keep_oracle", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper module from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


K = load_gp95_helper()
R = K.R
H = K.H


def parse_floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def optimal_scalar(torch: Any, exact: Any, cand: Any) -> float:
    denom = float(torch.dot(cand, cand).item())
    if denom <= 1e-30:
        return 0.0
    return float(torch.dot(exact, cand).item()) / denom


class Acc:
    def __init__(self) -> None:
        self.n = 0
        self.sum_rel = 0.0
        self.max_rel = 0.0
        self.sum_alpha = 0.0

    def add(self, rel: float, alpha: float = 1.0) -> None:
        self.n += 1
        self.sum_rel += rel
        self.max_rel = max(self.max_rel, rel)
        self.sum_alpha += alpha

    def merge_weighted_row(self, row: dict[str, Any]) -> None:
        n = int(row["rows"])
        self.n += n
        self.sum_rel += float(row["mean_rel_l2"]) * n
        self.max_rel = max(self.max_rel, float(row["max_rel_l2"]))
        self.sum_alpha += float(row.get("mean_alpha", 1.0)) * n

    def row(self) -> dict[str, Any]:
        return {
            "rows": self.n,
            "mean_rel_l2": self.sum_rel / self.n if self.n else 0.0,
            "max_rel_l2": self.max_rel,
            "mean_alpha": self.sum_alpha / self.n if self.n else 0.0,
        }


def rel_l2(torch: Any, exact: Any, cand: Any) -> float:
    return float(torch.linalg.vector_norm(cand - exact).item()) / max(float(torch.linalg.vector_norm(exact).item()), 1e-30)


def analyze_prompt(torch: Any, root: Path, inventory: dict[str, Any], lib: Any, keep_fracs: list[float], max_records: int):
    rows = K.load_rows(root / "act" / "activations.csv", max_records)
    bin_path = root / "act" / "activations.f32"
    by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[(row["tensor"], row["expert_idx"])].append(row)

    exact_outputs: dict[int, Any] = {}
    raw_outputs: dict[tuple[int, float], Any] = {}
    row_scaled_outputs: dict[tuple[int, float], Any] = {}
    row_acc: dict[str, dict[str, Acc]] = defaultdict(lambda: defaultdict(Acc))

    for (tensor, expert_idx), key_rows in sorted(by_key.items()):
        if tensor not in inventory:
            raise RuntimeError(f"tensor not in inventory: {tensor}")
        matrix, _ = R.dequant_to_torch(lib, inventory[tensor], expert_idx, torch)
        matrix = matrix.to(torch.float32)
        for row in key_rows:
            vec = K.load_vector(torch, bin_path, row)
            exact = K.down_matvec(torch, matrix, vec, row)
            exact_outputs[row["record_id"]] = exact
            for keep_frac in keep_fracs:
                keep = max(1, min(row["ne00"], int(round(row["ne00"] * keep_frac))))
                idx = torch.topk(vec.abs(), keep, largest=True, sorted=False).indices
                cand = K.down_partial_matvec(torch, matrix, vec, row, idx)
                raw_outputs[(row["record_id"], keep_frac)] = cand
                key = f"{keep_frac:g}"
                row_acc[key]["raw"].add(rel_l2(torch, exact, cand), 1.0)
                alpha = optimal_scalar(torch, exact, cand)
                scaled = cand * alpha
                row_scaled_outputs[(row["record_id"], keep_frac)] = scaled
                row_acc[key]["row_scalar"].add(rel_l2(torch, exact, scaled), alpha)
        del matrix

    group_acc: dict[str, dict[str, Acc]] = defaultdict(lambda: defaultdict(Acc))
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["mode"], row["call"], K.layer_name(row["tensor"]), row["tensor"], row["token_id"])
        groups[key].append(row)
    for group_rows in groups.values():
        group_rows.sort(key=lambda r: r["active_slot"])
        exact_sum = torch.stack([exact_outputs[row["record_id"]] for row in group_rows], dim=0).sum(dim=0)
        for keep_frac in keep_fracs:
            key = f"{keep_frac:g}"
            raw_sum = torch.stack([raw_outputs[(row["record_id"], keep_frac)] for row in group_rows], dim=0).sum(dim=0)
            row_scaled_sum = torch.stack([row_scaled_outputs[(row["record_id"], keep_frac)] for row in group_rows], dim=0).sum(dim=0)
            group_acc[key]["raw"].add(rel_l2(torch, exact_sum, raw_sum), 1.0)
            group_acc[key]["row_scalar_sum"].add(rel_l2(torch, exact_sum, row_scaled_sum), 1.0)
            alpha = optimal_scalar(torch, exact_sum, raw_sum)
            group_acc[key]["group_scalar"].add(rel_l2(torch, exact_sum, raw_sum * alpha), alpha)

    return {
        "prompt_id": root.name,
        "records": len(rows),
        "groups": len(groups),
        "unique_tensor_experts": len(by_key),
        "row_summary": {
            keep: {mode: acc.row() for mode, acc in modes.items()}
            for keep, modes in sorted(row_acc.items(), key=lambda kv: float(kv[0]))
        },
        "group_summary": {
            keep: {mode: acc.row() for mode, acc in modes.items()}
            for keep, modes in sorted(group_acc.items(), key=lambda kv: float(kv[0]))
        },
    }


def write_md(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Kimi scalar-corrected intermediate keep oracle",
        "",
        "This is a dev-only offline oracle. It does not change runtime behavior or claim SOTA.",
        "",
        f"- generated_at: `{result['generated_at']}`",
        f"- prompts: `{len(result['prompts'])}`",
        f"- keep fractions: `{result['keep_fracs']}`",
        "",
        "## Aggregate Group Error",
        "",
        "| keep | raw mean rel L2 | row-scalar-sum mean rel L2 | group-scalar mean rel L2 | group-scalar max rel L2 | mean group alpha |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for keep in result["keep_fracs"]:
        key = f"{keep:g}"
        group = result["aggregate"]["group"][key]
        lines.append(
            f"| `{key}` | `{group['raw']['mean_rel_l2']:.6f}` | "
            f"`{group['row_scalar_sum']['mean_rel_l2']:.6f}` | "
            f"`{group['group_scalar']['mean_rel_l2']:.6f}` | "
            f"`{group['group_scalar']['max_rel_l2']:.6f}` | "
            f"`{group['group_scalar']['mean_alpha']:.6f}` |"
        )
    lines += [
        "",
        "## Aggregate Row Error",
        "",
        "| keep | raw mean rel L2 | row-scalar mean rel L2 | row-scalar max rel L2 | mean row alpha |",
        "|---:|---:|---:|---:|---:|",
    ]
    for keep in result["keep_fracs"]:
        key = f"{keep:g}"
        row = result["aggregate"]["row"][key]
        lines.append(
            f"| `{key}` | `{row['raw']['mean_rel_l2']:.6f}` | "
            f"`{row['row_scalar']['mean_rel_l2']:.6f}` | "
            f"`{row['row_scalar']['max_rel_l2']:.6f}` | "
            f"`{row['row_scalar']['mean_alpha']:.6f}` |"
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
            "| keep | raw group rel L2 | group-scalar rel L2 | mean alpha |",
            "|---:|---:|---:|---:|",
        ]
        for keep in result["keep_fracs"]:
            key = f"{keep:g}"
            raw = prompt["group_summary"][key]["raw"]
            scalar = prompt["group_summary"][key]["group_scalar"]
            lines.append(
                f"| `{key}` | `{raw['mean_rel_l2']:.6f}` | "
                f"`{scalar['mean_rel_l2']:.6f}` | `{scalar['mean_alpha']:.6f}` |"
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
    parser.add_argument("--keep-fracs", default="0.35,0.4,0.45,0.5")
    parser.add_argument("--max-records-per-prompt", type=int, default=64)
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
        aggregate["row"][key] = {}
        aggregate["group"][key] = {}
        for mode in ("raw", "row_scalar"):
            acc = Acc()
            for prompt in prompts:
                acc.merge_weighted_row(prompt["row_summary"][key][mode])
            aggregate["row"][key][mode] = acc.row()
        for mode in ("raw", "row_scalar_sum", "group_scalar"):
            acc = Acc()
            for prompt in prompts:
                acc.merge_weighted_row(prompt["group_summary"][key][mode])
            aggregate["group"][key][mode] = acc.row()

    target_key = f"{args.target_keep_frac:g}"
    target = aggregate["group"].get(target_key, {}).get("group_scalar")
    if target and target["mean_rel_l2"] <= args.target_group_rel_l2:
        decision = (
            "The optimal group scalar passes the dev oracle gate at the target byte ratio. "
            "This is not deployable yet; next test whether alpha can be predicted from kept/dropped activation energy without exact output."
        )
    else:
        got = target["mean_rel_l2"] if target else float("nan")
        decision = (
            f"Reject scalar correction as a rescue for the 0.4x intermediate top-k path: "
            f"even the optimal group scalar has mean rel L2 {got:.6f}, above the {args.target_group_rel_l2:g} gate."
        )

    result = {
        "kind": "kimi_joint_intermediate_scalar_oracle",
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
