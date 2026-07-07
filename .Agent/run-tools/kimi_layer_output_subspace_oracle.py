#!/usr/bin/env python3
"""Layer-level MoE output subspace oracle.

This is a dev-only offline screen. It computes the exact summed down output for
each sampled MoE group, then tests whether outputs for the same layer can be
reconstructed from a small PCA basis trained on other dev prompts.
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


def parse_ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def rel_l2(torch: Any, exact: Any, cand: Any) -> float:
    return float(torch.linalg.vector_norm(cand - exact).item()) / max(float(torch.linalg.vector_norm(exact).item()), 1e-30)


class Acc:
    def __init__(self) -> None:
        self.n = 0
        self.sum_rel = 0.0
        self.max_rel = 0.0

    def add(self, rel: float) -> None:
        self.n += 1
        self.sum_rel += rel
        self.max_rel = max(self.max_rel, rel)

    def row(self) -> dict[str, Any]:
        return {
            "rows": self.n,
            "mean_rel_l2": self.sum_rel / self.n if self.n else 0.0,
            "max_rel_l2": self.max_rel,
        }


def compute_prompt_outputs(torch: Any, root: Path, inventory: dict[str, Any], lib: Any, max_down_records: int) -> list[dict[str, Any]]:
    rows = K.load_rows(root / "act" / "activations.csv", max_down_records)
    bin_path = root / "act" / "activations.f32"
    by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[(row["tensor"], row["expert_idx"])].append(row)

    exact_outputs: dict[int, Any] = {}
    for (tensor, expert_idx), key_rows in sorted(by_key.items()):
        if tensor not in inventory:
            raise RuntimeError(f"tensor not in inventory: {tensor}")
        matrix, _ = R.dequant_to_torch(lib, inventory[tensor], expert_idx, torch)
        matrix = matrix.to(torch.float32)
        for row in key_rows:
            vec = K.load_vector(torch, bin_path, row)
            exact_outputs[row["record_id"]] = K.down_matvec(torch, matrix, vec, row)
        del matrix

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        layer = K.layer_name(row["tensor"])
        key = (layer, row["mode"], row["call"], row["token_id"])
        groups[key].append(row)

    outputs: list[dict[str, Any]] = []
    for key, group_rows in sorted(groups.items()):
        group_rows.sort(key=lambda r: r["active_slot"])
        y = torch.stack([exact_outputs[row["record_id"]] for row in group_rows], dim=0).sum(dim=0)
        outputs.append({
            "prompt_id": root.name,
            "layer": key[0],
            "group_key": "|".join(map(str, key)),
            "active": len(group_rows),
            "vector": y,
        })
    return outputs


def fit_basis(torch: Any, train: Any, rank: int) -> tuple[Any, Any]:
    mean = train.mean(dim=0)
    centered = train - mean
    usable = max(0, min(rank, centered.shape[0] - 1, centered.shape[1]))
    if usable <= 0:
        return mean, None
    # train samples per layer are tiny in the smoke, so exact SVD is cheap and deterministic.
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    basis = vh[:usable, :].contiguous()
    return mean, basis


def reconstruct(torch: Any, y: Any, mean: Any, basis: Any) -> Any:
    if basis is None or basis.numel() == 0:
        return mean
    centered = y - mean
    coeff = torch.mv(basis, centered)
    return mean + torch.mv(basis.t().contiguous(), coeff)


def evaluate(torch: Any, outputs: list[dict[str, Any]], ranks: list[int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_layer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prompts = sorted({row["prompt_id"] for row in outputs})
    for row in outputs:
        by_layer[row["layer"]].append(row)

    accs: dict[int, Acc] = {rank: Acc() for rank in ranks}
    layer_accs: dict[tuple[str, int], Acc] = defaultdict(Acc)
    details: list[dict[str, Any]] = []

    for layer, layer_rows in sorted(by_layer.items()):
        for test_prompt in prompts:
            train_rows = [row for row in layer_rows if row["prompt_id"] != test_prompt]
            test_rows = [row for row in layer_rows if row["prompt_id"] == test_prompt]
            if len(train_rows) < 2 or not test_rows:
                continue
            train = torch.stack([row["vector"] for row in train_rows], dim=0).to(torch.float32)
            for rank in ranks:
                mean, basis = fit_basis(torch, train, rank)
                usable_rank = 0 if basis is None else int(basis.shape[0])
                for row in test_rows:
                    y = row["vector"].to(torch.float32)
                    recon = reconstruct(torch, y, mean, basis)
                    err = rel_l2(torch, y, recon)
                    accs[rank].add(err)
                    layer_accs[(layer, rank)].add(err)
                    details.append({
                        "layer": layer,
                        "test_prompt": test_prompt,
                        "group_key": row["group_key"],
                        "rank": rank,
                        "usable_rank": usable_rank,
                        "rel_l2": err,
                    })

    summary = {f"rank{rank}": {"rank": rank, **accs[rank].row()} for rank in ranks}
    layer_summary = {
        f"{layer}:rank{rank}": {"layer": layer, "rank": rank, **acc.row()}
        for (layer, rank), acc in sorted(layer_accs.items())
    }
    return {"summary": summary, "layer_summary": layer_summary}, details


def write_md(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Kimi layer MoE output subspace oracle",
        "",
        "This is a dev-only offline oracle. It does not change runtime behavior or claim SOTA.",
        "",
        f"- generated_at: `{result['generated_at']}`",
        f"- prompts: `{len(result['prompts'])}`",
        f"- groups: `{result['groups']}`",
        f"- ranks: `{result['ranks']}`",
        "",
        "## Leave-One-Prompt-Out Summary",
        "",
        "| rank | rows | mean rel L2 | max rel L2 |",
        "|---:|---:|---:|---:|",
    ]
    for rank in result["ranks"]:
        row = result["summary"][f"rank{rank}"]
        lines.append(f"| `{rank}` | `{row['rows']}` | `{row['mean_rel_l2']:.6f}` | `{row['max_rel_l2']:.6f}` |")
    lines += [
        "",
        "## Worst Layer Rows",
        "",
        "| layer | rank | rows | mean rel L2 | max rel L2 |",
        "|---|---:|---:|---:|---:|",
    ]
    layer_rows = sorted(result["layer_summary"].values(), key=lambda r: (int(r["rank"]), -float(r["mean_rel_l2"])))
    for row in layer_rows[:30]:
        lines.append(
            f"| `{row['layer']}` | `{row['rank']}` | `{row['rows']}` | "
            f"`{row['mean_rel_l2']:.6f}` | `{row['max_rel_l2']:.6f}` |"
        )
    lines += [
        "",
        "## Decision",
        "",
        result["decision"],
        "",
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
    parser.add_argument("--ranks", default="0,1,2,4")
    parser.add_argument("--max-down-records-per-prompt", type=int, default=64)
    parser.add_argument("--target-rank", type=int, default=4)
    parser.add_argument("--target-mean-rel-l2", type=float, default=0.10)
    parser.add_argument("--torch-threads", type=int, default=8)
    args = parser.parse_args()

    import torch

    torch.set_num_threads(args.torch_threads)
    ranks = parse_ints(args.ranks)
    inventory = H.load_inventory(args.inventory)
    lib = H.load_ggml(args.libggml_base)

    all_outputs: list[dict[str, Any]] = []
    prompts = []
    for root in args.prompt_root:
        outputs = compute_prompt_outputs(torch, root, inventory, lib, args.max_down_records_per_prompt)
        all_outputs.extend(outputs)
        prompts.append({
            "prompt_id": root.name,
            "root": str(root),
            "groups": len(outputs),
            "layers": len({row["layer"] for row in outputs}),
        })

    evaluated, details = evaluate(torch, all_outputs, ranks)
    target = evaluated["summary"].get(f"rank{args.target_rank}")
    if target and target["mean_rel_l2"] <= args.target_mean_rel_l2:
        decision = (
            "The layer-output subspace oracle passes the dev gate. "
            "This is not deployable yet; next test coefficient prediction from hidden states."
        )
    else:
        got = target["mean_rel_l2"] if target else float("nan")
        decision = (
            f"Reject tiny layer-output subspace as the next primary path: "
            f"rank {args.target_rank} leave-one-prompt-out mean rel L2 is {got:.6f}, "
            f"above the {args.target_mean_rel_l2:g} gate."
        )

    result = {
        "kind": "kimi_layer_output_subspace_oracle",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prompts": prompts,
        "groups": len(all_outputs),
        "ranks": ranks,
        "target_rank": args.target_rank,
        "target_mean_rel_l2": args.target_mean_rel_l2,
        "summary": evaluated["summary"],
        "layer_summary": evaluated["layer_summary"],
        "details": details,
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
