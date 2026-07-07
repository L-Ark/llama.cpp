#!/usr/bin/env python3
"""Shared down-projector oracle for Kimi MoE layers.

This is a dev-only offline screen. It tests whether the summed down output of a
layer can be predicted from aggregated down intermediate activations with one
shared per-layer projector, avoiding per-expert down tensor movement.
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


def load_helper() -> Any:
    path = Path(__file__).with_name("kimi_joint_intermediate_keep_oracle.py")
    spec = importlib.util.spec_from_file_location("kimi_joint_intermediate_keep_oracle", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


K = load_helper()
R = K.R
H = K.H


def parse_floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def feature_modes(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def rel_l2(torch: Any, exact: Any, cand: Any) -> float:
    return float(torch.linalg.vector_norm(cand - exact).item()) / max(float(torch.linalg.vector_norm(exact).item()), 1e-30)


def prompt_down_groups(torch: Any, root: Path, inventory: dict[str, Any], lib: Any, max_down_records: int) -> list[dict[str, Any]]:
    rows = K.load_rows(root / "act" / "activations.csv", max_down_records)
    rows = [row for row in rows if row.get("role") == "down"]
    bin_path = root / "act" / "activations.f32"
    by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[(row["tensor"], row["expert_idx"])].append(row)

    vecs: dict[int, Any] = {}
    exact_outputs: dict[int, Any] = {}
    for (tensor, expert_idx), key_rows in sorted(by_key.items()):
        if tensor not in inventory:
            raise RuntimeError(f"tensor not in inventory: {tensor}")
        matrix, _ = R.dequant_to_torch(lib, inventory[tensor], expert_idx, torch)
        matrix = matrix.to(torch.float32)
        for row in key_rows:
            vec = K.load_vector(torch, bin_path, row)
            vecs[row["record_id"]] = vec
            exact_outputs[row["record_id"]] = K.down_matvec(torch, matrix, vec, row)
        del matrix

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (K.layer_name(row["tensor"]), row["mode"], row["call"], row["tensor"], row["token_id"])
        grouped[key].append(row)

    out = []
    for key, group_rows in sorted(grouped.items()):
        if len(group_rows) < 2:
            continue
        h_stack = torch.stack([vecs[row["record_id"]] for row in group_rows], dim=0).to(torch.float32)
        y_stack = torch.stack([exact_outputs[row["record_id"]] for row in group_rows], dim=0).to(torch.float32)
        out.append(
            {
                "prompt_id": root.name,
                "layer": key[0],
                "group_key": "|".join(map(str, key)),
                "active": len(group_rows),
                "sum_h": h_stack.sum(dim=0).contiguous(),
                "sum_abs_h": h_stack.abs().sum(dim=0).contiguous(),
                "sum_sq_h": (h_stack * h_stack).sum(dim=0).contiguous(),
                "target": y_stack.sum(dim=0).contiguous(),
            }
        )
    return out


def make_features(torch: Any, rows: list[dict[str, Any]], mode: str) -> Any:
    chunks = []
    for row in rows:
        if mode == "sum_h":
            feat = row["sum_h"]
        elif mode == "sum_h_abs":
            feat = torch.cat([row["sum_h"], row["sum_abs_h"]], dim=0)
        elif mode == "sum_h_abs_sq":
            feat = torch.cat([row["sum_h"], row["sum_abs_h"], row["sum_sq_h"]], dim=0)
        else:
            raise RuntimeError(f"unknown feature mode: {mode}")
        chunks.append(feat)
    return torch.stack(chunks, dim=0).to(torch.float32)


def standardize(torch: Any, train_x: Any, test_x: Any) -> tuple[Any, Any]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (train_x - mean) / std, (test_x - mean) / std


def kernel_ridge_predict(torch: Any, train_x: Any, train_y: Any, test_x: Any, lam: float) -> Any:
    dim = max(float(train_x.shape[1]), 1.0)
    k_train = (train_x @ train_x.t()) / dim
    eye = torch.eye(k_train.shape[0], dtype=torch.float32)
    alpha = torch.linalg.solve(k_train + lam * eye, train_y)
    k_test = (test_x @ train_x.t()) / dim
    return k_test @ alpha


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


def evaluate(torch: Any, groups: list[dict[str, Any]], modes: list[str], lambdas: list[float]) -> dict[str, Any]:
    by_layer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prompts = sorted({row["prompt_id"] for row in groups})
    for row in groups:
        by_layer[row["layer"]].append(row)

    accs: dict[tuple[str, float], Acc] = defaultdict(Acc)
    layer_accs: dict[tuple[str, str, float], Acc] = defaultdict(Acc)
    details = []

    for layer, layer_rows in sorted(by_layer.items()):
        for test_prompt in prompts:
            train_rows = [row for row in layer_rows if row["prompt_id"] != test_prompt]
            test_rows = [row for row in layer_rows if row["prompt_id"] == test_prompt]
            if len(train_rows) < 2 or not test_rows:
                continue
            train_y = torch.stack([row["target"] for row in train_rows], dim=0).to(torch.float32)
            test_y = torch.stack([row["target"] for row in test_rows], dim=0).to(torch.float32)
            y_mean = train_y.mean(dim=0, keepdim=True)
            centered_train_y = train_y - y_mean
            for mode in modes:
                train_x = make_features(torch, train_rows, mode)
                test_x = make_features(torch, test_rows, mode)
                train_x, test_x = standardize(torch, train_x, test_x)
                for lam in lambdas:
                    pred = kernel_ridge_predict(torch, train_x, centered_train_y, test_x, lam) + y_mean
                    for idx, row in enumerate(test_rows):
                        err = rel_l2(torch, test_y[idx], pred[idx])
                        accs[(mode, lam)].add(err)
                        layer_accs[(layer, mode, lam)].add(err)
                        details.append(
                            {
                                "layer": layer,
                                "test_prompt": test_prompt,
                                "group_key": row["group_key"],
                                "mode": mode,
                                "lambda": lam,
                                "rel_l2": err,
                            }
                        )

    return {
        "summary": {f"{mode}:lambda={lam:g}": {"mode": mode, "lambda": lam, **acc.row()} for (mode, lam), acc in sorted(accs.items())},
        "layer_summary": {
            f"{layer}:{mode}:lambda={lam:g}": {"layer": layer, "mode": mode, "lambda": lam, **acc.row()}
            for (layer, mode, lam), acc in sorted(layer_accs.items())
        },
        "details": details,
    }


def mode_dim(mode: str, hidden: int = 2048) -> int:
    if mode == "sum_h":
        return hidden
    if mode == "sum_h_abs":
        return hidden * 2
    if mode == "sum_h_abs_sq":
        return hidden * 3
    raise RuntimeError(f"unknown mode: {mode}")


def write_md(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Kimi shared down-projector oracle",
        "",
        "This is a dev-only offline oracle. It does not change runtime behavior or claim SOTA.",
        "",
        f"- generated_at: `{result['generated_at']}`",
        f"- prompts: `{len(result['prompt_roots'])}`",
        f"- groups: `{result['groups']}`",
        f"- layers evaluated: `{result['layers']}`",
        f"- max down records per prompt: `{result['max_down_records_per_prompt']}`",
        "",
        "## Summary",
        "",
        "| mode | lambda | rows | mean rel L2 | max rel L2 | BF16 MiB/layer | BF16 GiB/60 layers |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, row in sorted(result["summary"].items(), key=lambda kv: (kv[1]["mode"], kv[1]["lambda"])):
        mib = result["resident_mib_per_layer"][row["mode"]]
        lines.append(
            f"| `{row['mode']}` | `{row['lambda']:g}` | `{row['rows']}` | "
            f"`{row['mean_rel_l2']:.6f}` | `{row['max_rel_l2']:.6f}` | "
            f"`{mib:.2f}` | `{mib * 60 / 1024:.2f}` |"
        )
    lines += [
        "",
        "## Worst Layer Rows",
        "",
        "| layer | mode | lambda | rows | mean rel L2 | max rel L2 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    worst = sorted(result["layer_summary"].values(), key=lambda r: -float(r["mean_rel_l2"]))
    for row in worst[:40]:
        lines.append(
            f"| `{row['layer']}` | `{row['mode']}` | `{row['lambda']:g}` | `{row['rows']}` | "
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
    parser.add_argument("--max-down-records-per-prompt", type=int, default=512)
    parser.add_argument("--modes", default="sum_h,sum_h_abs,sum_h_abs_sq")
    parser.add_argument("--lambdas", default="0.001,0.01,0.1,1.0")
    parser.add_argument("--torch-threads", type=int, default=8)
    args = parser.parse_args()

    import torch

    torch.set_num_threads(args.torch_threads)
    inventory = H.load_inventory(args.inventory)
    lib = H.load_ggml(args.libggml_base)
    modes = feature_modes(args.modes)
    lambdas = parse_floats(args.lambdas)

    groups = []
    for root in args.prompt_root:
        groups.extend(prompt_down_groups(torch, root, inventory, lib, args.max_down_records_per_prompt))

    eval_result = evaluate(torch, groups, modes, lambdas)
    resident_mib = {mode: mode_dim(mode) * 7168 * 2 / (1024 * 1024) for mode in modes}

    candidates = []
    for key, row in eval_result["summary"].items():
        candidates.append((row["mean_rel_l2"], row["mode"], row["lambda"], row["max_rel_l2"]))
    candidates.sort()
    if candidates and candidates[0][0] <= 0.10:
        decision = (
            f"Proceed: best shared down projector {candidates[0][1]} lambda={candidates[0][2]:g} "
            f"has mean rel L2 {candidates[0][0]:.6f}."
        )
    elif candidates:
        decision = (
            f"Reject as primary: best shared down projector {candidates[0][1]} lambda={candidates[0][2]:g} "
            f"has mean rel L2 {candidates[0][0]:.6f}, above the 0.10 gate."
        )
    else:
        decision = "Reject/defer: insufficient complete down groups for leave-one-prompt-out evaluation."

    result = {
        "kind": "kimi_shared_down_projector_oracle",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prompt_roots": [str(p) for p in args.prompt_root],
        "inventory": str(args.inventory),
        "libggml_base": str(args.libggml_base),
        "max_down_records_per_prompt": args.max_down_records_per_prompt,
        "modes": modes,
        "lambdas": lambdas,
        "groups": len(groups),
        "layers": len({row["layer"] for row in groups}),
        "resident_mib_per_layer": resident_mib,
        "summary": eval_result["summary"],
        "layer_summary": eval_result["layer_summary"],
        "decision": decision,
        "reproduce_command": " ".join(os.sys.argv),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_md(args.out_md, result)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    print(decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

