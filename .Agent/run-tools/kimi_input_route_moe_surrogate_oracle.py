#!/usr/bin/env python3
"""Input+route MoE-output surrogate oracle for Kimi.

This is a dev-only offline screen. It reconstructs the exact summed MoE output
for complete active-expert groups, then tests whether a small learned/prototype
surrogate can predict that output from runtime-available inputs: the pre-MoE
hidden vector and selected expert IDs.

It does not modify runtime behavior and does not claim a token-rate result.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
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


def parse_csv_floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def parse_csv_strings(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def load_rows(path: Path, max_records: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
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
        raise RuntimeError(f"no activation rows in {path}")
    return rows


def layer_name(tensor: str) -> str:
    parts = tensor.split(".")
    if len(parts) >= 2 and parts[0] == "blk":
        return f"{parts[0]}.{parts[1]}"
    return tensor


def group_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (layer_name(row["tensor"]), int(row["call"]), int(row["token_id"]))


def activation_dir(root: Path) -> Path:
    nested = root / "act"
    if (nested / "activations.csv").exists():
        return nested
    if (root / "activations.csv").exists():
        return root
    return nested


def rel_l2(torch: Any, exact: Any, cand: Any) -> float:
    return float(torch.linalg.vector_norm(cand - exact).item()) / max(float(torch.linalg.vector_norm(exact).item()), 1e-30)


class Acc:
    def __init__(self) -> None:
        self.rows = 0
        self.sum_rel = 0.0
        self.max_rel = 0.0

    def add(self, rel: float) -> None:
        self.rows += 1
        self.sum_rel += rel
        self.max_rel = max(self.max_rel, rel)

    def row(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "mean_rel_l2": self.sum_rel / self.rows if self.rows else 0.0,
            "max_rel_l2": self.max_rel,
        }


def build_prompt_groups(torch: Any, root: Path, inventory: dict[str, Any], lib: Any, max_records: int) -> list[dict[str, Any]]:
    act_dir = activation_dir(root)
    rows = load_rows(act_dir / "activations.csv", max_records)
    bin_path = act_dir / "activations.f32"
    by_group: dict[tuple[str, int, int], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["role"] in {"up", "gate", "down"} and row["mode"] == "decode":
            by_group[group_key(row)][row["role"]].append(row)

    # Dequantize only down tensors needed to reconstruct exact MoE outputs.
    down_rows = [row for row in rows if row["role"] == "down" and row["mode"] == "decode"]
    by_tensor_expert: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in down_rows:
        by_tensor_expert[(row["tensor"], row["expert_idx"])].append(row)

    down_outputs: dict[int, Any] = {}
    for (tensor, expert_idx), key_rows in sorted(by_tensor_expert.items()):
        if tensor not in inventory:
            raise RuntimeError(f"tensor not in inventory: {tensor}")
        matrix, _ = R.dequant_to_torch(lib, inventory[tensor], expert_idx, torch)
        matrix = matrix.to(torch.float32)
        for row in key_rows:
            vec = K.load_vector(torch, bin_path, row)
            down_outputs[row["record_id"]] = K.down_matvec(torch, matrix, vec, row)
        del matrix

    by_layer_token: dict[tuple[str, int], dict[str, list[tuple[int, list[dict[str, Any]]]]]] = defaultdict(
        lambda: {"up": [], "down": []}
    )
    for key, role_rows in sorted(by_group.items()):
        layer, call, token_id = key
        up = sorted(role_rows.get("up", []), key=lambda r: r["active_slot"])
        down = sorted(role_rows.get("down", []), key=lambda r: r["active_slot"])
        if up:
            by_layer_token[(layer, token_id)]["up"].append((call, up))
        if down:
            by_layer_token[(layer, token_id)]["down"].append((call, down))

    groups: list[dict[str, Any]] = []
    for (layer, token_id), role_groups in sorted(by_layer_token.items()):
        up_groups = sorted(role_groups["up"], key=lambda item: item[0])
        down_groups = sorted(role_groups["down"], key=lambda item: item[0])
        for pair_idx, ((up_call, up), (down_call, down)) in enumerate(zip(up_groups, down_groups)):
            if len(up) < 2 or len(down) < 2:
                continue
        # Prefer complete top-8 groups; otherwise keep smaller groups only when
        # the dump cap prevented complete capture.
            input_row = up[0]
            x = K.load_vector(torch, bin_path, input_row).to(torch.float32)
            y = torch.stack([down_outputs[row["record_id"]] for row in down], dim=0).sum(dim=0).to(torch.float32)
            expert_ids = [int(row["expert_idx"]) for row in up]
            down_expert_ids = [int(row["expert_idx"]) for row in down]
            groups.append(
                {
                    "prompt_id": root.name,
                    "layer": layer,
                    "call": up_call,
                    "down_call": down_call,
                    "pair_idx": pair_idx,
                    "token_id": token_id,
                    "active_up": len(up),
                    "active_down": len(down),
                    "expert_ids": expert_ids,
                    "down_expert_ids": down_expert_ids,
                    "input": x.contiguous(),
                    "target": y.contiguous(),
                }
            )
    return groups


def route_tensor(torch: Any, expert_ids: list[int]) -> Any:
    ids = torch.tensor(expert_ids, dtype=torch.float32)
    if ids.numel() == 0:
        ids = torch.zeros(1, dtype=torch.float32)
    norm = ids / 383.0
    stats = [
        norm.mean(),
        norm.std(unbiased=False),
        norm.min(),
        norm.max(),
        torch.tensor(float(len(expert_ids)) / 8.0, dtype=torch.float32),
    ]
    hashes = []
    for seed in (0.37, 1.19, 2.31, 3.73, 5.11, 7.07, 11.13, 13.17):
        hashes.append(torch.sin((ids + 1.0) * seed).mean())
        hashes.append(torch.cos((ids + 1.0) * seed).mean())
    return torch.stack(stats + hashes).to(torch.float32)


def make_feature(torch: Any, row: dict[str, Any], mode: str) -> Any:
    x = row["input"].to(torch.float32)
    r = route_tensor(torch, row["expert_ids"])
    if mode == "input":
        return x
    if mode == "input_stats":
        x_stats = torch.tensor(
            [
                float(torch.linalg.vector_norm(x).item()) / math.sqrt(max(x.numel(), 1)),
                float(x.mean().item()),
                float(x.std(unbiased=False).item()),
                float(x.abs().max().item()),
            ],
            dtype=torch.float32,
        )
        return torch.cat([x, x_stats], dim=0)
    if mode == "input_route_stats":
        x_stats = torch.tensor(
            [
                float(torch.linalg.vector_norm(x).item()) / math.sqrt(max(x.numel(), 1)),
                float(x.mean().item()),
                float(x.std(unbiased=False).item()),
                float(x.abs().max().item()),
            ],
            dtype=torch.float32,
        )
        return torch.cat([x, x_stats, r], dim=0)
    if mode == "input_route_gated":
        scales = r[:5]
        return torch.cat([x] + [x * scale for scale in scales], dim=0)
    raise RuntimeError(f"unknown feature mode: {mode}")


def make_features(torch: Any, rows: list[dict[str, Any]], mode: str) -> Any:
    return torch.stack([make_feature(torch, row, mode) for row in rows], dim=0).to(torch.float32)


def standardize(torch: Any, train_x: Any, test_x: Any) -> tuple[Any, Any]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (train_x - mean) / std, (test_x - mean) / std


def normalize(torch: Any, x: Any) -> Any:
    return x / torch.linalg.vector_norm(x, dim=1, keepdim=True).clamp_min(1e-12)


def nearest_predict(torch: Any, train_x: Any, train_y: Any, test_x: Any, scale: bool) -> tuple[Any, Any]:
    train_n = normalize(torch, train_x)
    test_n = normalize(torch, test_x)
    sim = test_n @ train_n.t()
    idx = sim.argmax(dim=1)
    pred = train_y[idx].clone()
    if scale:
        denom = (train_x[idx] * train_x[idx]).sum(dim=1, keepdim=True).clamp_min(1e-12)
        alpha = ((test_x * train_x[idx]).sum(dim=1, keepdim=True) / denom).clamp(-4.0, 4.0)
        pred = pred * alpha
    return pred, idx


def kernel_ridge_predict(torch: Any, train_x: Any, train_y: Any, test_x: Any, lam: float) -> Any:
    dim = max(float(train_x.shape[1]), 1.0)
    k_train = (train_x @ train_x.t()) / dim
    eye = torch.eye(k_train.shape[0], dtype=torch.float32)
    alpha = torch.linalg.solve(k_train + lam * eye, train_y)
    k_test = (test_x @ train_x.t()) / dim
    return k_test @ alpha


def evaluate(torch: Any, groups: list[dict[str, Any]], modes: list[str], lambdas: list[float]) -> dict[str, Any]:
    prompts = sorted({row["prompt_id"] for row in groups})
    by_layer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in groups:
        by_layer[row["layer"]].append(row)

    accs: dict[str, Acc] = defaultdict(Acc)
    layer_accs: dict[str, Acc] = defaultdict(Acc)
    details: list[dict[str, Any]] = []

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
                for scale in (False, True):
                    method = f"nn_{'scaled' if scale else 'raw'}:{mode}"
                    pred, nn_idx = nearest_predict(torch, train_x, train_y, test_x, scale=scale)
                    for idx, row in enumerate(test_rows):
                        err = rel_l2(torch, test_y[idx], pred[idx])
                        accs[method].add(err)
                        layer_accs[f"{layer}:{method}"].add(err)
                        details.append(
                            {
                                "layer": layer,
                                "test_prompt": test_prompt,
                                "method": method,
                                "rel_l2": err,
                                "nearest_train_prompt": train_rows[int(nn_idx[idx])]["prompt_id"],
                            }
                        )
                for lam in lambdas:
                    method = f"krr:{mode}:lambda={lam:g}"
                    pred = kernel_ridge_predict(torch, train_x, centered_train_y, test_x, lam) + y_mean
                    for idx, row in enumerate(test_rows):
                        err = rel_l2(torch, test_y[idx], pred[idx])
                        accs[method].add(err)
                        layer_accs[f"{layer}:{method}"].add(err)
                        details.append(
                            {
                                "layer": layer,
                                "test_prompt": test_prompt,
                                "method": method,
                                "rel_l2": err,
                            }
                        )

    return {
        "summary": {key: {"method": key, **acc.row()} for key, acc in sorted(accs.items())},
        "layer_summary": {
            key: {"layer": key.split(":", 1)[0], "method": key.split(":", 1)[1], **acc.row()}
            for key, acc in sorted(layer_accs.items())
        },
        "details": details,
    }


def feature_dim(mode: str, hidden: int = 2048) -> int:
    if mode == "input":
        return hidden
    if mode == "input_stats":
        return hidden + 4
    if mode == "input_route_stats":
        return hidden + 4 + 21
    if mode == "input_route_gated":
        return hidden * 6
    raise RuntimeError(f"unknown feature mode: {mode}")


def write_md(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Kimi input-route MoE-output surrogate oracle",
        "",
        "This is a dev-only offline oracle. It does not change runtime behavior or claim SOTA.",
        "",
        f"- generated_at: `{result['generated_at']}`",
        f"- prompts: `{len(result['prompt_roots'])}`",
        f"- groups: `{result['groups']}`",
        f"- layers evaluated: `{result['layers']}`",
        f"- max activation records per prompt: `{result['max_records_per_prompt']}`",
        f"- complete-group only: `{result['complete_group_only']}`",
        "",
        "## Summary",
        "",
        "| method | rows | mean rel L2 | max rel L2 |",
        "|---|---:|---:|---:|",
    ]
    for row in sorted(result["summary"].values(), key=lambda r: (float(r["mean_rel_l2"]), r["method"])):
        lines.append(
            f"| `{row['method']}` | `{row['rows']}` | `{row['mean_rel_l2']:.6f}` | `{row['max_rel_l2']:.6f}` |"
        )
    lines += [
        "",
        "## Prototype Resident Estimate",
        "",
        "| feature mode | feature dim | BF16 bytes/prototype with output | MiB/layer @ K=64 | GiB/60 layers @ K=64 |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode, est in result["resident_estimate"].items():
        lines.append(
            f"| `{mode}` | `{est['feature_dim']}` | `{est['bf16_bytes_per_prototype_with_output']}` | "
            f"`{est['mib_per_layer_k64']:.3f}` | `{est['gib_60_layers_k64']:.3f}` |"
        )
    lines += [
        "",
        "## Worst Layer Rows",
        "",
        "| layer | method | rows | mean rel L2 | max rel L2 |",
        "|---|---|---:|---:|---:|",
    ]
    worst = sorted(result["layer_summary"].values(), key=lambda r: -float(r["mean_rel_l2"]))
    for row in worst[:50]:
        lines.append(
            f"| `{row['layer']}` | `{row['method']}` | `{row['rows']}` | "
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
    parser.add_argument("--max-records-per-prompt", type=int, default=0)
    parser.add_argument("--modes", default="input,input_route_stats,input_route_gated")
    parser.add_argument("--lambdas", default="0.1,1.0,10.0,100.0")
    parser.add_argument("--complete-group-only", action="store_true")
    parser.add_argument("--target-rel-l2", type=float, default=0.10)
    parser.add_argument("--torch-threads", type=int, default=8)
    args = parser.parse_args()

    import torch

    torch.set_num_threads(args.torch_threads)
    inventory = H.load_inventory(args.inventory)
    lib = H.load_ggml(args.libggml_base)
    modes = parse_csv_strings(args.modes)
    lambdas = parse_csv_floats(args.lambdas)

    groups = []
    for root in args.prompt_root:
        groups.extend(build_prompt_groups(torch, root, inventory, lib, args.max_records_per_prompt))
    if args.complete_group_only:
        groups = [row for row in groups if row["active_up"] == 8 and row["active_down"] == 8]

    eval_result = evaluate(torch, groups, modes, lambdas)
    resident_estimate = {}
    for mode in modes:
        dim = feature_dim(mode)
        bytes_per = (dim + 2048) * 2
        mib_layer = bytes_per * 64 / (1024 * 1024)
        resident_estimate[mode] = {
            "feature_dim": dim,
            "bf16_bytes_per_prototype_with_output": bytes_per,
            "mib_per_layer_k64": mib_layer,
            "gib_60_layers_k64": mib_layer * 60 / 1024,
        }

    candidates = sorted(
        ((row["mean_rel_l2"], row["method"], row["max_rel_l2"]) for row in eval_result["summary"].values()),
        key=lambda x: x[0],
    )
    if candidates and candidates[0][0] <= args.target_rel_l2:
        decision = (
            f"Proceed: best input-route surrogate {candidates[0][1]} has mean rel L2 "
            f"{candidates[0][0]:.6f}, within the {args.target_rel_l2:g} gate. "
            "Do not use held-out until a concrete default-off runtime path is designed."
        )
    elif candidates:
        decision = (
            f"Reject as primary: best input-route surrogate {candidates[0][1]} has mean rel L2 "
            f"{candidates[0][0]:.6f}, above the {args.target_rel_l2:g} gate. "
            "This closes the small prototype/kernel full-MoE-output surrogate family for now."
        )
    else:
        decision = "Reject/defer: no complete leave-one-prompt-out groups were available."

    result = {
        "kind": "kimi_input_route_moe_surrogate_oracle",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prompt_roots": [str(p) for p in args.prompt_root],
        "inventory": str(args.inventory),
        "libggml_base": str(args.libggml_base),
        "max_records_per_prompt": args.max_records_per_prompt,
        "complete_group_only": args.complete_group_only,
        "modes": modes,
        "lambdas": lambdas,
        "groups": len(groups),
        "layers": len({row["layer"] for row in groups}),
        "summary": eval_result["summary"],
        "layer_summary": eval_result["layer_summary"],
        "resident_estimate": resident_estimate,
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
