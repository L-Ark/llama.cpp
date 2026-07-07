#!/usr/bin/env python3
"""Activation-weighted cluster-base residual oracle for Kimi down experts.

This is a dev-only offline screen. It does not change runtime behavior and does
not use held-out prompts. The goal is to test whether a D2MoE-style
cluster-base plus low-bit residual representation can preserve actual down
outputs while moving only 0.30x-0.40x of the current IQ3_S expert bytes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_helper(name: str, filename: str) -> Any:
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


K = load_helper("kimi_joint_intermediate_keep_oracle", "kimi_joint_intermediate_keep_oracle.py")
C = load_helper("kimi_d2moe_cluster_base_bound", "kimi_d2moe_cluster_base_bound.py")
R = K.R
H = K.H


def parse_ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def load_prompt_rows(prompt_root: Path, max_records: int) -> list[dict[str, Any]]:
    rows = K.load_rows(prompt_root / "act" / "activations.csv", max_records)
    for row in rows:
        row["prompt_id"] = prompt_root.name
    return rows


def rel_l2(torch: Any, exact: Any, cand: Any) -> float:
    return float(torch.linalg.vector_norm(cand - exact).item()) / max(float(torch.linalg.vector_norm(exact).item()), 1e-30)


def quantize_blockwise(torch: Any, values: Any, bits: int, block: int) -> tuple[Any, int]:
    flat = values.reshape(-1).to(torch.float32)
    n = flat.numel()
    pad = (block - (n % block)) % block
    if pad:
        flat_padded = torch.nn.functional.pad(flat, (0, pad))
    else:
        flat_padded = flat
    grouped = flat_padded.reshape(-1, block)
    scales = grouped.abs().amax(dim=1).clamp_min(1e-12)
    if bits == 1:
        q = torch.sign(grouped)
        recon = q * scales[:, None]
    else:
        qmax = (1 << (bits - 1)) - 1
        q = torch.round(grouped / scales[:, None] * qmax).clamp(-qmax, qmax)
        recon = q * (scales[:, None] / qmax)
    recon = recon.reshape(-1)[:n].reshape_as(values).contiguous()
    payload_bytes = math.ceil(n * bits / 8)
    scale_bytes = scales.numel() * 2
    return recon, int(payload_bytes + scale_bytes)


class Acc:
    def __init__(self) -> None:
        self.n = 0
        self.sum_rel = 0.0
        self.max_rel = 0.0
        self.sum_ratio = 0.0

    def add(self, rel: float, ratio: float) -> None:
        self.n += 1
        self.sum_rel += rel
        self.max_rel = max(self.max_rel, rel)
        self.sum_ratio += ratio

    def row(self) -> dict[str, Any]:
        return {
            "rows": self.n,
            "mean_rel_l2": self.sum_rel / self.n if self.n else 0.0,
            "max_rel_l2": self.max_rel,
            "mean_moved_ratio": self.sum_ratio / self.n if self.n else 0.0,
        }


def choose_tensors(rows: list[dict[str, Any]], max_tensors: int) -> list[str]:
    counts = Counter(row["tensor"] for row in rows)
    return [tensor for tensor, _ in counts.most_common(max_tensors)]


def make_sketches(torch: Any, matrices: list[Any], sketch_dim: int, seed: int) -> Any:
    n = matrices[0].numel()
    usable = min(sketch_dim, n)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    idx = torch.randperm(n, generator=gen)[:usable]
    sketches = []
    for matrix in matrices:
        sketches.append(matrix.reshape(-1)[idx].to(torch.float32).clone())
    return C.normalize_features(torch, torch.stack(sketches, dim=0))


def group_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row["prompt_id"], row["mode"], row["call"], K.layer_name(row["tensor"]), row["tensor"], row["token_id"])


def evaluate_tensor(
    torch: Any,
    tensor: str,
    rows: list[dict[str, Any]],
    inventory: dict[str, Any],
    lib: Any,
    clusters_values: list[int],
    bits_values: list[int],
    block: int,
    sketch_dim: int,
    seed: int,
) -> dict[str, Any]:
    row_meta = inventory[tensor]
    by_expert: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["tensor"] == tensor:
            by_expert[int(row["expert_idx"])].append(row)

    experts = sorted(by_expert)
    matrices = []
    exact_outputs: dict[tuple[str, int], Any] = {}
    activation_cache: dict[tuple[str, int], Any] = {}
    weights = torch.tensor([len(by_expert[expert]) for expert in experts], dtype=torch.float32)

    for expert in experts:
        matrix, _ = R.dequant_to_torch(lib, row_meta, expert, torch)
        matrices.append(matrix.to(torch.float32).contiguous())
        for row in by_expert[expert]:
            root = Path(row["prompt_root"])
            key = (row["prompt_id"], row["record_id"])
            vec = K.load_vector(torch, root / "act" / "activations.f32", row)
            activation_cache[key] = vec
            exact_outputs[key] = K.down_matvec(torch, matrices[-1], vec, row)

    sketches = make_sketches(torch, matrices, sketch_dim, seed)
    result = {
        "tensor": tensor,
        "layer": row_meta.layer,
        "kind": row_meta.kind,
        "type": row_meta.type_name,
        "shape": row_meta.shape,
        "expert_bytes": row_meta.expert_bytes,
        "experts": experts,
        "rows": sum(len(v) for v in by_expert.values()),
        "cluster_results": [],
    }

    for clusters in clusters_values:
        k = min(clusters, len(experts))
        assignments = C.weighted_kmeans(torch, sketches, weights, k, iters=12)
        cluster_members: dict[int, list[int]] = defaultdict(list)
        for idx, cid in enumerate(assignments):
            cluster_members[cid].append(idx)

        bases: dict[int, Any] = {}
        for cid, member_indices in cluster_members.items():
            w = weights[torch.tensor(member_indices, dtype=torch.long)]
            base = torch.zeros_like(matrices[member_indices[0]])
            for local_idx, weight in zip(member_indices, w):
                base += matrices[int(local_idx)] * float(weight.item())
            bases[cid] = (base / float(w.sum().item())).contiguous()

        candidate_outputs: dict[tuple[str, int, str], Any] = {}
        candidate_ratios: dict[str, float] = {"base_only": 0.0}
        moved_bytes_sum: dict[str, int] = defaultdict(int)
        moved_rows: dict[str, int] = defaultdict(int)

        for local_idx, expert in enumerate(experts):
            cid = assignments[local_idx]
            base = bases[cid]
            residual = matrices[local_idx] - base
            residual_candidates: dict[str, Any] = {"base_only": base}
            residual_bytes: dict[str, int] = {"base_only": 0}
            for bits in bits_values:
                q_residual, q_bytes = quantize_blockwise(torch, residual, bits, block)
                name = f"residual_{bits}bit"
                residual_candidates[name] = (base + q_residual).contiguous()
                residual_bytes[name] = q_bytes

            for row in by_expert[expert]:
                key = (row["prompt_id"], row["record_id"])
                vec = activation_cache[key]
                for name, matrix in residual_candidates.items():
                    candidate_outputs[(row["prompt_id"], row["record_id"], name)] = K.down_matvec(torch, matrix, vec, row)
                    moved_bytes_sum[name] += residual_bytes[name]
                    moved_rows[name] += 1

        for name, total_bytes in moved_bytes_sum.items():
            if moved_rows[name]:
                candidate_ratios[name] = (total_bytes / moved_rows[name]) / max(float(row_meta.expert_bytes), 1.0)

        row_acc: dict[str, Acc] = defaultdict(Acc)
        for row in [r for r in rows if r["tensor"] == tensor]:
            key = (row["prompt_id"], row["record_id"])
            exact = exact_outputs[key]
            for name in candidate_ratios:
                cand = candidate_outputs[(row["prompt_id"], row["record_id"], name)]
                row_acc[name].add(rel_l2(torch, exact, cand), candidate_ratios[name])

        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in [r for r in rows if r["tensor"] == tensor]:
            groups[group_key(row)].append(row)
        group_acc: dict[str, Acc] = defaultdict(Acc)
        for group_rows in groups.values():
            exact_sum = torch.stack(
                [exact_outputs[(row["prompt_id"], row["record_id"])] for row in group_rows], dim=0
            ).sum(dim=0)
            for name in candidate_ratios:
                cand_sum = torch.stack(
                    [candidate_outputs[(row["prompt_id"], row["record_id"], name)] for row in group_rows], dim=0
                ).sum(dim=0)
                group_acc[name].add(rel_l2(torch, exact_sum, cand_sum), candidate_ratios[name])

        base_bf16_mib = k * matrices[0].numel() * 2 / (1024 * 1024)
        result["cluster_results"].append(
            {
                "clusters": k,
                "base_bf16_mib": base_bf16_mib,
                "row_summary": {key: acc.row() for key, acc in sorted(row_acc.items())},
                "group_summary": {key: acc.row() for key, acc in sorted(group_acc.items())},
                "cluster_sizes": {str(cid): len(members) for cid, members in sorted(cluster_members.items())},
            }
        )

        del bases

    del matrices
    return result


def write_md(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Kimi activation-weighted cluster-base residual oracle",
        "",
        "This is a dev-only offline oracle. It does not change runtime behavior or claim SOTA.",
        "",
        f"- generated_at: `{result['generated_at']}`",
        f"- prompt roots: `{len(result['prompt_roots'])}`",
        f"- max records per prompt: `{result['max_records_per_prompt']}`",
        f"- tensors evaluated: `{len(result['tensors'])}`",
        f"- bits: `{result['bits']}`",
        f"- block: `{result['block']}`",
        "",
        "## Summary",
        "",
        "| tensor | clusters | candidate | moved ratio | base BF16 MiB | row mean rel L2 | group mean rel L2 | group max rel L2 |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for tensor in result["tensors"]:
        for cluster in tensor["cluster_results"]:
            for candidate, group in cluster["group_summary"].items():
                row = cluster["row_summary"][candidate]
                lines.append(
                    f"| `{tensor['tensor']}` | `{cluster['clusters']}` | `{candidate}` | "
                    f"`{group['mean_moved_ratio']:.4f}` | `{cluster['base_bf16_mib']:.2f}` | "
                    f"`{row['mean_rel_l2']:.6f}` | `{group['mean_rel_l2']:.6f}` | "
                    f"`{group['max_rel_l2']:.6f}` |"
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
    parser.add_argument("--max-records-per-prompt", type=int, default=96)
    parser.add_argument("--max-tensors", type=int, default=2)
    parser.add_argument("--clusters", default="1,4,8")
    parser.add_argument("--bits", default="1,2")
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--sketch-dim", type=int, default=4096)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    import torch

    torch.set_num_threads(args.torch_threads)
    inventory = H.load_inventory(args.inventory)
    lib = H.load_ggml(args.libggml_base)
    clusters_values = parse_ints(args.clusters)
    bits_values = parse_ints(args.bits)

    rows: list[dict[str, Any]] = []
    for prompt_root in args.prompt_root:
        prompt_rows = load_prompt_rows(prompt_root, args.max_records_per_prompt)
        for row in prompt_rows:
            row["prompt_root"] = str(prompt_root)
        rows.extend(prompt_rows)
    tensors = choose_tensors(rows, args.max_tensors)

    result = {
        "kind": "kimi_cluster_base_activation_oracle",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prompt_roots": [str(p) for p in args.prompt_root],
        "inventory": str(args.inventory),
        "libggml_base": str(args.libggml_base),
        "max_records_per_prompt": args.max_records_per_prompt,
        "max_tensors": args.max_tensors,
        "clusters": clusters_values,
        "bits": bits_values,
        "block": args.block,
        "sketch_dim": args.sketch_dim,
        "reproduce_command": " ".join(os.sys.argv),
        "tensors": [],
        "decision": "",
    }

    for tensor in tensors:
        result["tensors"].append(
            evaluate_tensor(
                torch,
                tensor,
                rows,
                inventory,
                lib,
                clusters_values,
                bits_values,
                args.block,
                args.sketch_dim,
                args.seed,
            )
        )

    best_under_target = []
    for tensor in result["tensors"]:
        for cluster in tensor["cluster_results"]:
            for name, group in cluster["group_summary"].items():
                if 0.30 <= group["mean_moved_ratio"] <= 0.40:
                    best_under_target.append((group["mean_rel_l2"], tensor["tensor"], cluster["clusters"], name))
    if best_under_target:
        best_under_target.sort()
        best = best_under_target[0]
        if best[0] <= 0.10:
            result["decision"] = (
                f"Proceed: best target-ratio candidate {best[1]} clusters={best[2]} "
                f"{best[3]} has group mean rel L2 {best[0]:.6f}."
            )
        else:
            result["decision"] = (
                f"Reject as primary: best target-ratio candidate {best[1]} clusters={best[2]} "
                f"{best[3]} has group mean rel L2 {best[0]:.6f}, above the 0.10 gate."
            )
    else:
        result["decision"] = "Reject/defer: no evaluated candidate landed in the 0.30x-0.40x moved-byte target."

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_md(args.out_md, result)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    print(result["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

