#!/usr/bin/env python3
"""Oracle for exact top-activation input-channel expert matvecs.

This is a dev-only offline screen. For each dumped activation vector, it keeps
the largest-magnitude input channels and computes the expert matvec from only
those exact columns/rows. The keep fraction is an optimistic moved-byte ratio
for an activation-guided partial-read representation.
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


def load_vector(torch: Any, bin_path: Path, row: dict[str, Any]) -> Any:
    with bin_path.open("rb") as f:
        f.seek(row["offset_bytes"])
        raw = f.read(row["nbytes"])
    expected = row["ne00"] * 4
    if len(raw) != expected:
        raise RuntimeError(f"short activation read for record {row['record_id']}: {len(raw)} != {expected}")
    return torch.frombuffer(bytearray(raw), dtype=torch.float32).clone()


def expert_matvec(torch: Any, matrix: Any, vec: Any, row: dict[str, Any]) -> Any:
    if matrix.shape[1] == row["ne00"] and matrix.shape[0] == row["ne01"]:
        return torch.mv(matrix, vec)
    if matrix.shape[0] == row["ne00"] and matrix.shape[1] == row["ne01"]:
        return torch.mv(matrix.t().contiguous(), vec)
    raise RuntimeError(
        f"shape mismatch for {row['tensor']} expert {row['expert_idx']}: "
        f"matrix={tuple(matrix.shape)} activation ne00/ne01={row['ne00']}/{row['ne01']}"
    )


def top_input_matvec(torch: Any, matrix: Any, vec: Any, row: dict[str, Any], keep_frac: float) -> Any:
    n_input = row["ne00"]
    keep = max(1, min(n_input, int(round(n_input * keep_frac))))
    idx = torch.topk(vec.abs(), keep, largest=True, sorted=False).indices
    if matrix.shape[1] == n_input and matrix.shape[0] == row["ne01"]:
        return torch.mv(matrix[:, idx].contiguous(), vec[idx])
    if matrix.shape[0] == n_input and matrix.shape[1] == row["ne01"]:
        return torch.mv(matrix[idx, :].t().contiguous(), vec[idx])
    raise RuntimeError(
        f"shape mismatch for top input matvec: matrix={tuple(matrix.shape)} "
        f"activation ne00/ne01={row['ne00']}/{row['ne01']}"
    )


def silu(torch: Any, x: Any) -> Any:
    return x * torch.sigmoid(x)


class Acc:
    def __init__(self) -> None:
        self.rows = 0
        self.rel_sum = 0.0
        self.rel_max = 0.0
        self.abs_sum = 0.0
        self.abs_max = 0.0
        self.ratio_sum = 0.0

    def add(self, rel: float, abs_mean: float, abs_max: float, ratio: float) -> None:
        self.rows += 1
        self.rel_sum += rel
        self.rel_max = max(self.rel_max, rel)
        self.abs_sum += abs_mean
        self.abs_max = max(self.abs_max, abs_max)
        self.ratio_sum += ratio

    def row(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "mean_rel_l2": self.rel_sum / self.rows if self.rows else 0.0,
            "max_rel_l2": self.rel_max,
            "mean_abs_error": self.abs_sum / self.rows if self.rows else 0.0,
            "max_abs_error": self.abs_max,
            "mean_byte_ratio": self.ratio_sum / self.rows if self.rows else 0.0,
        }


def error_metrics(torch: Any, exact: Any, cand: Any) -> tuple[float, float, float]:
    diff = cand - exact
    rel = float(torch.linalg.vector_norm(diff).item()) / max(float(torch.linalg.vector_norm(exact).item()), 1e-30)
    return rel, float(diff.abs().mean().item()), float(diff.abs().max().item())


def write_md(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Kimi exact input-channel keep oracle",
        "",
        "This is a dev-only offline oracle. It does not change runtime behavior or claim SOTA.",
        "",
        f"- generated_at: `{result['generated_at']}`",
        f"- activation_csv: `{result['activation_csv']}`",
        f"- records_loaded: `{result['records_loaded']}`",
        f"- unique tensor experts: `{result['unique_tensor_experts']}`",
        f"- keep fractions: `{result['keep_fracs']}`",
        "",
        "## Matvec Output Error",
        "",
        "| role | keep | rows | byte ratio | mean rel L2 | max rel L2 | mean abs err | max abs err |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, row in result["aggregate"].items():
        role, keep = key.split(":")
        lines.append(
            f"| `{role}` | `{keep}` | `{row['rows']}` | `{row['mean_byte_ratio']:.4f}` | "
            f"`{row['mean_rel_l2']:.6f}` | `{row['max_rel_l2']:.6f}` | "
            f"`{row['mean_abs_error']:.6g}` | `{row['max_abs_error']:.6g}` |"
        )
    lines += [
        "",
        "## Fused Up/Gate Output Error",
        "",
        "| keep | pairs | byte ratio | mean rel L2 | max rel L2 | mean abs err | max abs err |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, row in result["fused_up_gate"].items():
        _, keep = key.split(":")
        lines.append(
            f"| `{keep}` | `{row['rows']}` | `{row['mean_byte_ratio']:.4f}` | "
            f"`{row['mean_rel_l2']:.6f}` | `{row['max_rel_l2']:.6f}` | "
            f"`{row['mean_abs_error']:.6g}` | `{row['max_abs_error']:.6g}` |"
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
    parser.add_argument("--activation-csv", type=Path, required=True)
    parser.add_argument("--activation-bin", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--libggml-base", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--keep-fracs", default="0.01,0.02,0.05,0.1,0.2,0.4")
    parser.add_argument("--max-records", type=int, default=512)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--target-keep-frac", type=float, default=0.40)
    parser.add_argument("--target-mean-rel-l2", type=float, default=0.10)
    args = parser.parse_args()

    import torch

    torch.set_num_threads(args.torch_threads)
    keep_fracs = parse_floats(args.keep_fracs)
    inventory = H.load_inventory(args.inventory)
    lib = H.load_ggml(args.libggml_base)
    rows = load_rows(args.activation_csv, args.max_records)

    by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[(row["tensor"], row["expert_idx"])].append(row)

    aggregate: dict[str, Acc] = defaultdict(Acc)
    fused_acc: dict[str, Acc] = defaultdict(Acc)
    exact_outputs: dict[int, Any] = {}
    cand_outputs: dict[tuple[int, float], Any] = {}

    for (tensor, expert_idx), key_rows in sorted(by_key.items()):
        if tensor not in inventory:
            raise RuntimeError(f"tensor not in inventory: {tensor}")
        matrix, _ = R.dequant_to_torch(lib, inventory[tensor], expert_idx, torch)
        matrix = matrix.to(torch.float32)
        vectors = {row["record_id"]: load_vector(torch, args.activation_bin, row) for row in key_rows}
        for row in key_rows:
            rid = row["record_id"]
            exact = expert_matvec(torch, matrix, vectors[rid], row)
            exact_outputs[rid] = exact
            for keep_frac in keep_fracs:
                cand = top_input_matvec(torch, matrix, vectors[rid], row, keep_frac)
                cand_outputs[(rid, keep_frac)] = cand
                rel, abs_mean, abs_max = error_metrics(torch, exact, cand)
                aggregate[f"{row['role']}:{keep_frac:g}"].add(rel, abs_mean, abs_max, keep_frac)
        del matrix

    pair_index: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["role"] not in ("up", "gate"):
            continue
        key = (row["mode"], row["call"], row["expert_idx"], row["active_slot"], row["dst_id"], row["token_id"])
        pair_index[key][row["role"]] = row

    for pair in pair_index.values():
        if "up" not in pair or "gate" not in pair:
            continue
        up = pair["up"]
        gate = pair["gate"]
        exact = exact_outputs[up["record_id"]] * silu(torch, exact_outputs[gate["record_id"]])
        for keep_frac in keep_fracs:
            cand = cand_outputs[(up["record_id"], keep_frac)] * silu(torch, cand_outputs[(gate["record_id"], keep_frac)])
            rel, abs_mean, abs_max = error_metrics(torch, exact, cand)
            fused_acc[f"fused_up_gate:{keep_frac:g}"].add(rel, abs_mean, abs_max, keep_frac)

    aggregate_rows = {key: acc.row() for key, acc in sorted(aggregate.items())}
    fused_rows = {key: acc.row() for key, acc in sorted(fused_acc.items())}
    passing = [
        (key, row)
        for key, row in {**aggregate_rows, **fused_rows}.items()
        if row["mean_byte_ratio"] <= args.target_keep_frac and row["mean_rel_l2"] <= args.target_mean_rel_l2
    ]
    if passing:
        decision = (
            "At least one exact input-channel keep candidate passes the offline error gate. "
            "Do not use held-out yet; next build a multi-prompt byte and runtime feasibility plan."
        )
    else:
        decision = (
            "No exact input-channel keep candidate passes the offline error gate within the target byte ratio. "
            "Do not implement activation-guided partial exact reads as the next primary runtime path."
        )

    result = {
        "kind": "kimi_exact_input_keep_oracle",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "activation_csv": str(args.activation_csv),
        "activation_bin": str(args.activation_bin),
        "inventory": str(args.inventory),
        "libggml_base": str(args.libggml_base),
        "records_loaded": len(rows),
        "unique_tensor_experts": len(by_key),
        "keep_fracs": keep_fracs,
        "aggregate": aggregate_rows,
        "fused_up_gate": fused_rows,
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
