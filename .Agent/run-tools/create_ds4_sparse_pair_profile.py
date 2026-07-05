#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/root/lfz/ik_llama/gguf-py")
from gguf import GGUFReader


DEFAULT_MODEL = "/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf"
DEFAULT_TOUCH_PROFILE = (
    "/root/lfz/runs/vendor-ds4-16gb/"
    "20260705T033155Z-20260705_current_head_sota44_touch_split/"
    "france-touch-cpu40-vram0gb/fallback_profile.csv"
)


def run_text(cmd: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(cmd, cwd=cwd, text=True).strip()


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_layer(tensor: str) -> int:
    m = re.search(r"blk\.(\d+)\.", tensor)
    if not m:
        raise ValueError(f"cannot parse layer from tensor name: {tensor}")
    return int(m.group(1))


def tensor_role(tensor: str) -> str | None:
    if "ffn_up_exps" in tensor:
        return "up"
    if "ffn_down_exps" in tensor:
        return "down"
    return None


def load_pairs(fallback_profile: Path) -> list[dict]:
    by_pair: dict[tuple[int, int], dict] = {}
    with fallback_profile.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            role = tensor_role(row["tensor"])
            if row["phase"] != "decode" or role is None:
                continue
            layer = tensor_layer(row["tensor"])
            expert = int(row["expert_idx"])
            pair = by_pair.setdefault((layer, expert), {
                "layer": layer,
                "expert": expert,
                "up": None,
                "down": None,
            })
            pair[role] = {
                "tensor": row["tensor"],
                "calls": int(row["calls"]),
                "hot_ms": int(row["fallback_us"]) / 1000.0,
                "touch_ms": int(row.get("touch_us", 0)) / 1000.0,
                "expert_bytes": int(row["expert_bytes"]),
                "gguf_calls": int(row.get("gguf_calls", 0)),
                "pack_mmap_calls": int(row.get("pack_mmap_calls", 0)),
            }

    out = []
    for pair in by_pair.values():
        if pair["up"] is None or pair["down"] is None:
            continue
        up = pair["up"]
        down = pair["down"]
        fused_calls = min(up["calls"], down["calls"])
        out.append({
            "layer": pair["layer"],
            "expert": pair["expert"],
            "up": up,
            "down": down,
            "up_calls": up["calls"],
            "down_calls": down["calls"],
            "fused_calls": fused_calls,
            "hot_ms": up["hot_ms"] + down["hot_ms"],
            "up_hot_ms": up["hot_ms"],
            "down_hot_ms": down["hot_ms"],
            "up_touch_ms": up["touch_ms"],
            "down_touch_ms": down["touch_ms"],
            "up_expert_bytes": up["expert_bytes"],
            "down_expert_bytes": down["expert_bytes"],
        })
    out.sort(key=lambda r: r["hot_ms"], reverse=True)
    return out


def tensor_offset(reader: GGUFReader, tensor_name: str, expert: int) -> dict:
    tensors = {t.name: t for t in reader.tensors}
    tensor = tensors.get(tensor_name)
    if tensor is None:
        raise RuntimeError(f"missing tensor in model: {tensor_name}")
    shape = [int(x) for x in tensor.shape.tolist()]
    if len(shape) < 3:
        raise RuntimeError(f"unexpected tensor shape for {tensor_name}: {shape}")
    n_expert = int(shape[2])
    if expert < 0 or expert >= n_expert:
        raise RuntimeError(f"expert out of range for {tensor_name}: {expert}/{n_expert}")
    expert_bytes = int(tensor.n_bytes) // n_expert
    if expert_bytes * n_expert != int(tensor.n_bytes):
        raise RuntimeError(f"tensor nbytes not divisible by experts: {tensor_name}")
    return {
        "model_offset": int(tensor.data_offset) + expert * expert_bytes,
        "nbytes": expert_bytes,
        "shape": "x".join(str(x) for x in shape),
        "n_expert": n_expert,
    }


def write_tsv(path: Path, pairs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow([
            "rank", "layer", "expert", "hot_ms", "up_hot_ms", "down_hot_ms",
            "up_calls", "down_calls", "fused_calls", "up_tensor", "down_tensor",
            "up_nbytes", "down_nbytes",
        ])
        for i, p in enumerate(pairs, 1):
            w.writerow([
                i, p["layer"], p["expert"], f"{p['hot_ms']:.6f}",
                f"{p['up_hot_ms']:.6f}", f"{p['down_hot_ms']:.6f}",
                p["up_calls"], p["down_calls"], p["fused_calls"],
                p["up"]["tensor"], p["down"]["tensor"],
                p["up_expert_bytes"], p["down_expert_bytes"],
            ])


def write_manifest(path: Path, pairs: list[dict], reader: GGUFReader) -> list[dict]:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for rank, pair in enumerate(pairs, 1):
        for role in ["up", "down"]:
            item = pair[role]
            off = tensor_offset(reader, item["tensor"], pair["expert"])
            if off["nbytes"] != item["expert_bytes"]:
                raise RuntimeError(
                    f"expert byte mismatch {item['tensor']} expert={pair['expert']} "
                    f"profile={item['expert_bytes']} gguf={off['nbytes']}"
                )
            rows.append({
                "rank": rank,
                "pair_key": f"blk.{pair['layer']}.expert.{pair['expert']}",
                "layer": pair["layer"],
                "expert": pair["expert"],
                "role": role,
                "tensor": item["tensor"],
                "model_offset": off["model_offset"],
                "nbytes": off["nbytes"],
                "shape": off["shape"],
                "n_expert": off["n_expert"],
                "hot_ms": f"{item['hot_ms']:.6f}",
                "calls": item["calls"],
                "fused_calls": pair["fused_calls"],
            })
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "rank", "pair_key", "layer", "expert", "role", "tensor", "model_offset",
            "nbytes", "shape", "n_expert", "hot_ms", "calls", "fused_calls",
        ])
        w.writeheader()
        w.writerows(rows)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL))
    ap.add_argument("--fallback-profile", type=Path, default=Path(DEFAULT_TOUCH_PROFILE))
    ap.add_argument("--top-n", type=int, default=64)
    ap.add_argument("--tsv-out", type=Path, required=True)
    ap.add_argument("--manifest-out", type=Path, required=True)
    ap.add_argument("--profile-json-out", type=Path, required=True)
    ap.add_argument("--summary-out", type=Path, required=True)
    args = ap.parse_args()

    pairs_all = load_pairs(args.fallback_profile)
    pairs = pairs_all[:args.top_n]
    reader = GGUFReader(str(args.model), "r")

    write_tsv(args.tsv_out, pairs)
    manifest_rows = write_manifest(args.manifest_out, pairs, reader)

    total_payload_bytes = sum(int(r["nbytes"]) for r in manifest_rows)
    layer_counts: dict[int, int] = {}
    for pair in pairs:
        layer_counts[pair["layer"]] = layer_counts.get(pair["layer"], 0) + 1

    profile = {
        "format": "ds4_sparse_pair_profile_v1",
        "source": {
            "repo": str(args.repo),
            "branch": run_text(["git", "rev-parse", "--abbrev-ref", "HEAD"], args.repo),
            "head": run_text(["git", "rev-parse", "HEAD"], args.repo),
            "status": run_text(["git", "status", "--short"], args.repo),
        },
        "model": str(args.model),
        "fallback_profile": str(args.fallback_profile),
        "top_n": args.top_n,
        "payload_bytes": total_payload_bytes,
        "payload_mib": total_payload_bytes / (1024 * 1024),
        "active_layers": len(layer_counts),
        "max_pairs_per_layer": max(layer_counts.values()) if layer_counts else 0,
        "per_layer_pair_counts": dict(sorted(layer_counts.items())),
        "pairs": [
            {
                "rank": rank,
                "layer": p["layer"],
                "expert": p["expert"],
                "pair_key": f"blk.{p['layer']}.expert.{p['expert']}",
                "hot_ms": p["hot_ms"],
                "up_hot_ms": p["up_hot_ms"],
                "down_hot_ms": p["down_hot_ms"],
                "up_calls": p["up_calls"],
                "down_calls": p["down_calls"],
                "fused_calls": p["fused_calls"],
                "up_tensor": p["up"]["tensor"],
                "down_tensor": p["down"]["tensor"],
            }
            for rank, p in enumerate(pairs, 1)
        ],
    }
    args.profile_json_out.parent.mkdir(parents=True, exist_ok=True)
    args.profile_json_out.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = {
        "artifact": str(args.summary_out),
        "profile_json": str(args.profile_json_out),
        "tsv": str(args.tsv_out),
        "manifest": str(args.manifest_out),
        "top_n": args.top_n,
        "entries": len(manifest_rows),
        "pairs": len(pairs),
        "complete_pairs_available": len(pairs_all),
        "payload_bytes": total_payload_bytes,
        "payload_mib": total_payload_bytes / (1024 * 1024),
        "active_layers": len(layer_counts),
        "max_pairs_per_layer": max(layer_counts.values()) if layer_counts else 0,
        "per_layer_pair_counts": dict(sorted(layer_counts.items())),
        "hot_ms_sum": sum(p["hot_ms"] for p in pairs),
        "sha256": {},
    }
    for label, path in [
        ("profile_json", args.profile_json_out),
        ("tsv", args.tsv_out),
        ("manifest", args.manifest_out),
    ]:
        summary["sha256"][label] = sha256_path(path)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
