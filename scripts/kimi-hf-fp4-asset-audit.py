#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any


SMALL_JSON_FILES = (
    "config.json",
    "hf_quant_config.json",
    "model.safetensors.index.json",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def to_gib(n: int | None) -> float | None:
    if n is None:
        return None
    return n / (1024.0 ** 3)


def sibling_name(sibling: Any) -> str:
    return getattr(sibling, "rfilename", None) or getattr(sibling, "path", "")


def sibling_size(sibling: Any) -> int | None:
    size = getattr(sibling, "size", None)
    return int(size) if size is not None else None


def head_size(repo_id: str, filename: str, revision: str, timeout: float = 30.0) -> int | None:
    from huggingface_hub import hf_hub_url

    url = hf_hub_url(repo_id=repo_id, filename=filename, revision=revision, repo_type="model")
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = response.headers.get("content-length")
            return int(value) if value and value.isdigit() else None
    except Exception:
        return None


def load_small_json(repo_id: str, revision: str, filename: str, max_bytes: int) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    info: dict[str, Any] = {"filename": filename, "present": False}
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            repo_type="model",
        )
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
        return None, info

    size = os.path.getsize(path)
    info.update({"present": True, "cache_path": path, "size": size})
    if size > max_bytes:
        info["error"] = f"small JSON limit exceeded: {size} > {max_bytes}"
        return None, info

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f), info


def sample_matching(names: list[str], pattern: str, limit: int = 20) -> list[str]:
    rx = re.compile(pattern)
    return [name for name in names if rx.search(name)][:limit]


def tensor_name_summary(weight_map: dict[str, str]) -> dict[str, Any]:
    names = sorted(weight_map)
    expert_names = [name for name in names if "expert" in name or ".ffn." in name or ".mlp." in name]
    scale_names = [name for name in names if "scale" in name.lower()]

    patterns = {
        "w1": r"(^|[.])w1([.]|$)",
        "w2": r"(^|[.])w2([.]|$)",
        "w3": r"(^|[.])w3([.]|$)",
        "up_proj": r"up_proj",
        "gate_proj": r"gate_proj",
        "down_proj": r"down_proj",
        "ffn_experts": r"[.]ffn[.]experts[.]",
        "mlp_experts": r"[.]mlp[.]experts[.]",
        "weight_scale": r"(weight_)?scale",
    }

    counts = {}
    for key, pattern in patterns.items():
        rx = re.compile(pattern)
        counts[key] = sum(1 for name in names if rx.search(name))

    return {
        "tensor_count": len(names),
        "unique_shards_in_weight_map": len(set(weight_map.values())),
        "pattern_counts": counts,
        "samples": {
            "expert": expert_names[:30],
            "scale": scale_names[:30],
            "w1": sample_matching(names, patterns["w1"]),
            "w2": sample_matching(names, patterns["w2"]),
            "w3": sample_matching(names, patterns["w3"]),
            "up_proj": sample_matching(names, patterns["up_proj"]),
            "gate_proj": sample_matching(names, patterns["gate_proj"]),
            "down_proj": sample_matching(names, patterns["down_proj"]),
        },
    }


def summarize_safetensors(repo_id: str, revision: str, siblings: list[Any]) -> dict[str, Any]:
    files = []
    missing_sizes: list[str] = []

    for sibling in siblings:
        name = sibling_name(sibling)
        if not name.endswith(".safetensors"):
            continue
        size = sibling_size(sibling)
        if size is None:
            size = head_size(repo_id, name, revision)
        if size is None:
            missing_sizes.append(name)
        files.append({"name": name, "size": size, "size_gib": to_gib(size)})

    known_sizes = [entry["size"] for entry in files if entry["size"] is not None]
    total = int(sum(known_sizes))
    largest = max(files, key=lambda entry: entry["size"] or -1, default=None)

    return {
        "count": len(files),
        "known_size_count": len(known_sizes),
        "missing_size_count": len(missing_sizes),
        "missing_size_files": missing_sizes[:50],
        "total_bytes_known": total,
        "total_gib_known": to_gib(total),
        "largest_file": largest,
        "files_head": files[:10],
        "files_tail": files[-10:],
    }


def audit_repo(repo_id: str, revision: str, max_json_bytes: int) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.model_info(repo_id, revision=revision, files_metadata=True)
    siblings = list(getattr(info, "siblings", []) or [])

    small_json: dict[str, Any] = {}
    small_json_info: dict[str, Any] = {}
    for filename in SMALL_JSON_FILES:
        data, file_info = load_small_json(repo_id, revision, filename, max_json_bytes)
        small_json_info[filename] = file_info
        if data is not None:
            small_json[filename] = data

    config = small_json.get("config.json") or {}
    hf_quant_config = small_json.get("hf_quant_config.json")
    index = small_json.get("model.safetensors.index.json") or {}
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict):
        weight_map = {}

    safetensors = summarize_safetensors(repo_id, revision, siblings)
    shard_names = set(weight_map.values())
    shard_size_by_name = {
        entry["name"]: entry["size"]
        for entry in safetensors["files_head"] + safetensors["files_tail"]
        if entry.get("size") is not None
    }
    # Preserve all sibling sizes for exact weight-map coverage without storing every file twice.
    for sibling in siblings:
        name = sibling_name(sibling)
        if name in shard_names:
            size = sibling_size(sibling)
            if size is not None:
                shard_size_by_name[name] = size

    weight_map_known_total = sum(size for name, size in shard_size_by_name.items() if name in shard_names and size is not None)

    quant_config = config.get("quantization_config")
    if not isinstance(quant_config, dict):
        quant_config = {}

    return {
        "repo_id": repo_id,
        "requested_revision": revision,
        "resolved_sha": getattr(info, "sha", None),
        "private": getattr(info, "private", None),
        "tags": list(getattr(info, "tags", []) or []),
        "siblings_count": len(siblings),
        "safetensors": safetensors,
        "small_json": small_json_info,
        "config_summary": {
            "architectures": config.get("architectures"),
            "model_type": config.get("model_type"),
            "torch_dtype": config.get("torch_dtype"),
            "quantization_config": quant_config,
        },
        "hf_quant_config_summary": {
            "present": hf_quant_config is not None,
            "keys": sorted(hf_quant_config.keys()) if isinstance(hf_quant_config, dict) else [],
            "producer": hf_quant_config.get("producer") if isinstance(hf_quant_config, dict) else None,
            "quantization": hf_quant_config.get("quantization") if isinstance(hf_quant_config, dict) else None,
        },
        "index_summary": {
            "metadata": index.get("metadata") if isinstance(index, dict) else None,
            "tensor_count": len(weight_map),
            "unique_shard_count": len(shard_names),
            "known_weight_map_shard_bytes": weight_map_known_total,
            "known_weight_map_shard_gib": to_gib(weight_map_known_total),
        },
        "tensor_names": tensor_name_summary(weight_map),
    }


def add_decisions(result: dict[str, Any], disk_free_gib: float) -> None:
    for repo in result["repos"]:
        total_gib = repo["safetensors"]["total_gib_known"]
        quant_method = repo["config_summary"]["quantization_config"].get("quant_method")
        fits_with_margin = total_gib is not None and total_gib < disk_free_gib * 0.8
        name_counts = repo["tensor_names"]["pattern_counts"]
        likely_deepseek_v4_mxfp4 = (
            quant_method in {"quark", "fp8", "mxfp4"}
            and name_counts.get("w1", 0) > 0
            and name_counts.get("w2", 0) > 0
            and name_counts.get("w3", 0) > 0
        )
        likely_modelopt_nvfp4 = (
            quant_method == "modelopt"
            or (repo["hf_quant_config_summary"]["quantization"] or {}).get("quant_algo") == "NVFP4"
        )
        repo["decision_inputs"] = {
            "disk_free_gib_assumed": disk_free_gib,
            "fits_80_percent_disk_margin": fits_with_margin,
            "quant_method": quant_method,
            "likely_deepseek_v4_mxfp4_names": likely_deepseek_v4_mxfp4,
            "likely_modelopt_nvfp4": likely_modelopt_nvfp4,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Metadata-only audit for Kimi FP4 Hugging Face assets.")
    parser.add_argument("repos", nargs="+", help="Hugging Face model repo IDs")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-json-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--disk-free-gib", type=float, default=88.0)
    args = parser.parse_args()

    result: dict[str, Any] = {
        "generated_at": utc_now(),
        "revision": args.revision,
        "max_json_bytes": args.max_json_bytes,
        "repos": [],
    }

    for repo_id in args.repos:
        result["repos"].append(audit_repo(repo_id, args.revision, args.max_json_bytes))

    add_decisions(result, args.disk_free_gib)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for repo in result["repos"]:
        total_gib = repo["safetensors"]["total_gib_known"]
        quant_method = repo["decision_inputs"]["quant_method"]
        print(
            f"{repo['repo_id']}: sha={repo['resolved_sha']} quant_method={quant_method} "
            f"safetensors={repo['safetensors']['count']} total_gib={total_gib:.2f}"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
