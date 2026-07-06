#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


ALIGNMENT = 4096
PACK_HEADER_BYTES = 40
PACK_ENTRY_BYTES = 152


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def kind_from_tensor(name: str) -> str:
    if ".ffn_up_exps." in name:
        return "up"
    if ".ffn_gate_exps." in name:
        return "gate"
    if ".ffn_down_exps." in name:
        return "down"
    return "other"


def layer_from_tensor(name: str) -> int:
    parts = name.split(".")
    if len(parts) >= 2 and parts[0] == "blk":
        return int(parts[1])
    return -1


def load_remote_tensor_meta(plan_tsv: Path) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    with plan_tsv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            tensor = row["tensor"]
            if tensor in meta:
                continue
            meta[tensor] = {
                "layer": int(row["layer"]),
                "kind": row["kind"],
                "remote_nbytes": int(row["remote_nbytes"]),
                "remote_type": row["remote_type"],
                "remote_shard_name": row["remote_shard_name"],
                "remote_shard_size": int(row["remote_shard_size"]),
                "remote_tensor_offset": int(row["remote_tensor_offset"]),
            }
    return meta


def load_profiles(paths: list[Path], remote_meta: dict[str, dict], kinds: set[str]) -> tuple[dict, dict]:
    candidates: dict[tuple[str, int], dict] = {}
    prompts: dict[str, dict] = {}

    for path in paths:
        prompt = path.parent.name
        prompt_bucket = prompts.setdefault(prompt, {
            "current_bytes": 0,
            "by_key_current": defaultdict(int),
            "events": 0,
        })
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                tensor = row["tensor"]
                kind = kind_from_tensor(tensor)
                if kind not in kinds:
                    continue
                if tensor not in remote_meta:
                    continue
                expert = int(row["expert_idx"])
                count = int(row["count"])
                current_nbytes = int(row["expert_bytes"])
                remote_nbytes = int(remote_meta[tensor]["remote_nbytes"])
                current_bytes = count * current_nbytes
                remote_bytes = count * remote_nbytes
                key = (tensor, expert)
                prompt_bucket["current_bytes"] += current_bytes
                prompt_bucket["by_key_current"][key] += current_bytes
                prompt_bucket["events"] += count
                item = candidates.setdefault(key, {
                    "tensor": tensor,
                    "expert_idx": expert,
                    "kind": kind,
                    "layer": layer_from_tensor(tensor),
                    "current_nbytes": current_nbytes,
                    "remote_nbytes": remote_nbytes,
                    "remote_type": remote_meta[tensor]["remote_type"],
                    "remote_shard_name": remote_meta[tensor]["remote_shard_name"],
                    "remote_shard_size": remote_meta[tensor]["remote_shard_size"],
                    "remote_tensor_offset": remote_meta[tensor]["remote_tensor_offset"],
                    "events": 0,
                    "current_bytes": 0,
                    "remote_bytes": 0,
                    "per_prompt_current": defaultdict(int),
                    "per_prompt_remote": defaultdict(int),
                })
                item["events"] += count
                item["current_bytes"] += current_bytes
                item["remote_bytes"] += remote_bytes
                item["per_prompt_current"][prompt] += current_bytes
                item["per_prompt_remote"][prompt] += remote_bytes

    return candidates, prompts


def pack_estimate(selected: list[dict]) -> dict:
    index_bytes = len(selected) * PACK_ENTRY_BYTES
    data_start = align_up(PACK_HEADER_BYTES + index_bytes)
    offset = data_start
    payload = 0
    padding = data_start - (PACK_HEADER_BYTES + index_bytes)
    for item in selected:
        aligned = align_up(offset)
        padding += aligned - offset
        offset = aligned + int(item["remote_nbytes"])
        payload += int(item["remote_nbytes"])
    return {
        "entries": len(selected),
        "index_bytes": index_bytes,
        "data_start": data_start,
        "payload_bytes": payload,
        "padding_bytes": padding,
        "estimated_pack_bytes": offset,
    }


def select_greedy(candidates: dict, budget_bytes: int) -> list[dict]:
    scored = []
    for item in candidates.values():
        benefit = int(item["current_bytes"]) - int(item["remote_bytes"])
        if benefit <= 0:
            continue
        # Approximate storage cost used for selection. Exact pack size is recomputed after selection.
        storage_cost = align_up(int(item["remote_nbytes"])) + PACK_ENTRY_BYTES
        scored.append((benefit / storage_cost, benefit, storage_cost, item))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    selected: list[dict] = []
    used = align_up(PACK_HEADER_BYTES)
    for _score, _benefit, storage_cost, item in scored:
        if used + storage_cost > budget_bytes:
            continue
        selected.append(item)
        used += storage_cost

    # Trim if exact pack layout exceeds budget.
    while selected and pack_estimate(selected)["estimated_pack_bytes"] > budget_bytes:
        selected.pop()
    return selected


def summarize(selected: list[dict], prompts: dict, kinds: set[str]) -> dict:
    selected_keys = {(item["tensor"], int(item["expert_idx"])) for item in selected}
    prompt_rows = []
    agg_current = 0
    agg_hybrid = 0
    agg_events = 0
    for prompt, pinfo in sorted(prompts.items()):
        current = int(pinfo["current_bytes"])
        hybrid = current
        selected_current = 0
        selected_remote = 0
        for item in selected:
            cur = int(item["per_prompt_current"].get(prompt, 0))
            if cur == 0:
                continue
            rem = int(item["per_prompt_remote"].get(prompt, 0))
            selected_current += cur
            selected_remote += rem
            hybrid += rem - cur
        agg_current += current
        agg_hybrid += hybrid
        agg_events += int(pinfo["events"])
        prompt_rows.append({
            "prompt": prompt,
            "events": int(pinfo["events"]),
            "current_bytes": current,
            "hybrid_bytes": hybrid,
            "hybrid_byte_ratio": hybrid / current if current else 0.0,
            "selected_current_bytes": selected_current,
            "selected_remote_bytes": selected_remote,
            "selected_byte_coverage": selected_current / current if current else 0.0,
        })
    by_kind = defaultdict(lambda: {"entries": 0, "current_bytes": 0, "remote_bytes": 0})
    for item in selected:
        b = by_kind[item["kind"]]
        b["entries"] += 1
        b["current_bytes"] += int(item["current_nbytes"])
        b["remote_bytes"] += int(item["remote_nbytes"])
    return {
        "kinds": sorted(kinds),
        "selected_entries": len(selected),
        "selected_keys": len(selected_keys),
        "pack": pack_estimate(selected),
        "aggregate": {
            "events": agg_events,
            "current_bytes": agg_current,
            "hybrid_bytes": agg_hybrid,
            "hybrid_byte_ratio": agg_hybrid / agg_current if agg_current else 0.0,
        },
        "by_prompt": prompt_rows,
        "by_kind": dict(sorted(by_kind.items())),
    }


def write_selected_plan(path: Path, selected: list[dict]) -> None:
    fields = [
        "layer", "kind", "tensor", "expert_idx", "current_nbytes", "remote_nbytes",
        "remote_type", "remote_shard_name", "remote_shard_size", "remote_tensor_offset",
        "source_pack",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for item in sorted(selected, key=lambda x: (x["layer"], x["kind"], x["tensor"], x["expert_idx"])):
            writer.writerow({
                "layer": item["layer"],
                "kind": item["kind"],
                "tensor": item["tensor"],
                "expert_idx": item["expert_idx"],
                "current_nbytes": item["current_nbytes"],
                "remote_nbytes": item["remote_nbytes"],
                "remote_type": item["remote_type"],
                "remote_shard_name": item["remote_shard_name"],
                "remote_shard_size": item["remote_shard_size"],
                "remote_tensor_offset": item["remote_tensor_offset"],
                "source_pack": "dev-route-budgeted",
            })


def main() -> int:
    parser = argparse.ArgumentParser(description="Dev-only budgeted lower-byte Kimi hotset bound.")
    parser.add_argument("--remote-plan-tsv", type=Path, required=True)
    parser.add_argument("--profile", type=Path, action="append", required=True)
    parser.add_argument("--kind", default="up,gate,down")
    parser.add_argument("--budget-gib", type=float, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    kinds = {x.strip() for x in args.kind.split(",") if x.strip()}
    remote_meta = load_remote_tensor_meta(args.remote_plan_tsv)
    candidates, prompts = load_profiles(args.profile, remote_meta, kinds)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for budget_gib in args.budget_gib:
        budget_bytes = int(budget_gib * 1024**3)
        selected = select_greedy(candidates, budget_bytes)
        summary = summarize(selected, prompts, kinds)
        summary.update({
            "budget_gib": budget_gib,
            "budget_bytes": budget_bytes,
            "profiles": [str(p) for p in args.profile],
            "candidate_keys": len(candidates),
            "remote_plan_tsv": str(args.remote_plan_tsv),
        })
        tag = str(budget_gib).replace(".", "p")
        json_path = args.out_dir / f"budget-{tag}.json"
        plan_path = args.out_dir / f"budget-{tag}-selected-plan.tsv"
        json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_selected_plan(plan_path, selected)
        summary["json"] = str(json_path)
        summary["selected_plan_tsv"] = str(plan_path)
        reports.append(summary)

    md = [
        "# Kimi budgeted lower-byte hotset bound",
        "",
        f"- Remote plan: `{args.remote_plan_tsv}`",
        f"- Kinds: `{','.join(sorted(kinds))}`",
        f"- Profiles: `{len(args.profile)}`",
        f"- Candidate keys: `{len(candidates)}`",
        "",
        "| budget GiB | selected entries | pack GiB | hybrid byte ratio | current GiB | hybrid GiB |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for report in reports:
        md.append(
            f"| {report['budget_gib']:.3f} | {report['selected_entries']} | "
            f"{report['pack']['estimated_pack_bytes'] / 1024**3:.3f} | "
            f"{report['aggregate']['hybrid_byte_ratio']:.4f} | "
            f"{report['aggregate']['current_bytes'] / 1024**3:.3f} | "
            f"{report['aggregate']['hybrid_bytes'] / 1024**3:.3f} |"
        )
    md.extend(["", "## Worst Prompt Ratios", ""])
    for report in reports:
        worst = max(report["by_prompt"], key=lambda r: r["hybrid_byte_ratio"])
        best = min(report["by_prompt"], key=lambda r: r["hybrid_byte_ratio"])
        md.append(
            f"- `{report['budget_gib']:.3f} GiB`: worst `{worst['prompt']}` "
            f"`{worst['hybrid_byte_ratio']:.4f}`, best `{best['prompt']}` "
            f"`{best['hybrid_byte_ratio']:.4f}`."
        )
    (args.out_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    for report in reports:
        print(
            f"budget_gib={report['budget_gib']:.3f} "
            f"selected={report['selected_entries']} "
            f"pack_gib={report['pack']['estimated_pack_bytes'] / 1024**3:.3f} "
            f"hybrid_ratio={report['aggregate']['hybrid_byte_ratio']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
