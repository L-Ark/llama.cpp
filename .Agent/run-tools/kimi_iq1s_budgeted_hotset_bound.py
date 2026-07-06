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
N_EXPERTS = 384
EXPERT_ELEMS = 7168 * 2048
QK_K = 256
TYPE_BYTES = {
    "IQ1_S": 2 + QK_K // 8 + QK_K // 16,
    "Q2_K": 2 + 2 + QK_K // 16 + QK_K // 4,
}


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


def iq1s_remote_type(layer: int, kind: str) -> str:
    # GP32 metadata: up/gate are IQ1_S for all MoE layers; down is Q2_K for
    # three layers and IQ1_S for the remaining 57 layers.
    if kind == "down" and 1 <= layer <= 3:
        return "Q2_K"
    return "IQ1_S"


def remote_expert_bytes(remote_type: str) -> int:
    return EXPERT_ELEMS // QK_K * TYPE_BYTES[remote_type]


def load_profiles(paths: list[Path], kinds: set[str]) -> tuple[dict[tuple[str, int], dict], dict[str, dict]]:
    candidates: dict[tuple[str, int], dict] = {}
    prompts: dict[str, dict] = {}
    for path in paths:
        prompt = path.parent.name
        pinfo = prompts.setdefault(prompt, {
            "events": 0,
            "current_bytes": 0,
        })
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                tensor = row["tensor"]
                kind = kind_from_tensor(tensor)
                if kind not in kinds:
                    continue
                layer = layer_from_tensor(tensor)
                if layer < 1:
                    continue
                expert_idx = int(row["expert_idx"])
                count = int(row["count"])
                current_nbytes = int(row["expert_bytes"])
                rtype = iq1s_remote_type(layer, kind)
                remote_nbytes = remote_expert_bytes(rtype)
                key = (tensor, expert_idx)
                current_bytes = count * current_nbytes
                remote_bytes = count * remote_nbytes
                pinfo["events"] += count
                pinfo["current_bytes"] += current_bytes
                item = candidates.setdefault(key, {
                    "tensor": tensor,
                    "expert_idx": expert_idx,
                    "layer": layer,
                    "kind": kind,
                    "current_nbytes": current_nbytes,
                    "remote_nbytes": remote_nbytes,
                    "remote_type": rtype,
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
    data_start = align_up(PACK_HEADER_BYTES + len(selected) * PACK_ENTRY_BYTES)
    offset = data_start
    payload = 0
    padding = data_start - (PACK_HEADER_BYTES + len(selected) * PACK_ENTRY_BYTES)
    for item in selected:
        aligned = align_up(offset)
        padding += aligned - offset
        offset = aligned + int(item["remote_nbytes"])
        payload += int(item["remote_nbytes"])
    return {
        "entries": len(selected),
        "data_start": data_start,
        "payload_bytes": payload,
        "padding_bytes": padding,
        "estimated_pack_bytes": offset,
    }


def select_greedy(candidates: dict[tuple[str, int], dict], budget_bytes: int) -> list[dict]:
    scored = []
    for item in candidates.values():
        benefit = int(item["current_bytes"]) - int(item["remote_bytes"])
        if benefit <= 0:
            continue
        storage_cost = align_up(int(item["remote_nbytes"])) + PACK_ENTRY_BYTES
        scored.append((benefit / storage_cost, benefit, storage_cost, item))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    selected = []
    used = align_up(PACK_HEADER_BYTES)
    for _score, _benefit, storage_cost, item in scored:
        if used + storage_cost > budget_bytes:
            continue
        selected.append(item)
        used += storage_cost
    while selected and pack_estimate(selected)["estimated_pack_bytes"] > budget_bytes:
        selected.pop()
    return selected


def summarize_selection(selected: list[dict], prompts: dict[str, dict], budget_gib: float | None) -> dict:
    agg_current = 0
    agg_hybrid = 0
    by_prompt = []
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
        by_prompt.append({
            "prompt": prompt,
            "events": int(pinfo["events"]),
            "current_bytes": current,
            "hybrid_bytes": hybrid,
            "hybrid_byte_ratio": hybrid / current if current else 0.0,
            "selected_current_bytes": selected_current,
            "selected_remote_bytes": selected_remote,
            "selected_byte_coverage": selected_current / current if current else 0.0,
        })
    by_kind = defaultdict(lambda: {"entries": 0, "current_nbytes": 0, "remote_nbytes": 0})
    by_type = defaultdict(lambda: {"entries": 0, "current_nbytes": 0, "remote_nbytes": 0})
    mismatch_count = 0
    for item in selected:
        if int(item["current_nbytes"]) != int(item["remote_nbytes"]):
            mismatch_count += 1
        for bucket in (by_kind[item["kind"]], by_type[item["remote_type"]]):
            bucket["entries"] += 1
            bucket["current_nbytes"] += int(item["current_nbytes"])
            bucket["remote_nbytes"] += int(item["remote_nbytes"])
    return {
        "budget_gib": budget_gib,
        "selected_entries": len(selected),
        "selected_tensors": len({item["tensor"] for item in selected}),
        "pack": pack_estimate(selected),
        "runtime_nbytes_mismatch_count": mismatch_count,
        "current_iq3_runtime_compatible": mismatch_count == 0,
        "aggregate": {
            "current_bytes": agg_current,
            "hybrid_bytes": agg_hybrid,
            "hybrid_byte_ratio": agg_hybrid / agg_current if agg_current else 0.0,
        },
        "by_prompt": by_prompt,
        "by_kind": dict(sorted(by_kind.items())),
        "by_remote_type": dict(sorted(by_type.items())),
    }


def write_selected(path: Path, selected: list[dict]) -> None:
    fields = [
        "layer", "kind", "tensor", "expert_idx", "current_nbytes",
        "remote_nbytes", "remote_type", "source_pack",
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
                "source_pack": "mradermacher-i1-IQ1_S-dev-route-budgeted",
            })


def main() -> int:
    parser = argparse.ArgumentParser(description="Dev-only mradermacher i1-IQ1_S selected-hotset byte bound.")
    parser.add_argument("--profile", type=Path, action="append", required=True)
    parser.add_argument("--kind", default="up,gate,down")
    parser.add_argument("--budget-gib", type=float, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    kinds = {x.strip() for x in args.kind.split(",") if x.strip()}
    invalid = kinds - {"up", "gate", "down"}
    if invalid:
        raise RuntimeError(f"invalid kinds: {sorted(invalid)}")

    candidates, prompts = load_profiles(args.profile, kinds)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for budget_gib in args.budget_gib:
        selected = select_greedy(candidates, int(budget_gib * 1024**3))
        summary = summarize_selection(selected, prompts, budget_gib)
        summary.update({
            "profiles": [str(p) for p in args.profile],
            "kinds": sorted(kinds),
            "candidate_keys": len(candidates),
            "remote_asset": "mradermacher/Kimi-K2.7-Code-i1-GGUF Kimi-K2.7-Code.i1-IQ1_S.gguf",
            "source_header_record": ".Agent/runs/20260707-gp32-iq1s-header-preflight/parsed-header-full.json",
            "note": "The selected IQ1_S bytes are a theoretical bound for byte movement. They are not directly runnable with the IQ3_S main GGUF because expert-pack lookup and kernels require current nbytes/src0_type.",
        })
        tag = str(budget_gib).replace(".", "p")
        json_path = args.out_dir / f"budget-{tag}.json"
        plan_path = args.out_dir / f"budget-{tag}-selected-plan.tsv"
        json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_selected(plan_path, selected)
        summary["json"] = str(json_path)
        summary["selected_plan_tsv"] = str(plan_path)
        reports.append(summary)

    all_selected = list(candidates.values())
    all_summary = summarize_selection(all_selected, prompts, None)
    all_summary.update({
        "profiles": [str(p) for p in args.profile],
        "kinds": sorted(kinds),
        "candidate_keys": len(candidates),
        "remote_asset": "mradermacher/Kimi-K2.7-Code-i1-GGUF Kimi-K2.7-Code.i1-IQ1_S.gguf",
        "note": "All dev-profile candidate keys selected; still a theoretical bound, not directly runnable with IQ3_S.",
    })
    (args.out_dir / "all-candidates.json").write_text(json.dumps(all_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Kimi mradermacher IQ1_S budgeted hotset bound",
        "",
        "- Scope: dev profiles only; held-out test prompts are not used.",
        "- Asset: `mradermacher/Kimi-K2.7-Code-i1-GGUF` `i1-IQ1_S`.",
        "- Runtime status: theoretical byte bound only, not directly runnable with the IQ3_S main GGUF.",
        "",
        "| budget GiB | selected entries | pack GiB | hybrid byte ratio | current GiB | hybrid GiB | nbytes compatible |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for report in reports:
        lines.append(
            f"| {report['budget_gib']:.3f} | {report['selected_entries']} | "
            f"{report['pack']['estimated_pack_bytes'] / 1024**3:.3f} | "
            f"{report['aggregate']['hybrid_byte_ratio']:.4f} | "
            f"{report['aggregate']['current_bytes'] / 1024**3:.3f} | "
            f"{report['aggregate']['hybrid_bytes'] / 1024**3:.3f} | "
            f"{report['current_iq3_runtime_compatible']} |"
        )
    lines.extend([
        "",
        "## All Dev Candidate Keys",
        "",
        (
            f"- Entries: `{all_summary['selected_entries']}`; pack: "
            f"`{all_summary['pack']['estimated_pack_bytes'] / 1024**3:.3f} GiB`; "
            f"hybrid byte ratio: `{all_summary['aggregate']['hybrid_byte_ratio']:.4f}`."
        ),
        f"- Runtime nbytes mismatch count: `{all_summary['runtime_nbytes_mismatch_count']}`.",
        "",
        "## Worst Prompt Ratios",
        "",
    ])
    for report in reports:
        worst = max(report["by_prompt"], key=lambda r: r["hybrid_byte_ratio"])
        best = min(report["by_prompt"], key=lambda r: r["hybrid_byte_ratio"])
        lines.append(
            f"- `{report['budget_gib']:.3f} GiB`: worst `{worst['prompt']}` "
            f"`{worst['hybrid_byte_ratio']:.4f}`, best `{best['prompt']}` "
            f"`{best['hybrid_byte_ratio']:.4f}`."
        )
    lines.extend([
        "",
        "## Decision",
        "",
        "- A selected IQ1_S hotset can fit in the current free-disk envelope for small budgets.",
        "- It cannot be used directly by the current IQ3_S runtime because pack lookup keys include `nbytes`, and CUDA dispatch uses the main GGUF `src0_type`.",
        "- Therefore this is useful as a byte-reduction bound, but not as a runtime experiment unless mixed-type expert override kernels are implemented or the full IQ1_S model is available.",
    ])
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for report in reports:
        print(
            f"budget_gib={report['budget_gib']:.3f} "
            f"selected={report['selected_entries']} "
            f"pack_gib={report['pack']['estimated_pack_bytes'] / 1024**3:.3f} "
            f"hybrid_ratio={report['aggregate']['hybrid_byte_ratio']:.6f} "
            f"compatible={int(report['current_iq3_runtime_compatible'])}"
        )
    print(
        f"all_candidates={all_summary['selected_entries']} "
        f"all_pack_gib={all_summary['pack']['estimated_pack_bytes'] / 1024**3:.3f} "
        f"all_hybrid_ratio={all_summary['aggregate']['hybrid_byte_ratio']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
