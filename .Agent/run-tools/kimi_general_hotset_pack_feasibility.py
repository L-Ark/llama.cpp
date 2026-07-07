#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import pathlib
import re
import struct
from collections import defaultdict


PACK_MAGIC = b"GGMLMOEPACKv1\0\0\0"
PACK_HEADER = struct.Struct("<16sIIQQ")
PACK_ENTRY = struct.Struct("<128siIQQ")
ALIGNMENT = 4096
ROLE_RE = re.compile(r"ffn_(up|gate|down)_exps")
LAYER_RE = re.compile(r"blk\.(\d+)\.")


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def gib(value: int | float) -> float:
    return float(value) / 1024**3


def pct(num: int | float, den: int | float) -> float:
    return 100.0 * float(num) / float(den) if den else 0.0


def read_int(row: dict[str, str], key: str) -> int:
    try:
        return int(row.get(key, "0") or 0)
    except ValueError:
        return 0


def role_of(tensor: str) -> str:
    match = ROLE_RE.search(tensor)
    return match.group(1) if match else "other"


def layer_of(tensor: str) -> int:
    match = LAYER_RE.search(tensor)
    return int(match.group(1)) if match else -1


def parse_csv_ints(value: str) -> list[int]:
    out = []
    for item in value.split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    return out


def parse_csv_floats(value: str) -> list[float]:
    out = []
    for item in value.split(","):
        item = item.strip()
        if item:
            out.append(float(item))
    return out


def pack_size_bytes(keys: list[tuple[str, int, int]]) -> int:
    data_start = align_up(PACK_HEADER.size + len(keys) * PACK_ENTRY.size)
    offset = data_start
    for _tensor, _expert_idx, nbytes in sorted(keys):
        offset = align_up(offset)
        offset += nbytes
    return offset


def load_pack_keys(pack_paths: list[pathlib.Path], replace_duplicates: bool) -> tuple[set[tuple[str, int, int]], list[dict[str, object]]]:
    sources = []
    key_source: dict[tuple[str, int, int], int] = {}
    duplicate_count = 0

    for source_idx, path in enumerate(pack_paths):
        if not path.exists():
            raise RuntimeError(f"pack does not exist: {path}")
        entries = 0
        new_unique = 0
        duplicates = 0
        payload_bytes = 0
        with path.open("rb") as f:
            header = f.read(PACK_HEADER.size)
            if len(header) != PACK_HEADER.size:
                raise RuntimeError(f"{path}: short pack header")
            magic, version, header_size, n_entries, data_start = PACK_HEADER.unpack(header)
            if magic != PACK_MAGIC or version != 1:
                raise RuntimeError(f"{path}: invalid pack header")
            if header_size < PACK_HEADER.size:
                raise RuntimeError(f"{path}: invalid header_size={header_size}")
            if header_size > PACK_HEADER.size:
                f.seek(header_size)
            for _ in range(n_entries):
                raw = f.read(PACK_ENTRY.size)
                if len(raw) != PACK_ENTRY.size:
                    raise RuntimeError(f"{path}: short pack index")
                name_raw, expert_idx, _reserved, _offset, nbytes = PACK_ENTRY.unpack(raw)
                tensor = name_raw.split(b"\0", 1)[0].decode("utf-8", errors="replace")
                key = (tensor, int(expert_idx), int(nbytes))
                entries += 1
                if key in key_source:
                    duplicates += 1
                    duplicate_count += 1
                    if replace_duplicates and source_idx >= key_source[key]:
                        key_source[key] = source_idx
                else:
                    key_source[key] = source_idx
                    new_unique += 1
                    payload_bytes += int(nbytes)
        sources.append({
            "path": str(path),
            "entries": entries,
            "new_unique_keys": new_unique,
            "duplicates": duplicates,
            "payload_gib_new_unique": gib(payload_bytes),
            "data_start": int(data_start),
        })

    if duplicate_count and not replace_duplicates:
        raise RuntimeError(
            f"duplicate pack keys found: {duplicate_count}; use --replace-duplicates "
            "to mirror GGML_MOE_EXPERT_PACK_REPLACE_DUPLICATES=1"
        )
    return set(key_source), sources


def iter_route_profiles(route_root: pathlib.Path):
    if route_root.is_file():
        paths = [route_root]
    else:
        paths = sorted(route_root.glob("*/route-profile.csv"))
    if not paths:
        raise RuntimeError(f"no route-profile.csv found under {route_root}")
    for path in paths:
        prompt = path.parent.name
        with path.open(newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                tensor = row.get("tensor", "")
                expert_idx = read_int(row, "expert_idx")
                expert_bytes = read_int(row, "expert_bytes")
                count = read_int(row, "count")
                if not tensor or expert_bytes <= 0 or count <= 0:
                    continue
                yield {
                    "prompt": prompt,
                    "tensor": tensor,
                    "expert_idx": expert_idx,
                    "expert_bytes": expert_bytes,
                    "count": count,
                    "traffic_bytes": count * expert_bytes,
                    "key": (tensor, expert_idx, expert_bytes),
                    "role": role_of(tensor),
                    "layer": layer_of(tensor),
                }


def summarize_routes(route_root: pathlib.Path, existing_keys: set[tuple[str, int, int]]):
    by_key = {}
    prompt_totals = defaultdict(lambda: defaultdict(int))
    role_totals = defaultdict(lambda: defaultdict(int))
    layer_role_totals = defaultdict(lambda: defaultdict(int))

    for item in iter_route_profiles(route_root):
        key = item["key"]
        hit = key in existing_keys
        rec = by_key.setdefault(key, {
            "tensor": item["tensor"],
            "expert_idx": item["expert_idx"],
            "expert_bytes": item["expert_bytes"],
            "role": item["role"],
            "layer": item["layer"],
            "count": 0,
            "traffic_bytes": 0,
            "prompt_traffic": defaultdict(int),
            "prompt_count": defaultdict(int),
            "existing_hit": hit,
        })
        if rec["expert_bytes"] != item["expert_bytes"]:
            raise RuntimeError(f"conflicting bytes for key {key}")
        rec["count"] += item["count"]
        rec["traffic_bytes"] += item["traffic_bytes"]
        rec["prompt_traffic"][item["prompt"]] += item["traffic_bytes"]
        rec["prompt_count"][item["prompt"]] += item["count"]

        prefix = "hit" if hit else "miss"
        prompt_totals[item["prompt"]][f"{prefix}_traffic_bytes"] += item["traffic_bytes"]
        prompt_totals[item["prompt"]][f"{prefix}_events"] += item["count"]
        prompt_totals[item["prompt"]][f"{prefix}_unique_bytes"] += item["expert_bytes"]
        role_totals[item["role"]][f"{prefix}_traffic_bytes"] += item["traffic_bytes"]
        role_totals[item["role"]][f"{prefix}_events"] += item["count"]
        layer_role_totals[(item["layer"], item["role"])][f"{prefix}_traffic_bytes"] += item["traffic_bytes"]
        layer_role_totals[(item["layer"], item["role"])][f"{prefix}_events"] += item["count"]

    return by_key, prompt_totals, role_totals, layer_role_totals


def candidate_stats(name: str, selected: list[dict[str, object]], missing_records: list[dict[str, object]], prompt_totals) -> dict[str, object]:
    selected_keys = {rec["key"] for rec in selected}
    selected_traffic = sum(int(rec["traffic_bytes"]) for rec in selected)
    total_miss_traffic = sum(int(rec["traffic_bytes"]) for rec in missing_records)
    size = pack_size_bytes([rec["key"] for rec in selected])
    per_prompt_added = defaultdict(int)
    for rec in selected:
        for prompt, value in rec["prompt_traffic"].items():
            per_prompt_added[prompt] += value
    worst_prompt_added = 100.0
    for prompt, totals in prompt_totals.items():
        miss = totals.get("miss_traffic_bytes", 0)
        if miss:
            worst_prompt_added = min(worst_prompt_added, pct(per_prompt_added[prompt], miss))
    if not selected:
        worst_prompt_added = 0.0
    return {
        "name": name,
        "entries": len(selected),
        "pack_size_bytes": size,
        "unique_payload_bytes": sum(int(rec["expert_bytes"]) for rec in selected),
        "added_traffic_bytes": selected_traffic,
        "added_miss_traffic_pct": pct(selected_traffic, total_miss_traffic),
        "remaining_miss_traffic_bytes": total_miss_traffic - selected_traffic,
        "worst_prompt_added_miss_pct": worst_prompt_added,
        "selected_keys": selected_keys,
    }


def select_by_budget(missing_records: list[dict[str, object]], budget_bytes: int) -> list[dict[str, object]]:
    selected = []
    approx_size = align_up(PACK_HEADER.size)
    for rec in missing_records:
        # Conservative one-pass estimate. candidate_stats() recomputes exact v1
        # size after selection.
        entry_cost = PACK_ENTRY.size + align_up(int(rec["expert_bytes"]))
        if approx_size + entry_cost > budget_bytes:
            continue
        selected.append(rec)
        approx_size += entry_cost
    return selected


def write_tsv(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "rank", "tensor", "expert_idx", "expert_bytes", "role", "layer",
        "count", "traffic_gib", "entry",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for rank, rec in enumerate(rows, 1):
            writer.writerow({
                "rank": rank,
                "tensor": rec["tensor"],
                "expert_idx": rec["expert_idx"],
                "expert_bytes": rec["expert_bytes"],
                "role": rec["role"],
                "layer": rec["layer"],
                "count": rec["count"],
                "traffic_gib": f"{gib(rec['traffic_bytes']):.6f}",
                "entry": f"{rec['tensor']}:{rec['expert_idx']}",
            })


def write_entries(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    lines = [f"{rec['tensor']}:{rec['expert_idx']}" for rec in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_report(
    out_dir: pathlib.Path,
    route_root: pathlib.Path,
    pack_summaries: list[dict[str, object]],
    prompt_totals,
    role_totals,
    layer_role_totals,
    missing_records: list[dict[str, object]],
    candidates: list[dict[str, object]],
    assumed_direct_s_per_gib: float,
    top_rows: int,
) -> None:
    total_hit = sum(v.get("hit_traffic_bytes", 0) for v in prompt_totals.values())
    total_miss = sum(v.get("miss_traffic_bytes", 0) for v in prompt_totals.values())
    total = total_hit + total_miss

    lines = [
        "# GP54 prompt-agnostic hotset pack feasibility",
        "",
        "This is a dev-only simulation. It uses `route-profile.csv` files and",
        "expert-pack indexes; it does not run inference and does not inspect",
        "held-out test prompts.",
        "",
        "## Inputs",
        "",
        f"- route root: `{route_root}`",
        "- route traffic model: `count * expert_bytes` per `route-profile.csv` row",
        f"- assumed direct-copy cost: `{assumed_direct_s_per_gib:.2f} s/GiB`",
        "",
        "## Existing Pack Sources",
        "",
        "| path | entries | new unique keys | duplicates | new unique payload GiB |",
        "|---|---:|---:|---:|---:|",
    ]
    if pack_summaries:
        for item in pack_summaries:
            lines.append(
                f"| `{item['path']}` | {item['entries']} | {item['new_unique_keys']} | "
                f"{item['duplicates']} | {item['payload_gib_new_unique']:.2f} |"
            )
    else:
        lines.append("| none | 0 | 0 | 0 | 0.00 |")

    lines.extend([
        "",
        "## Current Dev Coverage",
        "",
        f"- routed traffic: `{gib(total):.2f} GiB`",
        f"- existing pack-hit traffic: `{gib(total_hit):.2f} GiB`",
        f"- existing pack-miss traffic: `{gib(total_miss):.2f} GiB`",
        f"- existing pack-hit traffic ratio: `{pct(total_hit, total):.1f}%`",
        f"- unique missing keys: `{len(missing_records)}`",
        "",
        "## Prompt Coverage",
        "",
        "| prompt | traffic GiB | hit GiB | hit traffic | miss GiB |",
        "|---|---:|---:|---:|---:|",
    ])
    for prompt, item in sorted(prompt_totals.items()):
        hit = item.get("hit_traffic_bytes", 0)
        miss = item.get("miss_traffic_bytes", 0)
        lines.append(
            f"| `{prompt}` | {gib(hit + miss):.2f} | {gib(hit):.2f} | "
            f"{pct(hit, hit + miss):.1f}% | {gib(miss):.2f} |"
        )

    lines.extend([
        "",
        "## Role Misses",
        "",
        "| role | traffic GiB | hit traffic | miss GiB |",
        "|---|---:|---:|---:|",
    ])
    for role, item in sorted(role_totals.items()):
        hit = item.get("hit_traffic_bytes", 0)
        miss = item.get("miss_traffic_bytes", 0)
        lines.append(f"| {role} | {gib(hit + miss):.2f} | {pct(hit, hit + miss):.1f}% | {gib(miss):.2f} |")

    lines.extend([
        "",
        "## Top Missing Layer/Role Buckets",
        "",
        "| layer | role | miss GiB | miss events |",
        "|---:|---|---:|---:|",
    ])
    ranked_layer_role = sorted(
        layer_role_totals.items(),
        key=lambda item: item[1].get("miss_traffic_bytes", 0),
        reverse=True,
    )
    for (layer, role), item in ranked_layer_role[:40]:
        lines.append(
            f"| {layer} | {role} | {gib(item.get('miss_traffic_bytes', 0)):.2f} | "
            f"{item.get('miss_events', 0)} |"
        )

    lines.extend([
        "",
        "## Candidate Hotsets",
        "",
        "| candidate | entries | pack GiB | added miss traffic GiB | added miss traffic | worst prompt added miss | optimistic direct-copy save s | remaining miss GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for item in candidates:
        added_gib = gib(item["added_traffic_bytes"])
        lines.append(
            f"| `{item['name']}` | {item['entries']} | {gib(item['pack_size_bytes']):.2f} | "
            f"{added_gib:.2f} | {item['added_miss_traffic_pct']:.1f}% | "
            f"{item['worst_prompt_added_miss_pct']:.1f}% | "
            f"{added_gib * assumed_direct_s_per_gib:.1f} | "
            f"{gib(item['remaining_miss_traffic_bytes']):.2f} |"
        )

    lines.extend([
        "",
        "## Top Missing Keys",
        "",
        "| rank | traffic GiB | count | tensor | expert | bytes |",
        "|---:|---:|---:|---|---:|---:|",
    ])
    for rank, rec in enumerate(missing_records[:top_rows], 1):
        lines.append(
            f"| {rank} | {gib(rec['traffic_bytes']):.2f} | {rec['count']} | "
            f"`{rec['tensor']}` | {rec['expert_idx']} | {rec['expert_bytes']} |"
        )

    lines.extend([
        "",
        "## Reproduction",
        "",
        "Run from repository root after replacing pack paths as needed:",
        "",
        "```bash",
        "python3 .Agent/run-tools/kimi_general_hotset_pack_feasibility.py \\",
        f"  --route-root {route_root} \\",
        "  --pack /root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-france-l12-upgate-v2.expert-pack \\",
        "  --pack /root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-l1l2down-overlay.expert-pack \\",
        f"  --out-dir {out_dir}",
        "```",
        "",
        "The generated `top-missing-entries.txt` can be converted to repeated",
        "`--entry` arguments for `scripts/kimi-build-missing-down-overlay.py`.",
        "",
        "## Interpretation",
        "",
        "- A high added miss-traffic percentage means the candidate can remove many",
        "  GGUF direct-copy fallback events if the generated overlay is loaded in the",
        "  runtime pack list.",
        "- The estimate is an upper bound: replacement iouring wait, H2D, scheduling,",
        "  TTFT, disk space, and 16GB host-RAM cold-start gates still require a real",
        "  runtime run before any SOTA claim.",
        "- Candidate selection is based only on dev prompts. Held-out prompts remain",
        "  reserved for final validation.",
        "",
    ])
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate prompt-agnostic Kimi general hotset expert-pack coverage.")
    parser.add_argument("--route-root", type=pathlib.Path, required=True, help="Directory containing dev prompt subdirs with route-profile.csv.")
    parser.add_argument("--pack", action="append", type=pathlib.Path, default=[], help="Existing v1 expert pack in runtime load order.")
    parser.add_argument("--replace-duplicates", action="store_true", help="Mirror GGML_MOE_EXPERT_PACK_REPLACE_DUPLICATES=1.")
    parser.add_argument("--topk", default="512,1024,2048,4096,8192,16384", help="Comma-separated top-K candidate sizes.")
    parser.add_argument("--budget-gib", default="4,8,16,32,64", help="Comma-separated pack-size budgets for greedy candidates.")
    parser.add_argument("--assumed-direct-s-per-gib", type=float, default=2.70, help="Measured direct-copy cost used for optimistic savings.")
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--top-rows", type=int, default=80)
    args = parser.parse_args()

    existing_keys, pack_summaries = load_pack_keys(args.pack, args.replace_duplicates) if args.pack else (set(), [])
    by_key, prompt_totals, role_totals, layer_role_totals = summarize_routes(args.route_root, existing_keys)
    missing_records = [
        {
            **rec,
            "key": key,
            "prompt_traffic": dict(rec["prompt_traffic"]),
            "prompt_count": dict(rec["prompt_count"]),
        }
        for key, rec in by_key.items()
        if not rec["existing_hit"]
    ]
    missing_records.sort(key=lambda rec: (rec["traffic_bytes"], rec["expert_bytes"]), reverse=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.out_dir / "top-missing.tsv", missing_records)
    write_entries(args.out_dir / "top-missing-entries.txt", missing_records)

    candidates = []
    for topk in parse_csv_ints(args.topk):
        candidates.append(candidate_stats(f"top{topk}", missing_records[:topk], missing_records, prompt_totals))
        write_entries(args.out_dir / f"top{topk}-entries.txt", missing_records[:topk])
    for budget_gib in parse_csv_floats(args.budget_gib):
        selected = select_by_budget(missing_records, int(budget_gib * 1024**3))
        candidates.append(candidate_stats(f"budget{budget_gib:g}gib", selected, missing_records, prompt_totals))
        write_entries(args.out_dir / f"budget{budget_gib:g}gib-entries.txt", selected)

    with (args.out_dir / "candidates.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "name", "entries", "pack_size_gib", "unique_payload_gib",
            "added_traffic_gib", "added_miss_traffic_pct",
            "worst_prompt_added_miss_pct", "optimistic_direct_save_s",
            "remaining_miss_traffic_gib",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in candidates:
            added_gib = gib(item["added_traffic_bytes"])
            writer.writerow({
                "name": item["name"],
                "entries": item["entries"],
                "pack_size_gib": f"{gib(item['pack_size_bytes']):.6f}",
                "unique_payload_gib": f"{gib(item['unique_payload_bytes']):.6f}",
                "added_traffic_gib": f"{added_gib:.6f}",
                "added_miss_traffic_pct": f"{item['added_miss_traffic_pct']:.6f}",
                "worst_prompt_added_miss_pct": f"{item['worst_prompt_added_miss_pct']:.6f}",
                "optimistic_direct_save_s": f"{added_gib * args.assumed_direct_s_per_gib:.6f}",
                "remaining_miss_traffic_gib": f"{gib(item['remaining_miss_traffic_bytes']):.6f}",
            })

    write_report(
        args.out_dir,
        args.route_root,
        pack_summaries,
        prompt_totals,
        role_totals,
        layer_role_totals,
        missing_records,
        candidates,
        args.assumed_direct_s_per_gib,
        args.top_rows,
    )
    print(args.out_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
