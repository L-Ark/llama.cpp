#!/usr/bin/env python3
import argparse
import csv
import pathlib
import re
import struct
from collections import Counter, defaultdict


PACK_HEADER = struct.Struct("<16sIIQQ")
PACK_ENTRY = struct.Struct("<128siIQQ")
ROLE_RE = re.compile(r"ffn_(up|gate|down)_exps")
LAYER_RE = re.compile(r"blk\.(\d+)\.")


def role_of(tensor: str) -> str:
    match = ROLE_RE.search(tensor)
    return match.group(1) if match else "other"


def layer_of(tensor: str) -> str:
    match = LAYER_RE.search(tensor)
    return match.group(1) if match else ""


def read_int(row: dict[str, str], key: str) -> int:
    try:
        return int(row.get(key, "0") or 0)
    except ValueError:
        return 0


def load_pack_keys(pack_paths: list[pathlib.Path], replace_duplicates: bool) -> tuple[set[tuple[str, int, int]], list[dict[str, object]]]:
    keys: dict[tuple[str, int, int], int] = {}
    summaries = []
    duplicate_count = 0
    for source_idx, path in enumerate(pack_paths):
        with path.open("rb") as f:
            header = f.read(PACK_HEADER.size)
            if len(header) != PACK_HEADER.size:
                raise RuntimeError(f"{path}: short pack header")
            magic, version, header_size, n_entries, data_start = PACK_HEADER.unpack(header)
            if not magic.startswith(b"GGMLMOEPACKv1") or version != 1:
                raise RuntimeError(f"{path}: invalid pack header")
            if header_size < PACK_HEADER.size:
                raise RuntimeError(f"{path}: invalid header_size={header_size}")
            if header_size > PACK_HEADER.size:
                f.seek(header_size)
            source_dups = 0
            source_keys = 0
            payload_bytes = 0
            for _ in range(n_entries):
                raw = f.read(PACK_ENTRY.size)
                if len(raw) != PACK_ENTRY.size:
                    raise RuntimeError(f"{path}: short pack index")
                name_raw, expert_idx, _reserved, _offset, nbytes = PACK_ENTRY.unpack(raw)
                tensor = name_raw.split(b"\0", 1)[0].decode("utf-8", errors="replace")
                key = (tensor, int(expert_idx), int(nbytes))
                if key in keys:
                    source_dups += 1
                    duplicate_count += 1
                    if replace_duplicates and source_idx >= keys[key]:
                        keys[key] = source_idx
                else:
                    keys[key] = source_idx
                    source_keys += 1
                    payload_bytes += int(nbytes)
            summaries.append({
                "path": str(path),
                "entries": int(n_entries),
                "new_unique_keys": source_keys,
                "duplicates": source_dups,
                "payload_gib_new_unique": payload_bytes / 1024**3,
                "data_start": int(data_start),
            })
    if duplicate_count and not replace_duplicates:
        raise RuntimeError(f"duplicate pack keys found: {duplicate_count}; rerun with --replace-duplicates to mirror runtime override mode")
    return set(keys), summaries


def iter_route_trace(path: pathlib.Path):
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            tensor = row.get("tensor", "")
            expert_idx = read_int(row, "expert_idx")
            expert_bytes = read_int(row, "expert_bytes")
            if tensor and expert_bytes > 0:
                yield tensor, expert_idx, expert_bytes


def analyze_prompt(trace_path: pathlib.Path, pack_keys: set[tuple[str, int, int]]) -> dict[str, object]:
    totals = defaultdict(int)
    by_role = defaultdict(lambda: defaultdict(int))
    by_layer_role = defaultdict(lambda: defaultdict(int))
    missing_keys = Counter()

    for tensor, expert_idx, expert_bytes in iter_route_trace(trace_path):
        key = (tensor, expert_idx, expert_bytes)
        hit = key in pack_keys
        prefix = "hit" if hit else "miss"
        totals[f"{prefix}_events"] += 1
        totals[f"{prefix}_bytes"] += expert_bytes
        role = role_of(tensor)
        layer = layer_of(tensor)
        by_role[role][f"{prefix}_events"] += 1
        by_role[role][f"{prefix}_bytes"] += expert_bytes
        by_layer_role[(layer, role)][f"{prefix}_events"] += 1
        by_layer_role[(layer, role)][f"{prefix}_bytes"] += expert_bytes
        if not hit:
            missing_keys[key] += 1

    return {
        "prompt": trace_path.parent.name,
        "trace_path": str(trace_path),
        "totals": dict(totals),
        "by_role": {k: dict(v) for k, v in by_role.items()},
        "by_layer_role": {k: dict(v) for k, v in by_layer_role.items()},
        "missing_keys": missing_keys,
    }


def percent(num: int, den: int) -> float:
    return 100.0 * num / den if den else 0.0


def write_report(prompt_rows: list[dict[str, object]], pack_summaries: list[dict[str, object]], out: pathlib.Path):
    total = defaultdict(int)
    role_total = defaultdict(lambda: defaultdict(int))
    layer_role_total = defaultdict(lambda: defaultdict(int))
    missing = Counter()

    lines = [
        "# Kimi expert-pack coverage audit",
        "",
        "This report compares route-trace events against expert-pack index keys.",
        "It does not run inference and does not inspect held-out test prompts.",
        "",
        "## Pack Sources",
        "",
        "| path | entries | new unique keys | duplicates | new unique payload GiB |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in pack_summaries:
        lines.append(
            f"| `{item['path']}` | {item['entries']} | {item['new_unique_keys']} | "
            f"{item['duplicates']} | {item['payload_gib_new_unique']:.2f} |"
        )

    lines.extend([
        "",
        "## Prompt Coverage",
        "",
        "| prompt | events | event hit | GiB | byte hit | miss GiB |",
        "|---|---:|---:|---:|---:|---:|",
    ])

    for row in prompt_rows:
        t = row["totals"]
        for key, value in t.items():
            total[key] += value
        for role, item in row["by_role"].items():
            for key, value in item.items():
                role_total[role][key] += value
        for key_tuple, item in row["by_layer_role"].items():
            for key, value in item.items():
                layer_role_total[key_tuple][key] += value
        missing.update(row["missing_keys"])

        events = t.get("hit_events", 0) + t.get("miss_events", 0)
        bytes_total = t.get("hit_bytes", 0) + t.get("miss_bytes", 0)
        lines.append(
            f"| `{row['prompt']}` | {events} | {percent(t.get('hit_events', 0), events):.1f}% | "
            f"{bytes_total / 1024**3:.2f} | {percent(t.get('hit_bytes', 0), bytes_total):.1f}% | "
            f"{t.get('miss_bytes', 0) / 1024**3:.2f} |"
        )

    events = total.get("hit_events", 0) + total.get("miss_events", 0)
    bytes_total = total.get("hit_bytes", 0) + total.get("miss_bytes", 0)
    lines.extend([
        "",
        "## Aggregate",
        "",
        f"- events: `{events}`",
        f"- event hit rate: `{percent(total.get('hit_events', 0), events):.1f}%`",
        f"- routed bytes: `{bytes_total / 1024**3:.2f} GiB`",
        f"- byte hit rate: `{percent(total.get('hit_bytes', 0), bytes_total):.1f}%`",
        f"- miss bytes: `{total.get('miss_bytes', 0) / 1024**3:.2f} GiB`",
        "",
        "## By Role",
        "",
        "| role | events | event hit | GiB | byte hit | miss GiB |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for role, item in sorted(role_total.items()):
        role_events = item.get("hit_events", 0) + item.get("miss_events", 0)
        role_bytes = item.get("hit_bytes", 0) + item.get("miss_bytes", 0)
        lines.append(
            f"| {role} | {role_events} | {percent(item.get('hit_events', 0), role_events):.1f}% | "
            f"{role_bytes / 1024**3:.2f} | {percent(item.get('hit_bytes', 0), role_bytes):.1f}% | "
            f"{item.get('miss_bytes', 0) / 1024**3:.2f} |"
        )

    lines.extend([
        "",
        "## Top Missing Layer/Role Buckets",
        "",
        "| layer | role | miss events | miss GiB | hit events | byte hit |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    ranked_layer_role = sorted(
        layer_role_total.items(),
        key=lambda item: item[1].get("miss_bytes", 0),
        reverse=True,
    )
    for (layer, role), item in ranked_layer_role[:40]:
        layer_bytes = item.get("hit_bytes", 0) + item.get("miss_bytes", 0)
        lines.append(
            f"| {layer} | {role} | {item.get('miss_events', 0)} | "
            f"{item.get('miss_bytes', 0) / 1024**3:.2f} | {item.get('hit_events', 0)} | "
            f"{percent(item.get('hit_bytes', 0), layer_bytes):.1f}% |"
        )

    lines.extend([
        "",
        "## Top Missing Keys",
        "",
        "| count | GiB | tensor | expert | bytes |",
        "|---:|---:|---|---:|---:|",
    ])
    for (tensor, expert_idx, expert_bytes), count in missing.most_common(40):
        lines.append(
            f"| {count} | {count * expert_bytes / 1024**3:.2f} | `{tensor}` | "
            f"{expert_idx} | {expert_bytes} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `pack_hit` coverage is independent from VRAM-cache coverage. This audit",
        "  only measures whether a routed expert can use the optimized expert-pack",
        "  source path after it misses VRAM.",
        "- High miss bytes here mean the runtime must materialize selected experts",
        "  from GGUF-backed tensor storage for those route events.",
        "- A model-wide same-quant pack would remove these misses but currently needs",
        "  separate disk feasibility. A GGUF-offset alias pack would target the same",
        "  miss bytes without duplicating the payload.",
        "",
    ])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Audit route-trace expert-pack key coverage.")
    parser.add_argument("--pack", action="append", type=pathlib.Path, required=True, help="Expert-pack file; may be repeated in runtime load order.")
    parser.add_argument("--trace", action="append", type=pathlib.Path, help="route-trace.csv; may be repeated.")
    parser.add_argument("--runs-root", type=pathlib.Path, help="Directory containing prompt subdirs with route-trace.csv.")
    parser.add_argument("--replace-duplicates", action="store_true", help="Mirror GGML_MOE_EXPERT_PACK_REPLACE_DUPLICATES=1.")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    trace_paths = list(args.trace or [])
    if args.runs_root:
        trace_paths.extend(sorted(args.runs_root.glob("*/route-trace.csv")))
    if not trace_paths:
        raise SystemExit("no route traces supplied")

    pack_keys, pack_summaries = load_pack_keys(args.pack, args.replace_duplicates)
    rows = [analyze_prompt(path, pack_keys) for path in sorted(trace_paths)]
    write_report(rows, pack_summaries, args.out)
    print(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
