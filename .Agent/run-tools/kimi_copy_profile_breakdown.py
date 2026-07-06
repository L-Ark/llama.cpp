#!/usr/bin/env python3
import argparse
import csv
import json
import pathlib
import re
from collections import Counter, defaultdict


ROLE_RE = re.compile(r"ffn_(up|gate|down)_exps")
LAYER_RE = re.compile(r"blk\.(\d+)\.")


def role_of(tensor: str) -> str:
    match = ROLE_RE.search(tensor)
    return match.group(1) if match else "other"


def layer_of(tensor: str) -> str:
    match = LAYER_RE.search(tensor)
    return match.group(1) if match else ""


def read_float(row, key: str) -> float:
    try:
        return float(row.get(key, 0) or 0)
    except ValueError:
        return 0.0


def read_metrics(run_dir: pathlib.Path) -> dict:
    path = run_dir / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def analyze_copy_profile(path: pathlib.Path):
    by_path = defaultdict(lambda: defaultdict(float))
    by_layer_role = defaultdict(lambda: defaultdict(float))
    by_tensor = Counter()

    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            tensor = row.get("tensor", "")
            role = role_of(tensor)
            key = (row.get("op", ""), role, row.get("pack_hit", ""), row.get("iouring", ""))
            layer_key = (row.get("op", ""), layer_of(tensor), role, row.get("pack_hit", ""), row.get("iouring", ""))
            for field in (
                "bytes",
                "slot_wait_ms",
                "host_ms",
                "io_wait_ms",
                "enqueue_ms",
                "h2d_ms",
                "wall_ms",
            ):
                by_path[key][field] += read_float(row, field)
                by_layer_role[layer_key][field] += read_float(row, field)
            by_path[key]["calls"] += 1
            by_layer_role[layer_key]["calls"] += 1
            by_tensor[(row.get("op", ""), tensor, row.get("pack_hit", ""), row.get("iouring", ""))] += read_float(row, "wall_ms")

    return by_path, by_layer_role, by_tensor


def row_total_ms(rows, field: str, pred) -> float:
    return sum(values[field] for key, values in rows.items() if pred(key))


def write_report(run_dir: pathlib.Path, out: pathlib.Path):
    copy_profile = run_dir / "copy-profile.csv"
    if not copy_profile.exists():
        raise SystemExit(f"missing {copy_profile}")

    metrics = read_metrics(run_dir)
    by_path, by_layer_role, by_tensor = analyze_copy_profile(copy_profile)

    pack_miss_wall = row_total_ms(by_path, "wall_ms", lambda key: key[2] == "0")
    pack_hit_wall = row_total_ms(by_path, "wall_ms", lambda key: key[2] == "1")
    iouring_wall = row_total_ms(by_path, "wall_ms", lambda key: key[3] == "1")
    non_iouring_wall = row_total_ms(by_path, "wall_ms", lambda key: key[3] == "0")
    total_wall = row_total_ms(by_path, "wall_ms", lambda key: True)

    pack_miss_gib = row_total_ms(by_path, "bytes", lambda key: key[2] == "0") / 1024**3
    pack_hit_gib = row_total_ms(by_path, "bytes", lambda key: key[2] == "1") / 1024**3

    lines = [
        "# Kimi copy-profile breakdown",
        "",
        "This diagnostic report uses one dev prompt run with copy profiling enabled.",
        "It is not a held-out test result and is not an accepted SOTA metric.",
        "",
        "Important caveat: the diagnostic run used `COPY_PROFILE_H2D=1`, which",
        "synchronizes H2D copies for measurement. Token rate from this run is not",
        "directly comparable to normal SOTA runs, but the split between expert-pack",
        "hits and misses identifies where host-side materialization time is spent.",
        "",
        "## Run",
        "",
        f"- run dir: `{run_dir}`",
        f"- prompt id: `{metrics.get('prompt_id', run_dir.name)}`",
        f"- token rate: `{float(metrics.get('token_rate', 0) or 0):.2f} tok/s`",
        f"- TTFT: `{float(metrics.get('ttft_ms', 0) or 0):.2f} ms`",
        f"- decode: `{float(metrics.get('decode_ms', 0) or 0):.2f} ms / {metrics.get('decode_runs', '')}`",
        f"- memory peak: `{metrics.get('memory.peak', '')}`",
        "",
        "## High-Level Split",
        "",
        f"- total profiled copy wall: `{total_wall:.0f} ms`",
        f"- expert-pack miss wall: `{pack_miss_wall:.0f} ms` over `{pack_miss_gib:.2f} GiB`",
        f"- expert-pack hit wall: `{pack_hit_wall:.0f} ms` over `{pack_hit_gib:.2f} GiB`",
        f"- iouring wall: `{iouring_wall:.0f} ms`",
        f"- non-iouring wall: `{non_iouring_wall:.0f} ms`",
        "",
        "## By Copy Path",
        "",
        "| op | role | pack hit | iouring | calls | GiB | host ms | io wait ms | H2D ms | enqueue ms | slot wait ms | wall ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for (op, role, pack_hit, iouring), values in sorted(by_path.items(), key=lambda item: -item[1]["wall_ms"]):
        lines.append(
            f"| `{op}` | {role} | {pack_hit} | {iouring} | {int(values['calls'])} | "
            f"{values['bytes'] / 1024**3:.2f} | {values['host_ms']:.0f} | "
            f"{values['io_wait_ms']:.0f} | {values['h2d_ms']:.0f} | "
            f"{values['enqueue_ms']:.0f} | {values['slot_wait_ms']:.0f} | {values['wall_ms']:.0f} |"
        )

    lines.extend([
        "",
        "## Top Pack-Miss Layer/Role Buckets",
        "",
        "| op | layer | role | iouring | calls | GiB | host ms | H2D ms | wall ms |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    pack_miss_layers = [
        (key, values)
        for key, values in by_layer_role.items()
        if key[3] == "0"
    ]
    for (op, layer, role, _pack_hit, iouring), values in sorted(pack_miss_layers, key=lambda item: -item[1]["wall_ms"])[:30]:
        lines.append(
            f"| `{op}` | {layer} | {role} | {iouring} | {int(values['calls'])} | "
            f"{values['bytes'] / 1024**3:.2f} | {values['host_ms']:.0f} | "
            f"{values['h2d_ms']:.0f} | {values['wall_ms']:.0f} |"
        )

    lines.extend([
        "",
        "## Top Tensor Copy Wall",
        "",
        "| wall ms | op | tensor | pack hit | iouring |",
        "|---:|---|---|---:|---:|",
    ])
    for (op, tensor, pack_hit, iouring), wall_ms in by_tensor.most_common(30):
        lines.append(f"| {wall_ms:.0f} | `{op}` | `{tensor}` | {pack_hit} | {iouring} |")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- The largest measured copy cost comes from `pack_hit=0` paths. These are",
        "  tensors absent from the current expert pack and therefore materialized",
        "  through the slower GGUF-backed path rather than the optimized pack path.",
        "- This is different from a VRAM hotset miss. A tensor can miss VRAM cache",
        "  but still be served efficiently if it exists in the expert pack. The",
        "  current general-prompt profile shows many misses are also expert-pack",
        "  misses, which makes fixed VRAM hotset tuning insufficient.",
        "- A prompt-specific/dev-union pack would be useful only as a diagnostic",
        "  bound. An accepted candidate must use prompt-independent coverage, such",
        "  as a model-wide expert pack or a GGUF-offset direct-read pack alias.",
        "- The next implementation should first remove or reduce the pack-miss path",
        "  for general prompts, then rerun the strict n96 dev baseline before any",
        "  held-out test evaluation.",
        "",
    ])

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Summarize Kimi COPY_PROFILE run copy-path costs.")
    parser.add_argument("--run-dir", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    write_report(args.run_dir, args.out)
    print(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
