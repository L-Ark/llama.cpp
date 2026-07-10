#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


LAYER_RE = re.compile(r"blk\.(\d+)\.")


def fnum(value: str | None) -> float:
    try:
        return float(value or 0)
    except ValueError:
        return 0.0


def inum(value: str | None) -> int:
    try:
        return int(float(value or 0))
    except ValueError:
        return 0


def layer_of(tensor: str) -> int:
    match = LAYER_RE.search(tensor or "")
    return int(match.group(1)) if match else -1


def role_of(tensor: str) -> str:
    if ".ffn_up_exps." in tensor:
        return "up"
    if ".ffn_gate_exps." in tensor:
        return "gate"
    if ".ffn_down_exps." in tensor:
        return "down"
    return "other"


def read_metrics(run_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    path = run_dir / "metrics.txt"
    if not path.exists():
        return out
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key] = value
    return out


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


@dataclass
class Bucket:
    prompts: set[str]
    calls: int = 0
    wall_ms: float = 0.0
    stage_ms: float = 0.0
    kernel_ms: float = 0.0
    wait_ms: float = 0.0
    hits: int = 0
    misses: int = 0
    staged_jobs: int = 0
    unique_experts: set[tuple[str, int]] | None = None
    observed_bytes: int = 0
    route_hits: int = 0

    def __post_init__(self) -> None:
        if self.unique_experts is None:
            self.unique_experts = set()


def bucket_key(layer: int, role: str) -> tuple[int, str]:
    return (layer, role)


def main() -> int:
    ap = argparse.ArgumentParser(description="Rank Kimi layer/role cache-admission candidates by corrected decode wall time.")
    ap.add_argument("--profile-root", required=True, type=Path)
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--out-md", required=True, type=Path)
    ap.add_argument("--decode-down-max-active", type=int, default=8)
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    run_dirs = sorted(
        p for p in args.profile_root.iterdir()
        if p.is_dir() and (p / "metrics.txt").exists()
    )
    buckets: dict[tuple[int, str], Bucket] = defaultdict(lambda: Bucket(prompts=set()))
    total_decode_runs = 0
    total_decode_ms = 0.0

    route_counts: Counter[tuple[int, str, str, int]] = Counter()
    route_bytes: dict[tuple[str, int], int] = {}

    for run_dir in run_dirs:
        metrics = read_metrics(run_dir)
        prompt_id = metrics.get("prompt_id", run_dir.name)
        total_decode_runs += inum(metrics.get("decode_runs"))
        total_decode_ms += fnum(metrics.get("decode_ms"))

        for row in read_csv(run_dir / "up-gate-profile.csv"):
            if row.get("mode") != "decode":
                continue
            layer = layer_of(row.get("up_tensor", ""))
            if layer < 0:
                continue
            b = buckets[bucket_key(layer, "upgate")]
            b.prompts.add(prompt_id)
            b.calls += 1
            b.wall_ms += fnum(row.get("wall_ms"))
            b.stage_ms += fnum(row.get("stage_ms"))
            b.kernel_ms += fnum(row.get("kernel_ms"))
            b.wait_ms += fnum(row.get("up_wait_ms")) + fnum(row.get("gate_wait_ms"))
            b.hits += inum(row.get("up_cache_hits")) + inum(row.get("gate_cache_hits"))
            b.misses += inum(row.get("up_cache_misses")) + inum(row.get("gate_cache_misses"))
            b.staged_jobs += inum(row.get("up_stage_jobs")) + inum(row.get("gate_stage_jobs"))

        for row in read_csv(run_dir / "down-batch-profile.csv"):
            if inum(row.get("n_active")) > args.decode_down_max_active:
                continue
            tensor = row.get("tensor", "")
            layer = layer_of(tensor)
            role = role_of(tensor)
            if layer < 0 or role != "down":
                continue
            b = buckets[bucket_key(layer, "down")]
            b.prompts.add(prompt_id)
            b.calls += 1
            b.wall_ms += fnum(row.get("wall_ms"))
            b.stage_ms += fnum(row.get("stage_ms"))
            b.kernel_ms += fnum(row.get("kernel_ms"))
            b.hits += inum(row.get("cache_hits"))
            b.misses += inum(row.get("cache_misses"))
            b.staged_jobs += inum(row.get("staged_jobs"))

        for row in read_csv(run_dir / "route-trace.csv"):
            tensor = row.get("tensor", "")
            role = role_of(tensor)
            layer = layer_of(tensor)
            if layer < 0 or role not in {"up", "gate", "down"}:
                continue
            expert = inum(row.get("expert_idx"))
            nbytes = inum(row.get("expert_bytes"))
            route_counts[(layer, role, tensor, expert)] += 1
            route_bytes[(tensor, expert)] = nbytes

    for (layer, role, tensor, expert), count in route_counts.items():
        target_role = "upgate" if role in {"up", "gate"} else "down"
        b = buckets.get(bucket_key(layer, target_role))
        if not b:
            continue
        stable = (tensor, expert)
        if stable not in b.unique_experts:
            b.unique_experts.add(stable)
            b.observed_bytes += route_bytes.get(stable, 0)
        b.route_hits += count

    rows: list[dict[str, object]] = []
    decode_ms_per_token = total_decode_ms / total_decode_runs if total_decode_runs else 0.0
    for (layer, role), b in buckets.items():
        denom = b.hits + b.misses
        hit_rate = 100.0 * b.hits / denom if denom else 0.0
        ms_per_token = b.wall_ms / total_decode_runs if total_decode_runs else 0.0
        stage_ms_per_token = b.stage_ms / total_decode_runs if total_decode_runs else 0.0
        remaining = max(decode_ms_per_token - ms_per_token, 0.001)
        rows.append({
            "layer": layer,
            "role": role,
            "prompts": len(b.prompts),
            "calls": b.calls,
            "wall_ms": f"{b.wall_ms:.3f}",
            "ms_per_token": f"{ms_per_token:.3f}",
            "stage_ms": f"{b.stage_ms:.3f}",
            "stage_ms_per_token": f"{stage_ms_per_token:.3f}",
            "kernel_ms": f"{b.kernel_ms:.3f}",
            "wait_ms": f"{b.wait_ms:.3f}",
            "hits": b.hits,
            "misses": b.misses,
            "hit_rate": f"{hit_rate:.2f}",
            "staged_jobs": b.staged_jobs,
            "unique_experts": len(b.unique_experts or []),
            "observed_mib": f"{b.observed_bytes / (1024 * 1024):.2f}",
            "route_hits": b.route_hits,
            "ideal_tok_s_if_removed": f"{1000.0 / remaining:.3f}",
        })
    rows.sort(key=lambda r: float(r["ms_per_token"]), reverse=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "layer", "role", "prompts", "calls", "wall_ms", "ms_per_token",
        "stage_ms", "stage_ms_per_token", "kernel_ms", "wait_ms",
        "hits", "misses", "hit_rate", "staged_jobs", "unique_experts",
        "observed_mib", "route_hits", "ideal_tok_s_if_removed",
    ]
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Kimi Wait-Weighted Admission Screen",
        "",
        f"- profile root: `{args.profile_root}`",
        f"- runs: `{len(run_dirs)}`",
        f"- decode runs: `{total_decode_runs}`",
        f"- decode ms/token: `{decode_ms_per_token:.3f}`",
        f"- decode down filter: `n_active <= {args.decode_down_max_active}`",
        "",
        "This is an offline screen only. It does not select held-out prompts and",
        "does not claim a SOTA result.",
        "",
        "| layer | role | prompts | ms/token | stage ms/token | hit rate | misses | unique experts | observed MiB | ideal tok/s if removed |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows[: args.top]:
        lines.append(
            f"| {row['layer']} | {row['role']} | {row['prompts']} | "
            f"{row['ms_per_token']} | {row['stage_ms_per_token']} | "
            f"{row['hit_rate']}% | {row['misses']} | {row['unique_experts']} | "
            f"{row['observed_mib']} | {row['ideal_tok_s_if_removed']} |"
        )
    lines.extend([
        "",
        "Interpretation:",
        "",
        "- Prefer candidates that are high ms/token, appear in all dev prompts,",
        "  have substantial misses, and have an observed footprint small enough to",
        "  fit in VRAM or RAM without displacing higher-value entries.",
        "- This screen ranks layer/role buckets, not individual experts. A runtime",
        "  candidate still needs a profile-generation step and paired cold-start",
        "  A/B validation.",
        "",
    ])
    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print(args.out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
