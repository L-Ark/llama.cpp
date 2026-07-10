#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


MIB = 1024 * 1024
LAYER_RE = re.compile(r"blk\.(\d+)\.")


@dataclass(frozen=True)
class Key:
    tensor: str
    expert_idx: int
    expert_bytes: int

    @property
    def role(self) -> str:
        if ".ffn_up_exps." in self.tensor:
            return "up"
        if ".ffn_gate_exps." in self.tensor:
            return "gate"
        if ".ffn_down_exps." in self.tensor:
            return "down"
        return "other"

    @property
    def bucket_role(self) -> str:
        return "upgate" if self.role in {"up", "gate"} else self.role

    @property
    def layer(self) -> int:
        match = LAYER_RE.search(self.tensor)
        return int(match.group(1)) if match else -1

    @property
    def stable_id(self) -> tuple[str, int]:
        return (self.tensor, self.expert_idx)


@dataclass
class Candidate:
    key: Key
    count: int = 0
    score: float = 0.0
    layer_ms_per_token: float = 0.0


def fnum(value: str | None) -> float:
    try:
        return float(value or 0)
    except ValueError:
        return 0.0


def inum(value: str | None, default: int = 0) -> int:
    try:
        return int(float(value or 0))
    except ValueError:
        return default


def read_screen(path: Path, min_prompts: int, max_layers_per_role: int) -> dict[tuple[int, str], float]:
    by_role: dict[str, list[tuple[tuple[int, str], float, int]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            layer = inum(row.get("layer"), -1)
            role = row.get("role", "")
            prompts = inum(row.get("prompts"))
            ms = fnum(row.get("ms_per_token"))
            if layer < 0 or role not in {"upgate", "down"}:
                continue
            if prompts < min_prompts or ms <= 0:
                continue
            by_role[role].append(((layer, role), ms, prompts))
    out: dict[tuple[int, str], float] = {}
    for role, rows in by_role.items():
        rows.sort(key=lambda item: item[1], reverse=True)
        for bucket, ms, _prompts in rows[:max_layers_per_role]:
            out[bucket] = ms
    return out


def find_route_profiles(root: Path, exclude_re: re.Pattern[str]) -> list[Path]:
    if root.is_file():
        return [root]
    paths = []
    for path in sorted(root.glob("*/route-profile.csv")):
        prompt_id = path.parent.name
        if exclude_re.search(prompt_id):
            raise SystemExit(f"Refusing held-out/test-looking route profile: {path}")
        paths.append(path)
    if not paths:
        raise SystemExit(f"No route-profile.csv found under {root}")
    return paths


def read_candidates(route_profiles: list[Path], bucket_weights: dict[tuple[int, str], float]) -> dict[tuple[str, int], Candidate]:
    candidates: dict[tuple[str, int], Candidate] = {}
    for path in route_profiles:
        with path.open(newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                tensor = row.get("tensor", "")
                if not tensor:
                    continue
                key = Key(
                    tensor=tensor,
                    expert_idx=inum(row.get("expert_idx")),
                    expert_bytes=inum(row.get("expert_bytes")),
                )
                bucket = (key.layer, key.bucket_role)
                layer_ms = bucket_weights.get(bucket)
                if layer_ms is None or key.role not in {"up", "gate", "down"}:
                    continue
                stable_id = key.stable_id
                cand = candidates.get(stable_id)
                if cand is None:
                    cand = Candidate(key=key, layer_ms_per_token=layer_ms)
                    candidates[stable_id] = cand
                count = inum(row.get("count"))
                cand.count += count
                cand.layer_ms_per_token = max(cand.layer_ms_per_token, layer_ms)
    for cand in candidates.values():
        # Route count captures reuse; layer ms/token captures criticality; bytes
        # keeps the candidate budget efficient.
        cand.score = cand.count * cand.layer_ms_per_token / max(cand.key.expert_bytes / MIB, 1.0)
    return candidates


def select_budgeted(candidates: dict[tuple[str, int], Candidate], budgets_mib: dict[str, float]) -> list[Candidate]:
    selected: list[Candidate] = []
    used: dict[str, int] = defaultdict(int)
    ranked = sorted(
        candidates.values(),
        key=lambda c: (c.score, c.count, c.layer_ms_per_token),
        reverse=True,
    )
    for cand in ranked:
        role = cand.key.bucket_role
        budget = int(budgets_mib.get(role, 0.0) * MIB)
        if budget <= 0:
            continue
        if used[role] + cand.key.expert_bytes > budget:
            continue
        selected.append(cand)
        used[role] += cand.key.expert_bytes
    selected.sort(key=lambda c: (c.key.bucket_role, -c.score, c.key.tensor, c.key.expert_idx))
    return selected


def write_profile(path: Path, selected: list[Candidate]) -> None:
    cumulative = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rank", "count", "expert_bytes", "cumulative_bytes", "tensor_base", "expert_idx", "tensor"],
        )
        writer.writeheader()
        for rank, cand in enumerate(selected, start=1):
            cumulative += cand.key.expert_bytes
            writer.writerow({
                "rank": rank,
                "count": cand.count,
                "expert_bytes": cand.key.expert_bytes,
                "cumulative_bytes": cumulative,
                "tensor_base": "0x0",
                "expert_idx": cand.key.expert_idx,
                "tensor": cand.key.tensor,
            })


def write_report(path: Path, selected: list[Candidate], bucket_weights: dict[tuple[int, str], float], budgets_mib: dict[str, float], route_profiles: list[Path]) -> None:
    by_role: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    by_bucket: dict[tuple[int, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for cand in selected:
        role = cand.key.bucket_role
        by_role[role]["entries"] += 1
        by_role[role]["bytes"] += cand.key.expert_bytes
        by_role[role]["route_count"] += cand.count
        by_bucket[(cand.key.layer, role)]["entries"] += 1
        by_bucket[(cand.key.layer, role)]["bytes"] += cand.key.expert_bytes
        by_bucket[(cand.key.layer, role)]["route_count"] += cand.count
        by_bucket[(cand.key.layer, role)]["layer_ms_per_token"] = cand.layer_ms_per_token

    lines = [
        "# Kimi Budgeted Hot Expert Profile",
        "",
        "This is a dev-only offline candidate profile. It does not use held-out",
        "prompts and does not claim a SOTA result.",
        "",
        "## Inputs",
        "",
        f"- route profiles: `{len(route_profiles)}`",
        f"- selected layer/role buckets: `{len(bucket_weights)}`",
        f"- budgets MiB: `{json.dumps(budgets_mib, sort_keys=True)}`",
        "",
        "## Selected By Role",
        "",
        "| role | entries | MiB | route count | budget MiB |",
        "|---|---:|---:|---:|---:|",
    ]
    for role in sorted(budgets_mib):
        stats = by_role.get(role, {})
        lines.append(
            f"| {role} | {int(stats.get('entries', 0))} | "
            f"{stats.get('bytes', 0) / MIB:.2f} | {int(stats.get('route_count', 0))} | {budgets_mib[role]:.1f} |"
        )
    lines.extend([
        "",
        "## Top Selected Buckets",
        "",
        "| layer | role | entries | MiB | route count | source ms/token |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    for (layer, role), stats in sorted(by_bucket.items(), key=lambda item: -item[1]["bytes"])[:40]:
        lines.append(
            f"| {layer} | {role} | {int(stats['entries'])} | "
            f"{stats['bytes'] / MIB:.2f} | {int(stats['route_count'])} | {stats['layer_ms_per_token']:.3f} |"
        )
    lines.extend([
        "",
        "## Top Entries",
        "",
        "| rank | role | layer | tensor | expert | count | MiB | score |",
        "|---:|---|---:|---|---:|---:|---:|---:|",
    ])
    for rank, cand in enumerate(sorted(selected, key=lambda c: c.score, reverse=True)[:80], start=1):
        lines.append(
            f"| {rank} | {cand.key.bucket_role} | {cand.key.layer} | `{cand.key.tensor}` | "
            f"{cand.key.expert_idx} | {cand.count} | {cand.key.expert_bytes / MIB:.2f} | {cand.score:.3f} |"
        )
    lines.extend([
        "",
        "Interpretation:",
        "",
        "- This profile is intentionally smaller than whole-layer admission.",
        "- A runtime A/B must verify whether these pinned/preloaded entries reduce",
        "  decode wall without increasing TTFT, RAM pressure, or residual time.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a budgeted Kimi GGML_MOE_VRAM_PROFILE from corrected dev profile data.")
    ap.add_argument("--route-root", required=True, type=Path)
    ap.add_argument("--screen-csv", required=True, type=Path)
    ap.add_argument("--out-profile", required=True, type=Path)
    ap.add_argument("--out-report", required=True, type=Path)
    ap.add_argument("--upgate-budget-mib", type=float, default=1024.0)
    ap.add_argument("--down-budget-mib", type=float, default=512.0)
    ap.add_argument("--min-prompts", type=int, default=3)
    ap.add_argument("--max-layers-per-role", type=int, default=12)
    ap.add_argument("--exclude-prompt-regex", default=r"heldout|test_")
    args = ap.parse_args()

    exclude_re = re.compile(args.exclude_prompt_regex)
    route_profiles = find_route_profiles(args.route_root, exclude_re)
    bucket_weights = read_screen(args.screen_csv, args.min_prompts, args.max_layers_per_role)
    if not bucket_weights:
        raise SystemExit("No layer/role buckets selected from screen")
    candidates = read_candidates(route_profiles, bucket_weights)
    budgets = {"upgate": args.upgate_budget_mib, "down": args.down_budget_mib}
    selected = select_budgeted(candidates, budgets)
    if not selected:
        raise SystemExit("No experts selected within budgets")
    write_profile(args.out_profile, selected)
    write_report(args.out_report, selected, bucket_weights, budgets, route_profiles)
    summary = {
        "route_profiles": len(route_profiles),
        "buckets": len(bucket_weights),
        "entries": len(selected),
        "bytes": sum(c.key.expert_bytes for c in selected),
        "mib": sum(c.key.expert_bytes for c in selected) / MIB,
        "out_profile": str(args.out_profile),
        "out_report": str(args.out_report),
    }
    args.out_profile.with_suffix(args.out_profile.suffix + ".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
