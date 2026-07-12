#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


GIB = 1024 ** 3
MIB = 1024 ** 2
ROLE_RE = re.compile(r"ffn_(up|gate|down)_exps")
LAYER_RE = re.compile(r"blk\.(\d+)\.")


@dataclass(frozen=True)
class Key:
    tensor: str
    expert_idx: int
    nbytes: int

    @property
    def role(self) -> str:
        m = ROLE_RE.search(self.tensor)
        return m.group(1) if m else "other"

    @property
    def layer(self) -> int:
        m = LAYER_RE.search(self.tensor)
        return int(m.group(1)) if m else -1


@dataclass
class Stat:
    key: Key
    prompts: set[str] = field(default_factory=set)
    calls: int = 0
    traffic_bytes: int = 0
    io_wait_ms: float = 0.0
    h2d_ms: float = 0.0
    wall_ms: float = 0.0
    ops: Counter[str] = field(default_factory=Counter)


def fnum(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def inum(value: str | None) -> int:
    try:
        return int(float(value or 0))
    except ValueError:
        return 0


def read_metrics(run_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    path = run_dir / "metrics.txt"
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key] = value
    return out


def discover_runs(root: Path, exclude_re: re.Pattern[str]) -> list[Path]:
    runs = []
    for run in sorted(root.iterdir()):
        if not run.is_dir() or not (run / "copy-profile.csv").exists():
            continue
        if exclude_re.search(run.name):
            raise SystemExit(f"refusing held-out/test-looking run: {run}")
        runs.append(run)
    if not runs:
        raise SystemExit(f"no run dirs with copy-profile.csv under {root}")
    return runs


def load_stats(runs: list[Path], roles: set[str], min_prompts: int) -> tuple[list[Stat], dict[str, float]]:
    by_key: dict[Key, Stat] = {}
    totals = {
        "decode_ms": 0.0,
        "decode_runs": 0.0,
        "copy_rows": 0.0,
        "candidate_rows": 0.0,
        "candidate_iowait_ms": 0.0,
        "candidate_h2d_ms": 0.0,
        "candidate_traffic_bytes": 0.0,
    }

    for run in runs:
        metrics = read_metrics(run)
        prompt = metrics.get("prompt_id", run.name)
        totals["decode_ms"] += fnum(metrics.get("decode_ms"))
        totals["decode_runs"] += fnum(metrics.get("decode_runs"))
        with (run / "copy-profile.csv").open(newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                totals["copy_rows"] += 1
                if row.get("pack_hit") != "1" or row.get("iouring") != "1" or row.get("ram_hit") == "1":
                    continue
                key = Key(row.get("tensor", ""), inum(row.get("expert_idx")), inum(row.get("bytes")))
                if key.role not in roles or key.nbytes <= 0:
                    continue
                stat = by_key.get(key)
                if stat is None:
                    stat = Stat(key)
                    by_key[key] = stat
                nbytes = inum(row.get("bytes"))
                io_wait = fnum(row.get("io_wait_ms"))
                h2d = fnum(row.get("h2d_ms"))
                wall = fnum(row.get("wall_ms"))
                stat.prompts.add(prompt)
                stat.calls += 1
                stat.traffic_bytes += nbytes
                stat.io_wait_ms += io_wait
                stat.h2d_ms += h2d
                stat.wall_ms += wall
                stat.ops[row.get("op", "")] += 1
                totals["candidate_rows"] += 1
                totals["candidate_iowait_ms"] += io_wait
                totals["candidate_h2d_ms"] += h2d
                totals["candidate_traffic_bytes"] += nbytes

    stats = [s for s in by_key.values() if len(s.prompts) >= min_prompts]
    stats.sort(key=lambda s: (s.io_wait_ms / max(s.key.nbytes, 1), s.io_wait_ms), reverse=True)
    return stats, totals


def select_for_budget(stats: list[Stat], budget_mib: int, ram_h2d_gib_s: float) -> dict[str, object]:
    budget = budget_mib * MIB
    used = 0
    selected: list[Stat] = []
    for stat in stats:
        if used + stat.key.nbytes > budget:
            continue
        selected.append(stat)
        used += stat.key.nbytes

    io_wait_ms = sum(s.io_wait_ms for s in selected)
    h2d_ms_measured = sum(s.h2d_ms for s in selected)
    traffic_bytes = sum(s.traffic_bytes for s in selected)
    ram_h2d_ms = 1000.0 * (traffic_bytes / GIB) / ram_h2d_gib_s if ram_h2d_gib_s > 0 else 0.0
    by_role: Counter[str] = Counter()
    by_layer_role: Counter[str] = Counter()
    by_op: Counter[str] = Counter()
    for stat in selected:
        by_role[stat.key.role] += 1
        by_layer_role[f"blk.{stat.key.layer}.{stat.key.role}"] += 1
        by_op.update(stat.ops)

    return {
        "budget_mib": budget_mib,
        "resident_bytes": used,
        "resident_gib": used / GIB,
        "selected_entries": len(selected),
        "selected_calls": sum(s.calls for s in selected),
        "selected_traffic_gib": traffic_bytes / GIB,
        "optimistic_saved_ms": io_wait_ms,
        "measured_current_h2d_ms": h2d_ms_measured,
        "conservative_ram_h2d_ms": ram_h2d_ms,
        "conservative_net_saved_ms": max(0.0, io_wait_ms - ram_h2d_ms),
        "by_role_entries": dict(sorted(by_role.items())),
        "by_op_calls": dict(sorted(by_op.items())),
        "top_layer_role_entries": by_layer_role.most_common(20),
        "top_entries": [
            {
                "rank": i + 1,
                "tensor": s.key.tensor,
                "expert_idx": s.key.expert_idx,
                "nbytes": s.key.nbytes,
                "prompts": sorted(s.prompts),
                "calls": s.calls,
                "traffic_gib": s.traffic_bytes / GIB,
                "io_wait_ms": s.io_wait_ms,
                "h2d_ms": s.h2d_ms,
                "score_ms_per_mib": s.io_wait_ms / max(s.key.nbytes / MIB, 1e-9),
            }
            for i, s in enumerate(selected[:80])
        ],
    }


def add_bounds(rows: list[dict[str, object]], totals: dict[str, float]) -> None:
    decode_ms = totals["decode_ms"]
    decode_runs = totals["decode_runs"]
    for row in rows:
        optimistic_decode = max(1.0, decode_ms - float(row["optimistic_saved_ms"]))
        conservative_decode = max(1.0, decode_ms - float(row["conservative_net_saved_ms"]))
        row["optimistic_saved_ms_per_token"] = float(row["optimistic_saved_ms"]) / decode_runs if decode_runs else 0.0
        row["conservative_saved_ms_per_token"] = float(row["conservative_net_saved_ms"]) / decode_runs if decode_runs else 0.0
        row["optimistic_tok_s"] = decode_runs / (optimistic_decode / 1000.0) if optimistic_decode else 0.0
        row["conservative_tok_s"] = decode_runs / (conservative_decode / 1000.0) if conservative_decode else 0.0


def write_report(path: Path, root: Path, stats: list[Stat], totals: dict[str, float], rows: list[dict[str, object]], target_tok_s: float) -> None:
    decode_runs = totals["decode_runs"]
    decode_ms = totals["decode_ms"]
    baseline_tok_s = decode_runs / (decode_ms / 1000.0) if decode_ms else 0.0
    target_decode_ms = 1000.0 * decode_runs / target_tok_s
    needed_saved = max(0.0, decode_ms - target_decode_ms)
    best_opt = max(rows, key=lambda r: float(r["optimistic_tok_s"])) if rows else None
    best_con = max(rows, key=lambda r: float(r["conservative_tok_s"])) if rows else None

    lines = [
        "# Kimi copy-profile RAM tier oracle",
        "",
        "This is a dev-only offline upper bound. It does not change runtime behavior or claim SOTA.",
        "",
        f"- root: `{root}`",
        f"- decode: `{decode_ms:.2f} ms / {int(decode_runs)}`",
        f"- baseline tok/s: `{baseline_tok_s:.3f}`",
        f"- target tok/s: `{target_tok_s:.3f}`",
        f"- needed saved: `{needed_saved:.2f} ms` total, `{needed_saved / decode_runs if decode_runs else 0.0:.3f} ms/token`",
        f"- candidate keys after min-prompt filter: `{len(stats)}`",
        f"- candidate traffic: `{totals['candidate_traffic_bytes'] / GIB:.3f} GiB`",
        f"- candidate io_wait: `{totals['candidate_iowait_ms']:.2f} ms`",
        "",
        "## Budget Bound",
        "",
        "| budget MiB | resident GiB | entries | traffic GiB | opt saved ms/token | opt tok/s | conservative saved ms/token | conservative tok/s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['budget_mib']} | {float(row['resident_gib']):.3f} | {row['selected_entries']} | "
            f"{float(row['selected_traffic_gib']):.3f} | {float(row['optimistic_saved_ms_per_token']):.3f} | "
            f"{float(row['optimistic_tok_s']):.3f} | {float(row['conservative_saved_ms_per_token']):.3f} | "
            f"{float(row['conservative_tok_s']):.3f} |"
        )

    lines.extend([
        "",
        "## Decision",
        "",
    ])
    if best_opt and float(best_opt["optimistic_saved_ms"]) < needed_saved:
        lines.append("- `reject_runtime_ab`: even the row-level optimistic bound cannot reach the 2 tok/s target.")
    elif best_con and float(best_con["conservative_net_saved_ms"]) < needed_saved:
        lines.append("- `reject_until_batch_or_h2d_gap_is_solved`: optimistic bound can help, but conservative RAM->VRAM bound does not clear target.")
    else:
        lines.append("- `candidate_for_io_trace_or_runtime_ab`: conservative bound clears target; require IO batch fragmentation analysis next.")
    lines.extend([
        "",
        "Caveats:",
        "",
        "- Row-level optimistic saving assumes each selected RAM entry removes its full profiled `io_wait_ms`.",
        "- Conservative saving subtracts RAM->VRAM H2D at the configured bandwidth, because CPU/RAM residency does not remove H2D.",
        "- This oracle does not model mixed SSD/RAM batch fragmentation; if a budget looks promising, the next step is an `io-read-trace.csv` complete-batch oracle before runtime A/B.",
        "",
        "## Best Budget Top Entries",
        "",
    ])
    if best_opt:
        lines.append(f"Best optimistic budget: `{best_opt['budget_mib']} MiB`")
        lines.append("")
        lines.append("| rank | tensor | expert | prompts | calls | traffic GiB | io wait ms | score ms/MiB |")
        lines.append("|---:|---|---:|---|---:|---:|---:|---:|")
        for entry in best_opt["top_entries"][:30]:
            lines.append(
                f"| {entry['rank']} | `{entry['tensor']}` | {entry['expert_idx']} | "
                f"{';'.join(entry['prompts'])} | {entry['calls']} | {entry['traffic_gib']:.3f} | "
                f"{entry['io_wait_ms']:.2f} | {entry['score_ms_per_mib']:.3f} |"
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--roles", default="up,gate,down")
    ap.add_argument("--min-prompts", type=int, default=2)
    ap.add_argument("--budgets-mib", default="1024,2048,4096,6144,8192,10240")
    ap.add_argument("--ram-h2d-gib-s", type=float, default=24.0)
    ap.add_argument("--target-tok-s", type=float, default=2.0)
    ap.add_argument("--exclude-prompt-regex", default=r"heldout|test_")
    args = ap.parse_args()

    exclude_re = re.compile(args.exclude_prompt_regex)
    roles = {x.strip() for x in args.roles.split(",") if x.strip()}
    runs = discover_runs(args.root, exclude_re)
    stats, totals = load_stats(runs, roles, args.min_prompts)
    budgets = [int(x) for x in args.budgets_mib.split(",") if x.strip()]
    rows = [select_for_budget(stats, budget, args.ram_h2d_gib_s) for budget in budgets]
    add_bounds(rows, totals)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "kind": "kimi_copy_profile_ram_tier_oracle",
        "root": str(args.root),
        "roles": sorted(roles),
        "min_prompts": args.min_prompts,
        "ram_h2d_gib_s": args.ram_h2d_gib_s,
        "target_tok_s": args.target_tok_s,
        "totals": totals,
        "rows": rows,
    }
    (args.out_dir / "oracle.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    with (args.out_dir / "oracle.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "budget_mib", "resident_gib", "selected_entries", "selected_calls",
            "selected_traffic_gib", "optimistic_saved_ms_per_token", "optimistic_tok_s",
            "conservative_saved_ms_per_token", "conservative_tok_s",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})
    write_report(args.out_dir / "oracle.md", args.root, stats, totals, rows, args.target_tok_s)
    print(args.out_dir / "oracle.md")


if __name__ == "__main__":
    main()
