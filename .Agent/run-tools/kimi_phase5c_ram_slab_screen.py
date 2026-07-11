#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable


TENSOR_RE = re.compile(r"blk\.(\d+)\.ffn_(up|gate|down)_exps\.weight")
MIB = 1024 * 1024
GIB = 1024 ** 3


@dataclass(frozen=True)
class ExpertKey:
    tensor: str
    expert_idx: int


@dataclass(frozen=True)
class ReadRow:
    prompt: str
    batch_seq: int
    jobs: int
    op: str
    tensor: str
    layer: int
    role: str
    expert_idx: int
    source_idx: int
    offset: int
    nbytes: int

    @property
    def key(self) -> ExpertKey:
        return ExpertKey(self.tensor, self.expert_idx)

    @property
    def end(self) -> int:
        return self.offset + self.nbytes


@dataclass
class Batch:
    prompt: str
    batch_seq: int
    wait_ms: float = 0.0
    rows: int = 0
    bytes: int = 0


@dataclass
class Candidate:
    kind: str
    name: str
    keys: set[ExpertKey]
    layer: int | str = ""
    role: str = ""
    source_idx: int | str = ""
    offset_start: int | str = ""
    offset_end: int | str = ""
    resident_bytes: int = 0
    read_bytes: int = 0
    prompts: set[str] = field(default_factory=set)
    rows: int = 0
    batches: set[tuple[str, int]] = field(default_factory=set)
    dominant_batches: int = 0
    ram_only_batches: int = 0
    proportional_wait_ms: float = 0.0
    full_hit_wait_ms: float = 0.0
    mixed_risk_batches: int = 0
    vram_overlap_entries: int = 0
    vram_overlap_bytes: int = 0


def fnum(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value or default)
    except ValueError:
        return default


def inum(value: str | None, default: int = 0) -> int:
    try:
        return int(float(value or default))
    except ValueError:
        return default


def parse_tensor(tensor: str) -> tuple[int, str]:
    match = TENSOR_RE.search(tensor or "")
    if not match:
        return -1, "other"
    return int(match.group(1)), match.group(2)


def prompt_id_for(path: pathlib.Path) -> str:
    return path.parent.name if path.name.endswith(".csv") else path.stem


def discover_io_roots(paths: Iterable[pathlib.Path], exclude_re: re.Pattern[str]) -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    seen: set[pathlib.Path] = set()
    for path in paths:
        if path.is_file() and path.name == "io-read-trace.csv":
            candidates = [path]
        elif path.is_dir():
            candidates = sorted(path.glob("*/io-read-trace.csv"))
            if (path / "io-read-trace.csv").exists():
                candidates.append(path / "io-read-trace.csv")
        else:
            candidates = []
        for candidate in candidates:
            prompt = prompt_id_for(candidate)
            if exclude_re.search(prompt):
                raise SystemExit(f"Refusing held-out/test-looking trace: {candidate}")
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            out.append(candidate)
    return out


def load_waits(path: pathlib.Path) -> dict[int, float]:
    waits: dict[int, float] = defaultdict(float)
    if not path.exists():
        return waits
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            if row.get("batch_seq") == "batch_seq":
                continue
            waits[inum(row.get("batch_seq"))] += fnum(row.get("wait_ms"))
    return waits


def load_reads(paths: list[pathlib.Path], max_jobs: int, roles: set[str]) -> tuple[list[ReadRow], dict[tuple[str, int], Batch], dict[ExpertKey, int]]:
    rows: list[ReadRow] = []
    batches: dict[tuple[str, int], Batch] = {}
    expert_bytes: dict[ExpertKey, int] = {}
    for path in paths:
        prompt = prompt_id_for(path)
        waits = load_waits(path.parent / "io-wait-trace.csv")
        with path.open(newline="", encoding="utf-8", errors="replace") as f:
            for raw in csv.DictReader(f):
                if raw.get("jobs") == "jobs" or not raw.get("tensor"):
                    continue
                jobs = inum(raw.get("jobs"))
                if max_jobs > 0 and jobs > max_jobs:
                    continue
                tensor = raw.get("tensor", "")
                layer, role = parse_tensor(tensor)
                if role not in roles:
                    continue
                row = ReadRow(
                    prompt=prompt,
                    batch_seq=inum(raw.get("batch_seq")),
                    jobs=jobs,
                    op=raw.get("op", ""),
                    tensor=tensor,
                    layer=layer,
                    role=role,
                    expert_idx=inum(raw.get("expert_idx")),
                    source_idx=inum(raw.get("source_idx")),
                    offset=inum(raw.get("offset")),
                    nbytes=inum(raw.get("nbytes")),
                )
                rows.append(row)
                expert_bytes[row.key] = max(expert_bytes.get(row.key, 0), row.nbytes)
                batch_key = (prompt, row.batch_seq)
                batch = batches.get(batch_key)
                if batch is None:
                    batch = Batch(prompt=prompt, batch_seq=row.batch_seq, wait_ms=waits.get(row.batch_seq, 0.0))
                    batches[batch_key] = batch
                batch.rows += 1
                batch.bytes += row.nbytes
    return rows, batches, expert_bytes


def read_route_profiles(paths: Iterable[pathlib.Path]) -> list[tuple[ExpertKey, int, int]]:
    rows: list[tuple[ExpertKey, int, int]] = []
    for path in paths:
        candidates: list[pathlib.Path]
        if path.is_file() and path.name == "route-profile.csv":
            candidates = [path]
        elif path.is_dir():
            candidates = sorted(path.glob("*/route-profile.csv"))
            if (path / "route-profile.csv").exists():
                candidates.append(path / "route-profile.csv")
        else:
            candidates = []
        for candidate in candidates:
            with candidate.open(newline="", encoding="utf-8", errors="replace") as f:
                for row in csv.DictReader(f):
                    if not row.get("tensor") or row.get("expert_idx") == "expert_idx":
                        continue
                    key = ExpertKey(row["tensor"], inum(row.get("expert_idx")))
                    rows.append((key, inum(row.get("expert_bytes")), inum(row.get("count"))))
    return rows


def role_cache_id(tensor: str, expert_bytes: int, split_max_mib: float) -> str:
    _layer, role = parse_tensor(tensor)
    if role in {"up", "gate"} and expert_bytes <= split_max_mib * MIB:
        return "upgate"
    if role == "down":
        return "down"
    return "other"


def simulate_vram_hotset(route_rows: list[tuple[ExpertKey, int, int]], down_slots: int, upgate_slots: int, split_max_mib: float) -> set[ExpertKey]:
    aggregate: dict[ExpertKey, tuple[int, int]] = {}
    for key, nbytes, count in route_rows:
        prev_bytes, prev_count = aggregate.get(key, (nbytes, 0))
        aggregate[key] = (max(prev_bytes, nbytes), prev_count + count)
    ranked = sorted(aggregate.items(), key=lambda item: (item[1][0] * item[1][1], item[1][1]), reverse=True)
    limits = {"down": down_slots, "upgate": upgate_slots}
    used = {"down": 0, "upgate": 0}
    selected: set[ExpertKey] = set()
    for key, (nbytes, _count) in ranked:
        cid = role_cache_id(key.tensor, nbytes, split_max_mib)
        if cid not in limits or used[cid] >= limits[cid]:
            continue
        selected.add(key)
        used[cid] += 1
    return selected


def unit_candidates(rows: list[ReadRow], expert_bytes: dict[ExpertKey, int]) -> list[Candidate]:
    by_unit: dict[tuple[str, str], set[ExpertKey]] = defaultdict(set)
    for row in rows:
        by_unit[(f"layer_role", f"blk.{row.layer}.{row.role}")].add(row.key)
        if row.role in {"up", "gate"}:
            by_unit[(f"layer_upgate", f"blk.{row.layer}.upgate")].add(row.key)
        by_unit[(f"layer_all", f"blk.{row.layer}.all")].add(row.key)
    out: list[Candidate] = []
    for (kind, name), keys in by_unit.items():
        m = re.search(r"blk\.(\d+)\.([A-Za-z_]+)", name)
        layer = int(m.group(1)) if m else ""
        role = m.group(2) if m else ""
        out.append(Candidate(
            kind=kind,
            name=name,
            keys=keys,
            layer=layer,
            role=role,
            resident_bytes=sum(expert_bytes.get(key, 0) for key in keys),
        ))
    return out


def contiguous_candidates(rows: list[ReadRow], expert_bytes: dict[ExpertKey, int], window_mib_values: list[int], max_per_source: int) -> list[Candidate]:
    unique_by_source_role: dict[tuple[int, str], dict[ExpertKey, ReadRow]] = defaultdict(dict)
    traffic: Counter[ExpertKey] = Counter()
    for row in rows:
        unique_by_source_role[(row.source_idx, row.role)].setdefault(row.key, row)
        traffic[row.key] += row.nbytes

    candidates: list[Candidate] = []
    for (source_idx, role), mapping in unique_by_source_role.items():
        items = sorted(mapping.values(), key=lambda r: (r.offset, r.expert_idx, r.tensor))
        for window_mib in window_mib_values:
            max_span = window_mib * MIB
            left = 0
            best: list[tuple[int, int, int, int, set[ExpertKey]]] = []
            keys: set[ExpertKey] = set()
            for right, row in enumerate(items):
                keys.add(row.key)
                while left <= right and items[right].end - items[left].offset > max_span:
                    keys.discard(items[left].key)
                    left += 1
                if not keys:
                    continue
                span = items[right].end - items[left].offset
                useful = sum(expert_bytes.get(key, 0) for key in keys)
                weighted = sum(traffic.get(key, 0) for key in keys)
                best.append((weighted, useful, span, left, set(keys)))
            best.sort(reverse=True, key=lambda item: (item[0] / max(item[2], 1), item[0], item[1]))
            for rank, (weighted, useful, span, left_idx, keyset) in enumerate(best[:max_per_source], 1):
                if not keyset:
                    continue
                first = min((mapping[key].offset for key in keyset), default=0)
                last = max((mapping[key].end for key in keyset), default=first)
                candidates.append(Candidate(
                    kind="contig_window",
                    name=f"src{source_idx}.{role}.win{window_mib}MiB.rank{rank}",
                    keys=keyset,
                    layer="mixed",
                    role=role,
                    source_idx=source_idx,
                    offset_start=first,
                    offset_end=last,
                    resident_bytes=span,
                    read_bytes=useful,
                ))
    return candidates


def index_rows_by_key(rows: list[ReadRow]) -> dict[ExpertKey, list[ReadRow]]:
    out: dict[ExpertKey, list[ReadRow]] = defaultdict(list)
    for row in rows:
        out[row.key].append(row)
    return out


def score_candidate(candidate: Candidate, rows_by_key: dict[ExpertKey, list[ReadRow]], batches: dict[tuple[str, int], Batch], expert_bytes: dict[ExpertKey, int], vram_hotset: set[ExpertKey]) -> None:
    batch_hits: dict[tuple[str, int], tuple[int, int]] = defaultdict(lambda: [0, 0])  # type: ignore[assignment]
    for key in candidate.keys:
        for row in rows_by_key.get(key, []):
            batch_key = (row.prompt, row.batch_seq)
            candidate.rows += 1
            candidate.read_bytes += row.nbytes
            candidate.prompts.add(row.prompt)
            candidate.batches.add(batch_key)
            batch_hits[batch_key][0] += 1  # type: ignore[index]
            batch_hits[batch_key][1] += row.nbytes  # type: ignore[index]
    if candidate.resident_bytes <= 0:
        candidate.resident_bytes = sum(expert_bytes.get(key, 0) for key in candidate.keys)
    for key in candidate.keys:
        if key in vram_hotset:
            candidate.vram_overlap_entries += 1
            candidate.vram_overlap_bytes += expert_bytes.get(key, 0)
    for batch_key, (hit_rows, _hit_bytes) in batch_hits.items():
        batch = batches[batch_key]
        row_fraction = hit_rows / max(batch.rows, 1)
        candidate.proportional_wait_ms += batch.wait_ms * row_fraction
        if hit_rows >= batch.rows:
            candidate.ram_only_batches += 1
            candidate.full_hit_wait_ms += batch.wait_ms
        elif row_fraction >= 0.75 and hit_rows >= 4:
            candidate.dominant_batches += 1
            candidate.full_hit_wait_ms += batch.wait_ms
        else:
            candidate.mixed_risk_batches += 1


def candidate_row(candidate: Candidate, total_prompts: int) -> dict[str, object]:
    resident_mib = candidate.resident_bytes / MIB
    return {
        "kind": candidate.kind,
        "name": candidate.name,
        "layer": candidate.layer,
        "role": candidate.role,
        "source_idx": candidate.source_idx,
        "offset_start": candidate.offset_start,
        "offset_end": candidate.offset_end,
        "entries": len(candidate.keys),
        "resident_mib": f"{resident_mib:.2f}",
        "read_gib": f"{candidate.read_bytes / GIB:.3f}",
        "prompts": len(candidate.prompts),
        "prompt_coverage": f"{len(candidate.prompts) / max(total_prompts, 1):.3f}",
        "rows": candidate.rows,
        "hit_batches": len(candidate.batches),
        "dominant_batches": candidate.dominant_batches,
        "ram_only_batches": candidate.ram_only_batches,
        "mixed_risk_batches": candidate.mixed_risk_batches,
        "proportional_wait_ms": f"{candidate.proportional_wait_ms:.3f}",
        "full_or_dominant_wait_ms": f"{candidate.full_hit_wait_ms:.3f}",
        "wait_ms_per_gib_resident": f"{candidate.proportional_wait_ms / max(candidate.resident_bytes / GIB, 1e-9):.3f}",
        "read_gib_per_gib_resident": f"{(candidate.read_bytes / GIB) / max(candidate.resident_bytes / GIB, 1e-9):.3f}",
        "vram_overlap_entries": candidate.vram_overlap_entries,
        "vram_overlap_mib": f"{candidate.vram_overlap_bytes / MIB:.2f}",
        "resident_after_vram_overlap_mib": f"{max(candidate.resident_bytes - candidate.vram_overlap_bytes, 0) / MIB:.2f}",
    }


def write_csv(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(path: pathlib.Path, rows: list[dict[str, object]], metadata: dict[str, object]) -> None:
    def top_by_kind(kind: str, limit: int = 12) -> list[dict[str, object]]:
        selected = [row for row in rows if row["kind"] == kind]
        selected.sort(key=lambda row: float(row["proportional_wait_ms_per_token"]), reverse=True)
        return selected[:limit]

    def budget_table(kind: str, budgets_mib: list[int]) -> list[str]:
        selected = [row for row in rows if row["kind"] == kind]
        selected.sort(key=lambda row: float(row["wait_ms_per_gib_resident"]), reverse=True)
        out = [
            f"### Greedy `{kind}` Budget Bound",
            "",
            "| budget MiB | selected | resident MiB | wait ms/token |",
            "|---:|---:|---:|---:|",
        ]
        for budget in budgets_mib:
            used = 0.0
            wait = 0.0
            count = 0
            for row in selected:
                size = float(row["resident_mib"])
                if used + size > budget:
                    continue
                used += size
                wait += float(row["proportional_wait_ms_per_token"])
                count += 1
            out.append(f"| {budget} | {count} | {used:.1f} | {wait:.2f} |")
        out.append("")
        return out

    lines = [
        "# Kimi Phase 5C RAM slab screen",
        "",
        "This is a dev-only offline screen. It does not use held-out prompts and does not claim a SOTA result.",
        "",
        "## Inputs",
        "",
        f"- prompts: `{metadata['prompt_count']}`",
        f"- decode-like rows: `{metadata['row_count']}`",
        f"- decode-like batches: `{metadata['batch_count']}`",
        f"- decode runs: `{metadata['decode_runs']}`",
        f"- total batch wait: `{metadata['total_wait_ms']:.3f} ms`",
        f"- total batch wait per decode token: `{metadata['total_wait_ms_per_token']:.3f} ms/token`",
        f"- max jobs: `{metadata['max_jobs']}`",
        f"- simulated VRAM hotset entries: `{metadata['vram_hotset_entries']}`",
        "",
        "## Top Candidates",
        "",
        "| kind | name | prompts | resident MiB | read GiB | wait ms/token | wait/GiB | hit batches | ram-only | mixed risk | VRAM overlap MiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows[:40]:
        lines.append(
            f"| `{row['kind']}` | `{row['name']}` | {row['prompts']} | {row['resident_mib']} | "
            f"{row['read_gib']} | {row['proportional_wait_ms_per_token']} | {row['wait_ms_per_gib_resident']} | "
            f"{row['hit_batches']} | {row['ram_only_batches']} | {row['mixed_risk_batches']} | {row['vram_overlap_mib']} |"
        )
    lines.extend([
        "",
        "## Layer/Role Slab Upper Bound",
        "",
        "These rows are more relevant for a pageable RAM slab than the tiny contiguous windows because they turn whole layer/role demand batches into RAM hits.",
        "",
        "| kind | name | prompts | resident MiB | wait ms/token | hit batches | ram-only | mixed risk | VRAM overlap MiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for kind in ("layer_upgate", "layer_role", "layer_all"):
        for row in top_by_kind(kind, 8):
            lines.append(
                f"| `{row['kind']}` | `{row['name']}` | {row['prompts']} | {row['resident_mib']} | "
                f"{row['proportional_wait_ms_per_token']} | {row['hit_batches']} | {row['ram_only_batches']} | "
                f"{row['mixed_risk_batches']} | {row['vram_overlap_mib']} |"
            )
    lines.extend([""])
    lines.extend(budget_table("layer_upgate", [2048, 4096, 8192, 10240]))
    lines.extend(budget_table("layer_all", [4096, 8192, 10240]))
    lines.extend([
        "",
        "## Decision Notes",
        "",
        "- Prefer candidates with high `wait/GiB`, broad dev prompt coverage, low VRAM overlap, and many dominant/ram-only batches.",
        "- Treat high `mixed_risk_batches` as a fragmentation warning: those candidates can reduce SSD bytes while making remaining SSD batches smaller and less efficient.",
        "- If the greedy budget bound is only a few tens of ms/token, RAM slabs alone cannot bridge the current gap to `2 tok/s`; they should be treated as a small candidate or rejected in favor of lower-byte expert representation.",
        "- Runtime A/B should start with pageable RAM and must reject candidates that raise TTFT, refaults, direct reclaim, or aggregate demand wait.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def load_decode_runs(trace_paths: list[pathlib.Path]) -> int:
    total = 0
    for path in trace_paths:
        metrics = path.parent / "metrics.json"
        if not metrics.exists():
            continue
        try:
            data = json.loads(metrics.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError:
            continue
        total += int(float(data.get("decode_runs", 0) or 0))
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description="Screen dev-only Kimi RAM slab candidates from foreground IO traces.")
    ap.add_argument("--input-root", action="append", type=pathlib.Path, default=[])
    ap.add_argument("--io-read-trace", action="append", type=pathlib.Path, default=[])
    ap.add_argument("--route-profile-root", action="append", type=pathlib.Path, default=[])
    ap.add_argument("--route-profile", action="append", type=pathlib.Path, default=[])
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--roles", default="up,gate,down")
    ap.add_argument("--max-jobs", type=int, default=8)
    ap.add_argument("--down-slots", type=int, default=723)
    ap.add_argument("--upgate-slots", type=int, default=1735)
    ap.add_argument("--split-max-mib", type=float, default=6.0)
    ap.add_argument("--contig-window-mib", type=int, nargs="+", default=[512, 1024, 2048])
    ap.add_argument("--contig-top-per-source", type=int, default=8)
    ap.add_argument("--exclude-prompt-regex", default=r"heldout|test_")
    args = ap.parse_args()

    deny = re.compile(args.exclude_prompt_regex)
    roles = {role.strip() for role in args.roles.split(",") if role.strip()}
    traces = discover_io_roots(args.input_root + args.io_read_trace, deny)
    if not traces:
        raise SystemExit("No io-read-trace.csv inputs")

    read_rows, batches, expert_bytes = load_reads(traces, args.max_jobs, roles)
    rows_by_key = index_rows_by_key(read_rows)
    route_rows = read_route_profiles(args.route_profile_root + args.route_profile)
    vram_hotset = simulate_vram_hotset(route_rows, args.down_slots, args.upgate_slots, args.split_max_mib) if route_rows else set()

    candidates = unit_candidates(read_rows, expert_bytes)
    candidates.extend(contiguous_candidates(read_rows, expert_bytes, args.contig_window_mib, args.contig_top_per_source))
    for candidate in candidates:
        score_candidate(candidate, rows_by_key, batches, expert_bytes, vram_hotset)

    prompt_count = len({path.parent.name for path in traces})
    decode_runs = load_decode_runs(traces)
    candidate_rows = [candidate_row(candidate, prompt_count) for candidate in candidates if candidate.rows > 0 and candidate.resident_bytes > 0]
    for row in candidate_rows:
        row["proportional_wait_ms_per_token"] = f"{float(row['proportional_wait_ms']) / max(decode_runs, 1):.3f}"
        row["full_or_dominant_wait_ms_per_token"] = f"{float(row['full_or_dominant_wait_ms']) / max(decode_runs, 1):.3f}"
    candidate_rows.sort(
        key=lambda row: (
            float(row["wait_ms_per_gib_resident"]),
            int(row["prompts"]),
            float(row["proportional_wait_ms"]),
        ),
        reverse=True,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "candidate_screen.csv", candidate_rows)
    metadata = {
        "trace_count": len(traces),
        "prompt_count": prompt_count,
        "row_count": len(read_rows),
        "batch_count": len(batches),
        "decode_runs": decode_runs,
        "total_wait_ms": sum(batch.wait_ms for batch in batches.values()),
        "total_wait_ms_per_token": sum(batch.wait_ms for batch in batches.values()) / max(decode_runs, 1),
        "max_jobs": args.max_jobs,
        "roles": sorted(roles),
        "route_profile_rows": len(route_rows),
        "vram_hotset_entries": len(vram_hotset),
        "inputs": [str(path) for path in traces],
        "route_inputs": [str(path) for path in args.route_profile_root + args.route_profile],
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(args.out_dir / "report.md", candidate_rows, metadata)


if __name__ == "__main__":
    main()
