#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def kind_from_tensor(name: str) -> str:
    if ".ffn_up_exps." in name:
        return "up"
    if ".ffn_gate_exps." in name:
        return "gate"
    if ".ffn_down_exps." in name:
        return "down"
    return "other"


def load_manifest(path: Path, kinds: set[str]) -> dict[tuple[str, int], dict]:
    out: dict[tuple[str, int], dict] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["kind"] not in kinds:
                continue
            key = (row["tensor"], int(row["expert_idx"]))
            out[key] = {
                "remote_nbytes": int(row["remote_nbytes"]),
                "remote_type": row["remote_type"],
                "kind": row["kind"],
            }
    return out


def analyze_profile(path: Path, manifest: dict[tuple[str, int], dict], kinds: set[str]) -> dict:
    by_kind = {
        kind: {
            "events": 0,
            "covered_events": 0,
            "current_bytes": 0,
            "hybrid_bytes": 0,
            "covered_current_bytes": 0,
            "covered_remote_bytes": 0,
        }
        for kind in sorted(kinds)
    }
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tensor = row["tensor"]
            kind = kind_from_tensor(tensor)
            if kind not in kinds:
                continue
            expert = int(row["expert_idx"])
            count = int(row["count"])
            current_nbytes = int(row["expert_bytes"])
            current = count * current_nbytes
            key = (tensor, expert)
            bucket = by_kind[kind]
            bucket["events"] += count
            bucket["current_bytes"] += current
            hit = manifest.get(key)
            if hit:
                remote = count * int(hit["remote_nbytes"])
                bucket["covered_events"] += count
                bucket["covered_current_bytes"] += current
                bucket["covered_remote_bytes"] += remote
                bucket["hybrid_bytes"] += remote
            else:
                bucket["hybrid_bytes"] += current

    total = {
        "events": 0,
        "covered_events": 0,
        "current_bytes": 0,
        "hybrid_bytes": 0,
        "covered_current_bytes": 0,
        "covered_remote_bytes": 0,
    }
    for bucket in by_kind.values():
        for key in total:
            total[key] += bucket[key]

    def finalize(bucket: dict) -> None:
        bucket["event_coverage"] = bucket["covered_events"] / bucket["events"] if bucket["events"] else 0.0
        bucket["byte_coverage"] = (
            bucket["covered_current_bytes"] / bucket["current_bytes"] if bucket["current_bytes"] else 0.0
        )
        bucket["hybrid_byte_ratio"] = bucket["hybrid_bytes"] / bucket["current_bytes"] if bucket["current_bytes"] else 0.0
        bucket["covered_remote_to_current_ratio"] = (
            bucket["covered_remote_bytes"] / bucket["covered_current_bytes"]
            if bucket["covered_current_bytes"] else 0.0
        )

    for bucket in by_kind.values():
        finalize(bucket)
    finalize(total)
    return {
        "profile": str(path),
        "prompt": path.parent.name,
        "by_kind": by_kind,
        "total": total,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate lower-byte subpack coverage from Kimi route-profile CSVs.")
    parser.add_argument("--manifest-tsv", type=Path, required=True)
    parser.add_argument("--profile", type=Path, action="append", required=True)
    parser.add_argument("--kind", default="up,gate")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--md", type=Path, required=True)
    args = parser.parse_args()

    kinds = {x.strip() for x in args.kind.split(",") if x.strip()}
    manifest = load_manifest(args.manifest_tsv, kinds)
    results = [analyze_profile(path, manifest, kinds) for path in args.profile]

    aggregate = {
        "events": 0,
        "covered_events": 0,
        "current_bytes": 0,
        "hybrid_bytes": 0,
        "covered_current_bytes": 0,
        "covered_remote_bytes": 0,
    }
    for result in results:
        for key in aggregate:
            aggregate[key] += result["total"][key]
    aggregate["event_coverage"] = aggregate["covered_events"] / aggregate["events"] if aggregate["events"] else 0.0
    aggregate["byte_coverage"] = (
        aggregate["covered_current_bytes"] / aggregate["current_bytes"] if aggregate["current_bytes"] else 0.0
    )
    aggregate["hybrid_byte_ratio"] = aggregate["hybrid_bytes"] / aggregate["current_bytes"] if aggregate["current_bytes"] else 0.0
    aggregate["covered_remote_to_current_ratio"] = (
        aggregate["covered_remote_bytes"] / aggregate["covered_current_bytes"]
        if aggregate["covered_current_bytes"] else 0.0
    )

    report = {
        "manifest_tsv": str(args.manifest_tsv),
        "kinds": sorted(kinds),
        "manifest_entries": len(manifest),
        "profiles": [str(p) for p in args.profile],
        "aggregate": aggregate,
        "results": results,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Kimi lower-byte subpack coverage",
        "",
        f"- Manifest: `{args.manifest_tsv}`",
        f"- Kinds: `{','.join(sorted(kinds))}`",
        f"- Manifest entries: `{len(manifest)}`",
        "",
        "## Aggregate",
        "",
        "| events | event coverage | byte coverage | hybrid byte ratio | current GiB | hybrid GiB |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {aggregate['events']} | {aggregate['event_coverage']:.4f} | "
            f"{aggregate['byte_coverage']:.4f} | {aggregate['hybrid_byte_ratio']:.4f} | "
            f"{aggregate['current_bytes'] / 1024**3:.3f} | {aggregate['hybrid_bytes'] / 1024**3:.3f} |"
        ),
        "",
        "## Per Prompt",
        "",
        "| prompt | events | event coverage | byte coverage | hybrid byte ratio | current GiB | hybrid GiB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        total = result["total"]
        lines.append(
            f"| {result['prompt']} | {total['events']} | {total['event_coverage']:.4f} | "
            f"{total['byte_coverage']:.4f} | {total['hybrid_byte_ratio']:.4f} | "
            f"{total['current_bytes'] / 1024**3:.3f} | {total['hybrid_bytes'] / 1024**3:.3f} |"
        )
    args.md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"profiles={len(results)}")
    print(f"manifest_entries={len(manifest)}")
    print(f"events={aggregate['events']}")
    print(f"event_coverage={aggregate['event_coverage']:.6f}")
    print(f"byte_coverage={aggregate['byte_coverage']:.6f}")
    print(f"hybrid_byte_ratio={aggregate['hybrid_byte_ratio']:.6f}")
    print(f"current_gib={aggregate['current_bytes'] / 1024**3:.3f}")
    print(f"hybrid_gib={aggregate['hybrid_bytes'] / 1024**3:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
