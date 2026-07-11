#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_csv_set(text: str) -> set[str]:
    return {item.strip() for item in text.split(",") if item.strip()}


def main() -> int:
    ap = argparse.ArgumentParser(description="Create a Kimi RAM-tier profile for full layer/role expert slabs from alias TSV.")
    ap.add_argument("--alias-tsv", type=Path, required=True)
    ap.add_argument("--layers", required=True, help="Comma-separated layer ids, e.g. 1 or 1,4.")
    ap.add_argument("--roles", required=True, help="Comma-separated roles: up,gate,down.")
    ap.add_argument("--count", type=int, default=1, help="Profile count assigned to each selected expert.")
    ap.add_argument("--out-profile", type=Path, required=True)
    ap.add_argument("--out-report", type=Path, required=True)
    args = ap.parse_args()

    layers = {int(item) for item in parse_csv_set(args.layers)}
    roles = parse_csv_set(args.roles)
    if not layers:
        raise SystemExit("--layers selected no layers")
    if not roles:
        raise SystemExit("--roles selected no roles")

    selected = []
    with args.alias_tsv.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                layer = int(row["layer"])
                expert_idx = int(row["expert_idx"])
                nbytes = int(row["nbytes"])
            except (KeyError, ValueError) as exc:
                raise SystemExit(f"bad alias row: {row}") from exc
            role = row.get("kind", "")
            if layer not in layers or role not in roles:
                continue
            selected.append(
                {
                    "layer": layer,
                    "role": role,
                    "tensor": row["tensor"],
                    "expert_idx": expert_idx,
                    "expert_bytes": nbytes,
                    "source_path": row.get("source_path", ""),
                    "offset": int(row.get("offset", "0") or 0),
                    "type": row.get("type", ""),
                    "shard_index": row.get("shard_index", ""),
                }
            )

    selected.sort(key=lambda row: (row["layer"], row["role"], row["expert_idx"]))
    if not selected:
        raise SystemExit("no selected experts")

    args.out_profile.parent.mkdir(parents=True, exist_ok=True)
    cumulative = 0
    with args.out_profile.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["rank", "count", "expert_bytes", "cumulative_bytes", "tensor_base", "expert_idx", "tensor"])
        for rank, row in enumerate(selected, 1):
            cumulative += row["expert_bytes"]
            writer.writerow([rank, args.count, row["expert_bytes"], cumulative, "0x0", row["expert_idx"], row["tensor"]])

    by_role: dict[str, dict[str, int]] = {}
    for row in selected:
        key = f"blk.{row['layer']}.{row['role']}"
        bucket = by_role.setdefault(key, {"entries": 0, "bytes": 0})
        bucket["entries"] += 1
        bucket["bytes"] += row["expert_bytes"]

    report = {
        "kind": "kimi_layer_ram_profile_from_alias",
        "alias_tsv": str(args.alias_tsv),
        "layers": sorted(layers),
        "roles": sorted(roles),
        "entries": len(selected),
        "bytes": cumulative,
        "mib": cumulative / (1024 * 1024),
        "gib": cumulative / (1024 ** 3),
        "by_role": by_role,
        "out_profile": str(args.out_profile),
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out_profile} entries={len(selected)} mib={report['mib']:.2f}")
    print(f"wrote {args.out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
