#!/usr/bin/env python3
"""Preflight selected IQ1_S/Q2_K Kimi expert overlays against the IQ3_S model.

This is a metadata gate only. It does not build a pack, download weights, or
change runtime behavior.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import time
from pathlib import Path


QK_K = 256
TYPE_BYTES = {
    "IQ1_S": 2 + QK_K // 8 + QK_K // 16,
    "Q2_K": 2 + 2 + QK_K // 16 + QK_K // 4,
}
CURRENT_FAST_PATH_TYPES = {"IQ3_XXS", "IQ3_S", "IQ2_S", "MXFP4", "F8_E4M3_B128"}


def parse_shape(text: str) -> list[int]:
    if "," in text:
        return [int(part) for part in text.split(",") if part.strip()]
    return [int(part) for part in text.split("x") if part.strip()]


def remote_expert_bytes(remote_type: str, shape: list[int]) -> tuple[int, int]:
    if remote_type not in TYPE_BYTES:
        return 0, 0
    if len(shape) < 2:
        return 0, 0
    ne00 = int(shape[0])
    ne01 = int(shape[1])
    if ne00 <= 0 or ne01 <= 0 or ne00 % QK_K != 0:
        return 0, 0
    nb01 = (ne00 // QK_K) * TYPE_BYTES[remote_type]
    return nb01 * ne01, nb01


def load_inventory(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            tensor = row["tensor"]
            shape = parse_shape(row["shape"])
            if len(shape) < 3:
                raise RuntimeError(f"inventory tensor has invalid shape: {tensor} shape={row['shape']}")
            out[tensor] = {
                "layer": int(row["layer"]),
                "kind": row["kind"],
                "current_type": row["type"],
                "current_tensor_bytes": int(row["tensor_bytes"]),
                "current_expert_bytes": int(row["expert_bytes"]),
                "n_experts": int(row["n_experts"]),
                "entries": int(row["entries"]),
                "shard_index": int(row["shard_index"]),
                "current_data_offset": int(row["data_offset"]),
                "shape": shape,
                "shape_text": row["shape"],
                "shard_path": row["shard_path"],
            }
    return out


def input_fieldnames(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        return next(reader)


def get_optional_int(row: dict, names: list[str]) -> int | None:
    for name in names:
        value = row.get(name)
        if value is None or value == "":
            continue
        return int(value)
    return None


def selected_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def classify(row: dict, inv: dict, source_fields: list[str]) -> dict:
    tensor = row.get("tensor", "")
    expert_idx = int(row.get("expert_idx", "-1"))
    current_nbytes = int(row.get("current_nbytes", "0"))
    remote_nbytes = int(row.get("remote_nbytes", "0"))
    remote_type = row.get("remote_type", "")
    source_pack = row.get("source_pack", "")
    source_offset = get_optional_int(row, ["source_offset", "remote_offset", "remote_abs_start", "model_offset"])

    errors: list[str] = []
    current = inv.get(tensor)
    if current is None:
        errors.append("missing_inventory")
        shape = []
        expected_remote_nbytes = 0
        override_nb01 = 0
        current_type = ""
        current_expert_bytes = 0
        current_data_offset = 0
        current_shard_path = ""
        layer = int(row.get("layer", "-1") or -1)
        kind = row.get("kind", "")
        n_experts = 0
    else:
        shape = current["shape"]
        layer = current["layer"]
        kind = current["kind"]
        current_type = current["current_type"]
        current_expert_bytes = current["current_expert_bytes"]
        current_data_offset = current["current_data_offset"]
        current_shard_path = current["shard_path"]
        n_experts = current["n_experts"]
        expected_remote_nbytes, override_nb01 = remote_expert_bytes(remote_type, shape)
        if int(row.get("layer", layer)) != layer:
            errors.append("layer_mismatch")
        if row.get("kind", kind) != kind:
            errors.append("kind_mismatch")
        if not (0 <= expert_idx < n_experts):
            errors.append("expert_idx_out_of_range")
        if current_nbytes != current_expert_bytes:
            errors.append("current_nbytes_mismatch")
        if remote_type not in TYPE_BYTES:
            errors.append("unsupported_override_type")
        if expected_remote_nbytes == 0:
            errors.append("bad_override_shape")
        if expected_remote_nbytes and remote_nbytes != expected_remote_nbytes:
            errors.append("remote_nbytes_mismatch")

    if source_offset is None:
        errors.append("missing_source_offset")
    if remote_type not in CURRENT_FAST_PATH_TYPES:
        errors.append("fast_path_type_not_admitted_now")
    if remote_nbytes != current_nbytes:
        errors.append("runtime_nbytes_differs_from_current")

    shape_ok = not any(err in errors for err in {
        "missing_inventory",
        "layer_mismatch",
        "kind_mismatch",
        "expert_idx_out_of_range",
        "current_nbytes_mismatch",
        "unsupported_override_type",
        "bad_override_shape",
        "remote_nbytes_mismatch",
    })
    runtime_ready_now = not errors
    if runtime_ready_now:
        decision = "runtime_ready_now"
    elif shape_ok:
        decision = "metadata_ok_runtime_blocked"
    else:
        decision = "reject_metadata"

    current_expert_offset = current_data_offset + expert_idx * current_expert_bytes if current else 0
    return {
        "decision": decision,
        "reject_reasons": ";".join(errors),
        "shape_ok": int(shape_ok),
        "runtime_ready_now": int(runtime_ready_now),
        "tensor": tensor,
        "expert_idx": expert_idx,
        "layer": layer,
        "kind": kind,
        "shape": ",".join(str(x) for x in shape),
        "ne00": shape[0] if len(shape) > 0 else 0,
        "ne01": shape[1] if len(shape) > 1 else 0,
        "n_experts": n_experts,
        "current_type": current_type,
        "current_nbytes": current_nbytes,
        "current_expected_nbytes": current_expert_bytes,
        "current_tensor_offset": current_data_offset,
        "current_expert_offset": current_expert_offset,
        "current_shard_path": current_shard_path,
        "override_type": remote_type,
        "override_nbytes": remote_nbytes,
        "override_expected_nbytes": expected_remote_nbytes,
        "override_nb01": override_nb01,
        "source_pack": source_pack,
        "source_offset": "" if source_offset is None else source_offset,
        "source_offset_available": int(source_offset is not None),
        "input_has_source_offset_field": int(any(name in source_fields for name in ["source_offset", "remote_offset", "remote_abs_start", "model_offset"])),
        "fast_path_admitted_now": int(remote_type in CURRENT_FAST_PATH_TYPES),
        "nbytes_match_current": int(remote_nbytes == current_nbytes),
        "byte_ratio": remote_nbytes / current_nbytes if current_nbytes > 0 else 0.0,
    }


def summarize(rows: list[dict]) -> dict:
    by_decision = collections.Counter(row["decision"] for row in rows)
    reason_counts: collections.Counter[str] = collections.Counter()
    by_type = collections.defaultdict(lambda: {"entries": 0, "current_bytes": 0, "override_bytes": 0})
    by_kind = collections.defaultdict(lambda: {"entries": 0, "current_bytes": 0, "override_bytes": 0})
    for row in rows:
        for reason in str(row["reject_reasons"]).split(";"):
            if reason:
                reason_counts[reason] += 1
        for bucket in (by_type[row["override_type"]], by_kind[row["kind"]]):
            bucket["entries"] += 1
            bucket["current_bytes"] += int(row["current_nbytes"])
            bucket["override_bytes"] += int(row["override_nbytes"])
    for bucket in list(by_type.values()) + list(by_kind.values()):
        bucket["byte_ratio"] = bucket["override_bytes"] / bucket["current_bytes"] if bucket["current_bytes"] else 0.0
    total_current = sum(int(row["current_nbytes"]) for row in rows)
    total_override = sum(int(row["override_nbytes"]) for row in rows)
    return {
        "entries": len(rows),
        "metadata_ok_entries": sum(1 for row in rows if row["shape_ok"]),
        "runtime_ready_now_entries": sum(1 for row in rows if row["runtime_ready_now"]),
        "source_offset_available_entries": sum(1 for row in rows if row["source_offset_available"]),
        "current_bytes": total_current,
        "override_bytes": total_override,
        "byte_ratio": total_override / total_current if total_current else 0.0,
        "by_decision": dict(sorted(by_decision.items())),
        "reason_counts": dict(reason_counts.most_common()),
        "by_override_type": dict(sorted(by_type.items())),
        "by_kind": dict(sorted(by_kind.items())),
    }


def write_manifest(path: Path, rows: list[dict]) -> None:
    fields = [
        "decision", "reject_reasons", "shape_ok", "runtime_ready_now",
        "tensor", "expert_idx", "layer", "kind", "shape", "ne00", "ne01", "n_experts",
        "current_type", "current_nbytes", "current_expected_nbytes",
        "current_tensor_offset", "current_expert_offset", "current_shard_path",
        "override_type", "override_nbytes", "override_expected_nbytes", "override_nb01",
        "source_pack", "source_offset", "source_offset_available",
        "fast_path_admitted_now", "nbytes_match_current", "byte_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def write_markdown(path: Path, result: dict) -> None:
    s = result["summary"]
    lines = [
        "# Kimi IQ1_S mixed-type overlay preflight",
        "",
        "This is a metadata-only gate. It does not download weights, build an expert pack, modify runtime behavior, or claim SOTA.",
        "",
        f"- generated_at: `{result['generated_at']}`",
        f"- inventory_tsv: `{result['inventory_tsv']}`",
        f"- selected_plan_tsv: `{result['selected_plan_tsv']}`",
        f"- manifest_tsv: `{result['manifest_tsv']}`",
        "",
        "## Summary",
        "",
        f"- entries: `{s['entries']}`",
        f"- metadata OK entries: `{s['metadata_ok_entries']}`",
        f"- runtime-ready-now entries: `{s['runtime_ready_now_entries']}`",
        f"- source-offset available entries: `{s['source_offset_available_entries']}`",
        f"- selected byte ratio: `{s['byte_ratio']:.4f}`",
        "",
        "## Decisions",
        "",
        "| decision | entries |",
        "|---|---:|",
    ]
    for key, value in s["by_decision"].items():
        lines.append(f"| `{key}` | {value} |")
    lines += [
        "",
        "## Blockers",
        "",
        "| reason | entries |",
        "|---|---:|",
    ]
    for key, value in s["reason_counts"].items():
        lines.append(f"| `{key}` | {value} |")
    lines += [
        "",
        "## By Override Type",
        "",
        "| type | entries | current GiB | override GiB | byte ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, value in s["by_override_type"].items():
        lines.append(
            f"| `{key}` | {value['entries']} | {value['current_bytes'] / 1024**3:.3f} | "
            f"{value['override_bytes'] / 1024**3:.3f} | {value['byte_ratio']:.4f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- `metadata_ok_runtime_blocked` means tensor shape, expert index, and override byte size are internally consistent.",
        "- `missing_source_offset` means this selected-plan TSV is not enough to read the override payload; a range builder must add source offsets or build a payload pack.",
        "- `fast_path_type_not_admitted_now` means current runtime would not dispatch this override type through the existing Kimi one-stream fast path.",
        "- `runtime_nbytes_differs_from_current` confirms the normal one-pack `(tensor, expert, nbytes)` lookup cannot use this as a drop-in replacement.",
        "",
        "## Reproduce",
        "",
        "```bash",
        result["reproduce_command"],
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight Kimi selected IQ1_S/Q2_K overlay metadata.")
    parser.add_argument("--inventory-tsv", type=Path, required=True)
    parser.add_argument("--selected-plan-tsv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    inv = load_inventory(args.inventory_tsv)
    source_fields = input_fieldnames(args.selected_plan_tsv)
    rows = [classify(row, inv, source_fields) for row in selected_rows(args.selected_plan_tsv)]
    rows.sort(key=lambda row: (row["layer"], row["kind"], row["tensor"], row["expert_idx"]))
    summary = summarize(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "mixed-type-preflight-manifest.tsv"
    json_path = args.out_dir / "report.json"
    md_path = args.out_dir / "report.md"
    write_manifest(manifest_path, rows)

    result = {
        "kind": "kimi_iq1s_mixed_type_preflight",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "inventory_tsv": str(args.inventory_tsv),
        "selected_plan_tsv": str(args.selected_plan_tsv),
        "manifest_tsv": str(manifest_path),
        "summary": summary,
        "reproduce_command": " ".join([
            ".Agent/run-tools/kimi_iq1s_mixed_type_preflight.py",
            f"--inventory-tsv {args.inventory_tsv}",
            f"--selected-plan-tsv {args.selected_plan_tsv}",
            f"--out-dir {args.out_dir}",
        ]),
    }
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(md_path, result)
    print(md_path)
    print(f"entries={summary['entries']}")
    print(f"metadata_ok={summary['metadata_ok_entries']}")
    print(f"runtime_ready_now={summary['runtime_ready_now_entries']}")
    print(f"byte_ratio={summary['byte_ratio']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
