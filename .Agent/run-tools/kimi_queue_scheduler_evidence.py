#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
from collections import defaultdict


LAYER_RE = re.compile(r"blk\.(\d+)\.")


def fnum(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def inum(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_csv(path: pathlib.Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def read_metrics(run_dir: pathlib.Path):
    path = run_dir / "metrics.json"
    if path.exists():
        try:
            return json.loads(path.read_text(errors="replace"))
        except json.JSONDecodeError:
            return {}
    return {}


def role_from_tensor(name: str) -> str:
    if ".ffn_up_exps." in name:
        return "up"
    if ".ffn_gate_exps." in name:
        return "gate"
    if ".ffn_down_exps." in name:
        return "down"
    return "other"


def layer_from_tensor(name: str):
    m = LAYER_RE.search(name or "")
    return int(m.group(1)) if m else None


def bucket_jobs(n: int) -> str:
    if n == 0:
        return "0"
    if n == 1:
        return "1"
    if n <= 4:
        return "2-4"
    if n <= 8:
        return "5-8"
    if n <= 16:
        return "9-16"
    if n <= 32:
        return "17-32"
    return "gt32"


def quantile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[idx]


def add(acc, key, row, fields):
    dst = acc[key]
    dst["rows"] += 1
    for field in fields:
        dst[field] += fnum(row.get(field))
    return dst


def fmt(v):
    return f"{v:.3f}"


def write_csv(path: pathlib.Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def summarize_copy(run_dir: pathlib.Path):
    rows = read_csv(run_dir / "copy-profile.csv")
    by_role = defaultdict(lambda: defaultdict(float))
    by_tensor = defaultdict(lambda: defaultdict(float))
    waits = []
    for row in rows:
        tensor = row.get("tensor", "")
        role = role_from_tensor(tensor)
        fields = ["bytes", "slot_wait_ms", "host_ms", "io_wait_ms", "enqueue_ms", "h2d_ms", "wall_ms"]
        add(by_role, role, row, fields)
        add(by_tensor, tensor, row, fields)
        by_role[role]["iouring"] += inum(row.get("iouring"))
        by_role[role]["ram_hit"] += inum(row.get("ram_hit"))
        by_role[role]["pack_hit"] += inum(row.get("pack_hit"))
        by_tensor[tensor]["iouring"] += inum(row.get("iouring"))
        waits.append(fnum(row.get("io_wait_ms")))
    role_rows = []
    for role, c in by_role.items():
        rows_n = c["rows"] or 1
        role_rows.append({
            "role": role,
            "rows": int(c["rows"]),
            "bytes_gib": fmt(c["bytes"] / 1024**3),
            "io_wait_ms": fmt(c["io_wait_ms"]),
            "io_wait_ms_per_row": fmt(c["io_wait_ms"] / rows_n),
            "h2d_ms": fmt(c["h2d_ms"]),
            "h2d_ms_per_row": fmt(c["h2d_ms"] / rows_n),
            "slot_wait_ms": fmt(c["slot_wait_ms"]),
            "enqueue_ms": fmt(c["enqueue_ms"]),
            "wall_ms": fmt(c["wall_ms"]),
            "iouring_rows": int(c["iouring"]),
            "ram_hit_rows": int(c["ram_hit"]),
        })
    role_rows.sort(key=lambda r: float(r["io_wait_ms"]), reverse=True)
    tensor_rows = []
    for tensor, c in by_tensor.items():
        rows_n = c["rows"] or 1
        tensor_rows.append({
            "tensor": tensor,
            "layer": layer_from_tensor(tensor) if layer_from_tensor(tensor) is not None else "",
            "role": role_from_tensor(tensor),
            "rows": int(c["rows"]),
            "bytes_gib": fmt(c["bytes"] / 1024**3),
            "io_wait_ms": fmt(c["io_wait_ms"]),
            "io_wait_ms_per_row": fmt(c["io_wait_ms"] / rows_n),
            "h2d_ms": fmt(c["h2d_ms"]),
            "wall_ms": fmt(c["wall_ms"]),
            "iouring_rows": int(c["iouring"]),
        })
    tensor_rows.sort(key=lambda r: float(r["io_wait_ms"]), reverse=True)
    return {
        "rows": rows,
        "role_rows": role_rows,
        "tensor_rows": tensor_rows,
        "wait_p50": quantile(waits, 0.50),
        "wait_p95": quantile(waits, 0.95),
        "wait_p99": quantile(waits, 0.99),
    }


def summarize_io_batch(run_dir: pathlib.Path, max_read_jobs: int | None = None):
    rows = read_csv(run_dir / "io-batch-profile.csv")
    if max_read_jobs is not None:
        rows = [r for r in rows if inum(r.get("read_jobs")) <= max_read_jobs]
    by_op = defaultdict(lambda: defaultdict(float))
    by_first_role = defaultdict(lambda: defaultdict(float))
    hist = defaultdict(int)
    batch_seqs = set()
    for row in rows:
        batch_seqs.add(inum(row.get("seq")))
        op = row.get("op", "")
        first = row.get("first_tensor", "")
        role = role_from_tensor(first)
        fields = [
            "jobs", "read_jobs", "ram_or_prefetch_jobs", "ram_or_prefetch_bytes",
            "initial_submit_jobs", "submit_calls", "wait_calls", "cqes",
            "slot_wait_ms", "submit_ms", "wait_ms", "enqueue_ms", "wall_ms",
        ]
        add(by_op, op, row, fields)
        add(by_first_role, role, row, fields)
        by_op[op]["inflight_avg_sum"] += fnum(row.get("inflight_avg"))
        by_op[op]["inflight_max"] = max(by_op[op]["inflight_max"], fnum(row.get("inflight_max")))
        by_first_role[role]["inflight_avg_sum"] += fnum(row.get("inflight_avg"))
        by_first_role[role]["inflight_max"] = max(by_first_role[role]["inflight_max"], fnum(row.get("inflight_max")))
        hist[bucket_jobs(inum(row.get("read_jobs")))] += 1
    def rows_from(acc, key_name):
        out = []
        for key, c in acc.items():
            rows_n = c["rows"] or 1
            read_jobs = c["read_jobs"] or 1
            out.append({
                key_name: key,
                "rows": int(c["rows"]),
                "jobs": int(c["jobs"]),
                "read_jobs": int(c["read_jobs"]),
                "avg_read_jobs": fmt(c["read_jobs"] / rows_n),
                "avg_initial_submit": fmt(c["initial_submit_jobs"] / rows_n),
                "wait_calls": int(c["wait_calls"]),
                "wait_ms": fmt(c["wait_ms"]),
                "wait_ms_per_read_job": fmt(c["wait_ms"] / read_jobs),
                "wall_ms": fmt(c["wall_ms"]),
                "submit_ms": fmt(c["submit_ms"]),
                "slot_wait_ms": fmt(c["slot_wait_ms"]),
                "inflight_avg": fmt(c["inflight_avg_sum"] / rows_n),
                "inflight_max": int(c["inflight_max"]),
            })
        out.sort(key=lambda r: float(r["wait_ms"]), reverse=True)
        return out
    return {
        "rows": rows,
        "op_rows": rows_from(by_op, "op"),
        "role_rows": rows_from(by_first_role, "first_role"),
        "hist": dict(hist),
        "batch_seqs": batch_seqs,
    }


def summarize_io_wait(run_dir: pathlib.Path, batch_seqs: set[int] | None = None):
    rows = read_csv(run_dir / "io-wait-trace.csv")
    if batch_seqs is not None:
        rows = [r for r in rows if inum(r.get("batch_seq")) in batch_seqs]
    waits = []
    total = defaultdict(float)
    top = []
    for row in rows:
        wait = fnum(row.get("wait_ms"))
        waits.append(wait)
        total["rows"] += 1
        total["wait_ms"] += wait
        total["drained_cqes"] += inum(row.get("drained_cqes"))
        total["low_inflight"] += 1 if inum(row.get("inflight_before")) <= 1 else 0
        total["zero_drain"] += 1 if inum(row.get("drained_cqes")) == 0 else 0
        total["queue_empty"] += 1 if inum(row.get("inflight_before")) == 0 else 0
        total["next_job_done"] += 1 if inum(row.get("next_job")) >= inum(row.get("read_jobs")) else 0
        top.append(row)
    top.sort(key=lambda r: fnum(r.get("wait_ms")), reverse=True)
    top_rows = []
    for row in top[:24]:
        top_rows.append({
            "batch_seq": row.get("batch_seq", ""),
            "op": row.get("op", ""),
            "first_tensor": row.get("first_tensor", ""),
            "last_tensor": row.get("last_tensor", ""),
            "read_jobs": row.get("read_jobs", ""),
            "inflight_before": row.get("inflight_before", ""),
            "next_job": row.get("next_job", ""),
            "drained_cqes": row.get("drained_cqes", ""),
            "wait_ms": fmt(fnum(row.get("wait_ms"))),
            "enqueue_ms": fmt(fnum(row.get("enqueue_ms"))),
        })
    n = total["rows"] or 1
    return {
        "rows": rows,
        "summary": {
            "wait_rows": int(total["rows"]),
            "wait_ms": total["wait_ms"],
            "wait_ms_per_row": total["wait_ms"] / n,
            "p50": quantile(waits, 0.50),
            "p95": quantile(waits, 0.95),
            "p99": quantile(waits, 0.99),
            "low_inflight_ratio": total["low_inflight"] / n,
            "zero_drain_ratio": total["zero_drain"] / n,
            "queue_empty_ratio": total["queue_empty"] / n,
            "next_job_done_ratio": total["next_job_done"] / n,
        },
        "top_rows": top_rows,
    }


def summarize_h2d(run_dir: pathlib.Path):
    rows = read_csv(run_dir / "h2d-coalesce-profile.csv")
    by_op = defaultdict(lambda: defaultdict(float))
    for row in rows:
        add(by_op, row.get("op", ""), row, [
            "jobs", "read_jobs", "copy_count", "coalesced_copy_count",
            "host_contiguous_pairs", "dst_contiguous_pairs", "both_contiguous_pairs",
            "bytes", "total_saved_copy_count",
        ])
    out = []
    for op, c in by_op.items():
        copies = c["copy_count"] or 1
        out.append({
            "op": op,
            "rows": int(c["rows"]),
            "read_jobs": int(c["read_jobs"]),
            "copy_count": int(c["copy_count"]),
            "coalesced_copy_count": int(c["coalesced_copy_count"]),
            "coalesced_ratio": fmt(c["coalesced_copy_count"] / copies),
            "host_contiguous_pairs": int(c["host_contiguous_pairs"]),
            "dst_contiguous_pairs": int(c["dst_contiguous_pairs"]),
            "both_contiguous_pairs": int(c["both_contiguous_pairs"]),
            "bytes_gib": fmt(c["bytes"] / 1024**3),
            "saved_copy_count": int(c["total_saved_copy_count"]),
        })
    out.sort(key=lambda r: float(r["bytes_gib"]), reverse=True)
    return {"rows": rows, "op_rows": out}


def summarize_locality(run_dir: pathlib.Path, max_read_jobs: int | None = None):
    rows = read_csv(run_dir / "io-locality-profile.csv")
    if max_read_jobs is not None:
        rows = [r for r in rows if inum(r.get("read_jobs")) <= max_read_jobs]
    by_op = defaultdict(lambda: defaultdict(float))
    by_role = defaultdict(lambda: defaultdict(float))
    top_gap = []
    for row in rows:
        op = row.get("op", "")
        role = role_from_tensor(row.get("first_tensor", ""))
        fields = [
            "jobs", "read_jobs", "source_switches", "unique_sources",
            "read_bytes", "span_bytes", "gap_bytes", "max_gap_bytes",
            "adjacent_pairs",
        ]
        add(by_op, op, row, fields)
        add(by_role, role, row, fields)
        by_op[op]["same_tensor"] += inum(row.get("same_tensor"))
        by_role[role]["same_tensor"] += inum(row.get("same_tensor"))
        read_bytes = max(fnum(row.get("read_bytes")), 1.0)
        top_gap.append({
            "op": op,
            "first_tensor": row.get("first_tensor", ""),
            "last_tensor": row.get("last_tensor", ""),
            "read_jobs": inum(row.get("read_jobs")),
            "unique_sources": inum(row.get("unique_sources")),
            "source_switches": inum(row.get("source_switches")),
            "read_mib": fnum(row.get("read_bytes")) / 1024**2,
            "span_mib": fnum(row.get("span_bytes")) / 1024**2,
            "gap_mib": fnum(row.get("gap_bytes")) / 1024**2,
            "gap_over_read": fnum(row.get("gap_bytes")) / read_bytes,
            "adjacent_pairs": inum(row.get("adjacent_pairs")),
            "same_tensor": inum(row.get("same_tensor")),
        })

    def rows_from(acc, key_name):
        out = []
        for key, c in acc.items():
            rows_n = c["rows"] or 1
            read_bytes = max(c["read_bytes"], 1.0)
            read_jobs = max(c["read_jobs"], 1.0)
            out.append({
                key_name: key,
                "rows": int(c["rows"]),
                "read_jobs": int(c["read_jobs"]),
                "read_gib": fmt(c["read_bytes"] / 1024**3),
                "span_gib": fmt(c["span_bytes"] / 1024**3),
                "gap_gib": fmt(c["gap_bytes"] / 1024**3),
                "gap_over_read": fmt(c["gap_bytes"] / read_bytes),
                "span_over_read": fmt(c["span_bytes"] / read_bytes),
                "source_switches_per_row": fmt(c["source_switches"] / rows_n),
                "unique_sources_per_row": fmt(c["unique_sources"] / rows_n),
                "adjacent_pairs_per_read_job": fmt(c["adjacent_pairs"] / read_jobs),
                "same_tensor_ratio": fmt(c["same_tensor"] / rows_n),
            })
        out.sort(key=lambda r: float(r["gap_gib"]), reverse=True)
        return out

    top_gap.sort(key=lambda r: r["gap_over_read"], reverse=True)
    top_rows = []
    for row in top_gap[:24]:
        top_rows.append({
            "op": row["op"],
            "first_tensor": row["first_tensor"],
            "last_tensor": row["last_tensor"],
            "read_jobs": row["read_jobs"],
            "unique_sources": row["unique_sources"],
            "source_switches": row["source_switches"],
            "read_mib": fmt(row["read_mib"]),
            "span_mib": fmt(row["span_mib"]),
            "gap_mib": fmt(row["gap_mib"]),
            "gap_over_read": fmt(row["gap_over_read"]),
            "adjacent_pairs": row["adjacent_pairs"],
            "same_tensor": row["same_tensor"],
        })
    return {
        "rows": rows,
        "op_rows": rows_from(by_op, "op"),
        "role_rows": rows_from(by_role, "first_role"),
        "top_gap_rows": top_rows,
    }


def summarize_upgate(run_dir: pathlib.Path):
    rows = [r for r in read_csv(run_dir / "up-gate-profile.csv") if r.get("mode") == "decode"]
    by_type = defaultdict(lambda: defaultdict(float))
    for row in rows:
        key = f"up={row.get('up_type')}/gate={row.get('gate_type')}/parallel={row.get('parallel_stage')}"
        add(by_type, key, row, [
            "n_active", "up_cache_hits", "up_cache_misses", "gate_cache_hits",
            "gate_cache_misses", "up_stage_jobs", "gate_stage_jobs", "up_wait_ms",
            "gate_wait_ms", "up_compute_ms", "gate_compute_ms", "wall_ms",
        ])
    out = []
    for key, c in by_type.items():
        calls = c["rows"] or 1
        out.append({
            "type": key,
            "calls": int(c["rows"]),
            "wall_ms": fmt(c["wall_ms"]),
            "wall_ms_per_call": fmt(c["wall_ms"] / calls),
            "up_miss_per_call": fmt(c["up_cache_misses"] / calls),
            "gate_miss_per_call": fmt(c["gate_cache_misses"] / calls),
            "up_wait_ms_per_call": fmt(c["up_wait_ms"] / calls),
            "gate_wait_ms_per_call": fmt(c["gate_wait_ms"] / calls),
            "up_compute_ms_per_call": fmt(c["up_compute_ms"] / calls),
            "gate_compute_ms_per_call": fmt(c["gate_compute_ms"] / calls),
        })
    out.sort(key=lambda r: float(r["wall_ms"]), reverse=True)
    return {"rows": rows, "type_rows": out}


def main():
    ap = argparse.ArgumentParser(description="Summarize Kimi expert queue/scheduler evidence from profiling CSVs.")
    ap.add_argument("--input", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument(
        "--decode-read-max-jobs",
        type=int,
        default=8,
        help="Treat IO batches with read_jobs <= this value as decode-like batches.",
    )
    args = ap.parse_args()

    run_dir = args.input
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    metrics = read_metrics(run_dir)
    copy = summarize_copy(run_dir)
    io_batch = summarize_io_batch(run_dir)
    decode_io_batch = summarize_io_batch(run_dir, args.decode_read_max_jobs)
    io_wait = summarize_io_wait(run_dir)
    decode_io_wait = summarize_io_wait(run_dir, decode_io_batch["batch_seqs"])
    h2d = summarize_h2d(run_dir)
    locality = summarize_locality(run_dir)
    decode_locality = summarize_locality(run_dir, args.decode_read_max_jobs)
    upgate = summarize_upgate(run_dir)

    write_csv(out / "copy_by_role.csv", copy["role_rows"], list(copy["role_rows"][0].keys()) if copy["role_rows"] else [])
    write_csv(out / "copy_by_tensor.csv", copy["tensor_rows"], list(copy["tensor_rows"][0].keys()) if copy["tensor_rows"] else [])
    write_csv(out / "io_batch_by_op.csv", io_batch["op_rows"], list(io_batch["op_rows"][0].keys()) if io_batch["op_rows"] else [])
    write_csv(out / "decode_io_batch_by_op.csv", decode_io_batch["op_rows"], list(decode_io_batch["op_rows"][0].keys()) if decode_io_batch["op_rows"] else [])
    write_csv(out / "io_batch_by_first_role.csv", io_batch["role_rows"], list(io_batch["role_rows"][0].keys()) if io_batch["role_rows"] else [])
    write_csv(out / "io_wait_top.csv", io_wait["top_rows"], list(io_wait["top_rows"][0].keys()) if io_wait["top_rows"] else [])
    write_csv(out / "decode_io_wait_top.csv", decode_io_wait["top_rows"], list(decode_io_wait["top_rows"][0].keys()) if decode_io_wait["top_rows"] else [])
    write_csv(out / "h2d_by_op.csv", h2d["op_rows"], list(h2d["op_rows"][0].keys()) if h2d["op_rows"] else [])
    write_csv(out / "io_locality_by_op.csv", locality["op_rows"], list(locality["op_rows"][0].keys()) if locality["op_rows"] else [])
    write_csv(out / "decode_io_locality_by_op.csv", decode_locality["op_rows"], list(decode_locality["op_rows"][0].keys()) if decode_locality["op_rows"] else [])
    write_csv(out / "io_locality_by_first_role.csv", locality["role_rows"], list(locality["role_rows"][0].keys()) if locality["role_rows"] else [])
    write_csv(out / "io_locality_top_gap.csv", locality["top_gap_rows"], list(locality["top_gap_rows"][0].keys()) if locality["top_gap_rows"] else [])
    write_csv(out / "upgate_by_type.csv", upgate["type_rows"], list(upgate["type_rows"][0].keys()) if upgate["type_rows"] else [])

    summary = {
        "run_dir": str(run_dir),
        "metrics": {
            "quality": metrics.get("quality"),
            "token_rate": metrics.get("token_rate"),
            "ttft_ms": metrics.get("ttft_ms"),
            "decode_ms": metrics.get("decode_ms"),
            "decode_runs": metrics.get("decode_runs"),
            "expert_pack_0": metrics.get("expert_pack_0"),
            "expert_pack_iouring_0": metrics.get("expert_pack_iouring_0"),
        },
        "copy_wait_p50": copy["wait_p50"],
        "copy_wait_p95": copy["wait_p95"],
        "copy_wait_p99": copy["wait_p99"],
        "io_wait": io_wait["summary"],
        "decode_io_wait": decode_io_wait["summary"],
        "io_batch_hist": io_batch["hist"],
        "decode_io_batch_hist": decode_io_batch["hist"],
        "io_locality_rows": len(locality["rows"]),
        "decode_io_locality_rows": len(decode_locality["rows"]),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Kimi Queue/Scheduler Evidence",
        "",
        f"Input: `{run_dir}`",
        "",
        "## Run Metrics",
        "",
        f"- quality: `{metrics.get('quality', 'unknown')}`",
        f"- token_rate: `{metrics.get('token_rate', 'unknown')}`",
        f"- TTFT ms: `{metrics.get('ttft_ms', 'unknown')}`",
        f"- decode: `{metrics.get('decode_ms', 'unknown')} ms / {metrics.get('decode_runs', 'unknown')} runs`",
        f"- expert_pack: `{metrics.get('expert_pack_0', '')}`",
        f"- expert_pack_iouring: `{metrics.get('expert_pack_iouring_0', '')}`",
        "",
        "## Copy By Role",
        "",
        "| role | rows | GiB | io wait ms | io wait/row | H2D ms | H2D/row | slot wait ms | wall ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in copy["role_rows"]:
        lines.append(
            f"| `{r['role']}` | {r['rows']} | {r['bytes_gib']} | {r['io_wait_ms']} | {r['io_wait_ms_per_row']} | "
            f"{r['h2d_ms']} | {r['h2d_ms_per_row']} | {r['slot_wait_ms']} | {r['wall_ms']} |"
        )
    lines += [
        "",
        "## IO Batch By Op",
        "",
        "| op | rows | read jobs | avg read jobs | avg initial submit | wait calls | wait ms | wait/read job | inflight avg | inflight max | slot wait ms | wall ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in io_batch["op_rows"]:
        lines.append(
            f"| `{r['op']}` | {r['rows']} | {r['read_jobs']} | {r['avg_read_jobs']} | {r['avg_initial_submit']} | "
            f"{r['wait_calls']} | {r['wait_ms']} | {r['wait_ms_per_read_job']} | {r['inflight_avg']} | "
            f"{r['inflight_max']} | {r['slot_wait_ms']} | {r['wall_ms']} |"
        )
    lines += [
        "",
        f"## Decode-Like IO Batch By Op (`read_jobs <= {args.decode_read_max_jobs}`)",
        "",
        "| op | rows | read jobs | avg read jobs | avg initial submit | wait calls | wait ms | wait/read job | inflight avg | inflight max | slot wait ms | wall ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in decode_io_batch["op_rows"]:
        lines.append(
            f"| `{r['op']}` | {r['rows']} | {r['read_jobs']} | {r['avg_read_jobs']} | {r['avg_initial_submit']} | "
            f"{r['wait_calls']} | {r['wait_ms']} | {r['wait_ms_per_read_job']} | {r['inflight_avg']} | "
            f"{r['inflight_max']} | {r['slot_wait_ms']} | {r['wall_ms']} |"
        )
    lines += [
        "",
        "## IO Wait Trace",
        "",
        f"- wait rows: `{io_wait['summary'].get('wait_rows', 0)}`",
        f"- wait ms: `{fmt(io_wait['summary'].get('wait_ms', 0.0))}`",
        f"- wait p50/p95/p99 ms: `{fmt(io_wait['summary'].get('p50', 0.0))}` / `{fmt(io_wait['summary'].get('p95', 0.0))}` / `{fmt(io_wait['summary'].get('p99', 0.0))}`",
        f"- low inflight ratio: `{fmt(io_wait['summary'].get('low_inflight_ratio', 0.0))}`",
        f"- queue empty ratio: `{fmt(io_wait['summary'].get('queue_empty_ratio', 0.0))}`",
        f"- zero drain ratio: `{fmt(io_wait['summary'].get('zero_drain_ratio', 0.0))}`",
        f"- next-job-done ratio: `{fmt(io_wait['summary'].get('next_job_done_ratio', 0.0))}`",
        "",
        f"## Decode-Like IO Wait Trace (`read_jobs <= {args.decode_read_max_jobs}`)",
        "",
        f"- wait rows: `{decode_io_wait['summary'].get('wait_rows', 0)}`",
        f"- wait ms: `{fmt(decode_io_wait['summary'].get('wait_ms', 0.0))}`",
        f"- wait p50/p95/p99 ms: `{fmt(decode_io_wait['summary'].get('p50', 0.0))}` / `{fmt(decode_io_wait['summary'].get('p95', 0.0))}` / `{fmt(decode_io_wait['summary'].get('p99', 0.0))}`",
        f"- low inflight ratio: `{fmt(decode_io_wait['summary'].get('low_inflight_ratio', 0.0))}`",
        f"- queue empty ratio: `{fmt(decode_io_wait['summary'].get('queue_empty_ratio', 0.0))}`",
        f"- zero drain ratio: `{fmt(decode_io_wait['summary'].get('zero_drain_ratio', 0.0))}`",
        f"- next-job-done ratio: `{fmt(decode_io_wait['summary'].get('next_job_done_ratio', 0.0))}`",
        "",
        "## H2D Coalesce By Op",
        "",
        "| op | rows | read jobs | copy count | coalesced ratio | bytes GiB | saved copies |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in h2d["op_rows"]:
        lines.append(
            f"| `{r['op']}` | {r['rows']} | {r['read_jobs']} | {r['copy_count']} | "
            f"{r['coalesced_ratio']} | {r['bytes_gib']} | {r['saved_copy_count']} |"
        )
    lines += [
        "",
        "## IO Locality By Op",
        "",
        "| op | rows | read jobs | read GiB | span/read | gap/read | source switches/row | adjacent pairs/read job | same tensor ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in locality["op_rows"]:
        lines.append(
            f"| `{r['op']}` | {r['rows']} | {r['read_jobs']} | {r['read_gib']} | "
            f"{r['span_over_read']} | {r['gap_over_read']} | {r['source_switches_per_row']} | "
            f"{r['adjacent_pairs_per_read_job']} | {r['same_tensor_ratio']} |"
        )
    lines += [
        "",
        f"## Decode-Like IO Locality By Op (`read_jobs <= {args.decode_read_max_jobs}`)",
        "",
        "| op | rows | read jobs | read GiB | span/read | gap/read | source switches/row | adjacent pairs/read job | same tensor ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in decode_locality["op_rows"]:
        lines.append(
            f"| `{r['op']}` | {r['rows']} | {r['read_jobs']} | {r['read_gib']} | "
            f"{r['span_over_read']} | {r['gap_over_read']} | {r['source_switches_per_row']} | "
            f"{r['adjacent_pairs_per_read_job']} | {r['same_tensor_ratio']} |"
        )
    lines += [
        "",
        "## Top Locality Gaps",
        "",
        "| op | first tensor | read jobs | sources | switches | read MiB | span MiB | gap/read | adjacent pairs |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in locality["top_gap_rows"][:16]:
        lines.append(
            f"| `{r['op']}` | `{r['first_tensor']}` | {r['read_jobs']} | {r['unique_sources']} | "
            f"{r['source_switches']} | {r['read_mib']} | {r['span_mib']} | {r['gap_over_read']} | "
            f"{r['adjacent_pairs']} |"
        )
    lines += [
        "",
        "## Up/Gate By Type",
        "",
        "| type | calls | wall ms | wall/call | up miss/call | gate miss/call | up wait/call | gate wait/call | up compute/call | gate compute/call |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in upgate["type_rows"]:
        lines.append(
            f"| `{r['type']}` | {r['calls']} | {r['wall_ms']} | {r['wall_ms_per_call']} | "
            f"{r['up_miss_per_call']} | {r['gate_miss_per_call']} | {r['up_wait_ms_per_call']} | "
            f"{r['gate_wait_ms_per_call']} | {r['up_compute_ms_per_call']} | {r['gate_compute_ms_per_call']} |"
        )
    lines += [
        "",
        "## Top Copy Tensors By IO Wait",
        "",
        "| tensor | role | rows | GiB | io wait ms | io wait/row | H2D ms |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in copy["tensor_rows"][:20]:
        lines.append(
            f"| `{r['tensor']}` | `{r['role']}` | {r['rows']} | {r['bytes_gib']} | "
            f"{r['io_wait_ms']} | {r['io_wait_ms_per_row']} | {r['h2d_ms']} |"
        )
    lines += [
        "",
        "## Interpretation Rules",
        "",
        "- Low inflight or queue-empty waits point toward queue continuity/co-submit.",
        "- High inflight with long waits points toward storage latency/throughput or refill policy.",
        "- Large H2D time with poor coalescing points toward staging layout.",
        "- High locality gap/read with same-tensor batches points toward expert pack layout or route-order locality, not larger cross-role batches.",
        "- Large upgate wait without corresponding IO wait points toward CUDA stream sync or cache slot readiness.",
        "",
        "This report is profiling evidence only and makes no SOTA claim.",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
