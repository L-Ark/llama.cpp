#!/usr/bin/env python3
import csv
import pathlib
import statistics
import sys


def pct(vals, p):
    if not vals:
        return 0
    vals = sorted(vals)
    idx = min(len(vals) - 1, int(round((len(vals) - 1) * p / 100)))
    return vals[idx]


def summarize(name, vals):
    vals = list(vals)
    if not vals:
        print(f"{name}: empty")
        return
    print(
        f"{name}: sum={sum(vals):.3f} avg={statistics.mean(vals):.3f} "
        f"p50={pct(vals, 50):.3f} p90={pct(vals, 90):.3f} "
        f"p99={pct(vals, 99):.3f} max={max(vals):.3f}"
    )


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: kimi_phase0_analyze_io_trace.py RUN_DIR")
    run = pathlib.Path(sys.argv[1])
    rows = list(csv.DictReader((run / "io-batch-profile.csv").open()))
    waits = list(csv.DictReader((run / "io-wait-trace.csv").open()))
    locs = list(csv.DictReader((run / "io-locality-profile.csv").open()))
    reads = list(csv.DictReader((run / "io-read-trace.csv").open()))

    print(f"run={run}")
    print(f"batch_rows={len(rows)} wait_rows={len(waits)} read_rows={len(reads)}")

    for col in [
        "jobs",
        "read_jobs",
        "initial_submit_jobs",
        "submit_calls",
        "wait_calls",
        "cqes",
        "inflight_avg",
        "inflight_max",
        "wait_ms",
        "wall_ms",
    ]:
        summarize(f"batch.{col}", [float(r[col]) for r in rows])

    hist = {}
    for r in rows:
        k = int(r["read_jobs"])
        hist[k] = hist.get(k, 0) + 1
    print("read_jobs_hist=" + ",".join(f"{k}:{hist[k]}" for k in sorted(hist)))

    inflight_hist = {}
    inflight_wait = {}
    for r in waits:
        k = int(r["inflight_before"])
        inflight_hist[k] = inflight_hist.get(k, 0) + 1
        inflight_wait[k] = inflight_wait.get(k, 0.0) + float(r["wait_ms"])
    print("wait_inflight_before_hist=" + ",".join(f"{k}:{inflight_hist[k]}" for k in sorted(inflight_hist)))
    print("wait_inflight_before_ms=" + ",".join(f"{k}:{inflight_wait[k]:.3f}" for k in sorted(inflight_wait)))

    wait_vals = [float(r["wait_ms"]) for r in waits]
    summarize("wait.wait_ms", wait_vals)

    first = []
    later = []
    tail = []
    for r in waits:
        val = float(r["wait_ms"])
        if int(r["completed_before"]) == 0:
            first.append(val)
        elif int(r["inflight_after"]) == 0:
            tail.append(val)
        else:
            later.append(val)
    summarize("wait.first_cqe", first)
    summarize("wait.later_cqe", later)
    summarize("wait.tail_cqe", tail)

    for col in [
        "source_switches",
        "unique_sources",
        "read_bytes",
        "span_bytes",
        "gap_bytes",
        "max_gap_bytes",
        "adjacent_pairs",
    ]:
        summarize(f"locality.{col}", [float(r[col]) for r in locs])

    kind_stats = {}
    for r in rows:
        tensor = r["first_tensor"]
        if "down" in tensor:
            kind = "down"
        elif "gate" in tensor:
            kind = "gate"
        elif "up" in tensor:
            kind = "up"
        else:
            kind = "other"
        stat = kind_stats.setdefault(kind, {"n": 0, "read": 0, "wait": 0.0, "wall": 0.0, "cqes": 0})
        stat["n"] += 1
        stat["read"] += int(r["read_jobs"])
        stat["wait"] += float(r["wait_ms"])
        stat["wall"] += float(r["wall_ms"])
        stat["cqes"] += int(r["cqes"])
    for kind in sorted(kind_stats):
        stat = kind_stats[kind]
        print(
            f"kind.{kind}: batches={stat['n']} read_jobs={stat['read']} cqes={stat['cqes']} "
            f"wait_ms={stat['wait']:.3f} wall_ms={stat['wall']:.3f}"
        )


if __name__ == "__main__":
    main()
