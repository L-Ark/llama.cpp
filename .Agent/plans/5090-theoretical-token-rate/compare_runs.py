#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_5090_matrix as runner


def load_run(path: Path) -> dict[str, object]:
    summary_path = path / "summary.json"
    stderr_path = path / "stderr.txt"
    record = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if stderr_path.exists():
        record.update(runner.parse_stderr(stderr_path.read_text(errors="replace")))
    record.setdefault("label", path.name)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    rows = []
    for run in args.runs:
        r = load_run(run)
        pinned = r.get("pinned_staging", {})
        row = {
            "label": r.get("label"),
            "eval_tps": r.get("eval_tokens_per_s"),
            "total_ms": r.get("total_ms"),
            "vram_hit": r.get("vram_cache_hit_rate_pct"),
            "reads": r.get("expert_pack_iouring_reads"),
            "bytes": r.get("expert_pack_iouring_bytes"),
            "wait_us": r.get("expert_pack_iouring_wait_us"),
            "submit_us": r.get("expert_pack_iouring_submit_us"),
            "batches": r.get("iouring_batches"),
            "inflight_avg": r.get("iouring_inflight_avg"),
            "rings": {},
        }
        if isinstance(pinned, dict):
            for name, data in pinned.items():
                if not isinstance(data, dict):
                    continue
                io = data.get("iouring", {})
                if isinstance(io, dict):
                    row["rings"][name] = {
                        "copies": data.get("copies"),
                        "waits": data.get("waits"),
                        "jobs": io.get("jobs"),
                        "batches": io.get("batches"),
                        "wait_calls": io.get("wait_calls"),
                        "inflight_avg": io.get("inflight_avg"),
                        "hist_1": io.get("hist_1"),
                        "hist_2_4": io.get("hist_2_4"),
                        "hist_5_8": io.get("hist_5_8"),
                    }
        rows.append(row)

    text = json.dumps(rows, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
