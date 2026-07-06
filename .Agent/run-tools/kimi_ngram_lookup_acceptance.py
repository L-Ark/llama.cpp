#!/usr/bin/env python3
"""Estimate prompt-lookup / ngram draft acceptance from dev answer text.

This is a feasibility bound, not a runtime benchmark. It intentionally avoids
held-out test prompts and does not load the Kimi model.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\sA-Za-z0-9_\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def find_self_draft(tokens: list[str], pos: int, ngram: int, n_draft: int) -> list[str]:
    if pos < ngram:
        return []
    key = tokens[pos - ngram : pos]
    for start in range(pos - ngram - 1, -1, -1):
        if tokens[start : start + ngram] == key:
            draft_start = start + ngram
            draft_end = min(draft_start + n_draft, pos)
            draft = tokens[draft_start:draft_end]
            if len(draft) >= 1:
                return draft
    return []


def common_prefix_len(a: list[str], b: list[str]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def simulate(tokens: list[str], ngram: int, n_draft: int) -> dict[str, float | int]:
    pos = 1
    steps = 0
    drafted = 0
    accepted = 0
    accepted_runs: list[int] = []

    while pos < len(tokens):
        draft = find_self_draft(tokens, pos, ngram, n_draft)
        truth = tokens[pos : min(len(tokens), pos + len(draft))]
        n_accept = common_prefix_len(draft, truth)

        steps += 1
        drafted += len(draft)
        accepted += n_accept
        accepted_runs.append(n_accept)
        pos += 1 + n_accept

    nonzero = [x for x in accepted_runs if x > 0]
    return {
        "tokens": len(tokens),
        "steps": steps,
        "drafted": drafted,
        "accepted": accepted,
        "accept_rate": accepted / drafted if drafted else 0.0,
        "avg_accepted_per_step": accepted / steps if steps else 0.0,
        "effective_tokens_per_step": len(tokens) / steps if steps else 0.0,
        "nonzero_step_fraction": len(nonzero) / steps if steps else 0.0,
        "max_accepted_run": max(accepted_runs) if accepted_runs else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers-glob", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--ngram", type=int, action="append", default=[])
    parser.add_argument("--draft", type=int, action="append", default=[])
    args = parser.parse_args()

    ngrams = args.ngram or [1, 2, 3, 4]
    drafts = args.draft or [4, 8, 16]
    paths = sorted(Path(".").glob(args.answers_glob))
    if not paths:
        raise SystemExit(f"no answer files matched: {args.answers_glob}")

    rows: list[dict[str, str | int | float]] = []
    aggregate: dict[str, dict[str, float | int]] = {}

    for path in paths:
        if "/test_" in str(path) or path.name.startswith("test_"):
            raise SystemExit(f"refusing to read held-out test path: {path}")
        text = path.read_text(encoding="utf-8", errors="replace")
        tokens = tokenize(text)
        prompt_name = path.parent.name
        for ngram in ngrams:
            for n_draft in drafts:
                stats = simulate(tokens, ngram, n_draft)
                row = {
                    "prompt": prompt_name,
                    "path": str(path),
                    "ngram": ngram,
                    "n_draft": n_draft,
                    **stats,
                }
                rows.append(row)
                key = f"ngram{ngram}_draft{n_draft}"
                agg = aggregate.setdefault(key, {
                    "ngram": ngram,
                    "n_draft": n_draft,
                    "prompts": 0,
                    "tokens": 0,
                    "steps": 0,
                    "drafted": 0,
                    "accepted": 0,
                    "max_accepted_run": 0,
                })
                agg["prompts"] = int(agg["prompts"]) + 1
                for metric in ("tokens", "steps", "drafted", "accepted"):
                    agg[metric] = int(agg[metric]) + int(stats[metric])
                agg["max_accepted_run"] = max(int(agg["max_accepted_run"]), int(stats["max_accepted_run"]))

    for agg in aggregate.values():
        drafted = int(agg["drafted"])
        steps = int(agg["steps"])
        tokens = int(agg["tokens"])
        accepted = int(agg["accepted"])
        agg["accept_rate"] = accepted / drafted if drafted else 0.0
        agg["avg_accepted_per_step"] = accepted / steps if steps else 0.0
        agg["effective_tokens_per_step"] = tokens / steps if steps else 0.0

    best = max(aggregate.values(), key=lambda x: float(x["effective_tokens_per_step"]))

    out = {
        "note": "dev-only text-token ngram self-history bound; not model-token runtime",
        "answers_glob": args.answers_glob,
        "prompt_count": len(paths),
        "best_by_effective_tokens_per_step": best,
        "aggregate": dict(sorted(aggregate.items())),
    }

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(out["best_by_effective_tokens_per_step"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
