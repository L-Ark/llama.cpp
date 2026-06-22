#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import heapq
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


TASK_DIR = Path(__file__).resolve().parent
REPO = TASK_DIR.parents[2]
TOP8_PROFILE = REPO / "presets/moe/groundtruth/wici-glm51-interactive-n84-expert-top8.runtime.csv"
UPGATE_BYTES = 4_030_464
DOWN_BYTES = 4_816_896
LAYER_RE = re.compile(r"blk\.(\d+)\.")


@dataclass(frozen=True)
class RowKey:
    tensor: str
    expert_idx: int
    expert_bytes: int


@dataclass(frozen=True)
class ExpertKey:
    layer: int
    expert_idx: int


def layer_of(tensor: str) -> int:
    match = LAYER_RE.search(tensor)
    if not match:
        raise ValueError(f"cannot parse layer from tensor name: {tensor}")
    return int(match.group(1))


def cache_id(row: RowKey) -> str:
    return "upgate" if row.expert_bytes <= UPGATE_BYTES else "down"


def load_trace(path: Path, events_per_token: int) -> list[list[RowKey]]:
    rows: list[RowKey] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(RowKey(row["tensor"], int(row["expert_idx"]), int(row["expert_bytes"])))
    usable = (len(rows) // events_per_token) * events_per_token
    return [rows[i:i + events_per_token] for i in range(0, usable, events_per_token)]


def load_split(split: str, events_per_token: int) -> dict[str, list[list[RowKey]]]:
    out: dict[str, list[list[RowKey]]] = {}
    for path in sorted((TASK_DIR / "traces" / split).glob("*.route.csv")):
        out[path.stem] = load_trace(path, events_per_token)
    return out


def token_experts(token: list[RowKey]) -> dict[int, set[int]]:
    out: dict[int, set[int]] = defaultdict(set)
    for row in token:
        out[layer_of(row.tensor)].add(row.expert_idx)
    return out


def token_expert_keys(token: list[RowKey]) -> set[ExpertKey]:
    keys: set[ExpertKey] = set()
    for row in token:
        keys.add(ExpertKey(layer_of(row.tensor), row.expert_idx))
    return keys


def train_global_hot(train: dict[str, list[list[RowKey]]]) -> dict[int, Counter[int]]:
    counts: dict[int, Counter[int]] = defaultdict(Counter)
    for tokens in train.values():
        for token in tokens:
            for layer, experts in token_experts(token).items():
                for expert in experts:
                    counts[layer][expert] += 1
    return counts


def train_layer_transition(train: dict[str, list[list[RowKey]]], early_layers: int) -> dict[tuple[int, tuple[int, ...]], Counter[int]]:
    mapping: dict[tuple[int, tuple[int, ...]], Counter[int]] = defaultdict(Counter)
    for tokens in train.values():
        for token in tokens:
            by_layer = token_experts(token)
            layers = sorted(by_layer)
            early = layers[:early_layers]
            signature = tuple(sorted(expert for layer in early for expert in by_layer[layer]))
            if not signature:
                continue
            for layer in layers[early_layers:]:
                for expert in by_layer[layer]:
                    mapping[(layer, signature)][expert] += 1
    return mapping


def predict_global(global_hot: dict[int, Counter[int]], m: int) -> set[ExpertKey]:
    preds: set[ExpertKey] = set()
    for layer, counts in global_hot.items():
        for expert, _count in counts.most_common(m):
            preds.add(ExpertKey(layer, expert))
    return preds


def predict_last(tokens: list[list[RowKey]], t: int, m: int) -> set[ExpertKey]:
    preds: set[ExpertKey] = set()
    if t < 0:
        return preds
    for layer, experts in token_experts(tokens[t]).items():
        for expert in sorted(experts)[:m]:
            preds.add(ExpertKey(layer, expert))
    return preds


def predict_recent(tokens: list[list[RowKey]], t: int, window: int, m: int) -> set[ExpertKey]:
    counts: dict[int, Counter[int]] = defaultdict(Counter)
    start = max(0, t - window + 1)
    for idx in range(start, t + 1):
        for layer, experts in token_experts(tokens[idx]).items():
            for expert in experts:
                counts[layer][expert] += 1
    preds: set[ExpertKey] = set()
    for layer, counter in counts.items():
        for expert, _count in counter.most_common(m):
            preds.add(ExpertKey(layer, expert))
    return preds


def actual_future(tokens: list[list[RowKey]], t: int, lookahead: int) -> set[ExpertKey]:
    actual: set[ExpertKey] = set()
    for idx in range(t + 1, min(len(tokens), t + 1 + lookahead)):
        actual.update(token_expert_keys(tokens[idx]))
    return actual


def predict_oracle(tokens: list[list[RowKey]], t: int, lookahead: int, m: int) -> set[ExpertKey]:
    counts: dict[int, Counter[int]] = defaultdict(Counter)
    for idx in range(t + 1, min(len(tokens), t + 1 + lookahead)):
        for key in token_expert_keys(tokens[idx]):
            counts[key.layer][key.expert_idx] += 1
    preds: set[ExpertKey] = set()
    for layer, counter in counts.items():
        for expert, _count in counter.most_common(m):
            preds.add(ExpertKey(layer, expert))
    return preds


def score_prediction(pred: set[ExpertKey], actual: set[ExpertKey]) -> tuple[int, int, int]:
    hit = len(pred & actual)
    return hit, len(pred), len(actual)


def evaluate_predictors(
    train: dict[str, list[list[RowKey]]],
    test: dict[str, list[list[RowKey]]],
    lookaheads: list[int],
    widths: list[int],
    windows: list[int],
) -> dict[str, object]:
    global_hot = train_global_hot(train)
    rows: list[dict[str, object]] = []

    for lookahead in lookaheads:
        for m in widths:
            predictors: list[tuple[str, object]] = [
                ("global-hot-per-layer", None),
                ("last-token-repeat", None),
                ("oracle-upper-bound", None),
            ]
            predictors.extend((f"recent-hot-w{window}", window) for window in windows)
            for name, param in predictors:
                hit = pred_total = actual_total = cases = 0
                for tokens in test.values():
                    for t in range(0, max(0, len(tokens) - lookahead)):
                        actual = actual_future(tokens, t, lookahead)
                        if not actual:
                            continue
                        if name == "global-hot-per-layer":
                            pred = predict_global(global_hot, m)
                        elif name == "last-token-repeat":
                            pred = predict_last(tokens, t, m)
                        elif name == "oracle-upper-bound":
                            pred = predict_oracle(tokens, t, lookahead, m)
                        else:
                            pred = predict_recent(tokens, t, int(param), m)
                        h, p, a = score_prediction(pred, actual)
                        hit += h
                        pred_total += p
                        actual_total += a
                        cases += 1
                rows.append({
                    "predictor": name,
                    "lookahead": lookahead,
                    "m": m,
                    "cases": cases,
                    "hits": hit,
                    "predicted": pred_total,
                    "actual": actual_total,
                    "top_m_recall": hit / actual_total if actual_total else 0.0,
                    "precision": hit / pred_total if pred_total else 0.0,
                    "wasted_prefetch": pred_total - hit,
                })

    transition = evaluate_layer_transition(train, test, widths)
    return {
        "predictor_metrics": rows,
        "layer_transition_metrics": transition,
        "train_trace_count": len(train),
        "test_trace_count": len(test),
        "train_tokens": sum(len(tokens) for tokens in train.values()),
        "test_tokens": sum(len(tokens) for tokens in test.values()),
    }


def evaluate_layer_transition(
    train: dict[str, list[list[RowKey]]],
    test: dict[str, list[list[RowKey]]],
    widths: list[int],
    early_layers: int = 8,
) -> list[dict[str, object]]:
    mapping = train_layer_transition(train, early_layers)
    rows: list[dict[str, object]] = []
    for m in widths:
        hit = pred_total = actual_total = cases = 0
        for tokens in test.values():
            for token in tokens:
                by_layer = token_experts(token)
                layers = sorted(by_layer)
                early = layers[:early_layers]
                late = layers[early_layers:]
                signature = tuple(sorted(expert for layer in early for expert in by_layer[layer]))
                actual: set[ExpertKey] = set()
                pred: set[ExpertKey] = set()
                for layer in late:
                    for expert in by_layer[layer]:
                        actual.add(ExpertKey(layer, expert))
                    for expert, _count in mapping.get((layer, signature), Counter()).most_common(m):
                        pred.add(ExpertKey(layer, expert))
                if not actual:
                    continue
                h, p, a = score_prediction(pred, actual)
                hit += h
                pred_total += p
                actual_total += a
                cases += 1
        rows.append({
            "predictor": "layer-transition",
            "early_layers": early_layers,
            "m": m,
            "cases": cases,
            "hits": hit,
            "predicted": pred_total,
            "actual": actual_total,
            "top_m_recall": hit / actual_total if actual_total else 0.0,
            "precision": hit / pred_total if pred_total else 0.0,
            "wasted_prefetch": pred_total - hit,
        })
    return rows


class ByteCache:
    def __init__(self, capacity: int, pinned: set[RowKey]) -> None:
        self.capacity = capacity
        self.used = 0
        self.pinned = set(pinned)
        self.entries: dict[RowKey, int] = {}
        self.heap: list[tuple[int, RowKey]] = []
        self.clock = 0
        self.hits = 0
        self.misses = 0
        self.prefetch_reads = 0
        self.prefetch_hits = 0
        self.evictions = 0
        self.wasted_evictions = 0

    def preload(self, row: RowKey) -> bool:
        if self.used + row.expert_bytes > self.capacity:
            return False
        self.clock += 1
        self.entries[row] = self.clock
        heapq.heappush(self.heap, (self.clock, row))
        self.used += row.expert_bytes
        self.pinned.add(row)
        return True

    def access(self, row: RowKey) -> bool:
        self.clock += 1
        if row in self.entries:
            self.entries[row] = self.clock
            heapq.heappush(self.heap, (self.clock, row))
            self.hits += 1
            return True
        self.misses += 1
        self._insert(row, is_prefetch=False)
        return False

    def prefetch(self, row: RowKey) -> bool:
        self.clock += 1
        if row in self.entries:
            self.entries[row] = self.clock
            heapq.heappush(self.heap, (self.clock, row))
            self.prefetch_hits += 1
            return True
        self.prefetch_reads += 1
        self._insert(row, is_prefetch=True)
        return False

    def _insert(self, row: RowKey, is_prefetch: bool) -> None:
        while self.used + row.expert_bytes > self.capacity:
            victim = self._victim()
            if victim is None:
                return
            self.used -= victim.expert_bytes
            self.entries.pop(victim, None)
            self.evictions += 1
            if is_prefetch:
                self.wasted_evictions += 1
        self.entries[row] = self.clock
        heapq.heappush(self.heap, (self.clock, row))
        self.used += row.expert_bytes

    def _victim(self) -> RowKey | None:
        while self.heap:
            last, row = heapq.heappop(self.heap)
            if self.entries.get(row) != last:
                continue
            if row in self.pinned:
                continue
            return row
        return None


def load_profile_rows(path: Path) -> list[RowKey]:
    rows: list[RowKey] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(RowKey(row["tensor"], int(row["expert_idx"]), int(row["expert_bytes"])))
    return rows


def build_row_db(train: dict[str, list[list[RowKey]]], test: dict[str, list[list[RowKey]]]) -> dict[ExpertKey, set[RowKey]]:
    db: dict[ExpertKey, set[RowKey]] = defaultdict(set)
    for split in (train, test):
        for tokens in split.values():
            for token in tokens:
                for row in token:
                    db[ExpertKey(layer_of(row.tensor), row.expert_idx)].add(row)
    return db


def make_caches(budget_mib: int, upgate_pct: int, reserve_pct: int) -> tuple[ByteCache, ByteCache]:
    total = budget_mib * 1024 * 1024
    up_cap = total * upgate_pct // 100
    down_cap = total - up_cap
    up = ByteCache(up_cap, set())
    down = ByteCache(down_cap, set())
    up_preload_budget = up_cap - (up_cap * reserve_pct) // 100
    down_preload_budget = down_cap - (down_cap * reserve_pct) // 100
    profile = load_profile_rows(TOP8_PROFILE)
    for row in profile:
        cache = up if cache_id(row) == "upgate" else down
        limit = up_preload_budget if cache is up else down_preload_budget
        old_cap = cache.capacity
        cache.capacity = limit
        cache.preload(row)
        cache.capacity = old_cap
    return up, down


def expand_predictions(pred: set[ExpertKey], row_db: dict[ExpertKey, set[RowKey]]) -> list[RowKey]:
    rows: list[RowKey] = []
    for key in sorted(pred, key=lambda k: (k.layer, k.expert_idx)):
        rows.extend(sorted(row_db.get(key, set()), key=lambda r: (r.tensor, r.expert_idx, r.expert_bytes)))
    return rows


def cache_for(row: RowKey, up: ByteCache, down: ByteCache) -> ByteCache:
    return up if cache_id(row) == "upgate" else down


def replay_policy(
    train: dict[str, list[list[RowKey]]],
    test: dict[str, list[list[RowKey]]],
    budget_mib: int,
    predictor: str,
    lookahead: int,
    m: int,
    window: int,
    upgate_pct: int,
    reserve_pct: int,
) -> dict[str, object]:
    global_hot = train_global_hot(train)
    row_db = build_row_db(train, test)
    up, down = make_caches(budget_mib, upgate_pct, reserve_pct)

    demand_events = 0
    for tokens in test.values():
        for t, token in enumerate(tokens):
            pred: set[ExpertKey] = set()
            if predictor == "global-hot-per-layer":
                pred = predict_global(global_hot, m)
            elif predictor == "last-token-repeat" and t > 0:
                pred = predict_last(tokens, t - 1, m)
            elif predictor.startswith("recent-hot") and t > 0:
                pred = predict_recent(tokens, t - 1, window, m)
            elif predictor == "oracle-upper-bound":
                pred = predict_oracle(tokens, t - 1, lookahead, m) if t > 0 else set()
            for row in expand_predictions(pred, row_db):
                cache_for(row, up, down).prefetch(row)
            for row in token:
                demand_events += 1
                cache_for(row, up, down).access(row)

    hits = up.hits + down.hits
    misses = up.misses + down.misses
    prefetch_reads = up.prefetch_reads + down.prefetch_reads
    prefetch_hits = up.prefetch_hits + down.prefetch_hits
    return {
        "budget_mib": budget_mib,
        "predictor": predictor,
        "lookahead": lookahead,
        "m": m,
        "recent_window": window if predictor.startswith("recent-hot") else None,
        "events": demand_events,
        "hits": hits,
        "misses": misses,
        "hit_rate_pct": 100.0 * hits / max(1, hits + misses),
        "prefetch_reads": prefetch_reads,
        "prefetch_cache_hits": prefetch_hits,
        "extra_reads": prefetch_reads,
        "evictions": up.evictions + down.evictions,
        "prefetch_evictions": up.wasted_evictions + down.wasted_evictions,
        "upgate": {"hits": up.hits, "misses": up.misses, "prefetch_reads": up.prefetch_reads},
        "down": {"hits": down.hits, "misses": down.misses, "prefetch_reads": down.prefetch_reads},
    }


def run_replay(
    train: dict[str, list[list[RowKey]]],
    test: dict[str, list[list[RowKey]]],
    budgets: list[int],
    widths: list[int],
    lookaheads: list[int],
    windows: list[int],
    upgate_pct: int,
    reserve_pct: int,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for budget in budgets:
        baseline = replay_policy(train, test, budget, "baseline", 1, 0, 0, upgate_pct, reserve_pct)
        rows.append(baseline)
        baseline_misses = int(baseline["misses"])
        baseline_hit = float(baseline["hit_rate_pct"])
        for lookahead in lookaheads:
            for m in widths:
                for predictor in ("global-hot-per-layer", "last-token-repeat", "oracle-upper-bound"):
                    row = replay_policy(train, test, budget, predictor, lookahead, m, 0, upgate_pct, reserve_pct)
                    row["miss_reduction_sim"] = baseline_misses - int(row["misses"])
                    row["miss_reduction_pct"] = 100.0 * (baseline_misses - int(row["misses"])) / max(1, baseline_misses)
                    row["hit_rate_delta_abs"] = float(row["hit_rate_pct"]) - baseline_hit
                    rows.append(row)
                for window in windows:
                    row = replay_policy(train, test, budget, f"recent-hot-w{window}", lookahead, m, window, upgate_pct, reserve_pct)
                    row["miss_reduction_sim"] = baseline_misses - int(row["misses"])
                    row["miss_reduction_pct"] = 100.0 * (baseline_misses - int(row["misses"])) / max(1, baseline_misses)
                    row["hit_rate_delta_abs"] = float(row["hit_rate_pct"]) - baseline_hit
                    rows.append(row)
    return {"cache_replay": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-per-token", type=int, default=1800)
    parser.add_argument("--lookaheads", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--metric-widths", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--metric-windows", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--replay-widths", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--replay-windows", type=int, nargs="+", default=[8])
    parser.add_argument("--budgets-mib", type=int, nargs="+", default=[8192, 12288, 14336, 20480])
    parser.add_argument("--upgate-pct", type=int, default=60)
    parser.add_argument("--reserve-pct", type=int, default=10)
    args = parser.parse_args()

    train = load_split("train", args.events_per_token)
    test = load_split("test", args.events_per_token)
    result = {
        "config": vars(args),
        **evaluate_predictors(train, test, args.lookaheads, args.metric_widths, args.metric_windows),
        **run_replay(train, test, args.budgets_mib, args.replay_widths, args.lookaheads, args.replay_windows, args.upgate_pct, args.reserve_pct),
    }

    results_dir = TASK_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "offline_predictors.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
