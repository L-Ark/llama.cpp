#!/usr/bin/env python3
import collections
import csv
import sys
from collections import OrderedDict, deque


CAPS = {"upgate": 1735, "down": 766}


def classify_tensor(name):
    if ".ffn_down_exps." in name:
        return "down"
    if ".ffn_up_exps." in name or ".ffn_gate_exps." in name:
        return "upgate"
    return None


def load_trace(path):
    events = {"upgate": [], "down": []}
    bytes_by_key = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            kind = classify_tensor(row["tensor"])
            if kind is None:
                continue
            key = (row["tensor"], int(row["expert_idx"]), int(row["expert_bytes"]))
            events[kind].append(key)
            bytes_by_key[key] = int(row["expert_bytes"])
    return events, bytes_by_key


def load_waits(path):
    waits = collections.Counter()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("op") != "runtime_load":
                continue
            kind = classify_tensor(row.get("first_tensor", ""))
            if kind is None:
                continue
            waits[kind] += float(row.get("wait_ms", 0) or 0)
    return waits


def lru(seq, cap, bytes_by_key):
    cache = OrderedDict()
    hits = misses = miss_bytes = 0
    for key in seq:
        if key in cache:
            hits += 1
            cache.move_to_end(key)
            continue
        misses += 1
        miss_bytes += bytes_by_key[key]
        if len(cache) >= cap:
            cache.popitem(last=False)
        cache[key] = None
    return hits, misses, miss_bytes


def admit2_lru(seq, cap, bytes_by_key):
    cache = OrderedDict()
    seen = collections.Counter()
    hits = misses = miss_bytes = 0
    for key in seq:
        seen[key] += 1
        if key in cache:
            hits += 1
            cache.move_to_end(key)
            continue
        misses += 1
        miss_bytes += bytes_by_key[key]
        if seen[key] < 2:
            continue
        if len(cache) >= cap:
            cache.popitem(last=False)
        cache[key] = None
    return hits, misses, miss_bytes


def lru2(seq, cap, bytes_by_key):
    cache = set()
    hist = collections.defaultdict(lambda: deque(maxlen=2))
    hits = misses = miss_bytes = 0
    for i, key in enumerate(seq):
        if key in cache:
            hits += 1
            hist[key].append(i)
            continue
        misses += 1
        miss_bytes += bytes_by_key[key]
        if len(cache) >= cap:
            def victim_score(k):
                h = hist[k]
                second_last = h[0] if len(h) == 2 else -1
                last = h[-1] if h else -1
                return (second_last, last)
            cache.remove(min(cache, key=victim_score))
        cache.add(key)
        hist[key].append(i)
    return hits, misses, miss_bytes


def slru(seq, cap, bytes_by_key, protected_pct):
    protected_cap = max(1, min(cap - 1, int(round(cap * protected_pct))))
    probation_cap = cap - protected_cap
    protected = OrderedDict()
    probation = OrderedDict()
    hits = misses = miss_bytes = 0

    def trim_probation():
        while len(probation) > probation_cap:
            probation.popitem(last=False)

    def trim_protected():
        while len(protected) > protected_cap:
            demote, _ = protected.popitem(last=False)
            probation[demote] = None
            probation.move_to_end(demote)
            trim_probation()

    for key in seq:
        if key in protected:
            hits += 1
            protected.move_to_end(key)
            continue
        if key in probation:
            hits += 1
            probation.pop(key, None)
            protected[key] = None
            trim_protected()
            continue
        misses += 1
        miss_bytes += bytes_by_key[key]
        probation[key] = None
        trim_probation()
    return hits, misses, miss_bytes


def twoq(seq, cap, bytes_by_key, a1in_pct):
    a1in_cap = max(1, min(cap - 1, int(round(cap * a1in_pct))))
    am_cap = cap - a1in_cap
    ghost_cap = cap
    a1in = OrderedDict()
    am = OrderedDict()
    a1out = OrderedDict()
    hits = misses = miss_bytes = 0

    def trim_ghost():
        while len(a1out) > ghost_cap:
            a1out.popitem(last=False)

    def evict_a1in():
        old, _ = a1in.popitem(last=False)
        a1out[old] = None
        trim_ghost()

    for key in seq:
        if key in am:
            hits += 1
            am.move_to_end(key)
            continue
        if key in a1in:
            hits += 1
            a1in.move_to_end(key)
            continue
        misses += 1
        miss_bytes += bytes_by_key[key]
        if key in a1out:
            a1out.pop(key, None)
            if len(am) >= am_cap:
                am.popitem(last=False)
            am[key] = None
            continue
        if len(a1in) >= a1in_cap:
            evict_a1in()
        a1in[key] = None
    return hits, misses, miss_bytes


def belady(seq, cap, bytes_by_key):
    positions = collections.defaultdict(deque)
    for i, key in enumerate(seq):
        positions[key].append(i)
    cache = set()
    hits = misses = miss_bytes = 0
    never = 10**18
    for key in seq:
        positions[key].popleft()
        if key in cache:
            hits += 1
            continue
        misses += 1
        miss_bytes += bytes_by_key[key]
        if len(cache) >= cap:
            victim = max(cache, key=lambda k: positions[k][0] if positions[k] else never)
            cache.remove(victim)
        cache.add(key)
    return hits, misses, miss_bytes


def hit_pct(hits, misses):
    total = hits + misses
    return 100.0 * hits / total if total else 0.0


def main():
    if len(sys.argv) != 3:
        print("usage: kimi-cache-policy-sim.py ROUTE_TRACE_CSV IO_BATCH_PROFILE_CSV", file=sys.stderr)
        return 2
    events, bytes_by_key = load_trace(sys.argv[1])
    waits = load_waits(sys.argv[2])
    policies = [
        ("lru", lambda s, c: lru(s, c, bytes_by_key)),
        ("admit2_lru", lambda s, c: admit2_lru(s, c, bytes_by_key)),
        ("lru2", lambda s, c: lru2(s, c, bytes_by_key)),
        ("slru_p50", lambda s, c: slru(s, c, bytes_by_key, 0.50)),
        ("slru_p70", lambda s, c: slru(s, c, bytes_by_key, 0.70)),
        ("slru_p80", lambda s, c: slru(s, c, bytes_by_key, 0.80)),
        ("twoq_a20", lambda s, c: twoq(s, c, bytes_by_key, 0.20)),
        ("twoq_a25", lambda s, c: twoq(s, c, bytes_by_key, 0.25)),
        ("twoq_a33", lambda s, c: twoq(s, c, bytes_by_key, 0.33)),
        ("belady", lambda s, c: belady(s, c, bytes_by_key)),
    ]

    baselines = {}
    results = {}
    for kind in ("upgate", "down"):
        seq = events[kind]
        cap = CAPS[kind]
        baselines[kind] = lru(seq, cap, bytes_by_key)
        print(f"{kind}: events={len(seq)} unique={len(set(seq))} cap={cap} wait_ms={waits[kind]:.3f}")
        for name, fn in policies:
            hits, misses, miss_bytes = fn(seq, cap)
            base_bytes = baselines[kind][2]
            saved_bytes = base_bytes - miss_bytes
            bound = waits[kind] * saved_bytes / base_bytes if base_bytes else 0.0
            results[(name, kind)] = bound
            print(
                f"  {name:10s} hits={hits:6d} misses={misses:6d} "
                f"hit_pct={hit_pct(hits, misses):6.2f} miss_gib={miss_bytes / 2**30:8.3f} "
                f"saved_gib={saved_bytes / 2**30:8.3f} bound_ms={bound:8.3f}"
            )
    print("totals_vs_lru:")
    for name, _ in policies:
        total = results[(name, "upgate")] + results[(name, "down")]
        print(f"  {name:10s} total_bound_ms={total:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
