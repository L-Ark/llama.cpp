#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
GIB = 1024 ** 3


def load_helper(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ROUTE_SIM = load_helper("moe_route_cache_sim", SCRIPT_DIR / "moe-route-cache-sim.py")
READ_SCHED = load_helper("moe_read_schedule_analyze", SCRIPT_DIR / "moe-read-schedule-analyze.py")


def pct_range(start: int, stop: int, step: int) -> list[int]:
    if step <= 0:
        raise SystemExit("--sweep-step must be positive")
    if start < 1 or stop > 99 or start > stop:
        raise SystemExit("--sweep-min/--sweep-max must define a range within 1..99")
    return list(range(start, stop + 1, step))


def load_runtime_costs(paths: list[Path]) -> tuple[Any | None, dict[str, float]]:
    if not paths:
        return None, {}
    reports = [ROUTE_SIM.parse_runtime_stderr(path) for path in paths]
    report = ROUTE_SIM.merge_runtime_reports(reports)
    return report, ROUTE_SIM.runtime_calibration_costs(report)


def cache_score(est: dict[str, float | int], costs: dict[str, float]) -> tuple[str, float]:
    if costs.get("down_stage_ms_per_gib", 0.0) > 0.0 and costs.get("upgate_wait_ms_per_gib", 0.0) > 0.0:
        down_ms = float(est["down_miss_gib"]) * costs["down_stage_ms_per_gib"]
        up_wait_ms = float(est["upgate_miss_gib"]) * costs["upgate_wait_ms_per_gib"]
        return "down_exposed_plus_upgate_wait_ms", down_ms + up_wait_ms
    return "weighted_miss_gib", float(est["weighted_miss_gib"])


def measured_budget_mib(run: Any) -> int | None:
    env_path = ROUTE_SIM.measured_env_path(run.path)
    if not env_path.exists():
        return None
    with env_path.open("r", errors="replace") as f:
        for line in f:
            if line.startswith("GGML_MOE_VRAM_CACHE_MIB="):
                value = line.split("=", 1)[1].strip()
                if value.isdigit():
                    return int(value)
    return None


def measured_candidate(
        candidates: list[tuple[dict[str, float | int], str, float]],
        measured_runs: list[Any],
        budgets: list[int]) -> tuple[dict[str, float | int], str, float, str] | None:
    if not measured_runs:
        return None
    by_key: dict[tuple[int, int], tuple[dict[str, float | int], str, float]] = {}
    for est, kind, score in candidates:
        budget = int(est["upgate_budget_mib"] + est["down_budget_mib"])
        by_key[(budget, int(est["pct"]))] = (est, kind, score)

    usable: list[tuple[Any, tuple[dict[str, float | int], str, float]]] = []
    single_budget = budgets[0] if len(set(budgets)) == 1 else None
    for run in measured_runs:
        budget = measured_budget_mib(run)
        if budget is None:
            budget = single_budget
        if budget is None:
            continue
        candidate = by_key.get((budget, run.pct))
        if candidate:
            usable.append((run, candidate))
    if not usable:
        return None

    best_run, (est, _kind, _score) = min(usable, key=lambda item: item[0].timing.eval_ms)
    return est, "measured_eval_ms", best_run.timing.eval_ms, str(best_run.path)


def cache_recommendation(
        args: argparse.Namespace,
        costs: dict[str, float],
        measured_runs: list[Any]) -> tuple[dict[str, float | int], str, float, str]:
    entries = ROUTE_SIM.load_entries(args.profile)
    trace = ROUTE_SIM.load_trace(args.trace)
    candidates: list[tuple[dict[str, float | int], str, float]] = []
    for budget_mib in args.budget_mib:
        for pct in pct_range(args.sweep_min, args.sweep_max, args.sweep_step):
            est = ROUTE_SIM.simulate_trace(
                entries,
                trace,
                budget_mib,
                pct,
                args.reserve_pct,
                args.protect_profile,
                args.policy,
                args.preload,
                args.admit_after,
                args.upgate_weight,
                args.down_weight,
            )
            kind, score = cache_score(est, costs)
            candidates.append((est, kind, score))

    measured_best = measured_candidate(candidates, measured_runs, args.budget_mib)
    if measured_best:
        return measured_best

    best = min(candidates, key=lambda item: (item[2], float(item[0]["miss_gib"]), int(item[0]["pct"])))
    if args.verbose_table:
        print("cache_candidate,budget_mib,pct,up_mib,down_mib,up_hit_pct,down_hit_pct,hit_pct,miss_gib,score_kind,score")
        for est, kind, score in candidates:
            print(
                f"cache_candidate,{est['upgate_budget_mib'] + est['down_budget_mib']},"
                f"{est['pct']},{est['upgate_budget_mib']},{est['down_budget_mib']},"
                f"{est['upgate_hit_pct']:.2f},{est['down_hit_pct']:.2f},{est['hit_pct']:.2f},"
                f"{est['miss_gib']:.2f},{kind},{score:.2f}"
            )
    return best[0], best[1], best[2], "model_score_unvalidated"


def coalesce_stats(misses: list[Any], pack: dict[tuple[str, int], Any], thresholds_mib: list[int]) -> list[dict[str, float | int]]:
    runs = READ_SCHED.grouped_tensor_runs(misses)
    base_bytes = sum(event.expert_bytes for event in misses)
    rows: list[dict[str, float | int]] = []
    for threshold_mib in thresholds_mib:
        threshold = threshold_mib * 1024 * 1024
        total = 0
        calls = 0
        saved_calls = 0
        extra = 0
        for run in runs:
            for parity in (0, 1):
                jobs = [event for i, event in enumerate(run) if i % 2 == parity]
                if not jobs:
                    continue
                jobs = sorted(jobs, key=lambda event: pack[(event.tensor, event.expert_idx)].offset)
                start = None
                end = None
                count = 0
                useful = 0
                for event in jobs:
                    entry = pack[(event.tensor, event.expert_idx)]
                    if start is None:
                        start = entry.offset
                        end = entry.offset + entry.nbytes
                        count = 1
                        useful = entry.nbytes
                        continue
                    assert end is not None
                    gap = entry.offset - end
                    if 0 <= gap <= threshold:
                        end = max(end, entry.offset + entry.nbytes)
                        count += 1
                        useful += entry.nbytes
                    else:
                        total += end - start
                        calls += 1
                        saved_calls += max(0, count - 1)
                        extra += (end - start) - useful
                        start = entry.offset
                        end = entry.offset + entry.nbytes
                        count = 1
                        useful = entry.nbytes
                assert start is not None and end is not None
                total += end - start
                calls += 1
                saved_calls += max(0, count - 1)
                extra += (end - start) - useful
        rows.append({
            "threshold_mib": threshold_mib,
            "read_gib": total / GIB,
            "extra_pct": 0.0 if base_bytes == 0 else (total / base_bytes - 1.0) * 100.0,
            "read_calls": calls,
            "saved_calls": saved_calls,
            "saved_call_pct": 100.0 * saved_calls / max(len(misses), 1),
            "extra_gib": extra / GIB,
        })
    return rows


def layout_stats(misses: list[Any], pack_order: list[tuple[str, int]], pack: dict[tuple[str, int], Any]) -> dict[str, dict[str, float]]:
    first_miss_order: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    counts: dict[tuple[str, int], int] = {}
    for event in misses:
        key = (event.tensor, event.expert_idx)
        counts[key] = counts.get(key, 0) + 1
        if key not in seen:
            seen.add(key)
            first_miss_order.append(key)
    freq_order = sorted(counts, key=lambda key: (counts[key], key[0], key[1]), reverse=True)
    layouts = {
        "current": {key: pack[key].offset for key in pack_order},
        "first_miss": READ_SCHED.assign_offsets(first_miss_order, pack_order, pack),
        "frequency": READ_SCHED.assign_offsets(freq_order, pack_order, pack),
    }
    return {name: READ_SCHED.locality_stats(misses, offsets) for name, offsets in layouts.items()}


def prediction_stats(
        profile: list[Any],
        trace: list[Any],
        budget_mib: int,
        upgate_pct: int,
        reserve_pct: int,
        protect_profile: bool,
        policy: str,
        thresholds: list[float]) -> list[dict[str, float | int]]:
    runs = READ_SCHED.grouped_tensor_runs(trace)
    up_calls: list[tuple[int, str, str, tuple[int, ...], tuple[int, ...]]] = []
    for i in range(0, len(runs) - 2, 3):
        up = runs[i]
        gate = runs[i + 1]
        if not up or not gate or ".ffn_up_exps." not in up[0].tensor or ".ffn_gate_exps." not in gate[0].tensor:
            continue
        layer = READ_SCHED.layer_from_tensor(up[0].tensor)
        if layer is None:
            continue
        up_calls.append((
            layer,
            up[0].tensor,
            gate[0].tensor,
            tuple(event.expert_idx for event in up),
            tuple(event.expert_idx for event in gate),
        ))

    prev: dict[int, set[int]] = {}
    layer_hits: dict[int, list[float]] = {}
    for layer, _up_tensor, _gate_tensor, up_experts, _gate_experts in up_calls:
        current = set(up_experts)
        if layer in prev and current:
            layer_hits.setdefault(layer, []).append(len(current & prev[layer]) / len(current))
        prev[layer] = current
    layer_score = {layer: sum(values) / len(values) for layer, values in layer_hits.items()}

    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        caches = READ_SCHED.build_caches(profile, trace, budget_mib, upgate_pct, reserve_pct, protect_profile, policy)
        upgate_cache = caches["upgate"]
        prev_experts: dict[int, set[int]] = {}
        layers_used: set[int] = set()
        pred_reads = 0
        useful_reads = 0
        actual_misses = 0
        for layer, up_tensor, gate_tensor, up_experts, gate_experts in up_calls:
            predicted: list[tuple[str, int]] = []
            if layer in prev_experts and layer_score.get(layer, 0.0) >= threshold:
                layers_used.add(layer)
                for expert in prev_experts[layer]:
                    predicted.append((up_tensor, expert))
                    predicted.append((gate_tensor, expert))
                predicted = [key for key in predicted if not upgate_cache.contains(key)]
                pred_reads += len(predicted)

            actual_keys = [(up_tensor, expert) for expert in up_experts] + [(gate_tensor, expert) for expert in gate_experts]
            actual_miss_keys = {key for key in actual_keys if not upgate_cache.contains(key)}
            useful_reads += sum(1 for key in predicted if key in actual_miss_keys)
            for key in actual_keys:
                if not upgate_cache.access(key, 0):
                    actual_misses += 1
            prev_experts[layer] = set(up_experts)

        rows.append({
            "threshold": threshold,
            "layers": len(layers_used),
            "pred_reads": pred_reads,
            "useful_reads": useful_reads,
            "waste_reads": pred_reads - useful_reads,
            "precision": useful_reads / pred_reads if pred_reads else 0.0,
            "miss_cover": useful_reads / actual_misses if actual_misses else 0.0,
            "actual_misses": actual_misses,
        })
    return rows


def read_mechanism_recommendations(args: argparse.Namespace, best_cache: dict[str, float | int]) -> None:
    profile = READ_SCHED.load_profile(args.profile)
    trace = READ_SCHED.load_trace(args.trace)
    pack_order, pack = READ_SCHED.load_pack(args.expert_pack)
    budget_mib = int(best_cache["upgate_budget_mib"] + best_cache["down_budget_mib"])
    upgate_pct = int(best_cache["pct"])
    misses, caches = READ_SCHED.miss_stream(
        profile,
        trace,
        budget_mib,
        upgate_pct,
        args.reserve_pct,
        args.protect_profile,
        args.policy,
    )
    miss_gib = sum(event.expert_bytes for event in misses) / GIB
    print(
        "cache_replay,"
        + ",".join(
            f"{name}_hits={cache.hits},{name}_misses={cache.misses},{name}_miss_gib={cache.miss_bytes / GIB:.2f}"
            for name, cache in caches.items()
        )
        + f",total_miss_gib={miss_gib:.2f}"
    )

    coalesce = coalesce_stats(misses, pack, args.coalesce_threshold_mib)
    safe_coalesce = [
        row for row in coalesce
        if float(row["extra_pct"]) <= args.coalesce_max_extra_pct
        and float(row["saved_call_pct"]) >= args.coalesce_min_saved_call_pct
    ]
    best_coalesce = max(coalesce, key=lambda row: (
        float(row["saved_call_pct"]) if float(row["extra_pct"]) <= args.coalesce_max_extra_pct else -1.0,
        -float(row["extra_pct"]),
    ))
    coalesce_status = "candidate" if safe_coalesce else "reject"
    print(
        "mechanism,coalesce,"
        f"status={coalesce_status},"
        f"threshold_mib={best_coalesce['threshold_mib']},"
        f"saved_call_pct={best_coalesce['saved_call_pct']:.2f},"
        f"extra_pct={best_coalesce['extra_pct']:.2f},"
        f"rule=saved_call_pct>={args.coalesce_min_saved_call_pct:.2f}_and_extra_pct<={args.coalesce_max_extra_pct:.2f}"
    )

    layouts = layout_stats(misses, pack_order, pack)
    current = layouts["current"]
    best_layout_name, best_layout = min(layouts.items(), key=lambda item: (item[1]["total_tib"], item[1]["backward_pct"]))
    layout_gain_pct = 0.0 if current["total_tib"] == 0.0 else 100.0 * (current["total_tib"] - best_layout["total_tib"]) / current["total_tib"]
    layout_ok = (
        best_layout_name != "current"
        and layout_gain_pct >= args.layout_min_gain_pct
        and best_layout["backward_pct"] <= current["backward_pct"] + args.layout_max_backward_regress_pct
    )
    print(
        "mechanism,pack_layout,"
        f"status={'candidate' if layout_ok else 'keep_current'},"
        f"best={best_layout_name},"
        f"current_total_tib={current['total_tib']:.2f},"
        f"best_total_tib={best_layout['total_tib']:.2f},"
        f"gain_pct={layout_gain_pct:.2f},"
        f"current_backward_pct={current['backward_pct']:.2f},"
        f"best_backward_pct={best_layout['backward_pct']:.2f}"
    )

    prediction = prediction_stats(
        profile,
        trace,
        budget_mib,
        upgate_pct,
        args.reserve_pct,
        args.protect_profile,
        args.policy,
        args.prediction_threshold,
    )
    eligible_prediction = [
        row for row in prediction
        if float(row["precision"]) >= args.prediction_min_precision
        and float(row["miss_cover"]) >= args.prediction_min_miss_cover
        and int(row["useful_reads"]) > int(row["waste_reads"])
    ]
    best_prediction = max(prediction, key=lambda row: (float(row["precision"]), float(row["miss_cover"]), int(row["useful_reads"])))
    print(
        "mechanism,previous_route_prediction,"
        f"status={'candidate' if eligible_prediction else 'reject'},"
        f"threshold={best_prediction['threshold']:.2f},"
        f"precision={best_prediction['precision']:.3f},"
        f"miss_cover={best_prediction['miss_cover']:.3f},"
        f"useful_reads={best_prediction['useful_reads']},"
        f"waste_reads={best_prediction['waste_reads']},"
        f"rule=precision>={args.prediction_min_precision:.2f}_miss_cover>={args.prediction_min_miss_cover:.2f}_useful>waste"
    )

    transition = READ_SCHED.layer_transition_prediction_stats(
        profile,
        trace,
        budget_mib,
        upgate_pct,
        args.reserve_pct,
        args.protect_profile,
        args.policy,
        args.transition_top_k,
        args.transition_min_obs,
        args.transition_threshold,
    )
    eligible_transition = [
        row for row in transition
        if float(row["precision"]) >= args.transition_min_precision
        and float(row["miss_cover"]) >= args.transition_min_miss_cover
        and int(row["useful_reads"]) > int(row["waste_reads"])
    ]
    best_transition = max(transition, key=lambda row: (
        int(row["useful_reads"]) > int(row["waste_reads"]),
        float(row["precision"]),
        float(row["miss_cover"]),
        int(row["useful_reads"]),
    ))
    best_cover_transition = max(transition, key=lambda row: (float(row["miss_cover"]), float(row["precision"])))
    print(
        "mechanism,layer_transition_prediction,"
        f"status={'candidate' if eligible_transition else 'reject'},"
        f"top_k={best_transition['top_k']},"
        f"min_obs={best_transition['min_obs']},"
        f"threshold={best_transition['threshold']:.2f},"
        f"precision={best_transition['precision']:.3f},"
        f"miss_cover={best_transition['miss_cover']:.3f},"
        f"useful_reads={best_transition['useful_reads']},"
        f"waste_reads={best_transition['waste_reads']},"
        f"best_cover={best_cover_transition['miss_cover']:.3f},"
        f"best_cover_precision={best_cover_transition['precision']:.3f},"
        f"rule=precision>={args.transition_min_precision:.2f}_miss_cover>={args.transition_min_miss_cover:.2f}_useful>waste"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recommend MoE inference params from route traces and mechanism gates, before hardware A/B runs.")
    parser.add_argument("--profile", type=Path, required=True, help="Route profile CSV used for protected hotset preload.")
    parser.add_argument("--trace", type=Path, required=True, help="Route trace CSV from GGML_MOE_ROUTE_TRACE_OUT.")
    parser.add_argument("--expert-pack", type=Path, required=True, help="MoE expert pack used for read locality analysis.")
    parser.add_argument("--runtime-stderr", type=Path, action="append", default=[], help="Benchmark stderr with timing/cache/profile counters.")
    parser.add_argument("--measured-stderr", type=Path, action="append", default=[], help="Benchmark stderr from validated A/B runs; overrides model-score recommendations when pct/budget match.")
    parser.add_argument("--budget-mib", type=int, action="append", help="Candidate expert-cache budget. Repeat to compare budgets.")
    parser.add_argument("--sweep-min", type=int, default=35)
    parser.add_argument("--sweep-max", type=int, default=75)
    parser.add_argument("--sweep-step", type=int, default=5)
    parser.add_argument("--reserve-pct", type=int, default=20)
    parser.add_argument("--no-protect-profile", dest="protect_profile", action="store_false")
    parser.add_argument("--policy", choices=("lru", "lfu_lru"), default="lfu_lru")
    parser.add_argument("--preload", choices=("protected", "none", "full"), default="protected")
    parser.add_argument("--admit-after", type=int, default=1)
    parser.add_argument("--upgate-weight", type=float, default=1.0)
    parser.add_argument("--down-weight", type=float, default=1.0)
    parser.add_argument("--coalesce-threshold-mib", type=int, nargs="*", default=[0, 4, 8, 16, 32, 64])
    parser.add_argument("--coalesce-max-extra-pct", type=float, default=0.25)
    parser.add_argument("--coalesce-min-saved-call-pct", type=float, default=2.0)
    parser.add_argument("--layout-min-gain-pct", type=float, default=10.0)
    parser.add_argument("--layout-max-backward-regress-pct", type=float, default=5.0)
    parser.add_argument("--prediction-threshold", type=float, nargs="*", default=[0.0, 0.25, 0.35, 0.45, 0.5, 0.6])
    parser.add_argument("--prediction-min-precision", type=float, default=0.70)
    parser.add_argument("--prediction-min-miss-cover", type=float, default=0.15)
    parser.add_argument("--transition-top-k", type=int, nargs="*", default=[2, 4, 8, 12, 16])
    parser.add_argument("--transition-min-obs", type=int, nargs="*", default=[1, 4, 8, 16, 32])
    parser.add_argument("--transition-threshold", type=float, nargs="*", default=[0.0, 0.05, 0.10, 0.15, 0.20])
    parser.add_argument("--transition-min-precision", type=float, default=0.65)
    parser.add_argument("--transition-min-miss-cover", type=float, default=0.05)
    parser.add_argument("--verbose-table", action="store_true")
    parser.set_defaults(protect_profile=True)
    args = parser.parse_args()

    if args.budget_mib is None:
        args.budget_mib = [2048]
    if any(budget <= 0 for budget in args.budget_mib):
        raise SystemExit("--budget-mib must be positive")
    if args.admit_after <= 0:
        raise SystemExit("--admit-after must be positive")

    _runtime_report, costs = load_runtime_costs(args.runtime_stderr)
    measured_runs = ROUTE_SIM.parse_measured_runs(args.measured_stderr) if args.measured_stderr else []
    print("rule,cache_split,minimize=down_exposed_plus_upgate_wait_ms_when_available_else_weighted_miss_gib")
    print("rule,coalesce,candidate_only_if_saved_calls_are_material_and_extra_bytes_are_near_zero")
    print("rule,pack_layout,candidate_only_if_jump_distance_improves_without_backward_seek_regression")
    print("rule,previous_route_prediction,candidate_only_if_precision_and_miss_coverage_clear_thresholds")
    print("rule,layer_transition_prediction,candidate_only_if_online_transition_prediction_has_high_precision_and_material_miss_coverage")

    best_cache, score_kind, score, source = cache_recommendation(args, costs, measured_runs)
    budget_mib = int(best_cache["upgate_budget_mib"] + best_cache["down_budget_mib"])
    print(
        "cache_recommend,"
        f"budget_mib={budget_mib},"
        f"upgate_pct={best_cache['pct']},"
        f"up_mib={best_cache['upgate_budget_mib']},"
        f"down_mib={best_cache['down_budget_mib']},"
        f"hit_pct={best_cache['hit_pct']:.2f},"
        f"up_hit_pct={best_cache['upgate_hit_pct']:.2f},"
        f"down_hit_pct={best_cache['down_hit_pct']:.2f},"
        f"miss_gib={best_cache['miss_gib']:.2f},"
        f"score_kind={score_kind},"
        f"score={score:.2f},"
        f"source={source}"
    )
    print(f"env,GGML_MOE_VRAM_CACHE_MIB={budget_mib}")
    print(f"env,GGML_MOE_VRAM_CACHE_UPGATE_PCT={best_cache['pct']}")
    print(f"env,GGML_MOE_VRAM_PROFILE_PROTECT={1 if args.protect_profile else 0}")
    print(f"env,GGML_MOE_VRAM_CACHE_POLICY={args.policy}")

    read_mechanism_recommendations(args, best_cache)


if __name__ == "__main__":
    main()
