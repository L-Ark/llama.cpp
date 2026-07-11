#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import pathlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass


TRACE_RE = re.compile(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight")
ROLES = ("gate", "up", "down")


def fmt(value: float) -> str:
    return f"{value:.4f}"


def write_csv(path: pathlib.Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@dataclass
class Segment:
    layer: int
    role: str
    experts: list[int]
    expert_bytes: int


@dataclass
class LayerVisit:
    layer: int
    role_experts: dict[str, set[int]]
    role_counts: dict[str, int]

    @property
    def experts(self) -> set[int]:
        out: set[int] = set()
        for values in self.role_experts.values():
            out.update(values)
        return out

    @property
    def max_role_count(self) -> int:
        return max(self.role_counts.values()) if self.role_counts else 0


@dataclass
class PromptTrace:
    name: str
    path: pathlib.Path
    decode_tokens: list[dict[int, LayerVisit]]
    skipped_groups: int
    prompt_like_groups: int
    incomplete_groups: int
    segments: int
    rows: int


def parse_segments(path: pathlib.Path) -> tuple[list[Segment], dict[tuple[int, str], int], int]:
    segments: list[Segment] = []
    bytes_by_layer_role: dict[tuple[int, str], int] = {}
    rows = 0

    current_key: tuple[int, str] | None = None
    current_experts: list[int] = []
    current_bytes = 0

    def flush() -> None:
        nonlocal current_key, current_experts, current_bytes
        if current_key is None:
            return
        layer, role = current_key
        segments.append(Segment(layer, role, current_experts, current_bytes))
        current_key = None
        current_experts = []
        current_bytes = 0

    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            rows += 1
            match = TRACE_RE.search(row.get("tensor", ""))
            if not match:
                continue
            key = (int(match.group(1)), match.group(2))
            expert_idx = int(row.get("expert_idx") or 0)
            expert_bytes = int(float(row.get("expert_bytes") or 0))
            bytes_by_layer_role[key] = max(bytes_by_layer_role.get(key, 0), expert_bytes)
            if key != current_key:
                flush()
                current_key = key
                current_bytes = expert_bytes
            current_experts.append(expert_idx)
    flush()
    return segments, bytes_by_layer_role, rows


def segments_to_visits(segments: list[Segment]) -> list[LayerVisit]:
    visits: list[LayerVisit] = []
    current_layer: int | None = None
    role_experts: dict[str, set[int]] = {}
    role_counts: dict[str, int] = {}

    def flush() -> None:
        nonlocal current_layer, role_experts, role_counts
        if current_layer is None:
            return
        visits.append(LayerVisit(current_layer, role_experts, role_counts))
        current_layer = None
        role_experts = {}
        role_counts = {}

    for seg in segments:
        if current_layer is None:
            current_layer = seg.layer
        if seg.layer != current_layer or seg.role in role_experts:
            flush()
            current_layer = seg.layer
        role_experts[seg.role] = set(seg.experts)
        role_counts[seg.role] = len(seg.experts)
    flush()
    return visits


def parse_prompt_trace(path: pathlib.Path, max_decode_experts: int, layers: int) -> tuple[PromptTrace, dict[tuple[int, str], int]]:
    segments, bytes_by_layer_role, rows = parse_segments(path)
    visits = segments_to_visits(segments)

    groups: list[list[LayerVisit]] = []
    current: list[LayerVisit] = []
    prev_layer = 0
    for visit in visits:
        if current and visit.layer <= prev_layer:
            groups.append(current)
            current = []
        current.append(visit)
        prev_layer = visit.layer
    if current:
        groups.append(current)

    decode_tokens: list[dict[int, LayerVisit]] = []
    prompt_like = 0
    incomplete = 0
    for group in groups:
        by_layer = {visit.layer: visit for visit in group}
        complete = len(by_layer) == layers and all(layer in by_layer for layer in range(1, layers + 1))
        max_role_count = max((visit.max_role_count for visit in group), default=0)
        if complete and max_role_count <= max_decode_experts:
            decode_tokens.append(by_layer)
        elif max_role_count > max_decode_experts:
            prompt_like += 1
        else:
            incomplete += 1

    name = path.parent.name
    return (
        PromptTrace(
            name=name,
            path=path,
            decode_tokens=decode_tokens,
            skipped_groups=prompt_like + incomplete,
            prompt_like_groups=prompt_like,
            incomplete_groups=incomplete,
            segments=len(segments),
            rows=rows,
        ),
        bytes_by_layer_role,
    )


def expert_bytes_for_layer(bytes_by_layer_role: dict[tuple[int, str], int], layer: int, bundle: str) -> int:
    if bundle == "all":
        roles = ROLES
    elif bundle == "upgate":
        roles = ("up", "gate")
    elif bundle == "down":
        roles = ("down",)
    else:
        raise ValueError(f"unknown byte bundle: {bundle}")
    return sum(bytes_by_layer_role.get((layer, role), 0) for role in roles)


class Predictor:
    def __init__(self, layers: int, horizon: int, train: list[PromptTrace]):
        self.layers = layers
        self.horizon = horizon
        self.global_counts: dict[int, Counter[int]] = defaultdict(Counter)
        self.cooc_counts: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
        self._train(train)

    def _train(self, traces: list[PromptTrace]) -> None:
        for trace in traces:
            for token in trace.decode_tokens:
                for layer in range(1, self.layers - self.horizon + 1):
                    source = token[layer].experts
                    target_layer = layer + self.horizon
                    target = token[target_layer].experts
                    self.global_counts[target_layer].update(target)
                    for src_expert in source:
                        self.cooc_counts[(layer, src_expert)].update(target)

    def _global_top(self, target_layer: int, budget: int, exclude: set[int] | None = None) -> list[int]:
        exclude = exclude or set()
        out: list[int] = []
        for expert, _ in sorted(self.global_counts[target_layer].items(), key=lambda kv: (-kv[1], kv[0])):
            if expert not in exclude:
                out.append(expert)
                if len(out) >= budget:
                    break
        return out

    def predict(self, kind: str, layer: int, source: set[int], budget: int) -> list[int]:
        target_layer = layer + self.horizon
        if kind == "global_hot":
            return self._global_top(target_layer, budget)
        if kind != "cooc":
            raise ValueError(f"unknown predictor: {kind}")

        scores: Counter[int] = Counter()
        for src_expert in source:
            scores.update(self.cooc_counts.get((layer, src_expert), Counter()))
        predicted = [expert for expert, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:budget]]
        if len(predicted) < budget:
            predicted.extend(self._global_top(target_layer, budget - len(predicted), set(predicted)))
        return predicted


def evaluate_fold(
    heldout: PromptTrace,
    train: list[PromptTrace],
    bytes_by_layer_role: dict[tuple[int, str], int],
    horizons: list[int],
    budgets: list[int],
    bundles: list[str],
    layers: int,
    io_wait_ms_per_token: float,
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    layer_rows: list[dict] = []
    for horizon in horizons:
        predictor = Predictor(layers, horizon, train)
        for kind in ("global_hot", "cooc"):
            for budget in budgets:
                totals = {bundle: Counter() for bundle in bundles}
                per_layer: dict[tuple[int, str], Counter] = defaultdict(Counter)
                for token in heldout.decode_tokens:
                    for layer in range(1, layers - horizon + 1):
                        source = token[layer].experts
                        target_layer = layer + horizon
                        target = token[target_layer].experts
                        pred_list = predictor.predict(kind, layer, source, budget)
                        predicted = set(pred_list)
                        hits = len(predicted & target)
                        pred_n = len(predicted)
                        target_n = len(target)
                        for bundle in bundles:
                            b = expert_bytes_for_layer(bytes_by_layer_role, target_layer, bundle)
                            c = totals[bundle]
                            c["samples"] += 1
                            c["targets"] += target_n
                            c["predicted"] += pred_n
                            c["hits"] += hits
                            c["target_bytes"] += target_n * b
                            c["predicted_bytes"] += pred_n * b
                            c["useful_bytes"] += hits * b
                        lc = per_layer[(target_layer, "experts")]
                        lc["samples"] += 1
                        lc["targets"] += target_n
                        lc["predicted"] += pred_n
                        lc["hits"] += hits

                for bundle, c in totals.items():
                    targets = c["targets"] or 1
                    predicted = c["predicted"] or 1
                    recall = c["hits"] / targets
                    precision = c["hits"] / predicted
                    rows.append({
                        "heldout_prompt": heldout.name,
                        "predictor": kind,
                        "horizon": horizon,
                        "budget": budget,
                        "byte_bundle": bundle,
                        "decode_tokens": len(heldout.decode_tokens),
                        "samples": int(c["samples"]),
                        "target_experts": int(c["targets"]),
                        "predicted_experts": int(c["predicted"]),
                        "hits": int(c["hits"]),
                        "recall": fmt(recall),
                        "precision": fmt(precision),
                        "target_gib": fmt(c["target_bytes"] / 1024**3),
                        "predicted_gib": fmt(c["predicted_bytes"] / 1024**3),
                        "useful_gib": fmt(c["useful_bytes"] / 1024**3),
                        "waste_gib": fmt(max(c["predicted_bytes"] - c["useful_bytes"], 0) / 1024**3),
                        "linear_wait_saving_ms_per_token": fmt(recall * io_wait_ms_per_token),
                    })

                for (target_layer, _), c in per_layer.items():
                    targets = c["targets"] or 1
                    predicted = c["predicted"] or 1
                    layer_rows.append({
                        "heldout_prompt": heldout.name,
                        "predictor": kind,
                        "horizon": horizon,
                        "budget": budget,
                        "target_layer": target_layer,
                        "samples": int(c["samples"]),
                        "target_experts": int(c["targets"]),
                        "predicted_experts": int(c["predicted"]),
                        "hits": int(c["hits"]),
                        "recall": fmt(c["hits"] / targets),
                        "precision": fmt(c["hits"] / predicted),
                    })
    return rows, layer_rows


def aggregate_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int, int, str], Counter] = defaultdict(Counter)
    for row in rows:
        key = (row["predictor"], int(row["horizon"]), int(row["budget"]), row["byte_bundle"])
        c = grouped[key]
        c["decode_tokens"] += int(row["decode_tokens"])
        for field in ("samples", "target_experts", "predicted_experts", "hits"):
            c[field] += int(row[field])
        for field in ("target_gib", "predicted_gib", "useful_gib", "waste_gib"):
            c[field] += float(row[field])
        # This is recomputed from aggregate recall in the report writer.
    out: list[dict] = []
    for (predictor, horizon, budget, bundle), c in grouped.items():
        targets = c["target_experts"] or 1
        predicted = c["predicted_experts"] or 1
        recall = c["hits"] / targets
        out.append({
            "heldout_prompt": "ALL_DEV_LOO",
            "predictor": predictor,
            "horizon": horizon,
            "budget": budget,
            "byte_bundle": bundle,
            "decode_tokens": int(c["decode_tokens"]),
            "samples": int(c["samples"]),
            "target_experts": int(c["target_experts"]),
            "predicted_experts": int(c["predicted_experts"]),
            "hits": int(c["hits"]),
            "recall": fmt(recall),
            "precision": fmt(c["hits"] / predicted),
            "target_gib": fmt(c["target_gib"]),
            "predicted_gib": fmt(c["predicted_gib"]),
            "useful_gib": fmt(c["useful_gib"]),
            "waste_gib": fmt(c["waste_gib"]),
            "predicted_gib_per_decode_token": fmt(c["predicted_gib"] / c["decode_tokens"] if c["decode_tokens"] else 0.0),
            "useful_gib_per_decode_token": fmt(c["useful_gib"] / c["decode_tokens"] if c["decode_tokens"] else 0.0),
            "waste_gib_per_decode_token": fmt(c["waste_gib"] / c["decode_tokens"] if c["decode_tokens"] else 0.0),
        })
    out.sort(key=lambda r: (int(r["horizon"]), int(r["budget"]), r["byte_bundle"], r["predictor"]))
    return out


def make_report(out: pathlib.Path, parse_rows: list[dict], aggregate: list[dict], args: argparse.Namespace) -> None:
    def select(bundle: str, horizon: int, budget: int) -> list[dict]:
        return [
            r for r in aggregate
            if r["byte_bundle"] == bundle and int(r["horizon"]) == horizon and int(r["budget"]) == budget
        ]

    lines: list[str] = []
    lines.append("# Kimi future expert predictability study")
    lines.append("")
    lines.append(f"Input traces: `{args.input_glob}`")
    lines.append(f"Max decode experts per role: `{args.max_decode_experts}`")
    lines.append(f"Horizons: `{','.join(map(str, args.horizons))}`")
    lines.append(f"Budgets: `{','.join(map(str, args.budgets))}`")
    lines.append(f"Linear wait reference: `{args.io_wait_ms_per_token:.3f} ms/token`")
    lines.append("")
    lines.append("## Parse summary")
    lines.append("")
    lines.append("| prompt | rows | segments | decode_tokens | prompt_like_groups | incomplete_groups |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in parse_rows:
        lines.append(
            f"| `{r['prompt']}` | {r['rows']} | {r['segments']} | {r['decode_tokens']} | "
            f"{r['prompt_like_groups']} | {r['incomplete_groups']} |"
        )
    lines.append("")
    lines.append("## Leave-one-dev-prompt-out aggregate")
    lines.append("")
    lines.append("Recall is expert-set recall for future target layers. Bytes are an estimate for the selected byte bundle.")
    lines.append("")
    lines.append("| horizon | budget | bundle | predictor | recall | precision | pred GiB/token | useful GiB/token | waste GiB/token |")
    lines.append("|---:|---:|---|---|---:|---:|---:|---:|---:|")
    for r in aggregate:
        if r["byte_bundle"] != "all":
            continue
        lines.append(
            f"| {r['horizon']} | {r['budget']} | {r['byte_bundle']} | {r['predictor']} | "
            f"{r['recall']} | {r['precision']} | {r['predicted_gib_per_decode_token']} | "
            f"{r['useful_gib_per_decode_token']} | {r['waste_gib_per_decode_token']} |"
        )
    lines.append("")
    lines.append("## Initial decision rule")
    lines.append("")
    h1_b16 = select("all", 1, 16)
    by_pred = {r["predictor"]: float(r["recall"]) for r in h1_b16}
    cooc = by_pred.get("cooc", 0.0)
    global_hot = by_pred.get("global_hot", 0.0)
    lift = cooc - global_hot
    linear_saving = cooc * args.io_wait_ms_per_token
    lines.append(f"- H=1/B=16 cooc recall: `{cooc:.4f}`.")
    lines.append(f"- H=1/B=16 global-hot recall: `{global_hot:.4f}`.")
    lines.append(f"- Cooc lift over global-hot: `{lift:.4f}`.")
    lines.append(f"- Linear upper-bound wait coverage at H=1/B=16: `{linear_saving:.1f} ms/token`.")
    for row in select("all", 1, 16):
        if row["predictor"] == "cooc":
            lines.append(
                "- H=1/B=16 cooc all-role prefetch volume: "
                f"`{row['predicted_gib_per_decode_token']} GiB/token` predicted, "
                f"`{row['useful_gib_per_decode_token']} GiB/token` useful, "
                f"`{row['waste_gib_per_decode_token']} GiB/token` waste."
            )
    for row in select("all", 1, 32):
        if row["predictor"] == "cooc":
            lines.append(
                "- H=1/B=32 cooc all-role prefetch volume: "
                f"`{row['predicted_gib_per_decode_token']} GiB/token` predicted, "
                f"`{row['useful_gib_per_decode_token']} GiB/token` useful, "
                f"`{row['waste_gib_per_decode_token']} GiB/token` waste."
            )
    if cooc >= 0.50 and lift >= 0.05:
        lines.append("- Verdict: predictor signal is strong enough to justify a default-off runtime prefetch prototype.")
    elif cooc >= 0.35 and lift >= 0.03:
        lines.append("- Verdict: predictor signal is marginal; prototype only if the runtime can cancel prefetch cheaply and prioritize demand IO.")
    else:
        lines.append("- Verdict: predictor signal is weak; do not implement runtime future-layer prefetch before trying RAM/VRAM layout work.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `parse_summary.csv`")
    lines.append("- `fold_metrics.csv`")
    lines.append("- `aggregate_metrics.csv`")
    lines.append("- `layer_metrics.csv`")
    lines.append("")
    out.joinpath("report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate prompt-general future-layer expert prediction from Kimi route traces.")
    ap.add_argument("--input-glob", required=True, help="Glob for dev-only route-trace.csv files.")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--layers", type=int, default=60)
    ap.add_argument("--max-decode-experts", type=int, default=16)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--budgets", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--byte-bundles", default="all,upgate,down")
    ap.add_argument("--io-wait-ms-per-token", type=float, default=374.779)
    args = ap.parse_args()

    paths = [pathlib.Path(p) for p in sorted(glob.glob(args.input_glob))]
    if len(paths) < 2:
        raise SystemExit("Need at least two dev route traces for leave-one-prompt-out analysis")

    bundles = [b.strip() for b in args.byte_bundles.split(",") if b.strip()]
    args.out.mkdir(parents=True, exist_ok=True)

    traces: list[PromptTrace] = []
    merged_bytes: dict[tuple[int, str], int] = {}
    seen_names: set[str] = set()
    for path in paths:
        trace, bytes_by_layer_role = parse_prompt_trace(path, args.max_decode_experts, args.layers)
        if trace.name in seen_names:
            continue
        seen_names.add(trace.name)
        traces.append(trace)
        for key, value in bytes_by_layer_role.items():
            merged_bytes[key] = max(merged_bytes.get(key, 0), value)

    parse_rows = [
        {
            "prompt": t.name,
            "path": str(t.path),
            "rows": t.rows,
            "segments": t.segments,
            "decode_tokens": len(t.decode_tokens),
            "prompt_like_groups": t.prompt_like_groups,
            "incomplete_groups": t.incomplete_groups,
            "skipped_groups": t.skipped_groups,
        }
        for t in traces
    ]
    write_csv(args.out / "parse_summary.csv", parse_rows)

    fold_rows: list[dict] = []
    layer_rows: list[dict] = []
    for heldout in traces:
        train = [t for t in traces if t.name != heldout.name]
        rows, per_layer = evaluate_fold(
            heldout,
            train,
            merged_bytes,
            args.horizons,
            args.budgets,
            bundles,
            args.layers,
            args.io_wait_ms_per_token,
        )
        fold_rows.extend(rows)
        layer_rows.extend(per_layer)

    aggregate = aggregate_rows(fold_rows)
    write_csv(args.out / "fold_metrics.csv", fold_rows)
    write_csv(args.out / "aggregate_metrics.csv", aggregate)
    write_csv(args.out / "layer_metrics.csv", layer_rows)
    make_report(args.out, parse_rows, aggregate, args)


if __name__ == "__main__":
    main()
