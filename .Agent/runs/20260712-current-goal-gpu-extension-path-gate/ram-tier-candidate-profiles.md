# Prompt-Agnostic RAM Tier Candidate Profiles

Date: 2026-07-12
Branch: `vendor/kimi-deepseek-41d205-additive`
Current HEAD when generated: `d2f1b0d08`

These profiles are A/B inputs only. They are not SOTA results and have not been
benchmarked yet.

## Goal

Prepare the first prompt-agnostic RAM/VRAM layer-role experiments from the
current CPU/defer GPU-extension path gate. The candidates are selected from the
dev aggregate N96 profile, not from a single prompt hotset.

Input evidence:

- profile report: `.Agent/runs/20260712-current-head-cpudefer-n96-audit/report.md`
- up/gate layer table:
  `.Agent/runs/20260712-current-head-cpudefer-n96-audit/top_upgate_layers.csv`
- down layer table:
  `.Agent/runs/20260712-current-head-cpudefer-n96-audit/top_down_layers.csv`

Selection rule:

- choose the top aggregate up/gate wait layer: `blk.14`;
- choose the top aggregate down stage layer: `blk.4`;
- include every expert in the selected layer/role, `expert_idx=0..383`;
- do not use per-prompt hot expert counts.

## Generated Profiles

Output directory:

`.Agent/profiles/kimi/ram-tier/20260712-current-goal-dev-aggregate-layer-role/`

| candidate | profile | entries | bytes | GiB | source |
|---|---|---:|---:|---:|---|
| `blk14_gate_full384` | `blk14_gate_full384.csv` | `384` | `2157969408` | `2.0098` | full layer-role |
| `blk14_upgate_full384` | `blk14_upgate_full384.csv` | `768` | `3963617280` | `3.6914` | full layer-role |
| `blk4_down_full384` | `blk4_down_full384.csv` | `384` | `2994733056` | `2.7891` | full layer-role |

Every CSV has one header plus all selected experts:

- `blk14_gate_full384.csv`: `385` lines;
- `blk14_upgate_full384.csv`: `769` lines;
- `blk4_down_full384.csv`: `385` lines.

## Reproduce Profile Generation

```bash
cd /root/lfz/llama.cpp-vendor-kimi
ALIAS=/root/lfz/runs/vendor-kimi-token-rate/20260706-131700Z-gp2-gguf-alias-generate/kimi-iq3s-all-experts.gguf-alias.tsv
OUT=.Agent/profiles/kimi/ram-tier/20260712-current-goal-dev-aggregate-layer-role
mkdir -p "$OUT"

python3 .Agent/run-tools/kimi_make_layer_ram_profile_from_alias.py \
  --alias-tsv "$ALIAS" --layers 14 --roles gate \
  --out-profile "$OUT/blk14_gate_full384.csv" \
  --out-report "$OUT/blk14_gate_full384.json"

python3 .Agent/run-tools/kimi_make_layer_ram_profile_from_alias.py \
  --alias-tsv "$ALIAS" --layers 14 --roles up,gate \
  --out-profile "$OUT/blk14_upgate_full384.csv" \
  --out-report "$OUT/blk14_upgate_full384.json"

python3 .Agent/run-tools/kimi_make_layer_ram_profile_from_alias.py \
  --alias-tsv "$ALIAS" --layers 4 --roles down \
  --out-profile "$OUT/blk4_down_full384.csv" \
  --out-report "$OUT/blk4_down_full384.json"
```

## Planned A/B Shape

Each candidate must be paired with a same-commit baseline. Use dev prompts first;
held-out prompts are reserved for a candidate that passes dev.

Candidate environment template:

```bash
GGML_MOE_RAM_TIER_MIB=<budget_mib>
GGML_MOE_RAM_TIER_PROFILE=<profile_csv>
GGML_MOE_RAM_TIER_SKIP=0
GGML_MOE_RAM_TIER_PIN=0
GGML_MOE_RAM_TIER_PRELOAD_DIRECT=1
GGML_MOE_RAM_TIER_PRELOAD_THREADS=<threads>
GGML_MOE_RAM_BATCH_PROFILE_OUT=$RUN/ram-batch-profile.csv
```

Initial budgets:

| candidate | budget MiB | preload threads |
|---|---:|---:|
| `blk14_gate_full384` | `2200` | `8` |
| `blk14_upgate_full384` | `4000` | `12` |
| `blk4_down_full384` | `3100` | `12` |

Cold-start command shape:

```bash
RUN=/root/lfz/runs/vendor-kimi-token-rate/<new-run>
systemd-run --wait --collect --same-dir \
  -p MemoryMax=15900000000 -p MemorySwapMax=0 \
  env REPO=/root/lfz/llama.cpp-vendor-kimi \
      RUN="$RUN" \
      PROMPT_ID=dev_france_regression \
      PROMPT_USER_TEXT="Please introduce France in a short paragraph." \
      QUALITY_KEYWORDS="france|french,paris|culture|europe" \
      N=96 PROFILE=0 COPY_PROFILE=0 RAM_AUDIT=1 \
      EXTRA_RUNTIME_ENV="<candidate environment block>" \
      .Agent/run-tools/kimi-general-prompt-repro.sh
```

If a candidate improves low-overhead dev token rate and passes RAM/TTFT/quality,
rerun with `PROFILE=1 COPY_PROFILE=1` to prove the saved time comes from lower
io_uring/staging wait rather than measurement noise.

## Acceptance Gate

- cold start only;
- `MemoryMax=15900000000`, `MemorySwapMax=0`;
- host RAM peak `<16 GB`, including file cache and RAM tier;
- France quality pass;
- TTFT `<= +20%` versus paired same-commit baseline;
- endpoint decode token rate improves, not only hit rate;
- fallback rows remain zero;
- if accepted as SOTA, reproduce from pushed commit and write full commit body
  with env, command, prompt split, RAM/VRAM, TTFT, rates, quality, and rollback.

## Decision

Profiles are ready for default-off A/B. Do not promote them until paired
cold-start runs prove an endpoint token-rate improvement under the hard gates.
