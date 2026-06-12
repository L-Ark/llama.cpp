# Reproduce GLM SER Midrange Sweep

This file records the commands needed to reproduce the completed SER midrange
sweep on the 5090 32 GB machine under the strict 2 GB host-RAM line.

## Environment

Repository:

```bash
cd /root/lfz/ik_llama
git branch --show-current
```

Expected branch:

```text
feat/glm51-5090-2gb-vram-standalone
```

Required model files:

```text
/root/lfz/ik_llama/models/GLM-5.1-UD-IQ3_XXS/GLM-5.1-UD-IQ3_XXS-00001-of-00007.gguf
/root/lfz/ik_llama/models/GLM-5.1-UD-IQ3_XXS/glm51-iq3xxs.expert-pack
```

Required datasets:

```text
/root/lfz/data/glm_resource_eval/processed/smoke_one.jsonl
/root/lfz/data/glm_resource_eval/processed/smoke.jsonl
```

Runner reused from the prior dynamic-k/cache-aware task:

```text
/root/lfz/ik_llama/.Agent/plans/glm-dynamic-k-cache-aware-routing/runs/ser-local-2gb/run_smoke_accuracy.py
```

All target runs must use:

- `MemoryMax=2G`
- `MemorySwapMax=0`
- `GGML_MOE_RAM_TIER_MIB=0`
- `GGML_MOE_VRAM_CACHE_MIB=12288`
- top8 runtime profile:
  `/root/lfz/ik_llama/presets/moe/groundtruth/wici-glm51-interactive-n84-expert-top8.runtime.csv`

## Preflight

Check no model run is still active:

```bash
pgrep -af 'run_smoke_accuracy|/llama-cli|moe-run.py|run_hit_rate_baseline' || true
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
```

Build check:

```bash
cmake --build /root/lfz/ik_llama/build-cuda --target llama-cli -j 8
```

## Reproduce Phase 1: Single-Item Screen

This screens the narrow threshold region quickly.

```bash
set -euo pipefail

RUNNER=/root/lfz/ik_llama/.Agent/plans/glm-dynamic-k-cache-aware-routing/runs/ser-local-2gb/run_smoke_accuracy.py
OUT=/root/lfz/ik_llama/.Agent/plans/glm-ser-midrange-sweep/runs/accuracy
DATA=/root/lfz/data/glm_resource_eval/processed/smoke_one.jsonl

mkdir -p "$OUT"

for ser in 1,0.96 1,0.97 1,0.975 1,0.98 1,0.99; do
  label="smoke-one-ser-${ser/,/-}"
  systemd-run --same-dir --wait --collect --quiet \
    -p MemoryMax=2G -p MemorySwapMax=0 \
    python3 "$RUNNER" \
      --inner-no-systemd \
      --dataset "$DATA" \
      --output-dir "$OUT" \
      --label "$label" \
      --ser "$ser"
done
```

Expected completed summaries:

| SER | accuracy | output | direct_reads | VRAM hit | prompt tok/s | eval tok/s | total_ms | wall_s |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1,0.96` | 1/1 | B | 4421 | 30.3% | 0.686 | 0.861 | 119645.46 | 122.81 |
| `1,0.97` | 1/1 | B | 4354 | 25.9% | 0.710 | 0.664 | 111323.09 | 114.61 |
| `1,0.975` | 1/1 | B | 4092 | 28.3% | 0.725 | 0.810 | 111552.06 | 114.00 |
| `1,0.98` | 0/1 | C | 3864 | 25.1% | 0.811 | 1.027 | 100273.15 | 102.93 |
| `1,0.99` | 0/1 | `,` | 3829 | 8.2% | 0.901 | 0.893 | 95433.68 | 97.58 |

## Reproduce Phase 2: 4-Item Smoke

Run the quality-preserving candidate:

```bash
RUNNER=/root/lfz/ik_llama/.Agent/plans/glm-dynamic-k-cache-aware-routing/runs/ser-local-2gb/run_smoke_accuracy.py
OUT=/root/lfz/ik_llama/.Agent/plans/glm-ser-midrange-sweep/runs/accuracy
DATA=/root/lfz/data/glm_resource_eval/processed/smoke.jsonl

systemd-run --same-dir --wait --collect --quiet \
  -p MemoryMax=2G -p MemorySwapMax=0 \
  python3 "$RUNNER" \
    --inner-no-systemd \
    --dataset "$DATA" \
    --output-dir "$OUT" \
    --label smoke4-ser-1-0.96 \
    --ser 1,0.96
```

Run the faster rejected candidate:

```bash
RUNNER=/root/lfz/ik_llama/.Agent/plans/glm-dynamic-k-cache-aware-routing/runs/ser-local-2gb/run_smoke_accuracy.py
OUT=/root/lfz/ik_llama/.Agent/plans/glm-ser-midrange-sweep/runs/accuracy
DATA=/root/lfz/data/glm_resource_eval/processed/smoke.jsonl

systemd-run --same-dir --wait --collect --quiet \
  -p MemoryMax=2G -p MemorySwapMax=0 \
  python3 "$RUNNER" \
    --inner-no-systemd \
    --dataset "$DATA" \
    --output-dir "$OUT" \
    --label smoke4-ser-1-0.975 \
    --ser 1,0.975
```

Expected completed summaries:

| SER | accuracy | direct_reads | VRAM hit | prompt tok/s | eval tok/s | total_ms | wall_s | read_failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1,0.96` | 3/4 | 18938 | 25.8% | 1.021 | 0.641 | 577421.79 | 589.08 | 0 |
| `1,0.975` | 2/4 | 16409 | 27.0% | 1.094 | 0.846 | 546904.13 | 559.95 | 0 |

The expected exact token rates may vary slightly run to run. The pass/fail
expectation for reproducing this task is:

- `SER=1,0.96` remains around `3/4` accuracy and has `read_failures=0`.
- `SER=1,0.975` remains faster but drops to around `2/4` accuracy.
- All runs complete under `MemoryMax=2G` and `MemorySwapMax=0`.

## Parse Existing Results

Use this command to summarize the stored result JSON files:

```bash
python3 - <<'PY'
import json
import pathlib

base = pathlib.Path('/root/lfz/ik_llama/.Agent/plans/glm-ser-midrange-sweep/runs/accuracy')
for p in sorted(base.glob('*.summary.json')):
    d = json.loads(p.read_text(encoding='utf-8'))
    a = d.get('aggregate', {})
    print(
        p.name,
        f"accuracy={d.get('correct')}/{d.get('scored')}",
        f"direct_reads={a.get('direct_reads')}",
        f"vram_hit={a.get('vram_cache_hit_rate_pct_avg')}",
        f"prompt_tok_s={a.get('prompt_eval_tokens_per_s')}",
        f"eval_tok_s={a.get('eval_tokens_per_s')}",
        f"total_ms={a.get('total_ms')}",
        f"wall_s={a.get('wall_s')}",
        f"read_failures={a.get('read_failures')}",
    )
PY
```

## Final Recommendation

Use `--smart-expert-reduction 1,0.96` as the current conservative candidate
for this experiment line. It preserved the 4-item smoke accuracy observed in
the previous `SER=1,0.95` run while reducing direct reads and improving token
rate modestly. Do not use `1,0.975` as a default despite its better speed; it
lost one additional smoke item.
