# Kimi CPU/defer GPU-extension Audit

Date: 2026-07-12
Branch: `vendor/kimi-deepseek-41d205-additive`
Commit: `ce92cce46`

## Goal

Validate whether the DeepSeek-style improvement path, `CPU/defer MoE main path + GPU expert-cache extension`, is still a high-value migration target for Kimi.

This is a profiling/audit result only. It is not a SOTA claim because `COPY_PROFILE_H2D=1` adds measurement synchronization.

## Reproduction

France:

```bash
cd /root/lfz/llama.cpp-vendor-kimi
systemd-run --wait --collect --same-dir \
  -p MemoryMax=15900000000 -p MemorySwapMax=0 \
  env RUN=/root/lfz/runs/vendor-kimi-token-rate/20260712-cpu-defer-gpu-ext-audit-france-n32-032258 \
      N=32 PROFILE=1 COPY_PROFILE=1 \
      PROMPT_ID=dev_france_regression \
      PROMPT_USER_TEXT='Please introduce France in a short paragraph.' \
      QUALITY_KEYWORDS='france|french,paris|culture|europe' \
      .Agent/run-tools/kimi-general-prompt-repro.sh
```

Intelligence:

```bash
cd /root/lfz/llama.cpp-vendor-kimi
systemd-run --wait --collect --same-dir \
  -p MemoryMax=15900000000 -p MemorySwapMax=0 \
  env RUN=/root/lfz/runs/vendor-kimi-token-rate/20260712-cpu-defer-gpu-ext-audit-intelligence-n32-032459 \
      N=32 PROFILE=1 COPY_PROFILE=1 \
      PROMPT_ID=dev_intelligence_general \
      PROMPT_USER_TEXT='What is intelligence?' \
      QUALITY_KEYWORDS='intelligence|ability|learn|reason|understand' \
      .Agent/run-tools/kimi-general-prompt-repro.sh
```

Audit:

```bash
ROOT=/root/lfz/runs/vendor-kimi-token-rate/20260712-cpu-defer-gpu-ext-audit-dev2-root
OUT=/root/lfz/runs/vendor-kimi-token-rate/20260712-cpu-defer-gpu-ext-audit-dev2-report
python3 .Agent/run-tools/kimi_cpu_defer_gpu_extension_audit.py --input "$ROOT" --out "$OUT" --top-n 20
python3 .Agent/run-tools/kimi_general_profile_breakdown.py --runs-root "$ROOT" --out "$OUT/general-profile-breakdown.md"
python3 .Agent/run-tools/kimi_copy_profile_breakdown.py --run-dir <run-dir> --out <copy-breakdown.md>
```

## Results

| Prompt | quality | tok/s | decode ms/token | TTFT ms | RAM peak GiB | fallback rows | extension miss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dev_france_regression` | pass | 1.510 | 660.167 | 13133.250 | 11.891 | 0 | 0 |
| `dev_intelligence_general` | pass | 1.500 | 664.768 | 10049.540 | 11.737 | 0 | 0 |

Weighted decode: `662.468 ms/token`, `1.510 tok/s`.

CPU/defer op acceptance:

| Prompt | op | calls | batch accept | batch decline | accept rate | decode calls | prompt calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dev_france_regression` | `down` | 2038 | 2038 | 0 | 1.000 | 1861 | 177 |
| `dev_france_regression` | `up_gate` | 1861 | 1861 | 0 | 1.000 | 1861 | 0 |
| `dev_intelligence_general` | `down` | 2038 | 2038 | 0 | 1.000 | 1861 | 177 |
| `dev_intelligence_general` | `up_gate` | 1861 | 1861 | 0 | 1.000 | 1861 | 0 |

## Copy Profile

Aggregate over both prompts:

| op | role | pack hit | io_uring | RAM hit | calls | GiB | io wait ms | H2D ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `runtime_load` | gate | 1 | 1 | 0 | 24054 | 118.813 | 83497.7 | 6904.7 |
| `runtime_load` | up | 1 | 1 | 0 | 24048 | 109.942 | 75896.0 | 6515.6 |
| `runtime_load` | down | 1 | 1 | 0 | 18872 | 124.691 | 72035.6 | 6960.7 |
| `current_down_overlap` | down | 1 | 1 | 0 | 8865 | 52.079 | 26657.7 | 2888.7 |

There are no `pack_hit=0` rows in either copy-profile. The slow path is not GGUF expert fallback or missing expert-pack coverage. It is VRAM-cache miss handling through expert-pack/io_uring demand reads plus H2D and synchronization around those reads.

Total moved bytes are about `405.5 GiB` over `62` decode tokens, or `6.54 GiB/token`. At the measured `~10.3 GiB/s` IO ceiling this alone implies roughly `635 ms/token`, so a pure scheduler change without reducing bytes cannot reach `2 tok/s`, and is far from `5 tok/s`.

## Source Path Evidence

- `ggml/src/ggml-cpu/ggml-cpu.c:6118` checks CUDA up/gate batch preconditions and calls `ggml_cuda_moe_stream_up_gate_batch`.
- `ggml/src/ggml-cpu/ggml-cpu.c:6153` records `cuda_batch_accepted`; the fallback loop starts only after nonzero residual `matrix_row_counts`.
- `ggml/src/ggml-cpu/ggml-cpu.c:5678` records down fallback/total timings after CUDA batch handling.
- `ggml/src/ggml-cuda/moe_stream_batch.cu:14920` implements batched fused up/gate, gathers active experts, records route detail, and stages expert cache slots.
- `ggml/src/ggml-cuda/moe_stream_batch.cu:17180` implements batched down, validates down/prompt candidates, gathers active experts, and stages missing slots.

The profiled runs show `batch_decline=0` and empty fallback CSVs, so the DeepSeek-style GPU extension already catches Kimi's up/gate/down CPU/defer ops in this path.

## Decision

Do not spend the next implementation cycle on a broad Kimi CPU fallback hook rewrite. It is already active and accepted for the profiled generalized prompts.

Next optimization should target bytes and demand-read exposure:

1. Reduce moved expert bytes per token via lower-byte exact/near-exact representations.
2. Improve RAM/VRAM tier policy only if it reduces critical-path `runtime_load` io wait, not just hit rate.
3. Investigate stronger predictor/router-score signals only if predicted reads do not inflate moved bytes enough to erase the benefit.
4. Keep any cache/prefetch A/B default-off until it passes generalized prompts, RAM, TTFT, and quality gates.

