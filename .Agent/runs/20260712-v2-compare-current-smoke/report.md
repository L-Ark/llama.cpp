# Kimi v2 lower-byte vs current-pack MMVQ compare smoke

Date: 2026-07-12
Branch: `vendor/kimi-deepseek-41d205-additive`
Head while running: `a652d84c8`

## Goal

Check whether the real tiny `GGMLMOEPACKv2` IQ1_S/Q2_K payloads are close enough
to the current IQ3-pack experts to justify wiring a runtime override path.

This is default-off and does not load the model. It compares CUDA MMVQ output for
the same tensor/expert and the same random activation vectors:

- current source: current IQ3 expert pack + overlay + GGUF alias;
- candidate source: tiny selected `GGMLMOEPACKv2` lower-byte overlay;
- kernel: exported `ggml_cuda_moe_stream_mmvq_dev`;
- activations per entry: `4`.

## Inputs

- v2 pack:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-iq1s-selected-payload-pack8/selected-iq1s-overlay-v2.expert-pack`
- v2 manifest:
  `/root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-iq1s-selected-payload-pack8/selected-iq1s-overlay-manifest.tsv`
- current inventory:
  `.Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv`

## Result

Overall:

- entries: `8`;
- activation rows: `32`;
- mean rel-L2: `0.48436786`;
- max rel-L2: `0.57039980`;
- mean cosine: `0.88373874`;
- current pack read failures: `0`;
- current reads from GGUF alias source: `0`.

By role:

| role | current type | lower type | byte ratio | mean rel-L2 range | cosine range |
|---|---|---|---:|---:|---:|
| down | Q3_K | Q2_K | `0.763636` | `0.3036-0.3067` | `0.9531-0.9537` |
| gate | IQ2_S | IQ1_S | `0.609756` | `0.5385-0.5544` | `0.8573-0.8618` |
| up | IQ2_S | IQ1_S | `0.609756` | `0.5192-0.5583` | `0.8540-0.8708` |

Artifacts:

- `.Agent/runs/20260712-v2-compare-current-smoke/compare.tsv`
- `.Agent/runs/20260712-v2-compare-current-smoke/compare.stderr`

## Decision

Reject direct tiny IQ1_S/Q2_K replacement as the next runtime bridge.

The payloads are runnable, but the output error is too large for a first
semantic-quality candidate. The most important mixed up/gate path shows
`~0.52-0.56` mean rel-L2 for IQ1_S versus current IQ2_S, so wiring this into
runtime is likely to fail the France/generalized quality gate before it becomes
a token-rate win.

The lower-byte direction is still the right class of optimization, but the next
candidate should use a safer byte ratio or activation-aware representation:

- test IQ2_S/IQ2_XXS-style up/gate instead of IQ1_S if available;
- build an activation-output gate for real candidate payloads before runtime;
- keep v2 runtime bridge work paused until the candidate passes this error gate.

## Reproduce

```bash
RUN=/root/lfz/runs/vendor-kimi-token-rate/20260712-v2-compare-current-smoke
mkdir -p "$RUN"

g++ -std=c++17 -O2 \
  -I/usr/local/cuda/include \
  .Agent/run-tools/kimi_moepack_v2_compare_current_smoke.cpp \
  -L/usr/local/cuda/targets/x86_64-linux/lib -lcudart -ldl \
  -o "$RUN/kimi_moepack_v2_compare_current_smoke"

env \
  GGML_MOE_EXPERT_PACK=/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-france-l12-upgate-v2.expert-pack \
  GGML_MOE_EXPERT_PACK_OVERLAY=/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-l1l2down-overlay.expert-pack \
  GGML_MOE_EXPERT_GGUF_ALIAS_TSV=/root/lfz/runs/vendor-kimi-token-rate/20260706-131700Z-gp2-gguf-alias-generate/kimi-iq3s-all-experts.gguf-alias.tsv \
  GGML_MOE_IO_ALIGNED_ALIAS_BATCH=1 \
  LD_LIBRARY_PATH=build-cuda-batch/bin:/usr/local/cuda/targets/x86_64-linux/lib \
  "$RUN/kimi_moepack_v2_compare_current_smoke" \
  build-cuda-batch/bin/libggml-cuda.so \
  /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-iq1s-selected-payload-pack8/selected-iq1s-overlay-v2.expert-pack \
  /root/lfz/runs/vendor-kimi-token-rate/20260711-kimi-phase5d-iq1s-selected-payload-pack8/selected-iq1s-overlay-manifest.tsv \
  .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv \
  0 \
  4 \
  > "$RUN/compare.tsv" \
  2> "$RUN/compare.stderr"
```
