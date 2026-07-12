# Kimi Route Score Trace Audit

Date: 2026-07-12
Branch: `vendor/kimi-deepseek-41d205-additive`
Base commit before this change: `15452681f`

## Goal

This audit implements the next default-off instrumentation step from
`.Agent/plans/kimi-cpu-defer-gpu-extension-cache-plan.md`: determine whether
Kimi can expose router-score information that is stronger than plain
route-history for predictor/prefetch experiments.

This is not a SOTA result and does not change the default decode path.

## Finding

The CUDA MoE extension path only receives selected expert IDs:

- `build_moe_ffn()` computes `ffn_moe_logits`, `selection_probs`,
  `selected_experts`, and final `weights`.
- `ggml_moe_up_gate()` passes only `selected_experts` into
  `GGML_OP_MOE_FUSED_UP_GATE`.
- `ggml_cuda_moe_stream_up_gate_batch()` and `batch_route_profile_hit()` see
  tensor name, expert id, and expert bytes, but not router scores or weights.

So score-aware prediction cannot be implemented inside the existing CUDA
extension alone. The correct low-risk hook is graph-side, before the score
information is discarded.

## Implementation

Added default-off route score trace controlled by:

```bash
GGML_MOE_ROUTE_SCORE_TRACE_OUT=/path/to/route-score-trace.csv
```

When the env var is unset, no extra route-score tensors are added as graph
outputs.

When enabled for Kimi/DeepSeek2 one-token MoE graphs, the trace records:

- selected top-k expert IDs;
- top-k `selection_probs` values before final normalization/scaling;
- final expert `weights` after normalization/scaling;
- layer, token position, sequence id, tensor shapes, and copy time.

CSV schema:

```text
call,seq_id,pos,n_tokens,n_seq_tokens,layer,topk,ids_ne0,ids_ne1,scores_ne0,scores_ne1,weights_ne0,weights_ne1,record_us,ids,scores,weights
```

## Validation

Build:

```bash
cd /root/lfz/llama.cpp-vendor-kimi
cmake --build build-cuda-batch --target llama-cli -j 8
```

Result: build passed.

Smoke command:

```bash
cd /root/lfz/llama.cpp-vendor-kimi
RUN=/root/lfz/runs/vendor-kimi-token-rate/20260712-route-score-trace-smoke-n2-034846
systemd-run --wait --collect --same-dir \
  -p MemoryMax=15900000000 -p MemorySwapMax=0 \
  env RUN=$RUN N=2 PROMPT_ID=route_score_smoke \
      PROMPT_USER_TEXT="Please introduce France in a short paragraph." \
      PROFILE=0 COPY_PROFILE=0 \
      EXTRA_RUNTIME_ENV="GGML_MOE_ROUTE_SCORE_TRACE_OUT=$RUN/route-score-trace.csv" \
      .Agent/run-tools/kimi-general-prompt-repro.sh
```

Smoke result:

- exit: `0`;
- trace file: `/root/lfz/runs/vendor-kimi-token-rate/20260712-route-score-trace-smoke-n2-034846/route-score-trace.csv`;
- trace size: `16K`;
- line count: `62` including header;
- RAM peak: `12690104320` bytes;
- quality: `too_short`, expected because this was `N=2` instrumentation smoke.

Sample row:

```text
1,0,17,1,1,1,8,8,1,1,8,1,8,2149,34|121|265|257|178|227|44|208,0.26105994|0.248345032|0.238068879|0.235007465|0.233171582|0.232927009|0.232699767|0.232241482,0.484462082|0.505437613|0.419913232|0.27435261|0.276968986|0.281503201|0.28932476|0.295037389
```

## Next Step

Run this trace on generalized dev prompts, not held-out prompts, then evaluate
offline predictor feasibility:

- recall of actual next-layer/next-token experts;
- precision and predicted/actual byte ratio;
- full-step cover rate;
- score-margin and entropy correlation with repeated expert use;
- theoretical saved exposed wait in ms/token.

Only if offline score-aware prediction shows at least `100 ms/token` exposed
wait saving without excessive moved bytes should runtime prefetch A/B be built.

## Dev Predictor Analysis

Dev trace root:

```text
/root/lfz/runs/vendor-kimi-token-rate/20260712-route-score-dev-n32-035310
```

Prompts:

- `dev_france_regression`: `Please introduce France in a short paragraph.`
- `dev_intelligence_general`: `What is intelligence?`

Trace collection results:

| prompt | quality | token rate | TTFT ms | decode ms / runs | RAM peak bytes | trace lines |
|---|---|---:|---:|---:|---:|---:|
| France | pass | 1.51 | 13538.01 | 20513.44 / 31 | 12773163008 | 1862 |
| Intelligence | pass | 1.54 | 10050.75 | 20072.41 / 31 | 12602880000 | 1862 |

Offline analysis artifact:

```text
.Agent/runs/20260712-route-score-trace-audit/dev-predictor-analysis/analysis.md
```

Main result:

| policy | recall | precision | pred/actual | full-step |
|---|---:|---:|---:|---:|
| `prev_token_top8` | 0.3398 | 0.3511 | 0.968 | 0.0000 |
| `prev_layer_top8` | 0.0192 | 0.0195 | 0.983 | 0.0000 |
| `hybrid_layer8_token8` | 0.3528 | 0.1826 | 1.932 | 0.0000 |

Decision:

- Simple score/history policies do not pass the runtime prefetch gate.
- Best recall is only `0.3528`, while moved bytes would be almost `1.932x`
  actual for the best hybrid policy.
- Full-step cover is `0.0000`, meaning these policies almost never predict all
  active experts for a layer.
- Score margin is only weakly informative: highest-margin quartile improves
  next-token same-layer recall only to `0.3758`.

Do not build a runtime prefetch A/B from these simple policies. The next
prediction route needs a stronger signal such as a draft router / small model,
or must pivot back to RAM/VRAM storage rebalance oracle and lower-byte expert
representation.
