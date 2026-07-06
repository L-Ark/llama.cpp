# GP44 GGMLMOEPACKv2 shadow coverage profile

Timestamp: `2026-07-07T06:43:32+08:00`.

Branch: `vendor/kimi-speculative-general-token-rate-16gb`.

Base commit:

```text
bb29c3f43 tools: add kimi moepack v2 read debug path
```

Purpose:

- Add default-off runtime shadow profiling for future selected low-byte v2
  expert packs.
- Measure whether a v2 pack would cover real active experts without changing
  decode behavior.

Implementation:

- Added env gate:

```text
GGML_MOE_EXPERT_PACK_V2_SHADOW_PROFILE_OUT=<csv>
```

- Added CSV helper:
  `expert_pack_v2_shadow_profile_record(...)`.
- Added rows from real active expert loops:
  - `upgate_stage`;
  - `upgate_plan`;
  - `current_down_overlap`;
  - `down`.
- Each row records:
  - phase;
  - tensor;
  - expert index;
  - logical type and logical bytes;
  - cache state;
  - v2 hit/miss;
  - packed type and packed bytes;
  - packed dims/row stride;
  - packed type support flag;
  - byte ratio in milli-units.

Runtime safety:

- With `GGML_MOE_EXPERT_PACK_V2_SHADOW_PROFILE_OUT` unset, the profiler returns
  immediately and does not initialize v2 metadata.
- With the env set, it only appends CSV rows.
- It does not change:
  - cache placement;
  - H2D source;
  - v1 expert-pack lookup;
  - iouring batching;
  - RAM tier;
  - compute type;
  - output.

Local validation:

```bash
python3 .Agent/run-tools/kimi_moepack_v2_synthetic_test.py \
  --out-dir .Agent/runs/20260707-gp44-v2-shadow-profile
python3 -m py_compile .Agent/run-tools/kimi_moepack_v2_synthetic_test.py
git diff --check .Agent/plans/kimi-token-rate-16gb-optimization-plan.md \
  ggml/src/ggml-cuda/moe_stream_batch.cu \
  .Agent/run-tools/kimi_moepack_v2_synthetic_test.py
```

Result:

- Synthetic v2 payload metadata/checksum round-trip passed:
  `entries=2`, `entry_size=184`.
- Python syntax check passed.
- Diff whitespace check passed.

Remote validation:

Remote worktree:

```text
/root/lfz/tmp/vendor-kimi-speculative-gp33
```

Synthetic and whitespace checks:

```bash
python3 .Agent/run-tools/kimi_moepack_v2_synthetic_test.py \
  --out-dir .Agent/runs/20260707-gp44-v2-shadow-profile
python3 -m py_compile .Agent/run-tools/kimi_moepack_v2_synthetic_test.py
git diff --check .Agent/plans/kimi-token-rate-16gb-optimization-plan.md \
  ggml/src/ggml-cuda/moe_stream_batch.cu
```

Result:

- Passed.

Targeted CUDA compile:

```bash
cd /root/lfz/tmp/vendor-kimi-speculative-gp33/build-gp42
CUDA_DEFINES=$(sed -n 's/^CUDA_DEFINES = //p' \
  ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/flags.make)
CUDA_INCLUDES=$(sed -n 's/^CUDA_INCLUDES = //p' \
  ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/flags.make)
CUDA_FLAGS=$(sed -n 's/^CUDA_FLAGS = //p' \
  ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/flags.make)
eval "ccache /usr/local/cuda-12.9/bin/nvcc \
  -forward-unknown-to-host-compiler \
  $CUDA_DEFINES \
  -DGGML_CUDA_MOE_STREAM_BATCH \
  $CUDA_INCLUDES \
  $CUDA_FLAGS \
  -x cu \
  -c /root/lfz/tmp/vendor-kimi-speculative-gp33/ggml/src/ggml-cuda/moe_stream_batch.cu \
  -o /tmp/moe_stream_batch_gp44_batch.cu.o"
```

Result:

- Compile succeeded.
- Output object:
  `/tmp/moe_stream_batch_gp44_batch.cu.o`, size `6.9M`.
- Warnings were existing unused/missing-declaration patterns; no compile errors.

Acceptance decision:

- Accepted as non-SOTA instrumentation progress.
- No held-out prompts were used.
- No token-rate or output-quality claim is made.
- Next valid step is a dev-only run with a real or synthetic selected v2 pack to
  produce coverage/byte-ratio CSV before any v2 runtime integration.
