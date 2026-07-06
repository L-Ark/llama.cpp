# GP43 GGMLMOEPACKv2 payload read debug path

Timestamp: `2026-07-07T06:34:28+08:00`.

Branch: `vendor/kimi-speculative-general-token-rate-16gb`.

Base commit:

```text
8dc41afe0 tools: add kimi moepack v2 metadata skeleton
```

Purpose:

- Add a default-off debug-only v2 payload read path.
- Preserve existing v1 expert-pack runtime behavior.
- Prepare for later selected low-byte expert payload experiments.

Implementation:

- `expert_pack_v2_load_source()` now keeps the successfully parsed source file
  handle open in the v2 source table.
- Added `expert_pack_v2_source_for_entry()`.
- Added exported debug function:

```c++
extern "C" bool ggml_cuda_moe_expert_pack_v2_read_debug(
    const char *tensor_name,
    int expert_idx,
    void *dst,
    size_t dst_capacity,
    size_t *nread);
```

- The function:
  - looks up v2 metadata by `(tensor_name, expert_idx)`;
  - validates caller buffer capacity against `packed_nbytes`;
  - seeks to the v2 entry offset;
  - reads exactly `packed_nbytes` into the caller buffer.

Runtime safety:

- No v2 payload is used by decode.
- No v2 payload is connected to H2D staging, RAM tier, iouring batch, up/gate,
  down, or fallback compute.
- Existing `GGMLMOEPACKv1` lookup/copy behavior is unchanged.

Local validation:

```bash
python3 .Agent/run-tools/kimi_moepack_v2_synthetic_test.py \
  --out-dir .Agent/runs/20260707-gp43-moepack-v2-read-debug
python3 -m py_compile .Agent/run-tools/kimi_moepack_v2_synthetic_test.py
git diff --check .Agent/plans/kimi-token-rate-16gb-optimization-plan.md \
  .Agent/run-tools/kimi_moepack_v2_synthetic_test.py \
  ggml/src/ggml-cuda/moe_stream_batch.cu
```

Result:

- Synthetic v2 payload metadata round-trip passed:
  `entries=2`, `entry_size=184`.
- Payload checksums:
  - `blk.1.ffn_up_exps.weight`, expert `7`:
    `3bf32f70544abc626049805ad2a3c4f8f7061a1186aedc406d4fb674b785d4af`;
  - `blk.1.ffn_down_exps.weight`, expert `11`:
    `3e14a183bdf8f0bd166a3df0be66d41bb4a8887cec81e9a5ca2c3a07b62ea20c`.
- Python syntax check passed.
- Diff whitespace check passed.

Remote validation:

Remote worktree:

```text
/root/lfz/tmp/vendor-kimi-speculative-gp33
```

Synthetic test:

```bash
python3 .Agent/run-tools/kimi_moepack_v2_synthetic_test.py \
  --out-dir .Agent/runs/20260707-gp43-moepack-v2-read-debug
python3 -m py_compile .Agent/run-tools/kimi_moepack_v2_synthetic_test.py
git diff --check .Agent/plans/kimi-token-rate-16gb-optimization-plan.md \
  .Agent/run-tools/kimi_moepack_v2_synthetic_test.py \
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
  -o /tmp/moe_stream_batch_gp43_batch.cu.o"
```

Result:

- Compile succeeded.
- Output object:
  `/tmp/moe_stream_batch_gp43_batch.cu.o`, size `6.9M`.
- Warnings were limited to existing unused/missing-declaration patterns plus
  the new v2 read debug missing declaration warning.

Acceptance decision:

- Accepted as non-SOTA infrastructure progress.
- No held-out prompts were used.
- No token-rate or quality claim is made.
- Next valid step is a default-off selected low-byte v2 H2D/compute prototype
  on dev prompts only, or a smaller synthetic C++ harness that calls the debug
  symbol directly before runtime integration.
