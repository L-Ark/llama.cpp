# GP42 GGMLMOEPACKv2 parser skeleton

Timestamp: `2026-07-07T06:27:58+08:00`.

Branch: `vendor/kimi-speculative-general-token-rate-16gb`.

Purpose:

- Add a default-off `GGMLMOEPACKv2` metadata parser skeleton for future
  selected mixed-type expert packs.
- Keep existing `GGMLMOEPACKv1` expert-pack behavior unchanged.
- Do not enable mixed-type runtime H2D or CUDA compute in this phase.

Changed files:

- `ggml/src/ggml-cuda/moe_stream_batch.cu`
- `.Agent/run-tools/kimi_moepack_v2_synthetic_test.py`
- `.Agent/runs/20260707-gp42-moepack-v2-parser-skeleton/synthetic-v2.json`

Implementation:

- Added an env-gated parser entry point:
  `GGML_MOE_EXPERT_PACK_V2=<path-or-colon-list>`.
- Added a separate v2 metadata table with per-entry:
  - tensor name;
  - expert index;
  - packed quant type;
  - payload offset and byte size;
  - logical dimensions;
  - row stride metadata.
- Added a debug lookup surface:
  `ggml_cuda_moe_expert_pack_v2_lookup_debug(...)`.
- Existing v1 lookup, read, H2D, fallback, and compute paths are not connected
  to the v2 table.

Local validation:

```bash
python3 .Agent/run-tools/kimi_moepack_v2_synthetic_test.py \
  --out-dir .Agent/runs/20260707-gp42-moepack-v2-parser-skeleton
python3 -m py_compile .Agent/run-tools/kimi_moepack_v2_synthetic_test.py
git diff --check .Agent/plans/kimi-token-rate-16gb-optimization-plan.md \
  .Agent/run-tools/kimi_moepack_v2_synthetic_test.py \
  ggml/src/ggml-cuda/moe_stream_batch.cu
```

Result:

- Synthetic v2 pack metadata round-trip passed:
  `entries=2`, `entry_size=184`.
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
  --out-dir .Agent/runs/20260707-gp42-moepack-v2-parser-skeleton
```

Result:

- Passed with `entries=2`, `entry_size=184`.

CMake configure:

```bash
cmake -S . -B build-gp42 \
  -DGGML_CUDA=ON \
  -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc
```

Result:

- Configure succeeded.
- A full `cmake --build build-gp42 --target ggml-cuda -j2` was attempted but
  was interrupted because it was too slow for this verification pass.

Targeted CUDA compile:

The generated CMake flags only define `GGML_CUDA_MOE_STREAM`, so the actual
batch branch was compiled by explicitly adding `GGML_CUDA_MOE_STREAM_BATCH`:

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
  -o /tmp/moe_stream_batch_gp42_batch.cu.o"
```

Result:

- Compile succeeded.
- Output object:
  `/tmp/moe_stream_batch_gp42_batch.cu.o`, size `6.9M`.
- Warnings were limited to existing unused/missing-declaration patterns plus
  the new debug lookup missing declaration warning.

Acceptance decision:

- Accepted as non-SOTA infrastructure progress.
- No held-out prompts were used.
- No runtime mixed-type pack path is enabled.
- No token-rate or quality claim is made from GP42.
