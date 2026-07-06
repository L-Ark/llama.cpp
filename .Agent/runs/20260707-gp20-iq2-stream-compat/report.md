# GP20: IQ2_XXS/IQ2_XS vendor MoE stream compatibility

Timestamp: `2026-07-07T03:31:00+0800`.

Status: implemented and build-verified; no token-rate improvement claimed.

## Rationale

GP17 showed the AesSedai lower-byte candidate has compatible Kimi/deepseek2
metadata at the first-shard header level. GP19 showed the server still needs a
disk/artifact decision before the full model can be downloaded.

Before downloading the full artifact, this step audited whether the current
vendor MoE stream path would accept IQ2 lower-byte tensors. The generic CUDA
matmul stack has IQ2 support, but `moe_stream_batch` still rejected
`GGML_TYPE_IQ2_XXS` and `GGML_TYPE_IQ2_XS` at its own fast-path gate.

## Source audit

Relevant evidence:

- `ggml/src/ggml-cuda/common.cuh` defines CUDA traits for:
  - `GGML_TYPE_IQ2_XXS`;
  - `GGML_TYPE_IQ2_XS`;
  - `GGML_TYPE_IQ2_S`.
- `ggml/src/ggml-cuda/mmq.cu` dispatches `mul_mat_q_case` for:
  - `GGML_TYPE_IQ2_XXS`;
  - `GGML_TYPE_IQ2_XS`;
  - `GGML_TYPE_IQ2_S`.
- `ggml/src/ggml-cuda/mmvq.cu` uses generic
  `mul_mat_vec_q_switch_type()`, so compact MMVQ can use the registered CUDA
  type implementation when the caller allows the type.
- `ggml/src/ggml-cuda/moe_stream_batch.cu` previously allowed only:
  - `IQ3_XXS`;
  - `IQ3_S`;
  - `IQ2_S`;
  - `Q3_K`;
  - `IQ4_XS`.

## Implementation

Changed file:

- `ggml/src/ggml-cuda/moe_stream_batch.cu`

Changes:

- Added `GGML_TYPE_IQ2_XXS` and `GGML_TYPE_IQ2_XS` to
  `moe_stream_type_supported()`.
- Added `GGML_TYPE_IQ2_XXS` and `GGML_TYPE_IQ2_XS` to
  `launch_moe_mmvq_compact_batch()`'s accepted type switch.

This keeps the existing IQ3_S SOTA path unchanged and does not alter runtime
defaults, cache policy, IO depth, current-down overlap, or prompt/test handling.

## Build verification

Remote host:

```text
ssh -p 51056 root@92.180.27.82
```

The main remote checkout was dirty, so validation used a clean temporary
worktree:

```bash
cd /root/lfz/llama.cpp-vendor-kimi
git fetch wici vendor/kimi-general-prompt-token-rate-16gb --quiet
git worktree add --detach /root/lfz/tmp/gp20-iq2-compat-wt FETCH_HEAD --quiet
cmake -S /root/lfz/tmp/gp20-iq2-compat-wt \
  -B /root/lfz/tmp/gp20-iq2-compat-build \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120a-real
cmake --build /root/lfz/tmp/gp20-iq2-compat-build --target ggml-cuda -j "$(nproc)"
```

Result:

- Configure passed with CUDA `12.9.86`.
- `ggml-cuda` target built successfully.
- The build compiled the relevant IQ2 paths:
  - `moe_stream_batch.cu.o`;
  - `mmvq.cu.o`;
  - `mmq-instance-iq2_xxs.cu.o`;
  - `mmq-instance-iq2_xs.cu.o`;
  - `mmq-instance-iq2_s.cu.o`.
- Warnings were pre-existing missing-declaration/unused-parameter style
  warnings, not new errors.

## Decision

- GP20 is accepted as a compatibility prerequisite, not as a performance SOTA.
- It removes a software gate that would otherwise make the lower-byte IQ2
  candidate fall off the vendor MoE stream path.
- No quality/token-rate/TTFT claim is made until the full IQ2 candidate exists
  locally and passes the planned cold-start dev gates.

Next gate after disk/artifact approval:

1. Download and verify all `IQ2_XXS` shards.
2. Verify full tensor metadata and actual tensor types; the header reports
   `general.file_type=20`, so the actual tensors may be `IQ2_XS` rather than
   `IQ2_XXS`.
3. Run cold-start `n32` dev smoke under 16 GB cgroup.
4. Confirm logs show vendor MoE stream active and no `unsupported_type`
   declines for IQ2 tensors.
5. Continue to expert-pack adaptation and dev `n96` only if quality passes.
