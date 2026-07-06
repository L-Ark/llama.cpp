# GP22: lower-byte vendor stream type compatibility

Timestamp: `2026-07-07T04:08:00+0800`.

Status: implemented and build-verified; no token-rate improvement claimed.

## Rationale

GP21 showed that AesSedai `IQ2_XXS` is a mixed low-bit GGUF directory, not a
single-type IQ2-only artifact. The selected current SOTA hotset maps to these
remote expert types:

- `up/IQ1_S`: `8698` entries, `23.226 GiB`
- `up/IQ2_XXS`: `1691` entries, `5.960 GiB`
- `gate/IQ1_S`: `4687` entries, `12.516 GiB`
- `gate/IQ2_XXS`: `5701` entries, `20.095 GiB`
- `down/IQ3_XXS`: `7224` entries, `37.809 GiB`
- `down/IQ2_S`: `3056` entries, `13.383 GiB`
- `down/IQ2_XS`: `384` entries, `1.518 GiB`
- `down/Q2_K`: `158` entries, `0.709 GiB`

GP20 covered `IQ2_XXS` and `IQ2_XS`, but the lower-byte model still needed:

- `IQ1_S` support for many up/gate tensors;
- `Q2_K` support for a small down subset;
- mixed up/gate support for `IQ1_S`/`IQ2_XXS` pairs.

## Implementation

Changed file:

- `ggml/src/ggml-cuda/moe_stream_batch.cu`

Changes:

- Added `GGML_TYPE_IQ1_S` and `GGML_TYPE_Q2_K` to
  `moe_stream_type_supported()`.
- Added `GGML_TYPE_IQ1_S` and `GGML_TYPE_Q2_K` to
  `launch_moe_mmvq_compact_batch()` accepted types.
- Allowed mixed up/gate pairs where the observed GP21 pair is
  `IQ1_S` with `IQ2_XXS`, in either order.

Scope not changed:

- no cache policy change;
- no IO depth change;
- no current-down overlap change;
- no SOTA runtime default change.

## Build verification

Remote host:

```text
ssh -p 51056 root@92.180.27.82
```

The main remote checkout is dirty, so validation used a clean temporary
worktree:

```bash
cd /root/lfz/llama.cpp-vendor-kimi
git fetch wici vendor/kimi-general-prompt-token-rate-16gb --quiet
git worktree add --detach /root/lfz/tmp/gp22-lowbyte-type-compat-wt FETCH_HEAD --quiet
cmake -S /root/lfz/tmp/gp22-lowbyte-type-compat-wt \
  -B /root/lfz/tmp/gp22-lowbyte-type-compat-build \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120a-real
cmake --build /root/lfz/tmp/gp22-lowbyte-type-compat-build --target ggml-cuda -j "$(nproc)"
```

Result:

- Configure passed with CUDA `12.9.86`.
- `ggml-cuda` target built successfully.
- Relevant compiled objects included:
  - `moe_stream_batch.cu.o`;
  - `mmvq.cu.o`;
  - `mmq-instance-iq1_s.cu.o`;
  - `mmq-instance-iq2_xxs.cu.o`;
  - `mmq-instance-iq2_xs.cu.o`;
  - `mmq-instance-iq2_s.cu.o`;
  - `mmq-instance-q2_k.cu.o`.
- Final link succeeded:
  `bin/libggml-cuda.so.0.10.0`.

Temporary validation directories were removed afterward:

```text
/root/lfz/tmp/gp22-lowbyte-type-compat-wt
/root/lfz/tmp/gp22-lowbyte-type-compat-build
```

## Decision

- GP22 is accepted as a lower-byte compatibility prerequisite.
- It is not a token-rate SOTA improvement.
- Future lower-byte runs can now test the actual AesSedai mixed low-bit expert
  types without being rejected by the vendor MoE stream type gate.
- The next blocking artifact issue remains disk:
  - current free space is about `86G`;
  - selected remote pack estimate from GP21 is `115.220 GiB`;
  - full GGUF download is `262.789 GiB`.
