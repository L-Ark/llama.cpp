# GP28 route detail instrumentation report

Timestamp: `2026-07-07T06:15:00+0800`.

Branch: `vendor/kimi-speculative-general-token-rate-16gb`

Plan commit: `695bb81c3`

## Scope

Instrumentation only. This does not claim token-rate improvement or SOTA.

## Change

Added a default-off route detail CSV controlled by:

```bash
GGML_MOE_ROUTE_DETAIL_OUT=/path/to/route-detail.csv
```

The existing `GGML_MOE_ROUTE_TRACE_OUT` format is unchanged.

New CSV fields:

```text
seq,call,mode,kind,tensor,layer,active_index,expert_idx,dst_id,flat_dst_id,token_id,n_active,expert_bytes,src0_type,ne01,ne00
```

Coverage:

- up/gate path records `token_id`, `dst_id`, `flat_dst_id`, layer, kind, and
  active expert for both tensors;
- down path records call/layer/kind/active expert with `token_id=-1`, because
  the current down batch function does not carry token ids.

## Build verification

Remote temporary worktree:

```text
/root/lfz/tmp/gp28-route-detail-build-1783366765
```

Patch source:

```bash
git diff -- ggml/src/ggml-cuda/moe_stream_batch.cu
```

Configure/build command:

```bash
cmake -S . -B build-cuda-batch -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120a-real \
  -DLLAMA_CURL=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_NATIVE=ON
cmake --build build-cuda-batch -t ggml-cuda -j2
```

Result:

- `moe_stream_batch.cu` compiled successfully.
- `ggml-cuda` target linked successfully.
- Final build lines:

```text
[149/150] Linking CUDA shared library bin/libggml-cuda.so.0.10.0
[150/150] Creating library symlink bin/libggml-cuda.so.0 bin/libggml-cuda.so
```

## Next

Run a small cold-start dev prompt with `GGML_MOE_ROUTE_DETAIL_OUT` enabled to
collect detail trace, then analyze per-layer same-token and next-token route
stability before implementing any predictor.
