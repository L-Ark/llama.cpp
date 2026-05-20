# ik_llama_fork — MoE expert prefetch experiments

Fork of ikawrakow/ik_llama.cpp adding opt-in machinery to push the
mmap-backed MoE expert reads closer to NVMe peak bandwidth.

All additions are OFF by default; enable via environment variables.

## Env vars

| variable | effect |
| --- | --- |
| `GGML_MOE_PREFETCH=1` | issue a prefetch hint on each active expert before its matmul |
| `GGML_MOE_PREFETCH_MODE=madvise\|fadvise\|readahead\|uring` | which mechanism to use; `madvise` is the cheapest, `uring` issues real async reads via io_uring at QD 64 |
| `GGML_MOE_PREFETCH_STRIDE=N` | cap bytes prefetched per expert range (default = whole expert) |
| `GGML_MOE_PARALLEL_EXPERTS=1` | each CPU worker thread takes a disjoint stripe of active experts and runs each one single-threaded — more concurrent file streams, but slower per-expert matmul |
| `GGML_MOE_PREDICT=1` | spin up a background thread that re-reads (via `pread`) the most recent token's expert ranges, layer-ordered.  Tries to warm the page cache one or more layers ahead of the compute path. |
| `GGML_MOE_PREFETCH_DEBUG=1` | print a startup line showing what's enabled |

## What's in the diff

- `ggml/src/ggml.c`
  - mapping registry: `ggml_vm_register_mapping` / `ggml_vm_lookup` translate an mmap address to (fd, offset) so we can hit the file's fd directly
  - `ggml_moe_prefetch_range` dispatcher for madvise / fadvise / readahead / io_uring
  - io_uring ring + reaper thread for `MODE=uring`
  - layer-ordered predictor bucket + worker thread (`GGML_MOE_PREDICT=1`)
  - parallel-experts branch inside `ggml_compute_forward_mul_mat_id` and `…_id_up_gate`
- `src/llama-mmap.cpp` — call `ggml_vm_register_mapping(addr, size, dup(fd))` right after each `mmap()` so the registry is populated.
- `ggml/src/CMakeLists.txt` — link `liburing` if present.

## Empirical results (Kimi-K2.6 IQ3_KS, 16 GB effective RAM, see `RESULTS_OPT.md`)

None of the prefetch hints beat the kernel's own readahead under tight memory.
The actual win is at the workload level: batching prefill from pp16 (0.41 t/s)
up to pp1024 (7.06 t/s).
