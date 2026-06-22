# DeepSeek V4-Flash SSD MoE Runtime Proceed Log

## 2026-06-15 11:31 CST

### Plan Compliance

- Active plan: `plan.md`
- Record-only file: `proceed.md`
- `plan.md` must not be modified during this phase.
- Work stays under `/home/wici/lfz`.
- Other users' GPU or RAM-heavy processes must not be stopped.

### Initial State

- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB total, 41 MiB used, 32,071 MiB free.
- GPU compute processes: none reported by `nvidia-smi`.
- RAM: 15 GiB total, 12 GiB available.
- Disk under `/home/wici/lfz`: 365 GiB available.
- Full DeepSeek V4-Flash weights: not present locally.
- Metadata-only DeepSeek V4-Flash snapshot: present at `models/DeepSeek-V4-Flash-metadata`.
- fastLLM runtime: not present locally.

### Immediate Work Items

1. Locate and clone fastLLM under `/home/wici/lfz`.
2. Inspect fastLLM DeepSeekV4 and disk-MoE support.
3. Prepare a full-model download path under `/home/wici/lfz`.
4. Start with fastLLM baseline if the runtime and model format are usable.
5. If fastLLM is blocked, document the blocker and continue to the KTransformers/SGLang MXFP4 SSD spike.

## 2026-06-15 11:32 CST

### User Account Note

- User provided GitHub identity:
  - username: `L-Ark`
  - email: `fliangae@connect.ust.hk`
- This identity was configured only in `/home/wici/lfz/fastllm/.git/config`.
- GitHub username/email alone is not an authentication credential for Hugging Face model downloads.
- DeepSeek V4-Flash is public on Hugging Face, so the model download can proceed without an HF token, subject to public rate limits.

### fastLLM Setup

- fastLLM cloned to `/home/wici/lfz/fastllm`.
- fastLLM revision: `1a3cbb94b74dcf20b9d81d02c9e460d03f741b85`.
- Installed `ftllm` wheel in `/home/wici/lfz/fastllm/.venv-ftllm`.
- Installed `ftllm` version: `0.1.7.0`.
- `docs/mixforward.md` confirms `DeepSeek-V4-Flash` mixed inference examples with `cuda`, `numa`, and `disk` in `--moe_device`.
- `docs/benchmark.md` confirms `ftllm bench` reports TTFT and token throughput.

### Download Preflight

- Target model source: `deepseek-ai/DeepSeek-V4-Flash`.
- Target model path: `/home/wici/lfz/models/DeepSeek-V4-Flash`.
- Repository revision from HF metadata: `553034d7dd9e06c2eeaee68cf85a17d6d4754cf0`.
- Repository size from HF file inventory: 148.671 GiB.
- Disk free before full download: 362 GiB.
- GPU state before download: RTX 5090, 41 MiB used, no compute process listed.
- RAM before download: 15 GiB total, 11 GiB available.
- Download tool availability: `aria2c` and `wget` are installed.

### Full Model Download Progress

- Started command:

```bash
FASTLLM_CACHEDIR=/home/wici/lfz/fastllm_cache \
/home/wici/lfz/fastllm/.venv-ftllm/bin/ftllm download deepseek-ai/DeepSeek-V4-Flash \
  --local-dir /home/wici/lfz/models/DeepSeek-V4-Flash \
  --tool aria2c -x 4 -j 2
```

- Observed throughput: roughly 40-53 MiB/s.
- Observed issue: intermittent 403 responses on HF/Xet signed range URLs when using aria2c. The overall aria2c job continued retrying and advancing to later shards.
- Checkpoint around 11:51 CST:
  - Local model directory size: 60 GiB.
  - `model-*.safetensors` files present: 20 including partial active downloads.
  - Active `.aria2` partial files: 2.
  - Disk free: 302 GiB.
  - GPU compute processes: none reported.
- Checkpoint around 12:14 CST:
  - Local model directory size: 124 GiB.
  - `model-*.safetensors` files present: 38 including partial active downloads.
  - Active `.aria2` partial files: 2.
  - Disk free: 239 GiB.
  - GPU compute processes: none reported.
  - Download is still running and has advanced to shards `model-00037-of-00046.safetensors` / `model-00038-of-00046.safetensors`.

## 2026-06-15 12:24 CST

### Full Model Download Completion

- `ftllm download` completed successfully.
- Final local model path: `/home/wici/lfz/models/DeepSeek-V4-Flash`.
- Final local model directory size: 149 GiB.
- `model-*.safetensors` file count: 46.
- Active `.aria2` partial files: 0.
- `model.safetensors.index.json` coverage check:
  - expected shards from index: 46.
  - actual shards on disk: 46.
  - missing shards: 0.
  - extra shards: 0.
- Safetensors header check:
  - parsed headers: 46.
  - failures: 0.
- Total shard bytes: 159,617,149,040.
- Disk free after download: 213 GiB.

### Runtime Gate Before Benchmark

- GPU check immediately after download:
  - RTX 5090 total VRAM: 32,607 MiB.
  - VRAM used: 30,763 MiB.
  - VRAM free: 1,349 MiB.
  - compute process reported by `nvidia-smi`: `/home/wici/lfz/ik_llama/build-cuda/bin/llama-cli`, about 30,714 MiB VRAM.
- Per user constraint, this process must not be stopped. The benchmark will wait until GPU resources are naturally released.
- RAM check while waiting:
  - total: 15 GiB.
  - available: 10-11 GiB.
  - swap used: about 1.4 GiB.

### Prepared fastLLM Baseline Command

First planned smoke/baseline command after GPU release:

```bash
FASTLLM_CACHEDIR=/home/wici/lfz/fastllm_cache \
FASTLLM_DISK_MOE_LOAD_THREADS=8 \
/home/wici/lfz/fastllm/.venv-ftllm/bin/ftllm bench /home/wici/lfz/models/DeepSeek-V4-Flash \
  --device cuda \
  --moe_device disk \
  --moe_device_layers -1 \
  --gpu_mem_ratio 0.85 \
  --cuda_slab 1024 \
  --input_tokens 128 \
  --output_tokens 256 \
  --batch 1 \
  --warmup 0 \
  --temperature 0 \
  --threads 8
```

- This follows the plan's fastLLM-first path and uses disk MoE without reducing top-k or expert count.
- The plan's 2 GiB CPU RAM tier is treated as a strict objective for the MoE tier; fastLLM exposes disk MoE placement but does not expose a direct RAM-cache-size knob in `ftllm bench`.

## 2026-06-15 13:11 CST

### fastLLM 128/256 Baseline And Optimization Results

Common setup:

- Model: `/home/wici/lfz/models/DeepSeek-V4-Flash`
- Runtime: fastLLM `0.1.7.0`, repo revision `1a3cbb94b74dcf20b9d81d02c9e460d03f741b85`
- GPU: RTX 5090 32 GiB
- Input tokens: 128
- Target output tokens: 256
- Batch: 1
- Warmup: 0
- Decoding: greedy (`sample=false`, `top_p=1.0`, `top_k=1`, effective temperature from benchmark output `1.0`)
- Top-k/expert count: not reduced.
- CPU: AVX2 only; AVX512/AMX unavailable.
- Temporary SSD swap used only to survive fastLLM load-time host-RAM spikes:
  - path: `/home/wici/lfz/deepseek-v4-fastllm.swap`
  - size: 32 GiB
  - reason: without it, fastLLM was killed during model loading at about 13 GiB RSS on a 15 GiB RAM host.

Failed pre-swap / low-memory attempts:

| Placement | Extra flags | Result | Notes |
|---|---:|---|---|
| `disk` | none | failed | killed by SIGKILL during loading near 99%; max RSS 13,165,060 KiB |
| `disk` | `--low` | failed | killed by SIGKILL during loading; max RSS 13,308,408 KiB |

Stable completed runs:

| Placement | Extra flags | TTFT avg | Decode after TTFT | Total tok/s | Prefill | Total time | Output |
|---|---|---:|---:|---:|---:|---:|---:|
| `disk` | `--low`, SSD swap | 18,118.25 ms | 1.06 tok/s | 0.99 tok/s | 7.06 tok/s | 259.6600 s | 256 |
| `{'cuda':1,'disk':9}` | `--low`, SSD swap | 17,100.70 ms | 1.21 tok/s | 1.13 tok/s | 7.49 tok/s | 227.5072 s | 256 |
| `{'cuda':15,'disk':85}` | `--low`, `--cuda_shared_expert true`, SSD swap | 16,139.05 ms | 1.28 tok/s | 1.19 tok/s | 7.93 tok/s | 215.0829 s | 256 |
| `{'cuda':16,'disk':84}` | `--low`, `--cuda_shared_expert true`, `--cuda_slab 256`, SSD swap | 16,107.86 ms | 1.34 tok/s | 1.24 tok/s | 7.95 tok/s | 207.0798 s | 256 |
| `{'cuda':16,'disk':84}` | same as above, repeat 2 | 15,876.57 ms | 1.37 tok/s | 1.26 tok/s | 8.06 tok/s | 202.4138 s | 256 |
| `{'cuda':16,'disk':84}` | same as above, repeat 3 | 15,980.78 ms | 1.29 tok/s | 1.20 tok/s | 8.01 tok/s | 213.7175 s | 256 |

Best stable configuration:

```bash
FASTLLM_CACHEDIR=/home/wici/lfz/fastllm_cache \
FASTLLM_DISK_MOE_LOAD_THREADS=8 \
/home/wici/lfz/fastllm/.venv-ftllm/bin/ftllm bench /home/wici/lfz/models/DeepSeek-V4-Flash \
  --low \
  --device cuda \
  --moe_device "{'cuda':16,'disk':84}" \
  --moe_device_layers -1 \
  --cuda_shared_expert true \
  --gpu_mem_ratio 0.98 \
  --cuda_slab 256 \
  --input_tokens 128 \
  --output_tokens 256 \
  --batch 1 \
  --warmup 0 \
  --temperature 0 \
  --threads 8
```

Best stable 3-repeat statistics:

| Metric | Values | p50 | p95 / worst observed |
|---|---:|---:|---:|
| TTFT | 15,876.57 / 15,980.78 / 16,107.86 ms | 15,980.78 ms | 16,107.86 ms |
| TPOP | 731.52 / 748.91 / 775.44 ms/token | 748.91 ms/token | 775.44 ms/token |
| Decode after TTFT | 1.29 / 1.34 / 1.37 tok/s | 1.34 tok/s | 1.29 tok/s worst |
| Total throughput | 1.20 / 1.24 / 1.26 tok/s | 1.24 tok/s | 1.20 tok/s worst |
| Prefill | 7.95 / 8.01 / 8.06 tok/s | 8.01 tok/s | 7.95 tok/s worst |
| Total time | 202.4138 / 207.0798 / 213.7175 s | 207.0798 s | 213.7175 s |

Unstable upper-bound attempts:

| Placement | Extra flags | Result | Notes |
|---|---|---|---|
| `{'cuda':2,'disk':8}` | `--low`, `--cuda_slab 1024` | failed | SIGSEGV during loading around 33% |
| `{'cuda':18,'disk':82}` | `--low`, `--cuda_shared_expert true`, `--cuda_slab 1024` | failed | CUDA OOM during warmup: needed 1010 MiB, only 810 MiB free |
| `{'cuda':18,'disk':82}` | `--low`, `--cuda_shared_expert true`, `--cuda_slab 256` | failed | still CUDA OOM during warmup: needed 1010 MiB, only 810 MiB free |

Interpretation:

- The poor token rate is expected for all-SSD routed experts. Every decode step requires reading active expert weights from NVMe; even with observed multi-GB/s NVMe reads, the per-token expert traffic dominates.
- Moving more MoE weight to VRAM improves both TTFT and token rate, but the RTX 5090 reaches the stability limit between 16% and 18% GPU MoE placement for this model/runtime.
- The plan's 10 tok/s target is not reachable on this host with the tested fastLLM SSD MoE path and 2 GiB CPU RAM-tier constraint. The configurations that could approach much higher rates require substantially more non-SSD expert residency, e.g. a large CPU/NUMA RAM tier or more VRAM/GPU parallelism.
- The best stable run improved decode from 1.06 tok/s to p50 1.34 tok/s, about +26% over all-disk, while keeping expert count unchanged.

### Cleanup

- Temporary swap file `/home/wici/lfz/deepseek-v4-fastllm.swap` was disabled and removed after benchmark completion.
- Final resource check:
  - GPU VRAM: 41 MiB used / 32,071 MiB free.
  - GPU compute processes: none reported.
  - RAM: 15 GiB total, 12 GiB available.
  - Swap: reverted to the original `/swap.img` only, 4.0 GiB total.
