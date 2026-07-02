# DeepSeek V4 Flash Vendor 冷启动 token rate 迭代计划（v16GB）

## 目标

- 继续在 `vendor` 实现上优化 DS4 cold-start 解码速度，当前 source-backed 可回退 SOTA 为 `2.6 tok/s`；旧 2026-07-01 `2.6 tok/s` 仍作为 historical/forensic observation 保留，但当前 SOTA 来自 2026-07-02 late10 top3 配置，已经通过 pushed-commit clean rebuild rerun。
- 严格保持：
  - Host RAM（含 page cache）`<= 16 GB`；
  - TTFT 不得高于当前 baseline 的 `20%` 阈值；
  - France 提示词输出语义正确且连贯；
  - 以 `vendor` 代码路径为准，不以 `ik_llama` 实现作为最终结果。

## 当前有效 SOTA / 基线（已确认）

- 当前可从 pushed source 干净重建并复跑的 cold-start 合规 SOTA：
  - `vendor` 框架
  - `cpu_moe=40`
  - `GGML_MOE_KEEP_TOPK_UPDOWN=4`
  - `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`
  - `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`
  - `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`
  - `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`
  - `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`
  - cold drop_caches
  - 16GB cgroup
- 当前记录：
  - `eval_tok_s=2.6`（pushed-commit clean rebuild rerun）
  - `prompt_tok_s=0.9`
  - `TTFT=37874.580124ms`
  - `memory_peak_bytes=16000000000`
  - `memory_file_bytes=15002906624`
  - `pgmajfault=288661`
  - `workingset_refault_file=3545116`
  - `ram_ok=true`
  - `correctness_ok=true`
  - pushed rerun commit: `19a0d36cd081feb0bfb90e26380005c74abe0151`
  - source-bearing code commit: `63caf68ba05b65120516ed03d4cd2c7b8dbc97d1`
  - binary version: `9118 (19a0d36cd)`
  - binary sha256: `c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`
  - pushed remote/branch: `https://github.com/wici-ai/ssd-llama.git` / `vendor/deepseek-token-rate-16gb`
  - pushed rerun dir: `/root/lfz/runs/vendor-ds4-16gb/20260702T005156Z-20260702_expert_keep_top3_late10_top4_rest_pushed_rerun/france-cpu40-vram0gb`
  - pushed rerun trace summary: `rows=35151`, `cache_hits=30180`, `cache_misses=4971`, `src0_ms=24305.283`, `total_ms=26552.328`
  - correctness manual review: pass; France answer is semantically correct, coherent, complete, and not repetitive/truncated.
- 旧 `2.6 tok/s` run 仍作为 forensic 排查对象，不作为当前可回退 SOTA：
  - run dir: `/root/lfz/runs/vendor-ds4-16gb/20260701T095526Z-cold-ds4-gate-stream-src1-rowmod-vram12-trace/france-cpu40-vram12gb`
  - 原记录 `eval_tok_s=2.6`, `TTFT=40836ms`, `memory_peak_bytes=16000000000`, `correctness_ok=true`
  - 但后续 clean rebuild/rerun 未复现，且旧 binary/source 状态没有完整可重建证据；找到 exact source/binary 前只能标记为 historical/forensic observation。
- 基准下线记录与回滚依据：
  - `vram13`：CUDA OOM（reject）
  - `vram14`：cache 插入失败、`0.8 tok/s`（reject）
  - `cache12800`：`1.6 tok/s`（reject）

## 2026-07-01 复现排查更新

- 旧 `2.6 tok/s` run 目录：
  - `/root/lfz/runs/vendor-ds4-16gb/20260701T095526Z-cold-ds4-gate-stream-src1-rowmod-vram12-trace/france-cpu40-vram12gb`
- 旧 run 的 `metadata.json` 记录 repo commit 为 `fcc937d01`，但 `stdout.txt` 中实际 binary build 为：
  - `build : b9079-7353439ea`
- 因此旧记录存在“repo HEAD commit”和“实际编译进二进制的 build commit”不一致，后续复现必须同时记录：
  - `git rev-parse HEAD`
  - `llama-cli` stdout 中的 `build : ...`
  - `build-ds4-moe-stream/bin/llama-cli` 的 mtime/sha256
- 干净重建并复现结果：
  - `aa8d6f916` / `f7f9cdc48` / `fcc937d01` 路径：France 输出正确，但 strict cold vram12 只能复现 `1.5~1.6 tok/s`，未复现 2.6。
  - 干净 `7353439ea` 路径：能复现快速 IO 形态（约 `4.0 tok/s`，major faults 约 `360k~378k`），但 France 输出退化为重复点号，`correctness_ok=false`，不可作为合格 SOTA。
- 最新 clean rerun：
  - `/root/lfz/runs/vendor-ds4-16gb/20260701T120246Z-20260701T_clean-aa8d6-vram12-rerun-after-probes`
  - clean `aa8d6f916` / build `b9085-aa8d6f916` / `drop_caches` / 16GB cgroup / same vram12 env；
  - `eval_tok_s=1.5`, `prompt_tok_s=0.6`, `TTFT=47397.847478ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14968168448`, `pgmajfault=615426`, `workingset_refault_file=12425111`, `correctness_ok=true`；
  - trace: `cache_hits=29624`, `cache_inserts=5129`, `src0_ms=26780.662`, `total_ms=29146.073`；
  - 结论：同源码、同命令、同 cgroup、同 cold procedure 仍未复现旧 `2.6 tok/s`。
- 旧 `2.6 tok/s` run 与当前正确输出 run 的核心差异：
  - 命令、环境、VRAM cache 容量一致；
  - VRAM cache 命中计数在旧正确 run 与新正确 run 均为 `hits=29624 misses=5129`；
  - 旧正确 run `pgmajfault≈309k`、`File system inputs≈154862552`、wall `1:29.13`；
  - 新正确 run `pgmajfault≈613k~621k`、`File system inputs≈255827216~259863288`、wall `2:04~2:11`；
  - 说明 2.6 差异主要不是算子选择，而是旧 binary/source/build 状态或 cold page/IO 状态未被完整记录。
- 当前可接受口径：
  - 在可从干净源码重建、France 正确、16GB cgroup 的条件下，当前复现有效值应暂按 `1.5~1.6 tok/s` 处理；
  - 旧 `2.6 tok/s` 保留为历史观测值，但在找到可重建的 exact binary/source 状态前，不应作为新的优化基线或可回退 SOTA。

## 当前 bottleneck 假设

1. 冷启动尾段仍受“专家权重分页/装载”主导：`stream src0` 等待明显，说明主瓶颈可能仍是 expert cache miss 引起的权重搬运与页错误抖动。
2. 计算路径本体（VDR/MMVQ）可能已较平，不应再做大范围改动，优先先级应在：
   1. `load`/`stream`/`prefetch` 的主导路径；
   2. `route` 级别“热点专家预置”与启动时页布局；
   3. `cgroup/cached pages` 下的启动态可复用。

## 实施规则（每次尝试前必做）

### SOTA 记录与发布红线

- **2026-07-02 用户新增硬性要求（最高优先级）：出现符合要求的新 SOTA 时，必须详细记录足以未来完全复现的所有信息，并立刻 push 对应源码到 `https://github.com/wici-ai/ssd-llama.git` 的 `vendor/deepseek-token-rate-16gb` 分支。该动作不能延后到后续实验之后；若没有完整复现信息、源码 commit、指定分支 push、pushed commit clean rebuild/rerun 证据，则该指标一律无效，不得称为当前 SOTA。**
- **新 SOTA 暂停线：任何实验一旦得到更高 token rate 且同时满足 RAM（含 page cache）`<=16GB`、France 正确率和 TTFT gate，必须立即暂停继续探索，先完成 run 目录复现包、源码 commit/push、pushed commit clean rebuild/rerun。只有这些完成后，才允许恢复下一轮优化。**
- **复现包最低粒度：未来操作者必须能只依赖 pushed source + run 记录复现指标；记录必须包含 exact source commit、remote/branch、dirty diff 为空或已纳入 commit、build command、run command、全部 env、模型文件指纹、binary/shared-library 指纹、cgroup 配置与 memory evidence、stdout/stderr、trace、summary、correctness 原文输出、人工判定、TTFT/token-rate/RAM 指标和开始/结束时间。**
- **最高优先级硬规则：出现符合要求的新 SOTA 时，必须当场详细记录完整复现信息，并立刻 push 对应源码到 `https://github.com/wici-ai/ssd-llama.git` 的 `vendor/deepseek-token-rate-16gb` 分支，确保未来一定可以从 pushed source + run 记录完全复现；没有完成这件事的高指标一律无效。**
- **新 SOTA 处理流程必须原子化执行：一旦某次 run 同时满足更高 token rate、16GB RAM（含 page cache）、France 正确率和 TTFT gate，必须立刻停止继续试新参数，先补齐 run 目录内的复现包、把源码和 plan/progress 记录 commit，并 push 到 `ssd` remote 的 `vendor/deepseek-token-rate-16gb` 分支；随后必须从该 pushed commit 干净重建并 rerun 通过，才允许把它写成“当前有效 SOTA”。**
- **任何新 SOTA 都不能只停留在临时服务器、临时二进制、未提交 diff、口头汇报或单次 run 目录中；必须有 pushed source commit、clean rebuild 命令、pushed-commit rerun 结果和完整指标证据。**
- **2026-07-02 强化要求：出现符合要求的新 SOTA 时，必须暂停后续优化，先完成完整复现记录、源码 commit、push 到 `ssd-llama` 指定分支，并从 pushed commit 干净重建复跑通过；否则该结果不能进入“当前最高 token rate”。**
- **不可弱化规则：只要出现符合要求的新 SOTA，必须把未来完全复现所需的信息详细写入 plan/progress/run 目录，并立刻 push 对应源码到 `https://github.com/wici-ai/ssd-llama.git` 的 `vendor/deepseek-token-rate-16gb` 分支；没有详细复现信息和 pushed source 的结果一律无效，不能作为当前最高 token rate。**
- **最高优先级门禁：任何符合 16GB RAM、正确率、TTFT gate 且 token rate 更高的新 SOTA，必须详细记录足以未来完全复现的全部信息，并立刻 push 对应源码到 `https://github.com/wici-ai/ssd-llama.git` 的 `vendor/deepseek-token-rate-16gb` 分支。没有完成详细复现记录和源码 push 的结果，即使指标更高，也只能标记为 `unpromoted observation`，不得作为当前 SOTA 或后续优化基线。**
- **出现符合要求的新 SOTA 时，必须先完成详细复现记录，再立刻 commit 并 push 源码。**
- **push 目标固定为 `https://github.com/wici-ai/ssd-llama.git` 的 `vendor/deepseek-token-rate-16gb` 分支；不能只 push 到个人 fork、不能只保留服务器临时分支、不能只保留二进制或 run 目录。**
- **run 目录必须能让未来从零复现该指标：包含 pushed commit、clean rebuild 命令、binary 指纹、完整运行命令/环境、16GB cgroup 证据、stdout/stderr、trace、summary、correctness 原文输出。**
- **若复现记录不完整、源码未 push、push 分支不对、clean rebuild/rerun 未过，结果一律不得叫 SOTA，只能标为 unpromoted/rejected。**
- **强调：任何新的合格 SOTA 必须在当场把复现信息记录到足够未来完全重建的粒度，并立刻 push 对应源码到 `ssd-llama`；如果未来无法仅凭 pushed source + run 记录复现该指标，则该指标无效。**
- **promote 顺序不可倒置：先写完整复现信息，再 commit/push 源码到 `https://github.com/wici-ai/ssd-llama.git` 的 `vendor/deepseek-token-rate-16gb`，再从 pushed commit 干净重建复跑；三步全部完成前，不得宣布“当前 SOTA”。**
- **复现记录必须足够具体到未来操作者不依赖记忆也能复跑：包括 exact source commit、pushed remote/branch、dirty diff 为空或已纳入 commit、build 命令、run 命令、env、模型文件指纹、binary/shared-library 指纹、cgroup 配置与内存证据、完整输出、trace 和 summary。**
- **如果某次 run 的指标看起来超过历史 SOTA，但记录缺任何一项，必须立即补齐并复跑；补不齐时必须降级为 forensic observation，不能写成 accepted/promoted SOTA。**

1. 先更新本计划，写明 `attempt_id / hypothesis / expected_delta / rollback`.
2. 建立 run 目录并写 `attempt_start_utc.txt`、`git_start_sha.txt`、`attempt_meta.env`。
3. 先运行一次基线计时切片（不改代码）确认环境状态，再改代码或参数。
4. 实验后必须写：
   - `attempt_end_utc.txt`、`wall_clock_elapsed_seconds.txt`
   - `result_status`（promoted / unpromoted / failed / reverted）
   - 指标、日志路径、commit/回滚状态。
5. 任何可复现的提升必须立即 `commit + push`；不满足限制则回退并记录原因。
6. 出现符合要求的新 SOTA 时，必须立刻记录完整复现信息并 push 源码，目标是未来一定能从记录和源码完全复现该指标。
7. **硬性 promote gate**：没有完整复现信息、没有 pushed source commit、没有 pushed branch/remote、没有 clean rebuild rerun 的结果，一律不得标记为 `promoted`，不得称为当前 SOTA，也不得作为后续优化基线。
8. 新 SOTA 的复现信息必须至少包含：
   - source repo path、remote URL、branch、`git rev-parse HEAD`、`git status --short`；
   - pushed commit SHA，以及已 push 到 `https://github.com/wici-ai/ssd-llama.git` 的 `vendor/...` 分支名；
   - `llama-cli` stdout 中的 `build : ...` 行；
   - `build-ds4-moe-stream/bin/llama-cli` 的 `sha256sum`、mtime、size；
   - exact build command、exact run command、完整环境变量、模型路径和模型 size/mtime；
   - cgroup `MemoryMax` / `MemorySwapMax`、`memory.current`、`memory.peak`、`memory.events`、`memory.stat`；
   - stdout/stderr、summary.json、results.tsv、trace 文件路径、correctness 原文输出。
9. 新 SOTA 必须能从 pushed source commit 干净重建并复跑；如果 binary build commit、repo HEAD、或 pushed commit 任一不一致，必须先修正记录并重新复跑，不能沿用该指标。
10. **不可省略的 SOTA 发布要求**：一旦出现满足 16GB RAM、正确率、TTFT gate 且 token rate 更高的新 SOTA，必须在同一阶段立刻提交并 push 对应源码到 `https://github.com/wici-ai/ssd-llama.git` 的 `vendor/deepseek-token-rate-16gb` 分支，不能只保留本地改动、远端服务器临时改动或二进制。run 目录必须完整记录 pushed source commit、build binary 指纹、clean rebuild/rerun 命令和结果，确保未来从源码、命令和记录一定可以复现该指标。
11. 如果 TTFT 超过 20% 但 token rate 有参考价值，可以 commit/push 并明确标记为 `not accepted`，但不得替代当前 accepted SOTA；后续必须把 TTFT 压回 gate 内才可 promoted。

## 下一轮 cold-start 实验计划（按优先级）

### 当前执行 attempt：clean-baseline-aa8d6-repro

- `attempt_id`: `20260701-clean-baseline-aa8d6-repro`
- `attempt_kind`: `baseline-repeat`
- `hypothesis`: 当前可干净重建且正确的 vendor DS4 vram12 baseline 应稳定在 `1.5~1.6 tok/s`；先用完整复现证据重新确认，后续优化只基于该 clean baseline 推进。
- `expected_delta`: 无性能提升预期；本次目标是建立可 push-source 复现的干净基线。
- `rollback`: 不改源码；若运行异常，仅记录为 failed，不改变当前分支。
- `required_evidence`: run 目录必须包含 source commit、binary sha256、binary build line、exact command/env、16GB cgroup memory 文件、stdout/stderr、summary/results/trace、France 输出。
- `result`: completed. Clean `aa8d6f916` / binary `b9085-aa8d6f916` strict cold vram12 baseline reproduced `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=47899.91054ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14956642304`, `correctness_ok=true`.
- `trace_bottleneck`: `src0_ms=26845.093ms`, `total_ms=29213.91ms`; `src0` is `91.89%` of traced MoE stream time, kernel is only `1.37%`. Next attempt should target page/refault behavior before compute kernels.

### 当前执行 attempt：dontneed0-aa8d6-vram12

- `attempt_id`: `20260701-dontneed0-aa8d6-vram12`
- `attempt_kind`: `config-probe`
- `hypothesis`: On the corrected clean DS4 gate-stream path, disabling `GGML_MOE_STREAM_DONTNEED` may reduce repeated source-page refaults and `src0_ms`, improving cold token rate while staying inside the 16GB cgroup.
- `expected_delta`: If `pgmajfault` and `src0_ms` drop materially without correctness/TTFT regression, possible gain from `1.6 tok/s` toward `>1.6 tok/s`; theoretical ceiling is bounded by removing up to `dontneed_ms≈1.27s` directly plus secondary refault reduction in the `26.8s` `src0` path.
- `rollback`: No source change. If RAM exceeds 16GB, correctness fails, TTFT exceeds clean baseline by >20%, or eval token rate does not improve, mark `unpromoted` and keep `GGML_MOE_STREAM_DONTNEED=1` as baseline.
- `required_evidence`: same full SOTA-grade run files as baseline, plus trace comparison against `20260701T112048Z-clean-baseline-aa8d6-repro`.
- `result`: completed and rejected. `eval_tok_s=1.5`, `prompt_tok_s=0.6`, `TTFT=47960.972348ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14978125824`, `correctness_ok=true`, but slower than clean baseline.
- `gap_analysis`: disabling `DONTNEED` removed direct `dontneed_ms` (`1273.292ms -> 0.063ms`) but increased `src0_ms` (`26845.093ms -> 28696.34ms`) and total traced time (`29213.91ms -> 29759.59ms`). This likely increases cgroup file-page pressure/refault churn rather than reducing it. Keep `GGML_MOE_STREAM_DONTNEED=1`.

### 当前执行 attempt：defer-sync-aa8d6-vram12

- `attempt_id`: `20260701-defer-sync-aa8d6-vram12`
- `attempt_kind`: `config-probe`
- `hypothesis`: Current stream path uses `defer_sync=0`, synchronizing every streamed expert before returning. Enabling `GGML_MOE_STREAM_DEFER=1` may pipeline up to `MOE_STREAM_NSLOTS=8` expert H2D/kernel/D2H operations and reduce wall time for cache misses.
- `expected_delta`: Direct `sync_ms+scatter_ms` is only about `371ms`, so a pure sync removal would be small. The larger possible gain is overlap of cache-miss H2D and kernel work; if pageable/mmap page faults block at `cudaMemcpyAsync`, gain will be limited. The theoretical upper bound is reducing part of the `5129` miss path (`src0_ms≈26845ms`) by stream overlap, but correctness risk exists because scatter is deferred.
- `rollback`: No source change. Reject if output correctness fails, RAM exceeds 16GB, TTFT exceeds clean baseline by >20%, or token rate does not exceed `1.6 tok/s`.
- `required_evidence`: exact env must show `GGML_MOE_STREAM_DEFER=1`; stderr must show `defer_sync=1`; run must include trace, cgroup memory files, output text, and source/binary reproducibility files.
- `result`: failed/rejected. Run `/root/lfz/runs/vendor-ds4-16gb/20260701T113029Z-defer-sync-aa8d6-vram12` reached only `hits=0 misses=32` and then exceeded acceptable runtime; it was terminated and marked `failed_timeout_defer_sync`.
- `gap_analysis`: `acquire_slot()` spins when all 8 slots are occupied. In `defer_sync=1`, slots are intentionally kept `in_use` until `ggml_cuda_moe_stream_sync()`, but sync is only called after the per-expert loop. Therefore the 9th streamed expert in a single op can wait forever. Need batch flush when slots are exhausted.

### 当前执行 attempt：defer-slot-flush-aa8d6-vram12

- `attempt_id`: `20260701-defer-slot-flush-aa8d6-vram12`
- `attempt_kind`: `implementation`
- `hypothesis`: If `acquire_slot()` flushes the current thread's pending deferred scatters when all slots are busy, `GGML_MOE_STREAM_DEFER=1` can pipeline experts in batches of up to 8 without deadlocking.
- `expected_delta`: Best case hides part of the cache-miss H2D/kernel/D2H latency across slots. Hard upper bound remains the `src0` page fault/read time; if `cudaMemcpyAsync` from mmap host blocks on page faults, measured gain may be small. Correctness should be unchanged because flush uses the existing `ggml_cuda_moe_stream_sync()` scatter path.
- `rollback`: If build fails, run hangs, correctness fails, RAM exceeds 16GB, TTFT exceeds baseline by >20%, or eval token rate does not exceed `1.6 tok/s`, revert the code change and record rejection. Only commit/push if the clean rebuild rerun passes all gates.
- `required_evidence`: source diff, build log, strict run with `GGML_MOE_STREAM_DEFER=1`, output text, cgroup memory files, trace summary, binary sha256/build line, pushed commit/branch if promoted.
- `result`: completed and rejected. The slot-flush fix prevented the previous hang, but strict run `/root/lfz/runs/vendor-ds4-16gb/20260701T113832Z-defer-slot-flush-aa8d6-vram12` produced only `eval_tok_s=1.5`, `prompt_tok_s=0.6`, `TTFT=47686.520289ms`, `memory_peak_bytes=16000000000`, `correctness_ok=true`.
- `gap_analysis`: per-expert trace showed `src0_ms=28309.514ms`, lower direct sync/scatter fields due deferred accounting, but end-to-end token rate still regressed. The batched flush cost and/or pageable mmap copy front-end still dominates; `GGML_MOE_STREAM_DEFER=1` is not accepted.
- `rollback_status`: source change reverted and remote binary rebuilt back to clean branch state. No commit or push.

### 当前执行 attempt：cache13000-aa8d6-vram12

- `attempt_id`: `20260701-cache13000-aa8d6-vram12`
- `attempt_kind`: `config-probe`
- `hypothesis`: Clean baseline has 2891 VRAM cache slots and 5129 misses. Trace simulation shows LRU could reduce misses as cache approaches ~3000-3100 slots; `GGML_MOE_STREAM_ONE_CACHE_MIB=13000` may fit below the `13GB` OOM cliff and reduce `src0_ms`.
- `expected_delta`: Belady lower bound at current sequence is 4534 misses; LRU at ~3060 slots is expected around ~5000 misses. Average miss cost is about `26845ms / 5129 ≈ 5.23ms`, so the hard expected gain is modest (~0.5-1.0s traced time). It is only worth promoting if end-to-end `eval_tok_s` exceeds `1.6` and all gates pass.
- `rollback`: No source change. If CUDA OOM, cache insertion cliff, correctness failure, TTFT > baseline +20%, RAM issue, or no token-rate improvement, mark `unpromoted`.
- `required_evidence`: env must include `GGML_MOE_STREAM_ONE_CACHE_MIB=13000`, stderr cache slot count, trace summary with hits/inserts/src0, cgroup memory evidence, output correctness text.
- `result`: completed and not promoted. Run `/root/lfz/runs/vendor-ds4-16gb/20260701T114441Z-cache13000-aa8d6-vram12` produced `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=47412.352435ms`, `memory_peak_bytes=16000000000`, `correctness_ok=true`.
- `gap_analysis`: cache increased to `12.7 GiB`, `3058` slots; misses dropped `5129 -> 4985` and `src0_ms` dropped `26845.093ms -> 25611.349ms`, but rounded token rate did not exceed clean baseline. Continue only as boundary mapping; not SOTA.

### 当前执行 attempt：cache13200-aa8d6-vram12

- `attempt_id`: `20260701-cache13200-aa8d6-vram12`
- `attempt_kind`: `config-probe`
- `hypothesis`: `13000MiB` fits and reduces misses; `13GB` (`13312MiB`) previously OOM. `13200MiB` may be close enough to the fit cliff to reduce additional misses without hitting CUDA OOM.
- `expected_delta`: Simulation suggests around `~4950` LRU misses near 3100 slots, i.e. only tens fewer misses than `13000MiB`. Expected gain is small; promote only if measured `eval_tok_s > 1.6` and all gates pass.
- `rollback`: No source change. Reject on CUDA OOM, cache insertion cliff, no token-rate improvement, correctness failure, TTFT > baseline +20%, or RAM issue.
- `required_evidence`: stderr cache slot count, trace summary, exact env with `GGML_MOE_STREAM_ONE_CACHE_MIB=13200`, cgroup memory files, output correctness.
- `result`: completed and not promoted. Run `/root/lfz/runs/vendor-ds4-16gb/20260701T114805Z-cache13200-aa8d6-vram12` produced `eval_tok_s=1.6`, `prompt_tok_s=0.6`, `TTFT=47405.45086ms`, `memory_peak_bytes=16000000000`, `correctness_ok=true`.
- `gap_analysis`: cache increased to `12.9 GiB`, `3105` slots; misses dropped to `4956` and `src0_ms=25293.202ms`, but token rate still did not exceed clean baseline. Cache-size-only gains are too small near the 13GB OOM cliff.

### 下一步设计方向：miss read-ahead / page fault hiding

- `finding`: At 3105 cache slots, remaining misses are still `4956`; `src0` still accounts for `91.58%` of trace time. LRU/Belady simulation shows only `~422` extra misses above theoretical unique-load lower bound at this capacity, so policy-only work has limited upside.
- `hypothesis`: For each `mul_mat_id`, the CPU loop already knows `matrix_row_counts[cur_a]` and processes experts in order. Issuing `MADV_WILLNEED` for the next few nonzero experts before the current `cudaMemcpyAsync` may overlap file page-in with current H2D/kernel/scatter and reduce later miss `src0_ms`.
- `risk`: The 16GB cgroup includes page cache. Excessive lookahead can increase file cache pressure and refaults; must use small depth and keep `DONTNEED=1`.
- `next_attempt_candidate`: implement default-off env `GGML_MOE_STREAM_WILLNEED_LOOKAHEAD=<N>` in the CPU expert loop, starting with `N=4` or `N=8`, and reject unless correctness/RAM/TTFT gates pass and token rate exceeds baseline.

### 当前执行 attempt：stream-willneed-lookahead4

- `attempt_id`: `20260701-stream-willneed-lookahead4`
- `attempt_kind`: `implementation`
- `hypothesis`: A narrow `MADV_WILLNEED` lookahead inside the GPU stream expert loop may overlap future expert page-in with current expert H2D/kernel work. This differs from the rejected broad `GGML_MOE_CPU_WILLNEED=1`, which prefetched all active experts for CPU fallback and increased reclaim pressure.
- `expected_delta`: Current miss path has `src0_ms≈25-27s`. A small lookahead can only help if Linux performs useful async readahead before the later `cudaMemcpyAsync` touches the mapped pages. The upper bound is a fraction of miss `src0_ms`; if reclaim contention dominates, it may regress.
- `implementation`: Add default-off env `GGML_MOE_STREAM_WILLNEED_LOOKAHEAD=<N>` in `ggml_compute_forward_mul_mat_id` when `use_gpu_stream` is true. For each current streamed expert, prefetch the next `N` nonzero expert source pages using existing `ggml_moe_cpu_willneed_pages`.
- `test_config`: start with `N=4`, clean branch `aa8d6f916`, `vram_cache=12GB`, `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace.
- `rollback`: If build fails, RAM exceeds 16GB, correctness fails, TTFT exceeds baseline by >20%, or `eval_tok_s <= 1.6`, revert source change and mark unpromoted. Commit/push only if clean rebuild rerun passes all SOTA gates.
- `required_evidence`: source diff, build log, exact env, stderr build line/cache stats, trace summary, cgroup memory files, output text, binary sha256/build line.
- `result`: completed and not promoted. Run `/root/lfz/runs/vendor-ds4-16gb/20260701T115559Z-20260701T_stream-willneed-lookahead4-aa8d6-vram12` produced `eval_tok_s=1.3`, `prompt_tok_s=0.7`, `TTFT=48627.600657ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14967099392`, `pgmajfault=424244`, `workingset_refault_file=15492625`, `correctness_ok=true`.
- `answer`: France output was coherent and semantically correct: France is described as a Western European country with rich history, culture, landmarks including the Eiffel Tower/Louvre/Versailles, cuisine, wine, fashion, EU membership, diverse landscape, and Paris as capital.
- `trace_summary`: `rows=34753`, `cache_hits=29624`, `cache_inserts=5129`, `src0_ms=22965.873`, `total_ms=25378.847`.
- `gap_analysis`: Direct traced `src0_ms` fell versus the clean baseline, but end-to-end generation regressed from `1.6` to `1.3 tok/s` and major faults/refault pressure rose. The likely cost moved outside the per-expert traced region into earlier page-in/reclaim/TTFT scheduling, so synchronous `MADV_WILLNEED` lookahead is rejected.
- `rollback_status`: revert `ggml/src/ggml-cpu/ggml-cpu.c` lookahead source change and rebuild clean; no commit/push.

### A. 算法/数据面复查（设计优先）

1. 在不改模型参数前提下，细分 `src0` 与 `src1` 上下游耗时：
   - 增加/核验 trace 分段字段，定位 `page fault`、`read wait`、`copy wait` 与核心算子的占比。
   - 输出：每段占比、可并行区间、是否存在串行等待链。
2. 结合 trace 估算瓶颈上界：
   - 用 `stream 读放大` 和 `热专家命中率` 做理论上界；
   - 计算“若 page cache 完美命中”可达到的 token rate。
3. 输出一页「瓶颈优先级列表」，确认下个代码尝试只改一件事。

### B. 冷启动路径执行优化（逐条）

### 当前执行 attempt：exact-2p6-binary-source-forensic

- `attempt_id`: `20260701-exact-2p6-binary-source-forensic`
- `attempt_kind`: `forensic-design`
- `hypothesis`: 旧 `2.6 tok/s` 正确 run 的 build line 为 `b9079-7353439ea`，但 clean `7353439ea` 输出退化，说明当时可能存在未提交源码、不同 shared object、旧 build 目录残留、或 run metadata 未记录的二进制状态。只有找到 exact source/binary 状态，才能把 `2.6` 从历史观测恢复为可回退 SOTA。
- `expected_delta`: 本步骤不期望直接提升 token rate；期望产出可验证的 exact binary/source 证据。如果找到旧 binary 或对应 dirty diff，理论上可复现 `2.6 tok/s`，并立刻按 SOTA 规则 clean rebuild/rerun、commit、push 到 `https://github.com/wici-ai/ssd-llama.git` 的 `vendor/deepseek-token-rate-16gb`。
- `search_scope`: old run directory, build directories, `/tmp` source backups, git reflog, branch history, run metadata, `strings` build lines, `sha256sum` of `llama-cli` and resolved shared libraries, and relevant diffs in `ggml/src/ggml-cuda/moe_stream.cu` / `ggml/src/ggml-cpu/ggml-cpu.c`.
- `rollback`: no source change unless an exact candidate is found. Any checkout/build probe must return to clean `aa8d6f916` state if rejected.
- `required_evidence`: file paths for any candidate binary/source, sha256/mtime/size, stdout build line, source commit/diff, exact old run command/env, and a strict 16GB cold rerun result.
- `finding_build_line`: `build : b9079-7353439ea` appears across multiple later run directories, including both incorrect fast DS4 stream trials and the corrected vram8/vram12 runs. Therefore this stdout build line is stale/incomplete and cannot identify exact source by itself.
- `finding_source_history`: the old 2.6 run happened after the src1 row-mod correctness fix was created and after vram8 evidence; recorded `.Agent/runs/.../accepted-cold-ds4-gate-stream-vram12.json` lists `base_commit=fcc937d01`. The only source files changed by the correctness fix versus `7353439ea` are `ggml/src/ggml-cpu/ggml-cpu.c` and `ggml/src/ggml-cuda/moe_stream.cu`.
- `finding_binary`: current server no longer has the old 09:55 binary/shared-object files; `build-ds4-moe-stream` has been overwritten by later clean rebuilds. No preserved candidate `llama-cli`/`libggml-cuda.so` from the exact 09:55 state was found in the repo/build directories.
- `finding_cold_gate`: old run `run_case.sh` records `sync; echo 3 > /proc/sys/vm/drop_caches`, and `memory.peak=16000000000`; resource samples show the cgroup stayed near the 16GB ceiling. The old result was not an obvious unrestricted warm-page-cache run.
- `finding_perf_delta`: old trace has `rows=34753`, `cache_hits=29624`, `cache_inserts=5129`, `src0_ms=23564.133`, `total_ms=25786.372`, `trace_span_ms=71223.061`, `pgmajfault=309216`, `workingset_refault_file=3371495`. Clean reruns have the same hit/insert counts but `pgmajfault≈600k+`, `workingset_refault_file≈12M+`, trace span `~108s+`, and `eval_tok_s=1.5~1.6`. This points to IO/page-fault/reclaim state or unrecorded binary/shared-object state, not routing/cache policy.
- `next_action`: run an exact copied version of the old `run_case.sh` under the same `systemd-run MemoryMax=16000000000 MemorySwapMax=0` and current clean source/binary to rule out strict runner drift. If it still reproduces only `1.5~1.6`, treat the remaining delta as storage/page-fault state or lost binary state and continue with cold IO/reclaim instrumentation.

### 当前执行 attempt：old-run-case-copy-rerun

- `attempt_id`: `20260701-old-run-case-copy-rerun`
- `attempt_kind`: `forensic-execution`
- `hypothesis`: The strict runner may have changed after 09:55 in a way that changes cold IO/page-cache behavior. Reusing the exact old generated `run_case.sh` structure with only paths/run name changed should reproduce the old behavior if runner drift is the cause.
- `expected_delta`: If runner drift caused the 2.6 result, this rerun should recover lower `pgmajfault` and higher token rate near `2.6`. If it remains `1.5~1.6`, runner drift is ruled out.
- `test_config`: current clean `aa8d6f916` source, clean rebuilt binary, copied old run script, same env (`GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, `GGML_MOE_VRAM_CACHE_GB=12`), cold `drop_caches`, 16GB cgroup, France prompt.
- `rollback`: no source change. Record result and reject unless it reproduces a valid SOTA and can then be tied to pushed source/binary evidence.
- `required_evidence`: copied script path, exact systemd command, stdout/stderr, summary, memory files, trace summary, binary/shared-object sha, source commit/status.
- `result`: completed and rejected. Run `/root/lfz/runs/vendor-ds4-16gb/20260701T150000Z-old-run-case-copy-rerun/france-cpu40-vram12gb` reused the old generated `run_case.sh` structure but current clean `aa8d6f916` source/binary. It produced `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=45537.303107ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14962831360`, `pgmajfault=610841`, `workingset_refault_file=12628329`, `correctness_ok=true`.
- `trace_summary`: same cache behavior (`cache_hits=29624`, `cache_inserts=5129`) but slower page/reclaim behavior: `src0_ms=25625.754`, `total_ms=27975.553`, `trace_span_ms=110830.339`.
- `conclusion`: strict runner drift is ruled out. The unreproduced 2.6 delta is not from the generated run script; it is either lost binary/shared-object state or storage/page-fault/reclaim state that was not captured by source/run metadata.

### 当前执行 attempt：cpu-moe38-vram10-gpu-layer-tradeoff

- `attempt_id`: `20260701-cpu-moe38-vram10-gpu-layer-tradeoff`
- `attempt_kind`: `config-probe`
- `hypothesis`: `--n-cpu-moe 40` keeps the first 40 MoE expert layers on CPU. The current bottleneck after gate streaming is the untraced up/down CPU fallback gap. Reducing to `cpu_moe=38` should move two MoE layers' experts back to GPU and reduce CPU fallback/page-fault work, while shrinking stream cache from 12GB to 10GB to stay within VRAM.
- `theoretical_upper_bound`: Gate-only control shows `~80s+` inter-stream gap. Removing 2 of 40 CPU-MoE layers can at most reduce roughly `5%` of CPU fallback layer work before accounting for layer imbalance. The cost is a smaller gate stream cache: old vram8 had `7110` inserts and vram12 has `5129`, so vram10 should land between them. Accept only if reduced CPU fallback gap outweighs extra gate misses.
- `test_config`: clean `aa8d6f916`, `cpu_moe=38`, `vram_cache=10GB`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace.
- `rollback`: no source change. Reject if CUDA OOM, RAM violation, correctness failure, TTFT > gate, or token rate does not exceed current reproducible clean `1.6 tok/s`.
- `required_evidence`: exact command/env, cgroup memory files, stdout answer, trace summary, binary/source sha, and result status.
- `result`: completed and rejected. Run `/root/lfz/runs/vendor-ds4-16gb/20260701T145202Z-20260701T_cpu-moe38-vram10-gpu-layer-tradeoff/france-cpu38-vram10gb` produced `eval_tok_s=0.9`, `prompt_tok_s=0.6`, `TTFT=57141.168244ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14975692800`, `pgmajfault=616783`, `workingset_refault_file=24626506`, `correctness_ok=true`.
- `trace_summary`: `rows=33007`, `cache_hits=0`, `cache_inserts=0`, `src0_ms=86128.942`, `total_ms=93662.348`, `trace_span_ms=183595.874`.
- `gap_analysis`: Lowering `cpu_moe` from 40 to 38 consumed enough VRAM that the gate stream cache stopped inserting entirely even at `vram_cache=10GB`. The intended CPU fallback reduction did not matter because gate streaming fell off the cache cliff. Do not continue lowering `cpu_moe` unless the stream cache size is reduced/fitted and the resulting miss count has a plausible upper bound.
- `next_action`: avoid further `cpu_moe<40` probes without first checking cache initialization/insertion in a short dry run. More promising next steps are cold IO/reclaim instrumentation or freeing VRAM from non-cache allocations while preserving `cpu_moe=40`.

### 当前执行 attempt：cpu-fallback-dontneed-after-op

- `attempt_id`: `20260701-cpu-fallback-dontneed-after-op`
- `attempt_kind`: `implementation-probe`
- `hypothesis`: The current traced gap is primarily non-streamed `ffn_up_exps.weight` and `ffn_down_exps.weight` CPU fallback. Those mmap expert pages remain in page cache after the CPU fallback op, competing with later stream/cache misses inside the same 16GB cgroup. A default-off `MADV_DONTNEED` after the CPU fallback op may reduce cgroup reclaim/refault churn for low-value up/down pages while preserving gate stream cache behavior.
- `theoretical_upper_bound`: This cannot reduce arithmetic time inside the current CPU fallback op. It can only reduce later page-cache pressure and subsequent fault/reclaim stalls. Upper bound is therefore some fraction of the inter-op gap/refault component (`workingset_refault_file≈12M+`, `gap_sum≈80s+`), but it may regress badly if the same expert pages are reused soon.
- `implementation`: Add default-off env `GGML_MOE_CPU_DONTNEED_AFTER_EXPERT=1` in `ggml_compute_forward_mul_mat_id`. After all CPU fallback chunks for an op are complete, synchronize threads, then thread 0 calls `madvise(MADV_DONTNEED)` on the still-CPU-computed expert source pages. Add a final barrier before returning.
- `test_config`: dirty default-compatible patch on clean `aa8d6f916`, `cpu_moe=40`, `vram_cache=12GB`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace, `GGML_MOE_CPU_DONTNEED_AFTER_EXPERT=1`.
- `rollback`: revert source and rebuild clean unless token rate exceeds current reproducible `1.6 tok/s` and all hard gates pass.
- `required_evidence`: source diff, exact env/command, cgroup memory files, France output, trace summary, binary/shared-object sha, result status.
- `result`: completed and rejected. Run `/root/lfz/runs/vendor-ds4-16gb/20260701T150136Z-20260701T_cpu-fallback-dontneed-after-op/france-cpu40-vram12gb` produced `eval_tok_s=1.4`, `prompt_tok_s=0.7`, `TTFT=49312.712067ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14964932608`, `pgmajfault=575185`, `workingset_refault_file=14882265`, `correctness_ok=true`.
- `answer`: France output remained semantically correct and coherent, covering France's history/culture, landmarks, cuisine/wine/fashion, EU membership, varied landscapes, and Paris as capital.
- `trace_summary`: gate stream behavior stayed mechanically identical (`cache_hits=29624`, `cache_inserts=5129`), with `src0_ms=23453.886`, `total_ms=25623.046`, but `trace_span_ms=121771.765`.
- `gap_analysis`: Dropping CPU fallback expert pages after each op reduced traced gate `src0_ms` but increased end-to-end span and refault pressure. This means the up/down pages still have near-term reuse value, and aggressive CPU fallback `MADV_DONTNEED` causes extra re-reads later. Do not promote.
- `rollback_status`: source change reverted and clean rebuild restored `ggml/src/ggml-cpu/ggml-cpu.c` / `ggml/src/ggml-cuda/moe_stream.cu` diff to `0`; clean `llama-cli` sha256 `c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`, `libggml-cpu.so.0.10.0` sha256 `38c2a657261227252ac214e1aab81f2397706017a91fc061b2703dc51e6089bc`. No commit or push.

### 当前执行 attempt：cpu-fallback-willneed-lookahead1

- `attempt_id`: `20260701-cpu-fallback-willneed-lookahead1`
- `attempt_kind`: `implementation-probe`
- `hypothesis`: Broad `GGML_MOE_CPU_WILLNEED=1` reduced major faults on vram12 (`~412k`) but did not improve token rate, likely because it prefetched all active up/down experts and created reclaim pressure. A per-expert CPU fallback lookahead of one active expert may overlap page-in for the next expert with CPU compute of the current expert while avoiding broad prefetch pressure.
- `theoretical_upper_bound`: It can only hide page-in/read stalls for non-streamed up/down fallback; it cannot reduce arithmetic. If one-expert lookahead hides a meaningful part of the `ffn_up/down` fallback wall (`~41s + ~36s` in op trace) without increasing reclaim, it could recover seconds. If CPU compute is not long enough to cover page-in, or if madvise work is synchronous/pressure-heavy, it will not improve.
- `implementation`: Add default-off env `GGML_MOE_CPU_WILLNEED_LOOKAHEAD=<N>` inside the CPU fallback expert loop. For `N=1`, thread 0 advises the next nonzero expert page before computing the current expert; no full-op prefetch and no source semantics change.
- `test_config`: dirty default-compatible patch on clean `aa8d6f916`, `GGML_MOE_CPU_WILLNEED_LOOKAHEAD=1`, `cpu_moe=40`, `vram_cache=12GB`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace.
- `rollback`: revert source and rebuild clean unless token rate exceeds current reproducible `1.6 tok/s` and all hard gates pass.
- `required_evidence`: source diff, exact env/command, cgroup memory files, France output, trace summary, binary/shared-object sha, result status.
- `result`: completed and not promoted. Run `/root/lfz/runs/vendor-ds4-16gb/20260701T150751Z-20260701T_cpu-fallback-willneed-lookahead1/france-cpu40-vram12gb` produced `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=45296.879382ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14967140352`, `pgmajfault=536298`, `workingset_refault_file=11838481`, `correctness_ok=true`.
- `answer`: France output remained semantically correct and coherent, with the same France paragraph structure as accepted baseline.
- `trace_summary`: `cache_hits=29624`, `cache_inserts=5129`, `src0_ms=25770.752`, `total_ms=28117.148`, `trace_span_ms=107430.152`.
- `gap_analysis`: Narrow lookahead reduced TTFT and refault counters versus clean baseline, but generation rate stayed at rounded `1.6 tok/s`. The trace span remained far above the old unreproduced `2.6` run (`71.2s`), so this is useful evidence but not a SOTA. A slightly larger lookahead could be explored only if the plan first sets a strict stop condition; broad `GGML_MOE_CPU_WILLNEED=1` was already not promoted.
- `rollback_status`: source change reverted and clean rebuild restored `ggml/src/ggml-cpu/ggml-cpu.c` / `ggml/src/ggml-cuda/moe_stream.cu` diff to `0`; clean `llama-cli` sha256 `c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`, `libggml-cpu.so.0.10.0` sha256 `38c2a657261227252ac214e1aab81f2397706017a91fc061b2703dc51e6089bc`. No commit or push.

### 当前执行 attempt：cpu-fallback-willneed-lookahead2

- `attempt_id`: `20260701-cpu-fallback-willneed-lookahead2`
- `attempt_kind`: `implementation-probe`
- `hypothesis`: `GGML_MOE_CPU_WILLNEED_LOOKAHEAD=1` improved TTFT/refault counters but did not improve rounded generation rate. Increasing the narrow lookahead to `2` may overlap more of the next expert page-in with current CPU fallback compute while still avoiding the broad all-active prefetch pressure that made `GGML_MOE_CPU_WILLNEED=1` unpromoted.
- `theoretical_upper_bound`: Same as lookahead1: only page-in/read stalls in non-streamed up/down fallback can be hidden. Expected gain is small unless two-expert page-in latency can overlap with enough CPU chunk work. If trace span remains near `~107s` and token rate remains `<=1.6`, this subdirection should stop.
- `implementation`: Reuse the default-off `GGML_MOE_CPU_WILLNEED_LOOKAHEAD=<N>` patch; test only `N=2`.
- `test_config`: dirty default-compatible patch on clean `aa8d6f916`, `GGML_MOE_CPU_WILLNEED_LOOKAHEAD=2`, `cpu_moe=40`, `vram_cache=12GB`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace.
- `stop_gate`: If `eval_tok_s <= 1.6` or `trace_span_ms >= 100000`, reject, revert source, and do not continue larger CPU fallback WILLNEED lookaheads without a new bottleneck measurement.
- `rollback`: revert source and rebuild clean unless token rate exceeds current reproducible `1.6 tok/s` and all hard gates pass.
- `required_evidence`: source diff, exact env/command, cgroup memory files, France output, trace summary, binary/shared-object sha, result status.
- `result`: completed and rejected. Run `/root/lfz/runs/vendor-ds4-16gb/20260701T151405Z-20260701T_cpu-fallback-willneed-lookahead2/france-cpu40-vram12gb` produced `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=59464.568311ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14961750016`, `pgmajfault=521625`, `workingset_refault_file=12015113`, `correctness_ok=true`.
- `answer`: France output remained semantically correct and coherent.
- `trace_summary`: `cache_hits=29624`, `cache_inserts=5129`, `src0_ms=26701.759`, `total_ms=29028.183`, `trace_span_ms=110591.227`.
- `gap_analysis`: Lookahead `2` did not improve rounded generation rate, worsened TTFT, and failed the stop gate (`trace_span_ms >= 100000`). The CPU fallback WILLNEED lookahead subdirection should stop until a new bottleneck measurement shows a different prefetch target.
- `rollback_status`: source change reverted and clean rebuild restored `ggml/src/ggml-cpu/ggml-cpu.c` / `ggml/src/ggml-cuda/moe_stream.cu` diff to `0`; clean `llama-cli` sha256 `c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`, `libggml-cpu.so.0.10.0` sha256 `38c2a657261227252ac214e1aab81f2397706017a91fc061b2703dc51e6089bc`. No commit or push.

### 当前执行 attempt：cpu-fallback-thread-count-32

- `attempt_id`: `20260701-cpu-fallback-thread-count-32`
- `attempt_kind`: `config-probe`
- `hypothesis`: Comparing old `2.6 tok/s`, current gate control, and profile-admission runs shows traced gate stream time can be similar (`~25.8s` old vs `~25.4s` profile) while trace span/wall time remains much slower (`~71s` old vs `~108s` profile). The gap is between gate stream calls and is likely CPU fallback work for non-streamed up/down experts plus other layer work. Increasing llama CPU threads from `20` to `32` may reduce this CPU fallback gap without changing model semantics.
- `theoretical_upper_bound`: Current profile/control trace gap is `~81-83s` versus old `~45s`; if this is CPU parallel work and scaling were ideal from 20 to 32 threads, the gap could drop by up to `~37.5%` (`~30s`). Real gain will be lower due memory bandwidth/page faults and cgroup page-cache pressure. Accepted result must improve token rate while keeping RAM and TTFT gates.
- `risk`: More CPU threads can increase page-cache contention, memory bandwidth pressure, scheduler overhead, and TTFT. It may also lower token rate despite reducing compute wait. This is a no-source config probe; reject if `eval_tok_s <= 1.6`, RAM exceeds 16GB, correctness fails, or TTFT exceeds baseline +20%.
- `test_config`: clean `aa8d6f916`, clean binary sha256 `c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`, `cpu_moe=40`, `vram_cache=12GB`, cold `drop_caches`, 16GB cgroup, France prompt, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, trace enabled, extra CLI args `-t 32 -tb 32`.
- `rollback`: No source change. Record and reject if no token-rate improvement or any hard gate fails. Commit/push only if a later source-backed SOTA passes all reproducibility requirements.
- `required_evidence`: exact command showing final `-t 32 -tb 32`, cgroup memory files, trace gap summary, France output, binary sha256/build line, and promoted/unpromoted status.
- `result`: completed and not promoted. Run `/root/lfz/runs/vendor-ds4-16gb/20260701T133803Z-20260701T_cpu-fallback-thread-count-32` produced `eval_tok_s=1.3`, `prompt_tok_s=0.7`, `TTFT=46279.289147ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14944657408`, `pgmajfault=817039`, `workingset_refault_file=12518918`, `correctness_ok=true`.
- `trace_summary`: `rows=34753`, `cache_hits=29624`, `cache_inserts=5129`, `src0_ms=26103.548`, `total_ms=28587.795`, `trace_span_ms=130873.316`, `gap_sum_ms=102290.221`.
- `gap_analysis`: Raising CPU threads from `20` to `32` worsened generation throughput and increased major faults. The traced gate stream time stayed similar, but inter-stream gap grew from gate-only control `~81.6s` to `~102.3s`. This confirms the current gap is memory/page-cache contention in non-streamed work, not insufficient CPU arithmetic parallelism. Do not pursue higher thread counts.

### 当前执行 attempt：cpu-fallback-thread-count-lower

- `attempt_id`: `20260701-cpu-fallback-thread-count-lower`
- `attempt_kind`: `config-probe`
- `hypothesis`: Since `-t 32 -tb 32` increased page faults and widened the non-streamed fallback gap, the fallback path may be memory/page-cache contention limited. Lowering CPU threads below the default `20` could reduce concurrent page pressure and improve end-to-end token rate even if arithmetic parallelism decreases.
- `theoretical_upper_bound`: If the gate-only `~81.6s` inter-stream gap includes memory contention, reducing contention could recover part of the historical old-run gap (`~45.4s`). If the gap is mostly required CPU arithmetic, lower thread counts will regress. This is a no-source probe and must beat `1.6 tok/s` plus all gates to be accepted.
- `test_config`: clean `aa8d6f916`, clean binary sha256 `c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`, `cpu_moe=40`, `vram_cache=12GB`, cold `drop_caches`, 16GB cgroup, France prompt, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, trace enabled, extra CLI args first `-t 16 -tb 16`, then `-t 12 -tb 12` only if t16 is promising or non-conclusive.
- `rollback`: No source change. Record and reject if token rate does not exceed `1.6`, correctness fails, TTFT exceeds gate, or 16GB cgroup is violated.
- `required_evidence`: exact command showing final thread args, cgroup memory files, trace gap summary, France output, binary sha/build line, result status.
- `result_t16`: completed and not promoted. Run `/root/lfz/runs/vendor-ds4-16gb/20260701T142358Z-20260701T_cpu-fallback-thread-count-16` produced `eval_tok_s=1.5`, `prompt_tok_s=0.7`, `TTFT=46552.546032ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14979936256`, `pgmajfault=828989`, `workingset_refault_file=12213467`, `correctness_ok=true`.
- `trace_summary_t16`: `rows=34753`, `cache_hits=29624`, `cache_inserts=5129`, `src0_ms=26216.355`, `total_ms=28584.152`, `gap_sum_ms=86614.756`, `trace_span_ms=115191.682`.
- `gap_analysis`: Lowering from 20 to 16 threads did not reduce memory/page-cache contention; it widened the fallback gap and reduced token rate. Since `t16` is already worse and `t32` is also worse, skip `t12` unless a later change alters the fallback scheduling model.

### 当前执行 attempt：cpu-fallback-chunk-size-32

- `attempt_id`: `20260701-cpu-fallback-chunk-size-32`
- `attempt_kind`: `implementation-probe`
- `hypothesis`: The CPU fallback `mul_mat_id` path currently uses fixed `chunk_size=16` for expert matrix chunks. For DS4 fallback experts, this creates many chunks per expert and uses an atomic work counter per expert. Raising chunk size to `32` may reduce scheduling/atomic overhead and reduce page interleave while preserving enough parallelism across 20 threads.
- `theoretical_upper_bound`: This can only reduce CPU fallback overhead/gap, not gate stream time. If chunk scheduling overhead is a meaningful part of the `~80s` inter-stream gap, larger chunks could recover seconds. If the gap is dominated by unavoidable memory bandwidth/page faults, token rate will not improve or may worsen from less balanced parallelism.
- `implementation`: Add default-off env `GGML_MOE_CPU_MUL_MAT_ID_CHUNK_SIZE=<N>` used only in `ggml_compute_forward_mul_mat_id`; unset preserves current `16` / `64` behavior. First test `N=32`.
- `test_config`: clean `aa8d6f916` plus default-compatible patch, `cpu_moe=40`, `vram_cache=12GB`, cold `drop_caches`, 16GB cgroup, France prompt, gate stream env, `GGML_MOE_CPU_MUL_MAT_ID_CHUNK_SIZE=32`, trace enabled.
- `rollback`: Revert source and rebuild clean unless token rate exceeds `1.6` and all hard gates pass. Commit/push only after a promoted clean rebuild/rerun from pushed source.
- `required_evidence`: source diff, exact env/command, cgroup memory files, trace gap summary, France output, binary sha/build line, result status.
- `result`: completed and not promoted. Run `/root/lfz/runs/vendor-ds4-16gb/20260701T143032Z-20260701T_cpu-fallback-chunk-size-32` produced `eval_tok_s=1.3`, `prompt_tok_s=0.7`, `TTFT=45371.525178ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14980497408`, `pgmajfault=1312719`, `workingset_refault_file=12487835`, `correctness_ok=true`.
- `trace_summary`: `rows=34753`, `cache_hits=29624`, `cache_inserts=5129`, `src0_ms=26231.393`, `total_ms=28614.213`, `gap_sum_ms=97395.035`, `trace_span_ms=126000.101`.
- `gap_analysis`: Larger CPU fallback chunks significantly worsened the inter-stream gap and major faults. This suggests chunk granularity is contributing to load balance/page wait behavior; do not pursue larger chunk sizes. Source was reverted and binary rebuilt clean (`llama-cli` sha256 `c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`).
- `next_action`: If continuing chunk-size exploration, only test smaller chunks (`8`) as a tail-latency/load-balance probe, with a strict reject if atomic overhead widens the gap or token rate fails to exceed `1.6`.

### 当前执行 attempt：cpu-fallback-chunk-size-8

- `attempt_id`: `20260701-cpu-fallback-chunk-size-8`
- `attempt_kind`: `implementation-probe`
- `hypothesis`: Since `chunk_size=32` widened the fallback gap, smaller chunks may improve load balancing and reduce long per-expert tail waits in the CPU fallback path. This trades more atomic scheduling overhead for finer-grained work distribution.
- `expected_delta`: Possible only if fallback tail imbalance dominates. If atomic overhead or page interleaving dominates, token rate will stay at or below `1.6`.
- `test_config`: same default-compatible chunk-size patch, `GGML_MOE_CPU_MUL_MAT_ID_CHUNK_SIZE=8`, clean cold vram12 gate stream, 16GB cgroup, France prompt.
- `rollback`: Revert source and rebuild clean unless token rate exceeds `1.6` and all hard gates pass.
- `required_evidence`: exact env/command, cgroup memory files, trace gap summary, France output, binary sha/build line, result status.
- `result`: completed and rejected. Run `/root/lfz/runs/vendor-ds4-16gb/20260701T143707Z-20260701T_cpu-fallback-chunk-size-8` produced `eval_tok_s=0.8`, `prompt_tok_s=0.4`, `TTFT=65633.647905ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14970347520`, `pgmajfault=3824341`, `workingset_refault_file=12110814`, `correctness_ok=true`.
- `answer`: France output remained semantically correct and coherent: it described France as a Western European country with rich history/culture, landmarks including the Eiffel Tower, Louvre and Versailles, cuisine/wine/fashion, EU membership, diverse landscapes, and Paris as capital.
- `trace_summary`: `rows=34753`, `cache_hits=29624`, `cache_inserts=5129`, `src0_ms=29957.612`, `total_ms=32434.924`, `gap_sum_ms=2477.312`.
- `gap_analysis`: Smaller chunks did not reduce the non-streamed fallback bottleneck. Gate stream cache behavior stayed mechanically the same, but end-to-end token rate collapsed and major faults rose sharply, indicating the finer chunking increased CPU fallback scheduling/page-fault churn rather than improving tail balance.
- `rollback_status`: source change reverted and clean rebuild restored `ggml/src/ggml-cpu/ggml-cpu.c` / `ggml/src/ggml-cuda/moe_stream.cu` diff to `0`; clean `llama-cli` sha256 is `c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`. No commit or push because this is not a valid SOTA.

### 当前执行 attempt：cpu-fallback-gap-trace

- `attempt_id`: `20260701-cpu-fallback-gap-trace`
- `attempt_kind`: `instrumentation`
- `hypothesis`: Old `2.6 tok/s` and current profile-admission have similar traced gate stream time, but current run has much larger untraced gap between gate stream calls. A default-off CPU fallback trace should measure where time is spent after gate experts are streamed, especially up/down expert CPU paths and non-MoE layer work.
- `expected_delta`: No direct performance gain; this is a bottleneck localization step. The expected output is per-op/per-tensor or per-fallback-stage timing that explains the `~35s` wall gap between old and current runs.
- `risk`: Instrumentation can perturb timing if too fine-grained. Keep it default-off, coarse enough for one France run, and reject any performance number from instrumented binary as non-SOTA unless rerun without tracing.
- `test_config`: clean source plus default-off trace patch, `cpu_moe=40`, `vram_cache=12GB`, cold `drop_caches`, 16GB cgroup, France prompt, gate stream trace enabled, CPU fallback trace enabled to a run-local file.
- `rollback`: Revert instrumentation after collecting trace and rebuild clean. No commit/push unless a later optimization based on this trace produces a valid SOTA and passes full reproduce gates.
- `required_evidence`: source diff, build line, CPU fallback trace file, gate trace file, cgroup memory files, France output, and explicit reverted/unpromoted status.
- `result_chunk_trace`: completed as instrumentation-only and not promoted. Existing default-off `GGML_MOE_CPU_CHUNK_TRACE_OUT` hook was used without source changes in `/root/lfz/runs/vendor-ds4-16gb/20260701T134411Z-20260701T_cpu-fallback-gap-trace`. Run produced `eval_tok_s=1.6`, `TTFT=45508.42127ms`, `memory_peak_bytes=16000000000`, `correctness_ok=true`; chunk trace hit the `1,000,000` row limit and showed only `ffn_up_exps.weight` and `ffn_down_exps.weight` fallback chunks.
- `result_op_trace`: completed as instrumentation-only and not promoted. Added a default-off coarse `GGML_MOE_CPU_OP_TRACE_OUT` hook, ran `/root/lfz/runs/vendor-ds4-16gb/20260701T134922Z-20260701T_cpu-fallback-op-gap-trace`, then reverted source and rebuilt clean (`llama-cli` sha256 back to `c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`). Run produced `eval_tok_s=1.5`, `TTFT=47815.061976ms`, `memory_peak_bytes=16000000000`, `correctness_ok=true`.
- `trace_summary`: op-level wall estimate: `ffn_gate_exps.weight` stream `~28.97s`, `ffn_up_exps.weight` fallback `~41.37s`, `ffn_down_exps.weight` fallback `~36.22s`; gate trace `gap_sum_ms=85034.607`, `src0_ms=26226.158`, `total_ms=28551.19`.
- `gap_analysis`: The missing wall time is not generic non-MoE overhead. It is primarily the non-streamed up/down expert CPU fallback. Higher CPU thread count worsened this path, so the next optimization should reduce up/down fallback work rather than increase CPU parallelism.
- `next_action`: Test partial multi-tensor streaming: gate+up and gate+down separately. Avoid naive all-FFN streaming because all-FFN already showed cache inserts explode (`36650`) and token rate falls to `0.9 tok/s`.

### 当前执行 attempt：stream-filter-list-gate-up

- `attempt_id`: `20260701-stream-filter-list-gate-up`
- `attempt_kind`: `implementation-probe`
- `hypothesis`: Op-level trace shows `ffn_up_exps.weight` CPU fallback accounts for about `41.37s` wall estimate while gate stream accounts for about `28.97s`. Streaming both gate and up experts may remove the largest CPU fallback component while leaving down on CPU, avoiding the worst cache churn seen in all-FFN.
- `theoretical_upper_bound`: If gate+up stream cost were approximately `gate stream 28.6s + up-only stream 36.3s` and down CPU fallback stayed `36.2s`, total could remain near current. The only way this wins is if streaming up avoids enough CPU fallback wall time and cache churn is sub-additive. Hard rejection if cache inserts/src0 approach all-FFN behavior or `eval_tok_s <= 1.6`.
- `implementation`: Extend `GGML_MOE_STREAM_ONE_NAME_FILTER` to accept comma-separated substrings while preserving the current single-substring behavior. Then run with `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps,ffn_up_exps`.
- `risk`: Gate+up may thrash the 12GB VRAM cache, increase page/refault pressure, or worsen TTFT. Correctness must be rechecked because combined stream order differs from single-filter probes.
- `test_config`: clean `aa8d6f916` plus default-compatible filter-list patch, `cpu_moe=40`, `vram_cache=12GB`, cold `drop_caches`, 16GB cgroup, France prompt, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps,ffn_up_exps`, trace enabled.
- `rollback`: Revert source and rebuild clean unless all hard gates pass and token rate improves. Commit/push only if accepted under full SOTA reproduce rules.
- `required_evidence`: source diff, exact command/env, cgroup memory files, trace tensor distribution, cache hits/inserts/src0, France output, binary sha256/build line, and promoted/unpromoted status.
- `result`: completed and not promoted. Run `/root/lfz/runs/vendor-ds4-16gb/20260701T135608Z-20260701T_stream-filter-list-gate-up` produced `eval_tok_s=1.2`, `prompt_tok_s=0.6`, `TTFT=49704.547061ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14953140224`, `pgmajfault=287764`, `workingset_refault_file=13570301`, `correctness_ok=true`.
- `trace_summary`: `rows=63700`, `cache_hits=47245`, `cache_inserts=16455`, `src0_ms=79235.371`, `total_ms=84675.874`, `gap_sum_ms=47467.04`. Tensor split: gate `rows=31850`, `inserts=8230`, `src0_ms=38398.126`; up `rows=31850`, `inserts=8225`, `src0_ms=40837.245`.
- `gap_analysis`: Gate+up reduces CPU fallback gap compared with gate-only, but stream/page-cache cost grows much faster. Inserts jump from gate-only `5129` to `16455`, so 12GB VRAM cache cannot hold the combined gate+up cold working set. Reject and do not promote.

### 当前执行 attempt：stream-filter-list-gate-down

- `attempt_id`: `20260701-stream-filter-list-gate-down`
- `attempt_kind`: `implementation-probe`
- `hypothesis`: Down CPU fallback wall estimate is `~36.22s`; gate+down may be less costly than gate+up if down has a better reuse/cache pattern.
- `test_config`: same filter-list patch as gate+up, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps,ffn_down_exps`, clean cold 16GB cgroup, `cpu_moe=40`, `vram_cache=12GB`, France prompt, trace enabled.
- `rollback`: Revert source and rebuild clean unless all hard gates pass and token rate improves.
- `result`: completed and not promoted. Run `/root/lfz/runs/vendor-ds4-16gb/20260701T140004Z-20260701T_stream-filter-list-gate-down` produced `eval_tok_s=1.1`, `prompt_tok_s=0.6`, `TTFT=47256.589488ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14954184704`, `pgmajfault=293519`, `workingset_refault_file=20356444`, `correctness_ok=true`.
- `trace_summary`: `rows=81954`, `cache_hits=60629`, `cache_inserts=21325`, `src0_ms=101047.733`, `total_ms=108176.084`, `gap_sum_ms=58626.246`. Tensor split: gate `rows=40977`, `inserts=10662`, `src0_ms=52522.603`; down `rows=40977`, `inserts=10663`, `src0_ms=48525.13`.
- `gap_analysis`: Gate+down is worse than gate+up because combined stream rows/inserts are even higher. Naive two-tensor streaming is rejected. Source was reverted and binary rebuilt clean (`llama-cli` sha256 `c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`); no commit/push.
- `next_action`: Do not stream full up/down tensors. If pursuing up/down, use selective expert-level stream/admission: gate stream remains, and only a profile-selected hot subset of up/down experts is allowed into the stream/cache so inserts stay near gate-only levels.

### 当前执行 attempt：selective-hot-updown64-stream

- `attempt_id`: `20260701-selective-hot-updown64-stream`
- `attempt_kind`: `implementation-probe`
- `hypothesis`: Full gate+up/down streaming is too slow because it streams too many cold experts. Offline LRU simulation on the all-FFN trace shows that keeping gate experts plus only the top-64 `ffn_up_exps` and top-64 `ffn_down_exps` experts by access frequency keeps cache inserts close to gate-only (`~5410` simulated inserts versus gate-only `~5113` on the same all-FFN order) while streaming about `7.3k` up rows and `7.3k` down rows. This may remove part of the CPU fallback gap without creating the page-cache churn seen in full gate+up/down.
- `theoretical_upper_bound`: From op trace, full up/down CPU fallback wall estimates are `~41.37s` and `~36.22s`. The top-64 selected experts cover a minority of up/down rows, so the best-case gain is bounded to a fraction of that fallback time. Simulated extra stream miss cost is small (`src0_miss_obs_ms` increases from `~26.47s` gate-only to `~27.97s` gate+top64+top64` on all-FFN order). If CPU fallback savings exceed this extra stream cost and overhead, token rate may improve.
- `implementation`: Add default-off `GGML_MOE_STREAM_ONE_EXPERT_ALLOWLIST=<path>`. When unset, behavior is unchanged. When set, `ffn_gate_exps` remains allowed by the name filter, while non-gate tensors are allowed only if `tensor_name<TAB>expert_index` appears in the allowlist. Run with `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_` and allowlist generated from all-FFN trace top-64 up/down experts.
- `risk`: The profile is prompt-specific and may not generalize. Even selected hot up/down experts can perturb gate cache residency. Correctness must be verified because mixed stream/fallback execution changes which experts are computed on GPU versus CPU.
- `test_config`: clean `aa8d6f916` plus default-compatible allowlist patch, `cpu_moe=40`, `vram_cache=12GB`, cold `drop_caches`, 16GB cgroup, France prompt, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_`, `GGML_MOE_STREAM_ONE_EXPERT_ALLOWLIST=<run/profile>`, trace enabled.
- `rollback`: Revert source and rebuild clean if build fails, correctness fails, RAM exceeds 16GB, TTFT exceeds baseline +20%, or `eval_tok_s <= 1.6`. Commit/push only if accepted under full SOTA reproduce rules.
- `required_evidence`: source diff, profile file + sha256, exact command/env, cgroup memory files, trace tensor distribution, cache hits/inserts/src0, France output, binary sha256/build line, and promoted/unpromoted status.
- `result`: completed and not promoted. Run `/root/lfz/runs/vendor-ds4-16gb/20260701T140952Z-20260701T_selective-hot-updown64-stream` produced `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=46637.632828ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14947753984`, `pgmajfault=587735`, `workingset_refault_file=12073959`, `correctness_ok=true`.
- `profile`: `/root/lfz/runs/vendor-ds4-16gb/20260701T140952Z-20260701T_selective-hot-updown64-stream/hot_updown64_allowlist.tsv`, `128` entries, sha256 `549c62e4d1547b42d84b97474353a145292aefa88a2642c757cdd51998d18e7c`.
- `trace_summary`: `rows=48225`, `cache_hits=42778`, `cache_inserts=5447`, `src0_ms=27615.152`, `total_ms=30520.422`, `gap_sum_ms=79461.835`. Split: gate `rows=34511`, `inserts=5319`, `src0_ms=26949.787`; up `rows=6857`, `inserts=64`, `src0_ms=376.106`; down `rows=6857`, `inserts=64`, `src0_ms=289.259`.
- `gap_analysis`: Expert allowlist worked mechanically and avoided the full up/down cache cliff. However, it only reduced fallback gap by about `2.1s` versus gate-only control while increasing traced stream time by about `1.8s`, so rounded token rate stayed at `1.6`. Not a SOTA.

### 当前执行 attempt：selective-hot-updown128-stream

- `attempt_id`: `20260701-selective-hot-updown128-stream`
- `attempt_kind`: `implementation-probe`
- `hypothesis`: Top64 selective stream was mechanically correct but too small to move end-to-end token rate. Offline all-FFN simulation for top128+top128 still keeps inserts moderate (`~5713`) while doubling selected up/down accesses versus top64, so it may reduce more CPU fallback gap without reaching the cache cliff.
- `test_config`: same allowlist patch, top-128 up and top-128 down profile from all-FFN trace, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_`, 12GB VRAM cache, 16GB cold cgroup, France prompt.
- `rollback`: Revert source and rebuild clean after the probe unless all SOTA gates pass and the result is accepted.
- `required_evidence`: profile sha256, exact command/env, cgroup memory files, trace split, France output, binary sha/build line, result status.
- `result`: completed and not promoted. Run `/root/lfz/runs/vendor-ds4-16gb/20260701T141433Z-20260701T_selective-hot-updown128-stream` produced `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=48035.524573ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14954897408`, `pgmajfault=573215`, `workingset_refault_file=12183659`, `correctness_ok=true`.
- `profile`: `/root/lfz/runs/vendor-ds4-16gb/20260701T141433Z-20260701T_selective-hot-updown128-stream/hot_updown128_allowlist.tsv`, `256` entries, sha256 `ab900b0e12d998d28f550d376c2c7333a4de82efed0ca520c1fc12972107c167`.
- `trace_summary`: `rows=57361`, `cache_hits=51338`, `cache_inserts=6023`, `src0_ms=30432.448`, `total_ms=33738.78`, `gap_sum_ms=78705.627`. Split: gate `rows=35227`, `inserts=5767`, `src0_ms=29142.625`; up `rows=11067`, `inserts=128`, `src0_ms=707.947`; down `rows=11067`, `inserts=128`, `src0_ms=581.876`.
- `gap_analysis`: Top128 streamed more up/down hot experts and still avoided the full cache cliff, but it increased gate inserts and total stream time enough that the end-to-end token rate remained `1.6` and TTFT worsened. Source was reverted and binary rebuilt clean (`llama-cli` sha256 `c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`); no commit/push.
- `conclusion`: Selective GPU stream is mechanically correct, but under the current per-expert stream path it does not reduce enough CPU fallback wall time to overcome extra stream/cache overhead. Next step should target CPU fallback page/read behavior directly.

### 当前执行 attempt：cpu-fallback-willneed-vram12

- `attempt_id`: `20260701-cpu-fallback-willneed-vram12`
- `attempt_kind`: `config-probe`
- `hypothesis`: Previous broad `GGML_MOE_CPU_WILLNEED=1` was rejected on a vram2 setup, but current traces show the remaining bottleneck under vram12 gate stream is specifically up/down CPU fallback page/read behavior. Re-testing `MADV_WILLNEED` on the current clean gate-stream vram12 setup may reduce fallback page stalls without changing semantics.
- `expected_delta`: No GPU stream increase; possible gain only if fallback page faults/read waits are reduced. Reject if `eval_tok_s <= 1.6`, TTFT worsens beyond gate, correctness fails, or 16GB cgroup is violated.
- `test_config`: clean `aa8d6f916`, clean binary sha256 `c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`, `cpu_moe=40`, `vram_cache=12GB`, cold `drop_caches`, 16GB cgroup, France prompt, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, `GGML_MOE_CPU_WILLNEED=1`, gate trace enabled.
- `rollback`: No source change. Record as rejected unless it exceeds the clean reproducible baseline and all gates pass.
- `required_evidence`: exact env/command, cgroup memory files, trace gap summary, France output, binary sha/build line, result status.
- `result`: completed and not promoted. Run `/root/lfz/runs/vendor-ds4-16gb/20260701T141848Z-20260701T_cpu-fallback-willneed-vram12` produced `eval_tok_s=1.5`, `prompt_tok_s=0.7`, `TTFT=44590.577779ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14980456448`, `pgmajfault=412718`, `workingset_refault_file=11983681`, `correctness_ok=true`.
- `trace_summary`: `rows=34753`, `cache_hits=29624`, `cache_inserts=5129`, `src0_ms=26350.437`, `total_ms=28686.233`, `gap_sum_ms=85923.734`, `trace_span_ms=114605.734`.
- `gap_analysis`: `MADV_WILLNEED` does not reduce the remaining CPU fallback gap on the current vram12 gate-stream setup. It lowers major faults relative to some controls but increases/does not improve the trace gap and reduces token rate to `1.5`. Do not pursue broad fallback WILLNEED.
- `next_action`: The remaining viable direction is lower-overhead CPU fallback execution or more structural batching/fusion. Simple stream expansion, selective hot stream, higher CPU thread counts, and broad prefetch have all failed to exceed the clean `1.6 tok/s` baseline.

### 当前执行 attempt：profile-guided-gate-cache-admission

- `attempt_id`: `20260701-profile-guided-gate-cache-admission`
- `attempt_kind`: `implementation-probe`
- `hypothesis`: Gate-only trace shows `4534` unique gate experts but `1418` appear only once. Current LRU inserts every miss into the 12GB VRAM cache, so one-shot experts can evict future reusable experts. A default-off profile-guided admission file keyed by `(tensor name, expert index)` can cache only experts whose previous France trace frequency is `>=2`, while still computing bypassed experts through the existing staging buffer. This should preserve correctness and may reduce later misses/refaults.
- `theoretical_upper_bound`: On `/root/lfz/runs/vendor-ds4-16gb/20260701T131641Z-20260701T_gate-control-before-all-ffn`, an oracle skip of total-frequency-1 experts changes simulated misses from `5129` to `4597` (`-532`). With measured miss `src0_avg≈5.14ms`, the hard upper-bound gain is about `2.7s` of traced cold miss time. This is not enough by itself to explain historical `2.6 tok/s`, but it is a targeted way to test whether cache pollution is a real component of the current gap.
- `risk`: This is France-profile-specific and may not generalize to other prompts. It must be default-off and cannot be called a general SOTA unless later validated on the small prompt set. If it improves France but remains below historical `2.6 tok/s`, record it as a probe, not the current cold SOTA.
- `test_config`: clean `aa8d6f916` plus default-off admission patch, `cpu_moe=40`, `vram_cache=12GB`, cold `drop_caches`, 16GB cgroup, France prompt, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, `GGML_MOE_STREAM_CACHE_ADMIT_PROFILE=<run_dir>/gate_freq_ge2.tsv`, trace enabled.
- `rollback`: Revert source and rebuild clean if build fails, correctness fails, RAM exceeds 16GB, TTFT exceeds baseline +20%, or token rate does not improve. Commit/push only if it passes all gates and is accepted as SOTA under the reproducibility rules above.
- `required_evidence`: source diff, generated profile file with checksum, exact build/run commands, cgroup memory files, trace summary including miss count reduction, France output, binary sha256/build line, and promoted/unpromoted status.
- `result`: completed and not promoted. Run `/root/lfz/runs/vendor-ds4-16gb/20260701T132950Z-20260701T_profile-guided-gate-cache-admission` produced `eval_tok_s=1.6`, `prompt_tok_s=0.6`, `TTFT=48597.836825ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14956589056`, `pgmajfault=615420`, `workingset_refault_file=11471439`, `correctness_ok=true`.
- `profile`: copied into `/root/lfz/runs/vendor-ds4-16gb/20260701T132950Z-20260701T_profile-guided-gate-cache-admission/gate_freq_ge2.tsv`, `3116` entries, sha256 `028bcf206e4b833a6c55d8e88c1a453ed3b760284b8e281bfc1dde39705d2652`.
- `trace_summary`: `rows=34753`, `cache_hits=30156`, `trace_misses=4597`, `cache_inserts=3179`, `src0_ms=23201.883`, `total_ms=25449.431`. Profile admission mechanically reduced trace misses versus gate-only control (`5129 -> 4597`) and reduced `src0_ms` (`26391.295 -> 23201.883`).
- `gap_analysis`: The targeted cache admission saved about `3.2s` of traced stream time, matching the oracle estimate, but end-to-end `eval_tok_s` stayed rounded at `1.6` and TTFT increased relative to the immediate gate-only control (`46260.936ms -> 48597.837ms`). This means cache pollution is real but not currently large enough to create an accepted token-rate SOTA. Source was reverted and binary rebuilt clean; no commit/push.

### 当前执行 attempt：stream-down-exps-only

- `attempt_id`: `20260701-stream-down-exps-only`
- `attempt_kind`: `config-probe`
- `hypothesis`: After `ffn_up_exps` proved semantically correct on the DS4 CUDA stream path, `ffn_down_exps` should also be tested independently. Down experts have different input/output orientation, so this is primarily a correctness and trace-shape probe before any multi-tensor stream attempt.
- `expected_delta`: Not expected to be final SOTA by itself. If correct, record row count, cache pressure, and token rate. If incorrect, do not attempt all-expert stream until down numeric correctness is fixed.
- `risk`: Down tensor orientation may not match the current MMVQ rows kernel assumptions, causing incorrect output or shape/layout errors.
- `test_config`: clean `aa8d6f916`, `cpu_moe=40`, `vram_cache=12GB`, cold `drop_caches`, 16GB cgroup, France prompt, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_down_exps`, trace enabled.
- `rollback`: No source change. Record and reject on correctness failure, RAM violation, TTFT > baseline +20%, or no token-rate improvement.
- `required_evidence`: exact env/command, cgroup memory files, stderr cache stats, trace tensor names, trace summary, France output, binary sha256/build line.
- `result`: completed and not promoted. Run `/root/lfz/runs/vendor-ds4-16gb/20260701T131228Z-20260701T_stream-down-exps-only` produced `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=44098.832411ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14969028608`, `pgmajfault=537987`, `workingset_refault_file=12068147`, `correctness_ok=true`.
- `trace_summary`: `rows=34258`, `cache_hits=28915`, `cache_inserts=5343`, `src0_ms=26919.54`, `total_ms=29331.987`; trace tensors were `blk.*.ffn_down_exps.weight`.
- `gap_analysis`: Down-only CUDA stream is semantically correct and TTFT is acceptable, but it only ties the current clean reproducible `1.6 tok/s` baseline. The trace shape is almost identical to gate-only (`src0≈26.8s~26.9s`), so the main bottleneck remains cold expert page/read time rather than down compute alone. Do not promote or push.

### 当前执行 attempt：stream-all-ffn-exps

- `attempt_id`: `20260701-stream-all-ffn-exps`
- `attempt_kind`: `config-probe`
- `hypothesis`: Since `ffn_gate_exps`, `ffn_up_exps`, and `ffn_down_exps` each produce semantically correct France output when streamed independently, setting `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_` should route all three expert tensors through the DS4 CUDA stream path. This may reduce CPU fallback compute, but it will also increase streamed rows, cache inserts, and cold page pressure.
- `expected_delta`: Theoretical best case is bounded by removing CPU fallback expert compute for up/down while keeping GPU stream correctness. Negative bound is large because individual traces already show `src0_ms≈26.9s` for gate/down and `≈33.1s` for up; if combined cache pressure is additive, total stream wait can exceed the baseline and token rate should fall. Treat as a boundary/correctness probe unless `eval_tok_s > 1.6` with all hard gates passing.
- `risk`: 12GB VRAM cache may thrash across gate/up/down tensors, increasing misses and page refaults; combined stream ordering may expose numeric/layout bugs not visible in single-filter probes.
- `test_config`: clean `aa8d6f916`, `cpu_moe=40`, `vram_cache=12GB`, cold `drop_caches`, 16GB cgroup, France prompt, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_`, trace enabled.
- `rollback`: No source change. Reject if correctness fails, RAM exceeds 16GB, TTFT exceeds baseline by >20%, or `eval_tok_s <= 1.6`. Commit/push only after a promoted candidate has full SOTA-grade evidence and clean rebuild/rerun from pushed source.
- `required_evidence`: exact env/command, cgroup memory files, stderr cache stats, trace tensor-name distribution, trace summary, France output, binary sha256/build line, and explicit promoted/unpromoted status.
- `pre_control`: Gate-only control run `/root/lfz/runs/vendor-ds4-16gb/20260701T131641Z-20260701T_gate-control-before-all-ffn` produced `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=46260.935924ms`, `memory_peak_bytes=16000000000`, `correctness_ok=true`; trace `rows=34753`, `cache_inserts=5129`, `src0_ms=26391.295`, `total_ms=28729.431`.
- `result`: completed and not promoted. Run `/root/lfz/runs/vendor-ds4-16gb/20260701T131912Z-20260701T_stream-all-ffn-exps` produced `eval_tok_s=0.9`, `prompt_tok_s=0.5`, `TTFT=54453.307693ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14978949120`, `pgmajfault=32133`, `workingset_refault_file=22006606`, `correctness_ok=true`.
- `trace_summary`: `rows=110697`, `cache_hits=74047`, `cache_inserts=36650`, `src0_ms=165616.927`, `total_ms=176750.374`; tensor distribution was `ffn_gate_exps.weight=36899`, `ffn_up_exps.weight=36899`, `ffn_down_exps.weight=36899`.
- `gap_analysis`: All-FFN stream is semantically correct but much slower. The 12GB VRAM cache cannot hold the combined gate/up/down cold working set under the 16GB host RAM constraint; cache inserts jump from `5129` gate-only to `36650`, and traced `src0` wait jumps from `~26.4s` to `~165.6s`. Do not pursue naive all-expert streaming until cache admission/pinning can keep a stable hot subset.

### 当前执行 attempt：stream-up-exps-only

- `attempt_id`: `20260701-stream-up-exps-only`
- `attempt_kind`: `config-probe`
- `hypothesis`: Current accepted DS4 stream uses `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, so only gate experts are offloaded to the CUDA stream path; `ffn_up_exps` and `ffn_down_exps` still run through the CPU fallback. The same MXFP4 stream kernel may also be correct for `ffn_up_exps`, which has the same input/output orientation as gate. If correct, this identifies a larger optimization surface than page-cache tuning.
- `expected_delta`: This is not expected to be final SOTA by itself because gate will fall back to CPU when the filter is changed to up-only. The useful evidence is correctness, trace shape, `src0_ms`, and end-to-end token rate relative to gate-only. If up-only is correct and comparable or faster, next attempt can test multi-filter/all-expert stream. Reject as final SOTA unless `eval_tok_s > 1.6` and all hard gates pass.
- `risk`: Up expert tensor may differ in quantization/layout enough to produce incorrect output. It may also shift VRAM cache pressure without reducing total CPU fallback time.
- `test_config`: clean `aa8d6f916`, `cpu_moe=40`, `vram_cache=12GB`, cold `drop_caches`, 16GB cgroup, France prompt, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_up_exps`, trace enabled.
- `rollback`: No source change. Record and reject on correctness failure, RAM violation, TTFT > baseline +20%, or no token-rate improvement.
- `required_evidence`: exact env/command, cgroup memory files, stderr cache stats, trace tensor names, trace summary, France output, binary sha256/build line.
- `result`: completed and not promoted. Run `/root/lfz/runs/vendor-ds4-16gb/20260701T130549Z-20260701T_stream-up-exps-only` produced `eval_tok_s=1.5`, `prompt_tok_s=0.7`, `TTFT=47678.896617ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14940020736`, `pgmajfault=763124`, `workingset_refault_file=20367670`, `correctness_ok=true`.
- `trace_summary`: `rows=47458`, `cache_hits=40966`, `cache_inserts=6492`, `src0_ms=33118.55`, `total_ms=36281.524`; trace tensors were `blk.*.ffn_up_exps.weight`.
- `gap_analysis`: Up-only CUDA stream is semantically correct, but it has more streamed rows and misses than gate-only and is slower. This confirms the CUDA kernel can handle up orientation, but a naive tensor swap is not useful. Do not promote.

### 当前执行 attempt：stream-src0-pinned-stage

- `attempt_id`: `20260701-stream-src0-pinned-stage`
- `attempt_kind`: `implementation`
- `hypothesis`: Current miss path passes mmap-backed expert weights directly into `cudaMemcpyAsync`. Cold page faults can then occur inside the CUDA copy front-end, and the copy source is pageable/file-backed memory. Staging each missed expert through a per-slot pinned host buffer may separate page faults into a CPU `memcpy` and let H2D run from pinned memory. This can improve miss latency if CUDA/page-fault interaction is the bottleneck.
- `expected_delta`: Each missed expert is `~4.25MiB`; with `5129` misses the extra CPU copy touches `~21.8GiB`, so the method only wins if pinned H2D plus cleaner fault behavior saves more than the added copy. Hard upper bound is the current miss-path gap versus the old fast observation (`src0_ms≈26.8s` current vs `≈23.5s` old). Reject unless `eval_tok_s > 1.6`, correctness/RAM pass, and TTFT stays within +20%.
- `implementation`: Add default-off env `GGML_MOE_STREAM_SRC0_PINNED_STAGE=1` in `ggml/src/ggml-cuda/moe_stream.cu`. On cache miss, allocate a per-slot pinned `h_src0_bounce`, `memcpy` from mmap source into it, and use that pinned pointer for VRAM cache insertion or fallback `ctx.d_src0` H2D. Default behavior remains unchanged.
- `test_config`: dirty default-compatible patch on clean `aa8d6f916`, `GGML_MOE_STREAM_SRC0_PINNED_STAGE=1`, `cpu_moe=40`, `vram_cache=12GB`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace.
- `rollback`: If build fails, correctness fails, RAM exceeds 16GB, TTFT exceeds baseline by >20%, or `eval_tok_s <= 1.6`, revert source change and rebuild clean. Commit/push only after promoted clean rebuild/rerun.
- `required_evidence`: source diff, build log, exact env/command, cgroup memory files, trace summary, France output, binary sha256/build line, and run-level rejection/promote status.
- `result`: completed and rejected. Run `/root/lfz/runs/vendor-ds4-16gb/20260701T125908Z-20260701T_stream-src0-pinned-stage` produced `eval_tok_s=1.5`, `prompt_tok_s=0.6`, `TTFT=47842.119386ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14929248256`, `pgmajfault=616571`, `workingset_refault_file=12540923`, `correctness_ok=true`.
- `trace_summary`: `rows=34753`, `cache_hits=29624`, `cache_inserts=5129`, `src0_ms=26849.372`, `src0_miss_avg=5.228`, `sync_ms=1074.997`, `total_ms=29868.045`.
- `gap_analysis`: Pinned staging did not reduce miss `src0` time versus clean baseline (`src0_miss_avg≈5.23ms` both ways) and increased later synchronization time, likely due to the extra CPU memcpy plus pinned H2D scheduling. The source change was reverted and binary rebuilt clean. No commit/push.

### 当前执行 attempt：stream-skip-hit-dontneed

- `attempt_id`: `20260701-stream-skip-hit-dontneed`
- `attempt_kind`: `implementation`
- `hypothesis`: On a VRAM cache hit, the stream path does not read `src0_data` from host memory, but the current code still calls `MADV_DONTNEED` on the source range after every expert. Baseline trace shows hit-path `dontneed_ms≈246ms` (`29624` hit rows) and miss-path `dontneed_ms≈1027ms`. Skipping `DONTNEED` for cache hits should reduce syscall/VMA overhead without increasing page-cache pressure, because no host source pages are touched on hits.
- `expected_delta`: Direct upper bound is about `0.25s` traced time, so token-rate improvement may be small. Promote only if end-to-end `eval_tok_s > 1.6` and all hard gates pass; otherwise keep as rejected micro-optimization.
- `implementation`: Add env `GGML_MOE_STREAM_DONTNEED_HITS=0|1` around the final `moe_stream_dontneed_source_pages()` call in `ggml/src/ggml-cuda/moe_stream.cu`; default keeps current behavior, test sets it to `0`.
- `test_config`: dirty default-compatible patch on clean `aa8d6f916`, `GGML_MOE_STREAM_DONTNEED_HITS=0`, `cpu_moe=40`, `vram_cache=12GB`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace.
- `rollback`: If build fails, correctness fails, RAM exceeds 16GB, TTFT exceeds baseline by >20%, or `eval_tok_s <= 1.6`, revert source change and rebuild clean. Commit/push only after promoted clean rebuild/rerun.
- `required_evidence`: source diff, build log, exact env/command, cgroup memory files, trace summary with hit/miss `dontneed_ms`, France output, binary sha256/build line.
- `result`: completed and not promoted. Run `/root/lfz/runs/vendor-ds4-16gb/20260701T125223Z-20260701T_stream-skip-hit-dontneed` produced `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=46542.864043ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14975078400`, `pgmajfault=609068`, `workingset_refault_file=12345443`, `correctness_ok=true`.
- `trace_summary`: `rows=34753`, `cache_hits=29624`, `cache_inserts=5129`, `src0_ms=26241.336`, `total_ms=28367.729`, `dontneed_hit_sum=0.02`, `dontneed_miss_sum=1054.358`.
- `gap_analysis`: The micro-optimization worked mechanically: hit-path `DONTNEED` dropped from about `246ms` to near zero. End-to-end token rate still did not exceed the clean `1.6 tok/s` baseline because miss `src0` time remains dominant. The source change was reverted and binary rebuilt clean. No commit/push.

### 当前执行 attempt：stream-src0-madv-random

- `attempt_id`: `20260701-stream-src0-madv-random`
- `attempt_kind`: `implementation`
- `hypothesis`: Clean cold runs show much larger file input/refault pressure than the useful expert bytes (`5129` misses * `4.25MiB` is only about `21.8GiB`, while run-level file input/refault evidence is far larger). The mmap source path may be doing harmful readahead for random expert ranges. Applying `MADV_RANDOM` to each missed expert source range before H2D may reduce read amplification and cgroup page-cache churn.
- `expected_delta`: The hard upper bound is the gap between current clean miss cost (`src0_miss_avg≈5.22ms`, `src0_ms≈26.8s`) and old fast observation (`src0_miss_avg≈4.59ms`, `src0_ms≈23.5s`), plus any non-traced reclaim/wall-time reduction. If Linux readahead is useful for the 4.25MiB contiguous copy, this can regress; promote only if `eval_tok_s > 1.6`, correctness/RAM pass, and TTFT stays within +20%.
- `implementation`: Add default-off env `GGML_MOE_STREAM_SRC0_MADVISE=random|sequential|normal|0` in `ggml/src/ggml-cuda/moe_stream.cu`. When a source expert is not served from VRAM cache, advise the aligned `src0_data/src0_bytes` range before `cudaMemcpyAsync`.
- `test_config`: clean `aa8d6f916` plus this dirty default-off patch, `GGML_MOE_STREAM_SRC0_MADVISE=random`, `cpu_moe=40`, `vram_cache=12GB`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace.
- `rollback`: If build fails, correctness fails, RAM exceeds 16GB, TTFT exceeds baseline by >20%, or `eval_tok_s <= 1.6`, revert source change and rebuild clean. Commit/push only after a promoted candidate is clean-rebuilt and rerun from pushed source.
- `required_evidence`: source diff, build log, exact env/command, cgroup memory files, stderr cache stats, trace summary, France output, binary sha256/build line, and run-level rejection/promote status.
- `result`: failed_timeout/rejected. Run `/root/lfz/runs/vendor-ds4-16gb/20260701T124201Z-20260701T_stream-src0-madv-random` was killed after more than `330s` without completing, far slower than the `~130s` clean baseline wall time.
- `partial_trace_summary`: `rows=3108`, `cache_hits=1252`, `cache_inserts=1856`, `src0_ms=316649.459`, `total_ms=317174.339`.
- `gap_analysis`: `MADV_RANDOM` destroyed per-miss performance. Only `1856` misses already accumulated `316s` of `src0_ms` (`~170ms/miss`) versus clean baseline `~5.2ms/miss`. Linux readahead is required for the 4.25MiB contiguous expert copy; disabling it is not viable.
- `rollback_status`: source change reverted and binary rebuilt clean (`aa8d6f916`). No commit/push.

### 当前执行 attempt：cpu40-vram13-c256-fit

- `attempt_id`: `20260701-cpu40-vram13-c256-fit`
- `attempt_kind`: `config-probe`
- `hypothesis`: Prior `cpu40/vram13` allocated the 13GB stream cache (`3132` slots) but later OOMed in CUDA `GET_ROWS` with only ~13MiB free. Reducing context/batch/microbatch from `-c 512 -b 64 -ub 64` to `-c 256 -b 32 -ub 32` should free KV/temporary GPU memory while still fitting the short France prompt plus `-n 192`, allowing the 13GB cache to run.
- `expected_delta`: 13GB cache has `3132` slots versus `2891` at 12GB. Simulations suggest only a few hundred fewer misses, so the likely direct `src0` gain is around `1s`; promote only if this translates into `eval_tok_s > 1.6` with correct output, 16GB RAM compliance, and TTFT within +20%.
- `risk`: Lower batch/microbatch can alter prompt eval timing and may not free enough VRAM; duplicated CLI args rely on later `--extra-arg` values overriding runner defaults. Reject on OOM, correctness failure, TTFT > baseline +20%, RAM violation, cache insertion cliff, or no token-rate improvement.
- `test_config`: clean `aa8d6f916`, `cpu_moe=40`, `vram_cache=13GB`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace, extra CLI args `-c 256 -b 32 -ub 32`.
- `rollback`: No source change. If successful but TTFT exceeds the gate, record as `needs_ttft_recovery`; otherwise reject unless all SOTA gates pass.
- `required_evidence`: exact command showing final `-c/-b/-ub`, stderr cache slots and no CUDA OOM, trace summary, cgroup memory files, output text, binary sha256/build line.
- `result`: completed and not promoted. Run `/root/lfz/runs/vendor-ds4-16gb/20260701T122657Z-20260701T_cpu40-vram13-c256-fit` produced `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=47654.476135ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14965723136`, `pgmajfault=609557`, `workingset_refault_file=11968118`, `correctness_ok=true`.
- `trace_summary`: `rows=34753`, `cache_hits=29819`, `cache_inserts=4934`, `src0_ms=25920.404`, `total_ms=28223.104`.
- `gap_analysis`: lowering context/batch allowed the 13GB cache to run (`3132` slots), reducing inserts from `5129 -> 4934`, but the gain was too small to exceed the clean `1.6 tok/s` baseline. GPU samples still showed about `492MiB` free during decode, so test one larger cache step only if additional batch/microbatch reduction frees enough temporary VRAM.

### 当前执行 attempt：cpu40-vram14-c256-b16-fit

- `attempt_id`: `20260701-cpu40-vram14-c256-b16-fit`
- `attempt_kind`: `config-probe`
- `hypothesis`: `vram13/c256/b32` runs with about `492MiB` GPU free. Reducing batch/microbatch to `-b 16 -ub 16` may free enough temporary GPU memory for a 14GB stream cache, avoiding the previous vram14 cache insertion cliff and reducing misses further.
- `expected_delta`: A 14GB cache would provide about `3370` slots versus `2891` at 12GB and `3132` at 13GB. The hard upper bound is still modest because LRU/Belady simulation shows only hundreds of avoidable misses, but this is the most direct way to use more VRAM without changing arithmetic. Promote only if `eval_tok_s > 1.6`, output correct, RAM <=16GB including page cache, and TTFT within +20%.
- `risk`: vram14 may still OOM or silently fail cache allocation. Smaller batch may affect prompt throughput/TTFT. Reject on OOM, cache `0 inserts`, correctness failure, TTFT > baseline +20%, RAM violation, or no token-rate improvement.
- `test_config`: clean `aa8d6f916`, `cpu_moe=40`, `vram_cache=14GB`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace, extra CLI args `-c 256 -b 16 -ub 16`.
- `rollback`: No source change. Record as boundary evidence unless all SOTA gates pass.
- `required_evidence`: exact command, stderr cache allocation/hits, GPU samples, trace summary, cgroup memory files, France output, binary sha256/build line.
- `result`: completed and rejected. Run `/root/lfz/runs/vendor-ds4-16gb/20260701T123109Z-20260701T_cpu40-vram14-c256-b16-fit` produced `eval_tok_s=0.8`, `prompt_tok_s=0.6`, `TTFT=50953.922647ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14966722560`, `pgmajfault=649822`, `workingset_refault_file=26898800`, `correctness_ok=true`.
- `trace_summary`: `rows=34753`, `cache_hits=0`, `cache_inserts=0`, `src0_ms=90503.638`, `total_ms=98356.556`.
- `gap_analysis`: stderr showed `[moe_stream] VRAM cache: cudaMalloc 14.0 GiB FAILED`, so vram14 remains a cache allocation cliff even with `-c 256 -b 16 -ub 16`. Do not continue larger integer GB cache attempts unless a source change frees substantial VRAM. Current best fit boundary is `vram13 + -c 256 -b 32 -ub 32`, but it only reaches `1.6 tok/s` and is not promoted.

### 当前执行 attempt：cpu39-vram12-fit-probe

- `attempt_id`: `20260701-cpu39-vram12-fit-probe`
- `attempt_kind`: `config-probe`
- `hypothesis`: `cpu_moe=40` with `vram_cache=12GB` leaves about `1.5GB` GPU free during decode. Reducing to `cpu_moe=39` may place one additional MoE layer on GPU resident memory, reducing streamed expert page misses and host page-cache/refault pressure while still using the same DS4 gate stream path for remaining CPU MoE layers.
- `expected_delta`: If exactly one of ~40 CPU MoE layers is removed from the streamed path, the hard upper bound from traced `src0_ms≈26.8s` is roughly `~0.7s` direct `src0` reduction plus fewer page faults/refaults. Larger gains are possible only if the layer moved to GPU also reduces non-traced scheduling/reclaim stalls. Reject unless `eval_tok_s > 1.6` with correct France output, RAM <=16GB including page cache, and TTFT <= clean baseline +20%.
- `risk`: Additional GPU-resident MoE weights may exceed remaining VRAM and abort during load or first generation. If OOM occurs, record as boundary evidence and try a smaller stream cache only after updating this plan.
- `test_config`: clean `aa8d6f916`, `cpu_moe=39`, `vram_cache=12GB`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace.
- `rollback`: No source change. Reject on CUDA OOM, correctness failure, RAM violation, TTFT > baseline +20%, or `eval_tok_s <= 1.6`.
- `required_evidence`: exact env/command, cgroup memory files, GPU samples, stderr cache slots/hits, trace summary, France output, binary sha256/build line.
- `result`: completed and rejected. Run `/root/lfz/runs/vendor-ds4-16gb/20260701T121018Z-20260701T_cpu39-vram12-fit-probe` produced `eval_tok_s=0.8`, `prompt_tok_s=0.6`, `TTFT=54706.878075ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14945218560`, `pgmajfault=632745`, `workingset_refault_file=25545339`, `correctness_ok=true`.
- `trace_summary`: `rows=33875`, `cache_hits=0`, `cache_inserts=0`, `src0_ms=87542.345`, `total_ms=95519.061`.
- `gap_analysis`: `cpu_moe=39` reduced streamed rows versus `cpu_moe=40` (`34753 -> 33875`), but the extra GPU-resident MoE layer consumed enough VRAM that the 12GB stream cache hit an allocation cliff and never inserted. This makes every streamed expert a host miss and regresses badly. Do not use `cpu_moe=39` with `vram_cache=12GB`.

### 当前执行 attempt：cpu39-vram11-cache-fit

- `attempt_id`: `20260701-cpu39-vram11-cache-fit`
- `attempt_kind`: `config-probe`
- `hypothesis`: `cpu_moe=39` can reduce streamed expert rows, but requires a smaller stream cache than 12GB to avoid the cache insertion cliff. `vram_cache=11GB` may restore cache hits while retaining one additional GPU-resident MoE layer.
- `expected_delta`: Compared with clean `cpu_moe=40/vram12`, this trades fewer streamed rows for fewer cache slots. It can only promote if restored cache hits reduce end-to-end misses enough to exceed `1.6 tok/s`; otherwise the row reduction is too small to offset cache loss.
- `risk`: Smaller cache may increase misses enough to lose versus `cpu_moe=40/vram12`; additional GPU layer may still keep VRAM too tight. Reject on cache insertion cliff, correctness failure, TTFT > baseline +20%, RAM violation, or `eval_tok_s <= 1.6`.
- `test_config`: clean `aa8d6f916`, `cpu_moe=39`, `vram_cache=11GB`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace.
- `rollback`: No source change. Record as boundary evidence only unless all SOTA gates pass.
- `required_evidence`: same as `cpu39-vram12-fit-probe`, with stderr proving cache slots and final hits/misses.
- `result`: completed and rejected. Run `/root/lfz/runs/vendor-ds4-16gb/20260701T121550Z-20260701T_cpu39-vram11-cache-fit` produced `eval_tok_s=0.8`, `prompt_tok_s=0.6`, `TTFT=55265.382051ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14979825664`, `pgmajfault=632281`, `workingset_refault_file=25570963`, `correctness_ok=true`.
- `trace_summary`: `rows=33875`, `cache_hits=0`, `cache_inserts=0`, `src0_ms=87468.374`, `total_ms=95074.491`.
- `gap_analysis`: stderr showed `[moe_stream] VRAM cache: cudaMalloc 11.0 GiB FAILED`; GPU samples during decode had only about `10538 MiB` free. Therefore `cpu_moe=39` requires cache budget below 11GB. Continue only one boundary probe at `vram_cache=10GB`.

### 当前执行 attempt：cpu39-vram10-cache-fit

- `attempt_id`: `20260701-cpu39-vram10-cache-fit`
- `attempt_kind`: `config-probe`
- `hypothesis`: With `cpu_moe=39`, GPU free memory is about `10.3GB`; `vram_cache=10GB` should restore VRAM cache insertion while still keeping one extra MoE layer GPU-resident. This tests whether fewer streamed rows can offset the smaller stream cache.
- `expected_delta`: The smaller cache will reduce slots from `2891` to roughly `2409`, increasing miss count. The row reduction is only `~2.5%` (`34753 -> 33875`), so the theoretical expectation is no promotion unless the extra GPU-resident layer removes a disproportionately expensive page/refault segment. Promote only if `eval_tok_s > 1.6` and all hard gates pass.
- `risk`: If cache still fails to allocate, performance remains around `0.8 tok/s`. If cache inserts, miss count may still be too high. Reject and stop the `cpu_moe=39` branch unless clear evidence suggests a nearby fit point.
- `test_config`: clean `aa8d6f916`, `cpu_moe=39`, `vram_cache=10GB`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace.
- `rollback`: No source change. Record boundary evidence only unless all SOTA gates pass.
- `required_evidence`: exact env/command, stderr cache allocation/hits, trace summary, cgroup memory files, France output, binary sha256/build line.
- `result`: completed and rejected. Run `/root/lfz/runs/vendor-ds4-16gb/20260701T122134Z-20260701T_cpu39-vram10-cache-fit` produced `eval_tok_s=1.5`, `prompt_tok_s=0.7`, `TTFT=50379.860843ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14978146304`, `pgmajfault=588245`, `workingset_refault_file=12769884`, `correctness_ok=true`.
- `trace_summary`: `rows=33875`, `cache_hits=28216`, `cache_inserts=5659`, `src0_ms=29048.074`, `total_ms=31496.396`.
- `gap_analysis`: `vram10` restored cache allocation, but reducing cache slots increased misses enough to erase the benefit of fewer streamed rows. Compared with clean `cpu40/vram12` (`rows=34753`, `inserts=5129`, `src0_ms≈26.8s`, `eval_tok_s=1.5~1.6`), `cpu39/vram10` has fewer rows but more inserts and worse `src0`. Stop the `cpu_moe=39` branch unless a separate change frees enough VRAM to keep at least a 12GB stream cache.

### 当前执行 attempt：batch-stream-ds4-decline-probe

- `attempt_id`: `20260701-batch-stream-ds4-decline-probe`
- `attempt_kind`: `source-probe`
- `status`: rejected/timeout
- `hypothesis`: Current CPU fallback only calls `ggml_cuda_moe_stream_one`; `ggml_cuda_moe_stream_batch` is compiled but unused. The batch path may reduce per-expert launch/sync overhead for decode, but its current type gate only allows `IQ3_XXS/IQ2_S`. A minimal gate-only, default-off probe should first prove whether the batch path declines DS4 as `unsupported_type` or can be safely reached before any larger kernel work.
- `expected_delta`: The first probe is diagnostic and not a SOTA candidate. If it only declines, expected token rate is unchanged except tiny overhead. If a later DS4-enabled batch path works, the upper bound is limited by non-stream CPU up/down fallback (`~77s` in prior trace) plus cold expert `src0` stalls; therefore even a perfect gate launch reduction cannot explain the old `2.6 tok/s` unless it also reduces page/refault behavior or replaces more CPU fallback work.
- `risk`: If DS4 type is force-enabled in batch without matching kernel support, output may become semantically wrong or CUDA may fail. Therefore the first source change must be env-gated and must fall back to the existing per-expert path on any decline/failure. Any DS4 batch enablement must be correctness-first using the France prompt; a bad output is an immediate reject and rollback.
- `test_config`: clean `aa8d6f916`, `cpu_moe=40`, `vram_cache=12GB`, cold `drop_caches`, 16GB cgroup, France prompt, `GGML_MOE_STREAM=1`, `GGML_MOE_STREAM_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_DECLINE_DEBUG=1`, plus a new env gate for the batch probe.
- `rollback`: Revert source and rebuild clean unless a later non-instrumented candidate exceeds `1.6 tok/s`, keeps RAM <=16GB including page cache, passes correctness, and TTFT <= clean baseline +20%. Commit/push only after full promoted rerun from pushed source.
- `required_evidence`: source diff, build log, stderr decline or activation message, France output, cgroup memory files, trace summary, binary sha256/build line, and explicit promoted/rejected status.
- `result`: completed and rejected. Full strict cold run `/root/lfz/runs/vendor-ds4-16gb/20260701T152744Z-20260701T_batch-stream-ds4-decline-probe/france-cpu40-vram12gb` produced `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=46433.035475ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14959099904`, `pgmajfault=594921`, `workingset_refault_file=12364862`, `correctness_ok=true`.
- `trace_summary`: `rows=34753`, `cache_hits=29624`, `cache_inserts=5129`, `src0_ms_sum=26988.359`, `total_ms_sum=29354.164`, `trace_span_ms=110357.838`.
- `batch_probe_detail`: A short `-n 1` diagnostic run proved the CPU-side probe can resolve and call `ggml_cuda_moe_stream_batch` (`fn=...350f10`), but `build-ds4-moe-stream/build.ninja` compiles `moe_stream_batch.cu` with `-DGGML_CUDA_MOE_STREAM` and without `-DGGML_CUDA_MOE_STREAM_BATCH`. Therefore the exported batch symbol is currently the stub branch from the top of `moe_stream_batch.cu`, which returns `false` without decline logs. The full batch decode implementation is not active in the current vendor build.
- `gap_analysis`: This cannot explain the unreproduced old `2.6 tok/s`: the full run stayed on the existing `ggml_cuda_moe_stream_one` per-expert gate stream path and reproduced the clean `1.6 tok/s` behavior. Enabling the full batch implementation would be a separate source experiment, but it is not expected to recover 2.6 alone because prior op trace attributes most remaining wall gap to non-streamed CPU up/down fallback and cold page/refault behavior.
- `rollback_status`: probe source reverted and clean rebuild restored source diff for `ggml/src/ggml-cpu/ggml-cpu.c`, `ggml/src/ggml-cuda/moe_stream.cu`, and `ggml/src/ggml-cuda/moe_stream_batch.cu` to `0`; clean hashes: `llama-cli=c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`, `libggml-cpu.so.0.10.0=38c2a657261227252ac214e1aab81f2397706017a91fc061b2703dc51e6089bc`, `libggml-cuda.so.0.10.0=113bd429caeb01d55680cbb41ba6dd206019b935f5ad588dc44fc06d7a305725`. No commit/push because there is no accepted SOTA.

### 当前执行 attempt：batch-full-impl-compile-decline-probe

- `attempt_id`: `20260701-batch-full-impl-compile-decline-probe`
- `attempt_kind`: `source-probe`
- `status`: rejected/tie
- `hypothesis`: `moe_stream_batch.cu` currently compiles only the stub branch because `GGML_CUDA_MOE_STREAM_BATCH` is not defined. Enabling the full implementation while keeping DS4 types disabled should produce a real `unsupported_type` decline for `MXFP4/F8`, proving the full batch path can be built before considering any DS4 kernel dispatch changes.
- `expected_delta`: No accepted token-rate gain is expected in this step. If build succeeds and the full implementation declines DS4, runtime should fall back to the existing `stream_one` path and remain near `1.6 tok/s`. A later DS4 batch experiment has a hard upper bound of removing only part of the current `~29s` gate stream trace unless it also changes page/refault behavior; it cannot be assumed to recover the old `2.6 tok/s`.
- `risk`: The full batch implementation may fail to compile in the current vendor branch, or it may initialize extra CUDA resources and perturb memory. This step must remain env-gated and must not be promoted. If compile fails or runtime changes correctness/RAM materially, revert immediately and rebuild clean.
- `test_config`: Add a default-off source patch that enables full `moe_stream_batch.cu` implementation and CPU gate-only batch probe under `GGML_MOE_STREAM_BATCH_PROBE=1`; run a short `-n 1` diagnostic first under 16GB cgroup with `GGML_MOE_STREAM_DECLINE_DEBUG=1`.
- `rollback`: Revert source and rebuild clean after diagnostics unless a separate non-instrumented candidate later exceeds `1.6 tok/s` and passes all gates. No commit/push for compile/decline diagnostics.
- `required_evidence`: source diff, build result, `build.ninja` defines or source macro evidence, stderr showing real batch decline reason or compile failure, cgroup memory evidence for any runtime probe, and clean rollback hashes.
- `result`: completed and rejected at build stage. Defining `GGML_CUDA_MOE_STREAM_BATCH` to compile the full implementation failed in `moe_stream_batch.cu` with `fatal error: iqk/iqk_mul_mat.h: No such file or directory`.
- `gap_analysis`: The current vendor branch does not have the include path/dependency set needed to compile the full batch implementation. Therefore the available exported `ggml_cuda_moe_stream_batch` remains a stub and cannot be used for DS4 gate batching without first porting missing IQK dependencies or rewriting/removing that dependency. This branch is not a near-term path to recover the old `2.6 tok/s`.
- `rollback_status`: source reverted and clean rebuild completed; source diff for `ggml/src/ggml-cpu/ggml-cpu.c`, `ggml/src/ggml-cuda/moe_stream.cu`, and `ggml/src/ggml-cuda/moe_stream_batch.cu` is `0`. Post-rollback hashes: `llama-cli=c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`, `libggml-cpu.so.0.10.0=38c2a657261227252ac214e1aab81f2397706017a91fc061b2703dc51e6089bc`, `libggml-cuda.so.0.10.0=cad386c2391210af2810c3d55c961a34e34bae7278ea494c6fee286c289059b8`. No commit/push because there is no accepted SOTA.

### 当前执行 attempt：dense-mmap-dontneed-after-load

- `attempt_id`: `20260701-dense-mmap-dontneed-after-load`
- `attempt_kind`: `source-probe`
- `status`: rejected/tie
- `hypothesis`: With `--defer-experts`, vendor already calls `MADV_DONTNEED` on expert mmap ranges after load, but non-expert/dense mmap pages that were faulted during GPU tensor upload may remain in the cgroup file cache. In strict 16GB cold runs, `memory_file_bytes≈14.9GB` and `workingset_refault_file≈12M`, so dense/GPU upload pages may be competing with CPU expert pages. Dropping non-expert mmap pages after load should free page-cache budget for later expert faults without changing tensor values, because GPU-resident tensors have already been copied and CPU-resident tensors can fault back from the mmap if needed.
- `expected_delta`: Hard upper bound comes from reducing cold page/refault stalls, not compute. If the old `2.6 tok/s` delta was mostly lower faults (`pgmajfault≈309k` old vs `~595k` current), this can help only if dense pages are a major reclaim competitor. A useful result should reduce `workingset_refault_file`, `pgmajfault`, or trace span versus clean `1.6` runs. Promote only if `eval_tok_s > 1.6`, correctness passes, RAM <=16GB including page cache, and TTFT <= clean baseline +20%.
- `risk`: Dropping dense mmap pages may make CPU-resident non-expert tensors refault later, increasing TTFT or generation time. It may also simply shift faults from load to decode with no net gain. The change must be default-off via an env var and must use `MADV_DONTNEED`, not `munmap`, so correctness is preserved by re-faulting pages if needed.
- `test_config`: Add env-gated loader method to call `MADV_DONTNEED` on the complement of expert ranges after `drop_mmap_expert_pages()`. Run clean `aa8d6f916`, `cpu_moe=40`, `vram_cache=12GB`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace, with `GGML_MOE_DROP_DENSE_MMAP_PAGES=1`.
- `rollback`: Revert source and rebuild clean unless the non-instrumented candidate exceeds `1.6 tok/s` and all hard gates pass. If it improves token rate but TTFT exceeds +20%, record as `not accepted` and keep only if explicitly useful for follow-up; otherwise revert.
- `required_evidence`: source diff, loader log showing dense ranges/pages dropped, exact env/command, cgroup memory files, `summary.json`, `trace_summary.json`, France output, binary sha256/build line, and pushed commit/branch only if promoted.
- `result`: completed and rejected. Run `/root/lfz/runs/vendor-ds4-16gb/20260701T155005Z-20260701T_dense-mmap-dontneed-after-load-stderr/france-cpu40-vram12gb` explicitly executed the patch and logged `[mmap_drop] dense_non_expert_dontneed_bytes=8978451008 gib=8.36`. Result: `eval_tok_s=1.5`, `prompt_tok_s=0.7`, `TTFT=46965.133138ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14964228096`, `pgmajfault=610090`, `workingset_refault_file=12412061`, `correctness_ok=true`.
- `trace_summary`: `rows=34753`, `cache_hits=29624`, `cache_inserts=5129`, `src0_ms_sum=25804.567`, `total_ms_sum=28174.497`, `trace_span_ms=111369.545`.
- `gap_analysis`: Dropping `8.36 GiB` of non-expert mmap pages did reduce traced stream `src0_ms` versus clean-ish runs, but end-to-end generation regressed to `1.5 tok/s` and `pgmajfault` stayed high. This means dense page eviction shifts or increases non-traced refault/load stalls rather than recovering the old `2.6 tok/s` state. Do not promote and do not continue broad dense-page eviction; any future page-cache work must be more selective and justified by a fresh trace.
- `rollback_status`: source reverted and clean rebuild completed; source diff for `src/llama-model-loader.h`, `src/llama-model-loader.cpp`, `src/llama-model.cpp`, `ggml/src/ggml-cpu/ggml-cpu.c`, and `ggml/src/ggml-cuda/moe_stream.cu` is `0`. Post-rollback hashes: `llama-cli=c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`, `libllama.so.0.0.9085=47fbb949f001c7ef903c10e432595d104bc95be710d9d297b8ddcaa2220b1c2c`, `libggml-cpu.so.0.10.0=38c2a657261227252ac214e1aab81f2397706017a91fc061b2703dc51e6089bc`, `libggml-cuda.so.0.10.0=cad386c2391210af2810c3d55c961a34e34bae7278ea494c6fee286c289059b8`. No commit/push because there is no accepted SOTA.

### 当前执行 attempt：fractional-vram-cache-13824mib

- `attempt_id`: `20260701-fractional-vram-cache-13824mib`
- `attempt_kind`: `config-probe`
- `status`: rejected/tie
- `hypothesis`: `vram_cache=13GB` with `-c 256 -b 32 -ub 32` runs and reduces inserts (`5129 -> 4934`) but remains `1.6 tok/s`; `vram_cache=14GB` with `-c 256 -b 16 -ub 16` fails cache allocation (`0 inserts`). The code supports `GGML_MOE_STREAM_ONE_CACHE_MIB`, so a fractional cache such as `13824 MiB` may fit between the known 13GB/14GB boundary and reduce misses without hitting the allocation cliff.
- `expected_delta`: 13824 MiB gives about `~3250` slots versus `3132` at 13GB and `2891` at 12GB. The hard upper bound is modest: if each avoided miss costs about `5ms` traced `src0`, even a few hundred fewer misses is around `1s`. Promote only if this translates to `eval_tok_s > 1.6` with correctness/RAM/TTFT gates passing.
- `risk`: The 13.5GB allocation may still fail, or it may fit but leave too little temporary VRAM and trigger later CUDA OOM. Smaller `-b/-ub` may also affect prompt/TTFT. Reject on OOM, `cache_inserts=0`, correctness failure, TTFT > clean baseline +20%, RAM violation, or no token-rate improvement.
- `test_config`: clean source, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13824`, unset `GGML_MOE_VRAM_CACHE_GB`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace, extra CLI args `-c 256 -b 16 -ub 16`.
- `rollback`: No source change. Record as boundary evidence only unless all SOTA gates pass. If this fails allocation, optionally test `13568 MiB` only after updating this plan.
- `required_evidence`: exact env proving `GGML_MOE_STREAM_ONE_CACHE_MIB=13824`, stderr cache slots/hits, GPU samples, cgroup memory files, `summary.json`, `trace_summary.json`, France output, binary sha256/build line.
- `result`: completed and rejected early. Run `/root/lfz/runs/vendor-ds4-16gb/20260701T155622Z-20260701T_fractional-vram-cache-13824mib/france-cpu40-cache13824mib` allocated the cache (`13.5 GiB`, `3252 slots`) but aborted during decode with CUDA OOM: `ggml_cuda_compute_forward: GET_ROWS failed`.
- `metrics`: exit `134`, no generation result, `memory_peak_bytes=16000000000`, cgroup `oom_kill=0`, stderr showed GPU free dropping to about `14 MiB` before abort. This is a VRAM boundary failure, not a host RAM violation.
- `gap_analysis`: 13.5GiB is above the safe decode temporary-memory boundary even with `-c 256 -b 16 -ub 16`. Do not retry 13824MiB. A single lower boundary probe at `13568 MiB` is justified because it frees 256MiB versus the OOM point while still adding 256MiB over the known 13GB run.

### 当前执行 attempt：fractional-vram-cache-13568mib

- `attempt_id`: `20260701-fractional-vram-cache-13568mib`
- `attempt_kind`: `config-probe`
- `status`: rejected/tie
- `hypothesis`: `13824 MiB` is too high and OOMs after cache allocation, while 13GB (`13312 MiB`) runs but does not improve beyond `1.6 tok/s`. `13568 MiB` may be the largest practical cache budget that avoids the GET_ROWS OOM and still reduces misses compared with 13GB.
- `expected_delta`: At most about `~60` additional expert slots over 13GB and `~180` over 12GB, so expected gain is small. Promote only if `eval_tok_s > 1.6`, output correct, RAM <=16GB including page cache, and TTFT within +20%.
- `risk`: It may still OOM, or it may fit with no rounded token-rate gain. Reject on OOM, `cache_inserts=0`, correctness failure, TTFT > gate, RAM violation, or `eval_tok_s <= 1.6`.
- `test_config`: clean source, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, unset `GGML_MOE_VRAM_CACHE_GB`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace, CLI args `-c 256 -b 16 -ub 16`.
- `rollback`: No source change. Stop fractional cache direction if this does not produce an accepted SOTA.
- `required_evidence`: exact env/command, stderr cache slots/hits or OOM, GPU/resource samples, cgroup memory files, output text if any, `summary.json`, `trace_summary.json`, binary sha256/build line.
- `result`: completed and rejected. Run `/root/lfz/runs/vendor-ds4-16gb/20260701T155822Z-20260701T_fractional-vram-cache-13568mib/france-cpu40-cache13568mib` completed with `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=46313.694937ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14955864064`, `pgmajfault=604994`, `workingset_refault_file=11947792`, `correctness_ok=true`.
- `cache_summary`: stderr showed `13.2 GiB`, `3192 slots`, `hits=29855`, `misses=4898`, `hit_rate=85.9%`. This is a real cache improvement versus clean 12GB (`5129 inserts`) and 13GB (`4934 inserts`) without OOM.
- `trace_summary`: `rows=34753`, `cache_hits=29855`, `cache_inserts=4898`, `src0_ms_sum=24701.713`, `total_ms_sum=27017.327`, `trace_span_ms=108516.974`.
- `gap_analysis`: Fractional cache improves traced gate stream time by about `2.1s` versus the clean 12GB run and avoids the 13.5GiB OOM boundary, but end-to-end generation remains rounded at `1.6 tok/s`. This confirms that gate-stream H2D misses are no longer the dominant remaining lever; the residual gap is in non-streamed CPU fallback / page-refault scheduling outside `src0_ms`. Stop fractional VRAM cache probing unless a later source change frees enough VRAM for a qualitatively different cache size or offloads more CPU fallback work.
- `rollback_status`: No source change. No commit/push because this is not an accepted SOTA.

### 当前执行 attempt：stream-early-updown-layers-0-2

- `attempt_id`: `20260702-stream-early-updown-layers-0-2`
- `attempt_kind`: `source-probe`
- `status`: rejected/regression
- `hypothesis`: The CPU op trace shows the largest non-streamed fallback costs in early up/down tensors, especially `blk.0/1/2.ffn_up_exps.weight` and `blk.0/1/2.ffn_down_exps.weight`. Previous all-FFN and broad hot-expert streaming hurt cache pressure, but a narrow layer-specific stream may remove a high-cost CPU fallback segment while keeping the gate stream cache viable.
- `expected_delta`: From the CPU chunk trace, early layer up/down tensors account for a large share of fallback thread time; the wall-time upper bound is several seconds if GPU stream avoids those CPU matvecs without cache thrash. The actual bound is lower because extra up/down streamed experts consume the same VRAM cache and may evict gate entries. Promote only if end-to-end `eval_tok_s > 1.6` with France correctness, RAM <=16GB including page cache, and TTFT <= clean baseline +20%.
- `risk`: Additional up/down stream entries may increase cache inserts enough to regress, or CUDA stream for up/down may be slower than CPU fallback for these shapes. Correctness risk is moderate because individual up/down stream probes were previously correct, but combined layer-specific routing must still be checked with the France prompt.
- `test_config`: Add a default-compatible filter parser so `GGML_MOE_STREAM_ONE_NAME_FILTER` can accept comma-separated substrings. Run clean `cpu_moe=40`, `vram_cache=12GB`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 stream trace, with filter `ffn_gate_exps,blk.0.ffn_up_exps,blk.1.ffn_up_exps,blk.2.ffn_up_exps,blk.0.ffn_down_exps,blk.1.ffn_down_exps,blk.2.ffn_down_exps`.
- `rollback`: Revert source and rebuild clean unless all hard gates pass and `eval_tok_s > 1.6`. No commit/push for ties or regressions.
- `required_evidence`: source diff, build log, exact env/command, stderr cache slots/hits, trace summary split showing added up/down stream rows, cgroup memory files, France output, binary sha256/build line, and pushed source only if promoted.
- `result`: completed and rejected. Run `/root/lfz/runs/vendor-ds4-16gb/20260701T160720Z-20260702T_stream-early-updown-layers-0-2/france-cpu40-vram12gb` produced `eval_tok_s=1.4`, `prompt_tok_s=0.6`, `TTFT=48129.989629ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14968569856`, `pgmajfault=526035`, `workingset_refault_file=12746125`, `correctness_ok=true`.
- `trace_summary`: `rows=41457`, `cache_hits=32610`, `cache_inserts=8847`, `src0_ms_sum=48016.681`, `total_ms_sum=51314.946`, `trace_span_ms=123627.723`.
- `gap_analysis`: The layer-specific stream did route the intended early up/down tensors through CUDA, but stream rows increased by `6704` and cache inserts rose from the clean `5129` to `8847`. The added H2D/cache pressure more than offset any CPU fallback reduction, so end-to-end generation regressed to `1.4 tok/s`. Do not continue up/down streaming by broad layer lists unless paired with a fundamentally different cache/offload strategy.
- `rollback_status`: source reverted and clean rebuild completed; source diff for `ggml/src/ggml-cuda/moe_stream.cu` and `ggml/src/ggml-cpu/ggml-cpu.c` is `0`. Post-rollback hashes: `llama-cli=c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`, `libllama.so.0.0.9085=47fbb949f001c7ef903c10e432595d104bc95be710d9d297b8ddcaa2220b1c2c`, `libggml-cpu.so.0.10.0=38c2a657261227252ac214e1aab81f2397706017a91fc061b2703dc51e6089bc`, `libggml-cuda.so.0.10.0=f7e5c6428e0b7829df4c5763fa5794450533cfad0dd5b096122ab266acd3a671`. No commit/push because this is not an accepted SOTA.

### 当前执行 attempt：cpu41-vram14-gate-cache-tradeoff

- `attempt_id`: `20260702-cpu41-vram14-gate-cache-tradeoff`
- `attempt_kind`: `config-probe`
- `status`: rejected/tie
- `hypothesis`: Prior `cpu_moe=39` moved one MoE layer to GPU and caused the stream cache allocation cliff; the opposite direction may free enough VRAM for a 14GB gate stream cache. This adds one CPU MoE layer worth of fallback work but may reduce gate stream misses if `vram14` becomes viable.
- `expected_delta`: Compared with `cpu40/vram12`, CPU fallback rows should increase by roughly one layer (`~2.5%`), while cache slots could increase from `2891` to about `3370`. The hard upper bound is modest; this can only help if fewer cold gate misses/page refaults offset the additional CPU fallback. Promote only if `eval_tok_s > 1.6`, France output correct, RAM <=16GB including page cache, and TTFT <= clean baseline +20%.
- `risk`: The 14GB cache may still fail, or it may allocate but leave too little VRAM for decode temporaries. If it runs, the extra CPU fallback may dominate. Reject on CUDA OOM, `cache_inserts=0`, correctness failure, TTFT > gate, RAM violation, or `eval_tok_s <= 1.6`.
- `test_config`: clean source, `cpu_moe=41`, `GGML_MOE_VRAM_CACHE_GB=14`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace, CLI args `-c 256 -b 16 -ub 16`.
- `rollback`: No source change. Record as boundary evidence only unless all gates pass and token rate improves. Stop this direction if it fails allocation or does not exceed `1.6`.
- `required_evidence`: exact env/command, stderr cache allocation/hits or OOM, GPU/resource samples, cgroup memory files, output text, `summary.json`, `trace_summary.json`, binary sha256/build line.
- `result`: completed and rejected. Run `/root/lfz/runs/vendor-ds4-16gb/20260701T161444Z-20260702T_cpu41-vram14-gate-cache-tradeoff/france-cpu41-vram14gb` completed with `eval_tok_s=1.6`, `prompt_tok_s=0.6`, `TTFT=43630.3318ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14964662272`, `pgmajfault=630762`, `workingset_refault_file=12287368`, `correctness_ok=true`.
- `cache_summary`: stderr showed `14.0 GiB`, `3373 slots`, `hits=30670`, `misses=4952`, `hit_rate=86.1%`.
- `trace_summary`: `rows=35622`, `cache_hits=30670`, `cache_inserts=4952`, `src0_ms_sum=25587.694`, `total_ms_sum=27914.145`, `trace_span_ms=112144.894`.
- `gap_analysis`: Increasing CPU MoE to 41 lets the 14GB cache fit and improves TTFT, but it adds enough CPU fallback work that generation remains tied at `1.6 tok/s` and trace span is worse than the `cpu40/13568MiB` boundary run (`112.1s` vs `108.5s`). Do not continue the “more CPU MoE to fit larger cache” direction unless a separate change reduces CPU fallback cost.
- `rollback_status`: No source change. No commit/push because this is not an accepted SOTA.

### 当前执行 attempt：cache13568-thread24

- `attempt_id`: `20260702-cache13568-thread24`
- `attempt_kind`: `config-probe`
- `status`: completed/rejected
- `hypothesis`: The best gate-cache boundary so far (`cpu40`, `13568MiB`, `-c 256 -b 16 -ub 16`) reduces stream `src0_ms` but remains `1.6 tok/s`; the residual gap is largely CPU up/down fallback. Prior broad thread probes showed `t16` and `t32` were worse on vram12, but `t24` may reduce fallback wall time without the oversubscription/memory-bandwidth penalty seen at `t32`.
- `expected_delta`: If CPU fallback wall time scales modestly from 20 to 24 threads, the upper bound is a few seconds. Promote only if `eval_tok_s > 1.6`, France output correct, RAM <=16GB including page cache, and TTFT <= clean baseline +20%.
- `risk`: Extra threads can increase memory bandwidth contention, page fault contention, or scheduling overhead; if so token rate may tie or regress. Reject on correctness failure, TTFT > gate, RAM violation, OOM, or `eval_tok_s <= 1.6`.
- `test_config`: clean source, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, unset `GGML_MOE_VRAM_CACHE_GB`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace, CLI args `-c 256 -b 16 -ub 16 -t 24 -tb 24`.
- `rollback`: No source change. Stop thread tuning around this boundary if `t24` does not exceed `1.6`.
- `required_evidence`: exact env/command, stderr cache slots/hits, cgroup memory files, output text, `summary.json`, `trace_summary.json`, binary sha256/build line.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T162021Z-20260702T_cache13568-thread24/france-cpu40-cache13568-t24`
- `result`: rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `first_answer_ms=46521.827057`, wall `2:08.10`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14960013312`, `max_rss_kb=15593800`, `pgmajfault=675157`, `workingset_refault_file=12179170`, `ram_ok=true`.
- `correctness`: pass. France answer is coherent and semantically correct: it describes France as a Western European country with rich history/culture, Eiffel Tower/Louvre/Versailles, cuisine/wine/fashion, EU membership, diverse geography, and Paris as capital. It contains a minor typo (`Rennowned`) but no semantic failure.
- `cache_summary`: stderr showed `13.2 GiB`, `3192 slots`, `hits=29855`, `misses=4898`, `hit_rate=85.9%`.
- `trace_summary`: `rows=34753`, `cache_hits=29855`, `cache_inserts=4898`, `src0_ms_sum=24671.366`, `total_ms_sum=27024.549`, `trace_span_ms=108080.685`.
- `gap_analysis`: `t24` improves traced `src0_ms` versus the earlier `13568MiB` boundary run (`~24.7s` vs `~25.7s`) but does not improve rounded end-to-end generation (`1.6 tok/s`). TTFT remains around `46.5s`, major faults remain high, and trace span is still near `108s`. Thread tuning at this cache boundary is therefore not enough; next work should target cold IO/refault behavior or reduce CPU fallback work structurally rather than continuing small thread-count probes.
- `rollback_status`: No source change. No commit/push because this is not an accepted SOTA and does not exceed the current clean reproducible `1.6 tok/s` baseline.

### 当前执行 attempt：warm-page-bound-diagnostic-cache13568

- `attempt_id`: `20260702-warm-page-bound-diagnostic-cache13568`
- `attempt_kind`: `diagnostic-only`
- `status`: completed_diagnostic
- `hypothesis`: The current cold bottleneck is either page/refault dominated or CPU fallback compute dominated. Running the same `cpu40/13568MiB` config without `drop_caches` but still inside the strict 16GB cgroup gives a practical upper bound for page-cache warmth. If token rate rises materially and `pgmajfault/workingset_refault_file` fall, the next cold optimization should focus on page residency/read scheduling. If token rate stays near `1.6`, the remaining bottleneck is more likely CPU fallback compute/kernel efficiency.
- `expected_delta`: This is not a SOTA candidate because it intentionally uses warm global page-cache state. The expected upper bound is the delta between cold `1.6 tok/s` and the warm/no-drop-caches rate under the same cgroup. Promote is forbidden regardless of token rate; use only to prioritize the next cold-start implementation.
- `risk`: Warm global page cache may include pages created outside the 16GB cgroup, so this cannot prove cold compliance. It is still useful as a diagnostic if the run archives cgroup `memory.peak`, `memory.stat`, page-fault counters, exact env/command, stdout/stderr, and trace.
- `test_config`: clean source, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, unset `GGML_MOE_VRAM_CACHE_GB`, **no** `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `rollback`: No source change. Always mark `diagnostic_only` / `not_promotable`; do not commit/push unless a later cold-source optimization passes all SOTA gates.
- `required_evidence`: exact env/command, explicit note that `drop_caches` was not used, stderr cache stats, `summary.json`, `trace_summary.json`, cgroup memory files including `memory.stat`, France answer, and comparison against cold `20260701-fractional-vram-cache-13568mib` and `20260702-cache13568-thread24`.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T163134Z-20260702_warm-page-bound-diagnostic-cache13568/france-cpu40-vram0gb`
- `result`: diagnostic-only / not promotable. `eval_tok_s=2.8`, `prompt_tok_s=0.8`, `first_answer_ms=40871.512337`, wall `1:26.42`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15064985600`, `pgmajfault=320963`, `workingset_refault_file=19159478`, `ram_ok=true`, `correctness_ok=true`.
- `correctness`: pass. France answer is coherent and semantically correct, with the same minor `Rennowned` typo as other accepted-correct runs.
- `cache_summary`: `13.2 GiB`, `3192 slots`, `hits=29855`, `misses=4898`, `hit_rate=85.9%`; same hit/miss shape as the cold `13568MiB` runs.
- `trace_summary`: `rows=34753`, `cache_hits=29855`, `cache_inserts=4898`, `src0_ms_sum=22225.714`, `total_ms_sum=24412.385`, `trace_span_ms=67656.44`.
- `gap_analysis`: Warm/no-drop state improves end-to-end generation from cold `1.6 tok/s` to `2.8 tok/s` without changing cache hits/misses or output, and lowers trace span from cold `~108s` to `~67.7s`. This proves the current gap is primarily cold page/refault/reclaim state rather than cache policy or arithmetic correctness. Because `drop_caches` was intentionally not used and the warm page cache may have been prepared outside this run, this result is not a valid cold SOTA and must not be committed/pushed as promoted.
- `next_action`: Test whether the same warm-state benefit can be prepared inside one strict 16GB cgroup after `drop_caches` by running a recorded warmup invocation and then a measured France invocation before the cgroup exits. Record both the measured test rate and the amortized wall time including warmup; this remains diagnostic unless a future acceptance rule explicitly includes the warmup cost and TTFT gate.

### 当前执行 attempt：same-cgroup-warmup-then-test-cache13568

- `attempt_id`: `20260702-same-cgroup-warmup-then-test-cache13568`
- `attempt_kind`: `diagnostic-only`
- `status`: rejected/tie
- `hypothesis`: If warm/no-drop speed comes from reusable page cache that can be built while staying inside one 16GB cgroup, then a script that starts from `drop_caches`, runs a warmup France invocation, and then runs the measured France invocation in the same systemd cgroup should approach the `2.8 tok/s` diagnostic rate on the second invocation. If it does not, the warm/no-drop benefit depends on page-cache state not reproducible under the strict 16GB workflow.
- `expected_delta`: The measured second invocation may approach `2.8 tok/s`, but this is still not a cold SOTA because it uses an explicit warmup. The amortized rate including warmup will likely remain below the single-run rate and TTFT including warmup will likely fail the accepted cold gate. Promote is forbidden; use only to decide whether a source-level in-cold prefetch/preload strategy is worth implementing.
- `risk`: The warmup may be killed or may evict useful pages within the same 16GB cgroup. If cgroup `memory.peak` exceeds the limit, `memory.events oom/oom_kill` increments, output degenerates, or the second invocation does not improve, reject as diagnostic evidence.
- `test_config`: one `systemd-run` service with `MemoryMax=16000000000` and `MemorySwapMax=0`; before service start run host `sync; echo 3 > /proc/sys/vm/drop_caches`; inside the service run warmup France then measured France with clean source, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, unset `GGML_MOE_VRAM_CACHE_GB`, DS4 gate stream trace, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `rollback`: No source change. Always mark `diagnostic_only` / `not_promotable`; do not commit/push.
- `required_evidence`: explicit `drop_caches` before service, same cgroup path for warmup and test, per-phase stdout/stderr, per-phase rates/answers, cgroup memory files after both phases, trace for measured phase, total wall time including warmup, and comparison against warm/no-drop `2.8` and cold `1.6`.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T163601Z-20260702-same-cgroup-warmup-then-test-cache13568/france-cpu40-cache13568-same-cgroup-warmup`
- `result`: diagnostic-only / rejected as an optimization path. One systemd service completed both phases with `memory_peak_bytes=16000000000`, `memory_file_bytes=14977691648`, `ram_ok=true`, no source change.
- `warmup_phase`: `eval_tok_s=1.6`, `prompt_tok_s=0.6`, `first_answer_ms=46899.346218`, `pgmajfault≈618456`, `workingset_refault_file=11982464`, `cache=13.2 GiB/3192 slots`, `hits=29855`, `misses=4898`, `trace_span_ms=109704.8`, `src0_ms_sum=25240.09`, correctness pass.
- `test_phase`: `eval_tok_s=1.6`, `prompt_tok_s=0.6`, `first_answer_ms=54576.825008`, phase `pgmajfault≈611371`, cumulative cgroup `pgmajfault=1229030`, `workingset_refault_file=44035197`, `cache=13.2 GiB/3192 slots`, `hits=29855`, `misses=4898`, `trace_span_ms=112260.763`, `src0_ms_sum=26796.072`, correctness pass.
- `gap_analysis`: A full warmup inside the same 16GB cgroup does not reproduce the global warm/no-drop `2.8 tok/s` state. The measured second invocation remains cold-like (`1.6 tok/s`) and still incurs about `611k` major faults. This means the useful global warm state is not simply “run once inside the 16GB cgroup”; the cgroup stays at the file-cache ceiling and reclaim/refault churn evicts or re-faults the pages needed by the second invocation. Do not pursue full warmup-as-optimization under the cold SOTA rules.
- `next_action`: Focus on reducing the cold working set or changing page/refault scheduling within a single first invocation. The next viable source attempts should be narrow and default-off: either a profile-guided prefetch/read scheduling experiment that does not add full warmup cost, or a structural CPU fallback change that reduces up/down host page touches without expanding the streamed cache working set.

### 当前执行 attempt：stream-willneed-lookahead1-cache13568

- `attempt_id`: `20260702-stream-willneed-lookahead1-cache13568`
- `attempt_kind`: `source-probe`
- `status`: rejected/tie
- `hypothesis`: Warm/no-drop reaches `2.8 tok/s` with the same gate cache hit/miss sequence, while same-cgroup full warmup stays cold-like. The remaining first-invocation gap is page/refault scheduling. A very narrow gate-stream `MADV_WILLNEED` lookahead of one future active expert may overlap a small amount of expert page-in with the current expert copy/kernel without creating the reclaim pressure seen by `lookahead4`.
- `theoretical_upper_bound`: The cold `13568MiB` run has about `4898` gate misses and `src0_ms≈24.7s`; global warm lowers trace span by about `40s` and lowers major faults to about `321k`. Lookahead `1` can only hide one future expert at a time, so expected direct gain is at most a fraction of `src0_ms` and likely a few seconds. It cannot remove CPU fallback work. Promote only if end-to-end `eval_tok_s > 1.6`, RAM/correctness pass, and TTFT is within the accepted gate.
- `risk`: Prior `GGML_MOE_STREAM_WILLNEED_LOOKAHEAD=4` reduced traced `src0_ms` but regressed token rate to `1.3`, likely by moving cost into reclaim/refault outside the traced region. Lookahead `1` may still increase cgroup pressure or TTFT. The implementation must be default-off and gate-filtered to `ffn_gate_exps` for this probe.
- `implementation`: Add default-off env `GGML_MOE_STREAM_WILLNEED_LOOKAHEAD=<N>` in `ggml_compute_forward_mul_mat_id` when the source tensor name contains `ffn_gate_exps`. Before streaming the current active expert, thread 0 issues `MADV_WILLNEED` for the next `N` active expert source pages using the existing `ggml_moe_cpu_willneed_pages()` helper.
- `test_config`: dirty default-compatible patch on clean `aa8d6f916`, `GGML_MOE_STREAM_WILLNEED_LOOKAHEAD=1`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, unset `GGML_MOE_VRAM_CACHE_GB`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `rollback`: Revert source and rebuild clean if build fails, correctness fails, RAM exceeds 16GB, TTFT exceeds gate, or `eval_tok_s <= 1.6`. Commit/push only if a clean rebuild/rerun from pushed source passes all SOTA gates.
- `required_evidence`: source diff, build log/hash, exact env/command, `summary.json`, `trace_summary.json`, cgroup memory files, France answer, comparison against cold `13568MiB`, warm/no-drop diagnostic, and same-cgroup warmup diagnostic.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T164838Z-20260702_stream-willneed-lookahead1-cache13568/france-cpu40-vram0gb`
- `result`: completed and rejected. `eval_tok_s=1.4`, `prompt_tok_s=0.6`, `first_answer_ms=48253.97287`, wall `2:22.27`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14978121728`, `pgmajfault=616083`, `workingset_refault_file=16346498`, `ram_ok=true`, `correctness_ok=true`.
- `correctness`: pass. France answer is coherent and semantically correct, with the same minor `Rennowned` typo.
- `cache_summary`: `13.2 GiB`, `3192 slots`, `hits=29855`, `misses=4898`, `hit_rate=85.9%`.
- `trace_summary`: `rows=34753`, `cache_hits=29855`, `cache_inserts=4898`, `src0_ms_sum=21259.698`, `total_ms_sum=23660.217`, `trace_span_ms=122386.105`.
- `gap_analysis`: Lookahead `1` reduced traced stream `src0_ms` versus cold `13568MiB` (`~24.7s -> ~21.3s`) but worsened end-to-end token rate (`1.6 -> 1.4`), TTFT (`~46.3s -> ~48.3s`), `workingset_refault_file`, and trace span (`~108.5s -> ~122.4s`). The prefetch again moves cost into untraced reclaim/refault gaps rather than improving decode. Stop synchronous `MADV_WILLNEED` stream lookahead unless a later mechanism can prove truly asynchronous IO without increasing cgroup reclaim.
- `rollback_status`: source change reverted and clean rebuild completed. Post-rollback hashes: `llama-cli=c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`, `libggml-cpu.so.0.10.0=38c2a657261227252ac214e1aab81f2397706017a91fc061b2703dc51e6089bc`. No commit/push because this is not an accepted SOTA.

### 当前执行 attempt：vram-cache-hash-lookup

- `attempt_id`: `20260702-vram-cache-hash-lookup`
- `attempt_kind`: `source-probe`
- `status`: promoted_candidate_pending_rerun
- `hypothesis`: `ggml/src/ggml-cuda/moe_stream.cu` defines a VRAM cache hash table, but `vram_cache_lookup()` currently scans every slot linearly and `vram_cache_insert()` does not maintain the hash table. With `3192` slots and about `34753` stream calls, this can produce tens of millions of pointer comparisons, mostly on cache hits. Replacing the linear scan with the existing open-addressed hash table should reduce CPU overhead in the stream path without changing cache policy, page-cache behavior, or numeric results.
- `theoretical_upper_bound`: This cannot reduce the dominant cold page/refault cost. The hard upper bound is limited to CPU lookup overhead inside `src0_ms`/stream calls. With `~29.9k` hits and `~4.9k` misses, an average linear scan of roughly half the cache is up to `~50M` slot checks. If each check is only tens of nanoseconds, the expected gain is sub-second to a few seconds at most. Promote only if this small fixed-cost reduction translates to `eval_tok_s > 1.6` and all gates pass.
- `risk`: A hash-table bug could cause stale cache hits and wrong output. The implementation must update hash entries on insert and eviction, keep the same LRU slot selection as current code, and fall back to miss when a key is absent. France correctness is mandatory.
- `implementation`: Implement `vram_cache_hash_key`, `vram_cache_ht_lookup`, `vram_cache_ht_remove`, and `vram_cache_ht_insert` using the existing `ht` array. Replace linear lookup with hash lookup; on insert, remove the evicted slot key then insert the new key. Keep `slot_key` and `slot_used` unchanged for LRU selection and reporting.
- `test_config`: dirty patch on clean `aa8d6f916`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, unset `GGML_MOE_VRAM_CACHE_GB`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `rollback`: Revert source and rebuild clean if build fails, correctness fails, RAM exceeds 16GB, TTFT exceeds gate, or `eval_tok_s <= 1.6`. Commit/push only if a clean rebuild/rerun from pushed source passes all SOTA gates.
- `required_evidence`: source diff, build log/hash, exact env/command, `summary.json`, `trace_summary.json`, cgroup memory files, France answer, and comparison against cold `13568MiB` and warm/no-drop diagnostic.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T165553Z-20260702_vram-cache-hash-lookup/france-cpu40-vram0gb`
- `result`: completed and rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `first_answer_ms=46636.224639`, wall `2:07.07`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14978424832`, `pgmajfault=608857`, `workingset_refault_file=12035462`, `ram_ok=true`, `correctness_ok=true`.
- `correctness`: pass. France answer is coherent and semantically correct, with the same minor `Rennowned` typo.
- `cache_summary`: `13.2 GiB`, `3192 slots`, `hits=29855`, `misses=4898`, `hit_rate=85.9%`.
- `trace_summary`: `rows=34753`, `cache_hits=29855`, `cache_inserts=4898`, `src0_ms_sum=25292.321`, `total_ms_sum=27569.415`, `trace_span_ms=108718.113`.
- `gap_analysis`: Hash lookup preserved cache behavior and correctness but did not exceed the clean reproducible `1.6 tok/s` baseline. The trace is essentially cold-baseline shaped, so linear cache lookup overhead is not a meaningful bottleneck compared with cold page/refault and CPU fallback gaps.
- `rollback_status`: source change reverted and clean rebuild completed; source diff for `ggml/src/ggml-cuda/moe_stream.cu` is `0`. Post-rollback hashes: `llama-cli=c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`, `libggml-cuda.so.0.10.0=f0c383ee4a785c0992662d8e4f4e127fa85348fa50aa3ad6a8cd85b4ac56cf4b`. No commit/push because this is not an accepted SOTA.

### 当前执行 attempt：cache13568-no-trace-cold-check

- `attempt_id`: `20260702-cache13568-no-trace-cold-check`
- `attempt_kind`: `config-diagnostic`
- `status`: completed / diagnostic_rejected_tie
- `hypothesis`: Most `13568MiB` boundary runs write `one_trace.csv` for every streamed expert. The trace itself may add file writes and page-cache pressure inside the same 16GB cgroup, potentially under-reporting deploy token rate. A strict cold run with the same config but without `GGML_MOE_STREAM_ONE_TRACE_OUT` will measure the low-instrumentation deploy path.
- `expected_delta`: If trace overhead is material, no-trace generation may exceed the traced `1.6 tok/s` boundary. This is not enough by itself to promote a SOTA because the plan requires detailed trace evidence for bottleneck accounting; a faster no-trace result should trigger a follow-up low-overhead evidence strategy. If no-trace also ties `1.6`, trace overhead is not the limiting issue.
- `risk`: Without `one_trace.csv`, cache hit/miss counts are only available from stderr aggregate cache reporting, not per-expert timing. Treat as diagnostic unless paired with a later accepted evidence run.
- `test_config`: clean source, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, unset `GGML_MOE_VRAM_CACHE_GB`, no `GGML_MOE_STREAM_ONE_TRACE_OUT`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `rollback`: No source change. Record metrics and reject as diagnostic unless all SOTA evidence requirements are later satisfied.
- `required_evidence`: exact env/command proving trace is disabled, stdout/stderr with aggregate cache stats, cgroup memory files, `summary.json`, France answer, binary sha256/build line, comparison against traced cold `13568MiB`.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T170258Z-20260702_cache13568-no-trace-cold-check/france-cpu40-vram0gb`
- `result`: completed and diagnostic rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.6`, `first_answer_ms=48779.7506`, wall `2:10.75`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14964670464`, `pgmajfault=619370`, `workingset_refault_file=11898996`, `ram_ok=true`, `correctness_ok=true`.
- `correctness`: pass. France answer is coherent and semantically correct, with the same minor `Rennowned` typo.
- `cache_summary`: `13.2 GiB`, `3192 slots`, `hits=29855`, `misses=4898`, `hit_rate=85.9%`.
- `trace_status`: disabled; environment contains no `GGML_MOE_STREAM_ONE_TRACE_OUT`.
- `gap_analysis`: Disabling `one_trace.csv` does not improve generation above the clean reproducible `1.6 tok/s` line and TTFT is worse than traced `13568MiB` runs. Trace file IO is therefore not the main reason cold runs are slow. Keep trace enabled when needed for bottleneck accounting; do not pursue no-trace as an optimization path.

### 当前执行 attempt：cpu39-cache10496mib-boundary

- `attempt_id`: `20260702-cpu39-cache10496mib-boundary`
- `attempt_kind`: `config-probe`
- `status`: completed / rejected_tie
- `hypothesis`: `cpu_moe=39` reduces one CPU fallback MoE layer, but prior `vram10` increased gate misses enough to regress to `1.5 tok/s`, while `vram11` failed cache allocation. A fractional cache at `10496 MiB` may fit below the allocation cliff and add about `~60` slots over 10GiB, slightly reducing misses while keeping the CPU fallback reduction.
- `expected_delta`: Hard upper bound is small. `cpu39/vram10` had `5659` inserts and `src0_ms≈29.0s`; `cpu40/cache13568` has `4898` inserts and `src0_ms≈24.7s`. Adding `~60` slots can only recover a fraction of the gap, likely less than a few seconds. Promote only if `eval_tok_s > 1.6`, output is correct, RAM <=16GB including page cache, and TTFT is within gate.
- `risk`: `10496 MiB` may still leave too little VRAM for decode temporaries, causing CUDA OOM, or fit but still remain below the `1.6` line. This is the final fractional cpu39 boundary probe unless it materially improves.
- `test_config`: clean source, `cpu_moe=39`, `GGML_MOE_STREAM_ONE_CACHE_MIB=10496`, unset `GGML_MOE_VRAM_CACHE_GB`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `rollback`: No source change. Reject on OOM, cache allocation failure/zero inserts, correctness failure, TTFT > gate, RAM violation, or `eval_tok_s <= 1.6`.
- `required_evidence`: exact env/command, stderr cache allocation/hits or OOM, `summary.json`, `trace_summary.json`, cgroup memory files, France answer, comparison against `cpu39/vram10`, `cpu39/vram11`, and `cpu40/cache13568`.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T170959Z-20260702_cpu39-cache10496mib-boundary/france-cpu39-vram0gb`
- `result`: rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `first_answer_ms=49989.869425`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14935244800`, `pgmajfault=589874`, `workingset_refault_file=12469552`, `ram_ok=true`, `correctness_ok=true`.
- `correctness`: pass. France output is semantically correct and coherent, with the same minor `Rennowned` typo.
- `cache_summary`: `10.2 GiB`, `2469 slots`, `hits=28344`, `misses=5531`, `hit_rate=83.7%`.
- `trace_summary`: `rows=33875`, `cache_hits=28344`, `cache_inserts=5531`, `trace_span_ms=109715.06`, `src0_ms_sum=28655.016`, `total_ms_sum=31059.553`.
- `gap_analysis`: Fractional cache improved the `cpu39` boundary versus `vram10` (`misses 5659 -> 5531`, `src0_ms≈29048ms -> 28655ms`) but only tied the clean reproducible `1.6 tok/s` line and worsened TTFT versus the `cpu40/cache13568` boundary. One fewer CPU-MoE layer is not enough to offset the smaller stream cache and cold refault behavior. Stop the `cpu39` fractional branch unless a separate change frees enough VRAM to use at least the `12GB`/`13568MiB` cache class.

### 当前执行 attempt：cache13824-c224-fit

- `attempt_id`: `20260702-cache13824-c224-fit`
- `attempt_kind`: `config-probe`
- `status`: planned
- `hypothesis`: `GGML_MOE_STREAM_ONE_CACHE_MIB=13824` previously allocated a `13.5 GiB` cache (`3252` slots) but aborted during decode with CUDA OOM when `-c 256 -b 16 -ub 16` left only about `14 MiB` free. The France prompt is short, so reducing context to `-c 224` should still fit the prompt plus `-n 192` while freeing KV/context VRAM for the larger cache.
- `theoretical_upper_bound`: `13568 MiB` has `3192` slots and `4898` misses; `13824 MiB` has `3252` slots, at most `60` more resident experts. At the measured cold miss cost of about `5ms` per miss, direct traced gain is likely only a few hundred milliseconds. The real value of this probe is testing whether the larger cache changes the refault/trace-span shape. Promote only if `eval_tok_s > 1.6`, France output is correct, host RAM including page cache remains `<=16GB`, and TTFT stays within the gate.
- `risk`: If `-c 224` is too small for prompt plus generation, output may truncate or context behavior may differ from the baseline. If the extra cache still leaves too little temporary VRAM, CUDA may OOM. Reject on OOM, cache insertion failure, correctness/truncation failure, TTFT > gate, RAM violation, or `eval_tok_s <= 1.6`.
- `test_config`: clean source, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13824`, unset `GGML_MOE_VRAM_CACHE_GB`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace, CLI args `-c 224 -b 16 -ub 16 -t 20 -tb 20`.
- `rollback`: No source change. Record as boundary evidence only unless all SOTA gates pass. If it produces a new accepted SOTA, immediately write full reproducibility evidence, commit, push to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from the pushed commit before promotion.
- `required_evidence`: exact env/command proving `-c 224` and `GGML_MOE_STREAM_ONE_CACHE_MIB=13824`, stderr cache slots/hits or CUDA OOM, GPU resource samples, `summary.json`, `trace_summary.json`, cgroup memory files, France answer, binary sha256/build line, and explicit promoted/rejected status.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T171931Z-20260702_cache13824-c224-fit/france-cpu40-vram0gb`
- `result`: failed/rejected before generation. The cache allocated as `13.5 GiB`, `3252 slots`, but CUDA aborted at the fifth streamed gate expert with `ggml_cuda_compute_forward: GET_ROWS failed`, `CUDA error: out of memory`.
- `metrics`: no `eval_tok_s`; no TTFT; `exit_status=134`; `memory_peak_bytes=16000000000`; `memory_file_bytes=15125835776`; cgroup `oom=0`, `oom_kill=0`, so this is a VRAM OOM, not a host RAM violation. `gpu_mem_free_mib` dropped to about `14 MiB` just before abort.
- `gap_analysis`: Reducing context from `256` to `224` did not free enough practical temporary VRAM for `13824 MiB`; the cache plus model still leaves too little decode headroom. Do not continue larger-cache integer/fractional probes unless a source change frees substantial VRAM. A `13696 MiB` probe would add only about `30` slots over `13568 MiB`, so its upper bound is too small to prioritize.

### 当前执行 attempt：cpu39-cache10496-profile-admission

- `attempt_id`: `20260702-cpu39-cache10496-profile-admission`
- `attempt_kind`: `source-probe`
- `status`: planned
- `hypothesis`: `cpu39/cache10496` keeps one additional MoE layer on GPU and has fewer streamed gate rows than `cpu40`, but its smaller cache raises misses to `5531`. Offline LRU simulation on the actual `cpu39/cache10496` trace shows profile-guided admission that skips experts with total frequency `1` can reduce misses from `5531` to `4819` while preserving the `cpu39` row reduction. This is a larger miss reduction than the `cpu40/13568` profile case and may finally move rounded generation above `1.6 tok/s`.
- `theoretical_upper_bound`: The simulated miss reduction is `712` fewer cold source loads. With measured `cpu39/cache10496` `src0_ms_sum=28655.016` over `5531` inserts (`~5.18ms/miss`), the direct traced upper bound is about `3.7s`. It cannot remove all untraced CPU fallback/refault gaps, so expected gain is modest. Promote only if `eval_tok_s > 1.6`, output is correct, host RAM including page cache is `<=16GB`, and TTFT is within gate.
- `risk`: The profile is France-trace-specific and may not generalize to other prompts. It must be default-off via `GGML_MOE_STREAM_CACHE_ADMIT_PROFILE`. If it improves France enough to become a candidate, it must then be validated on the small prompt set before making any broad claim. A profile parser/cache-admission bug could cause stale hits or no inserts; France correctness and trace cache counters are mandatory.
- `implementation`: Reintroduce a default-off profile admission path in `ggml/src/ggml-cuda/moe_stream.cu`. Load a TSV profile of `(tensor, expert)` keys at first use; on a cache miss, stream the expert as usual but insert it into the VRAM cache only when the key is present in the profile. Existing behavior must remain unchanged when the env var is unset.
- `test_config`: clean source plus default-off patch, generated profile from `/root/lfz/runs/vendor-ds4-16gb/20260701T170959Z-20260702_cpu39-cache10496mib-boundary/france-cpu39-vram0gb/one_trace.csv` with frequency `>=2`, `cpu_moe=39`, `GGML_MOE_STREAM_ONE_CACHE_MIB=10496`, unset `GGML_MOE_VRAM_CACHE_GB`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `rollback`: Revert source and clean rebuild unless all hard gates pass and `eval_tok_s > 1.6`. If it becomes a new accepted SOTA, first write full reproducibility evidence, then commit and push to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from the pushed commit before promotion.
- `required_evidence`: source diff, generated profile path/count/sha256, exact env/command, stderr cache slots/hits, `summary.json`, `trace_summary.json`, cgroup memory files, France answer, binary sha256/build line, and explicit promoted/rejected/rollback status.
- `profile`: `/root/lfz/runs/vendor-ds4-16gb/cpu39_gate_freq_ge2.tsv`, `3040` entries, sha256 `9c2e8253215457ba6889cfb77eb14cd654b62d2fdffb7d8c8013adc967af1e83`.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T172501Z-20260702_cpu39-cache10496-profile-admission/france-cpu39-vram0gb`
- `result`: completed and rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `first_answer_ms=50852.470994`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14977093632`, `pgmajfault=583564`, `workingset_refault_file=11409126`, `ram_ok=true`, `correctness_ok=true`.
- `correctness`: pass. France answer is semantically correct and coherent, with the same minor `Rennowned` typo.
- `cache_summary`: `10.2 GiB`, `2469 slots`, profile loaded `3040` entries, `hits=29056`, `misses=4819`, `hit_rate=85.8%`.
- `trace_summary`: `rows=33875`, `cache_hits=29056`, `trace_misses=4819`, `cache_inserts=3439`, `src0_ms_sum=24427.7`, `total_ms_sum=26693.099`.
- `gap_analysis`: Profile admission matched the offline prediction mechanically (`misses 5531 -> 4819`, `src0_ms 28655.016ms -> 24427.7ms`) and reduced `workingset_refault_file`, but rounded end-to-end generation still tied `1.6 tok/s` and TTFT worsened versus the `cpu40/cache13568` boundary. The remaining wall time is not dominated by gate cache pollution alone; untraced CPU fallback/refault gaps still prevent a SOTA. Source was reverted and clean rebuild restored `llama-cli=c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`, `libggml-cuda.so.0.10.0=baa7460b7bd1fef1aec3ad037b788d81397ed3eab230983466bfce72f22458ae`, `libggml-cpu.so.0.10.0=38c2a657261227252ac214e1aab81f2397706017a91fc061b2703dc51e6089bc`. No commit/push because this is not an accepted SOTA.

### 当前执行 attempt：cuda-graphs-enabled-cache13568

- `attempt_id`: `20260702-cuda-graphs-enabled-cache13568`
- `attempt_kind`: `config-probe`
- `status`: planned
- `hypothesis`: The strict runner disables CUDA graphs via `GGML_CUDA_DISABLE_GRAPHS=1`. Current cold runs are dominated by page/refault and CPU fallback gaps, but trace span remains much larger than summed gate stream time; enabling CUDA graphs may reduce part of the repeated CUDA graph scheduling/launch overhead outside the per-expert trace without changing model semantics.
- `theoretical_upper_bound`: This cannot reduce `src0` cold miss time (`~24-27s`) or CPU up/down page faults. The upper bound is only the non-stream CUDA launch/scheduler portion of the untraced gap. If page/refault still dominates, expected token rate remains near `1.6`. Promote only if `eval_tok_s > 1.6`, France output is correct, host RAM including page cache stays `<=16GB`, and TTFT is within gate.
- `risk`: CUDA graph capture may not be compatible with the dynamic MoE stream path or may add capture overhead/VRAM pressure. Reject on CUDA graph errors, OOM, correctness failure, TTFT > gate, RAM violation, or `eval_tok_s <= 1.6`.
- `test_config`: clean source, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_CUDA_DISABLE_GRAPHS=0`, unset `GGML_MOE_VRAM_CACHE_GB`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `rollback`: No source change. Record as rejected unless all hard gates pass and token rate improves. If this somehow becomes a new accepted SOTA, immediately record full reproducibility evidence, commit/push source to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun before promotion.
- `required_evidence`: exact env proving `GGML_CUDA_DISABLE_GRAPHS=0`, stdout/stderr graph/cuda messages, `summary.json`, `trace_summary.json`, cgroup memory files, France answer, binary sha256/build line, and explicit promoted/rejected status.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T173146Z-20260702_cuda-graphs-enabled-cache13568/france-cpu40-vram0gb`
- `result`: completed and rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.6`, `first_answer_ms=46589.645517`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14958915584`, `pgmajfault=608838`, `workingset_refault_file=12188046`, `ram_ok=true`, `correctness_ok=true`.
- `correctness`: pass. France output is semantically correct and coherent, with the same minor `Rennowned` typo.
- `cache_summary`: `13.2 GiB`, `3192 slots`, `hits=29855`, `misses=4898`, `hit_rate=85.9%`.
- `trace_summary`: `rows=34753`, `cache_hits=29855`, `cache_inserts=4898`, `trace_span_ms=108365.71`, `src0_ms_sum=25238.358`, `total_ms_sum=27545.916`.
- `gap_analysis`: Enabling CUDA graphs did not change cache shape, cold fault pressure, trace span, or rounded token rate. The current bottleneck is still cold page/refault plus CPU fallback gaps rather than CUDA launch/graph overhead. No source change and no commit/push.

### 当前执行 attempt：no-defer-experts-cache13568

- `attempt_id`: `20260702-no-defer-experts-cache13568`
- `attempt_kind`: `config-diagnostic`
- `status`: planned
- `hypothesis`: Current accepted-style runs use `--defer-experts`, which drops/deprioritizes expert mmap residency after load to keep host RSS/page cache inside the 16GB gate. The repeated cold refault shape may partly come from this defer/drop policy. Running the same cold `cpu40/cache13568` configuration without `--defer-experts` will show whether allowing normal mmap prefetch/residency improves decode or instead increases load/TTFT/reclaim enough to reject the path.
- `theoretical_upper_bound`: The warm/no-drop diagnostic proves that a better page-cache state can reach `2.8 tok/s`, so in principle avoiding expert page drops could reduce major faults and trace span. The hard downside is TTFT and cgroup reclaim: mapping/prefetching more expert pages may spend the same IO before first token and may still be limited to 16GB file cache. Promote only if `eval_tok_s > 1.6`, France output is correct, RAM including page cache stays `<=16GB`, and TTFT remains within gate.
- `risk`: Load time may become very long, TTFT may exceed gate, or the process may thrash under cgroup reclaim. Run under `MemoryMax=16000000000` and `MemorySwapMax=0`; reject on timeout, OOM, correctness failure, TTFT > gate, RAM violation, or no token-rate improvement.
- `test_config`: clean source, no `--defer-experts`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, unset `GGML_MOE_VRAM_CACHE_GB`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace, CLI args `-n 192 -c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `rollback`: No source change. Record as diagnostic/rejected unless all SOTA gates pass. If it somehow passes as a new SOTA, immediately write full reproducibility evidence and push source to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun before promotion.
- `required_evidence`: exact command proving absence of `--defer-experts`, environment, stdout/stderr, `summary.json`, `trace_summary.json`, cgroup memory files, France answer, binary sha256/build line, and explicit promoted/rejected status.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T173741Z-20260702_no-defer-experts-cache13568/france-cpu40-vram0gb`
- `result`: failed/rejected timeout. The run was killed after about `304` resource samples while still not producing a complete answer or timing summary. The exact command contains no `--defer-experts`.
- `partial_metrics`: `eval_tok_s=null`, `prompt_tok_s=null`, `first_answer_ms=null`, `correctness_ok=false`, `memory_peak_bytes_from_samples=16000000000`, `memory_current_max_from_samples=16000000000`, GPU free around `234 MiB` during the long run.
- `partial_trace_summary`: `rows=6656`, `cache_hits=0`, `cache_inserts=6656`, `trace_misses=6656`, `trace_span_ms=118575.8`, `src0_ms_sum=36159.303`, `total_ms_sum=37664.522`.
- `gap_analysis`: Without `--defer-experts`, the run keeps the cgroup at the 16GB ceiling and still has no VRAM cache hits in the early generated trace because it does not progress far enough to reuse cached experts. It is much slower than normal strict cold runs and fails TTFT/correctness by timeout. Keep `--defer-experts`; no-defer mmap residency is not viable under the 16GB host RAM gate.

### 当前执行 attempt：block-readahead-4096-cache13568

- `attempt_id`: `20260702-block-readahead-4096-cache13568`
- `attempt_kind`: `system-config-probe`
- `status`: planned
- `hypothesis`: Model file is on `/dev/vda1` backed by `/sys/block/vda`, and current block read-ahead is only `128KB`. Each streamed DS4 gate expert source is about `4.25MiB` contiguous. Prior `MADV_RANDOM` made miss latency catastrophically worse, proving Linux readahead is important; increasing block device read-ahead to `4096KB` may reduce per-miss major fault/read overhead for cold expert pages.
- `theoretical_upper_bound`: Cold `cpu40/cache13568` has `4898` misses and `src0_ms≈25s`, about `5.1ms` per expert. If larger read-ahead reduces only contiguous read/fault overhead by 20%, direct gain is about `5s`; if cgroup reclaim pressure increases, it may regress. Promote only if `eval_tok_s > 1.6`, France output is correct, host RAM including page cache stays `<=16GB`, and TTFT remains within gate.
- `risk`: Larger read-ahead may pull useless pages into the 16GB cgroup file cache, increasing reclaim/refault and TTFT. The system setting must be restored to the original `128KB` after the run regardless of result. This is a config probe; no source commit is expected unless a later source-backed SOTA uses it as part of a reproducible setup.
- `test_config`: temporarily set `/sys/block/vda/queue/read_ahead_kb=4096`, clean source, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, unset `GGML_MOE_VRAM_CACHE_GB`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`; restore read-ahead to `128` immediately after.
- `rollback`: Restore `/sys/block/vda/queue/read_ahead_kb=128`. Record as rejected unless all SOTA gates pass and token rate improves. If accepted, run directory must include the original/restored read-ahead values and exact sysfs command; source push still required only if source differs from clean branch.
- `required_evidence`: device mapping (`findmnt`, `lsblk`), original and changed read-ahead values, exact sysfs command, exact run command/env, `summary.json`, `trace_summary.json`, cgroup memory files, France answer, binary sha256/build line, and restored read-ahead value.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T174815Z-20260702_block-readahead-4096-cache13568/france-cpu40-vram0gb`
- `result`: failed/rejected timeout/regression. The run was killed after about `317` resource samples because it was far slower than normal cold runs and had no completed answer/timing summary.
- `system_evidence`: model is on `/dev/vda1` (`ext4`, `/`); original `/sys/block/vda/queue/read_ahead_kb=128`, changed to `4096` for the run, restored to `128` after termination.
- `partial_metrics`: `eval_tok_s=null`, `prompt_tok_s=null`, `first_answer_ms=null`, `correctness_ok=false`, `memory_peak_bytes_from_samples=16000000000`, `memory_current_max_from_samples=16000000000`, GPU free around `238 MiB` during the run.
- `partial_trace_summary`: `rows=12673`, `cache_hits=9354`, `cache_inserts=3319`, `trace_misses=3319`, `trace_span_ms=319194.37`, `src0_ms_sum=47524.562`, `total_ms_sum=48716.777`.
- `gap_analysis`: Increasing block read-ahead to `4096KB` made cgroup reclaim/page-cache behavior much worse: the partial run took about `319s` of trace span before reaching normal completion, versus about `108s` for a complete `13568MiB` cold run. Larger kernel readahead pulls too much data into the 16GB file-cache budget. Keep the default `128KB`; do not pursue larger block read-ahead under the current 16GB gate.

### 当前执行 attempt：block-readahead-512-cache13568

- `attempt_id`: `20260702-block-readahead-512-cache13568`
- `attempt_kind`: `system-config-probe`
- `status`: planned
- `hypothesis`: `4096KB` block read-ahead is too aggressive under the 16GB cgroup, but the default `128KB` may still under-read for `4.25MiB` contiguous expert pages. A moderate `512KB` read-ahead may reduce per-miss latency without pulling as many useless pages into the cgroup file cache.
- `theoretical_upper_bound`: Same base as the `4096KB` probe: cold `cpu40/cache13568` has `4898` misses and about `25s` `src0_ms`. A modest 5-10% miss-latency reduction would save `1.2-2.5s`, likely only enough for a tie unless it also reduces trace-span/refault gaps. Promote only if `eval_tok_s > 1.6`, France output is correct, host RAM including page cache stays `<=16GB`, and TTFT remains within gate.
- `risk`: Even `512KB` may increase cgroup file-cache pressure or change fault clustering unfavorably. Restore `/sys/block/vda/queue/read_ahead_kb=128` after the run regardless of result.
- `test_config`: temporarily set `/sys/block/vda/queue/read_ahead_kb=512`, clean source, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, unset `GGML_MOE_VRAM_CACHE_GB`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`; restore read-ahead to `128` immediately after.
- `rollback`: Restore `/sys/block/vda/queue/read_ahead_kb=128`. Record as rejected unless all SOTA gates pass and token rate improves. No source commit is expected for this config-only probe.
- `required_evidence`: original/changed/restored read-ahead values, exact sysfs command, exact run command/env, `summary.json`, `trace_summary.json`, cgroup memory files, France answer, binary sha256/build line, and explicit promoted/rejected status.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T175820Z-20260702_block-readahead-512-cache13568/france-cpu40-vram0gb`
- `result`: completed and rejected. `eval_tok_s=1.4`, `prompt_tok_s=0.5`, `first_answer_ms=52169.410561`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14980349952`, `pgmajfault=472154`, `workingset_refault_file=17012380`, `ram_ok=true`, `correctness_ok=true`.
- `system_evidence`: model is on `/dev/vda1`; original read-ahead `128`, changed to `512` for the run, restored/final value `128`.
- `correctness`: pass. France output is semantically correct and coherent, with the same minor `Rennowned` typo.
- `cache_summary`: `13.2 GiB`, `3192 slots`, `hits=29855`, `misses=4898`, unchanged versus baseline.
- `trace_summary`: `rows=34753`, `cache_hits=29855`, `cache_inserts=4898`, `trace_span_ms=130299.146`, `src0_ms_sum=23774.492`, `total_ms_sum=26043.127`.
- `gap_analysis`: `512KB` read-ahead reduced traced `src0_ms` versus some cold controls but increased trace span, TTFT, and end-to-end runtime. Like synchronous `WILLNEED`, the cost moves into untraced cgroup reclaim/refault gaps. Keep block read-ahead at `128KB`; do not pursue larger device read-ahead values under the 16GB gate.

### 当前执行 attempt：stream-madv-cold-after-use

- `attempt_id`: `20260702-stream-madv-cold-after-use`
- `attempt_kind`: `source-probe`
- `status`: planned
- `hypothesis`: Current stream path calls `MADV_DONTNEED` after each expert source range, which immediately discards source pages and can force later refaults. Disabling DONTNEED entirely was slower because file-cache pressure increased. `MADV_COLD` is a middle ground: it marks pages reclaimable under pressure without immediately discarding them. This may reduce refault churn while still letting the 16GB cgroup reclaim pages when needed.
- `theoretical_upper_bound`: Direct traced `dontneed_ms` is small, but repeated refault/reclaim dominates the untraced gap. Warm/no-drop shows an upper bound of `2.8 tok/s`; practical gain from changing only post-use advice is likely a few seconds of trace-span/refault reduction. Promote only if `eval_tok_s > 1.6`, France output is correct, host RAM including page cache stays `<=16GB`, and TTFT remains within gate.
- `risk`: `MADV_COLD` may be unavailable or may keep too many pages resident, recreating the `DONTNEED=0` regression. It may also reduce immediate `src0_ms` while increasing untraced reclaim gaps. The source change must be default-off and reverted unless the run passes all gates and improves token rate.
- `implementation`: Add default-off env `GGML_MOE_STREAM_DONTNEED_COLD=1` in `ggml/src/ggml-cuda/moe_stream.cu`. When enabled and `MADV_COLD` is defined, `moe_stream_dontneed_source_pages()` uses `madvise(..., MADV_COLD)` instead of `MADV_DONTNEED`; default behavior remains unchanged.
- `test_config`: clean source plus default-off patch, `GGML_MOE_STREAM_DONTNEED_COLD=1`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, unset `GGML_MOE_VRAM_CACHE_GB`, cold `drop_caches`, 16GB cgroup, France prompt, DS4 gate stream trace, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `rollback`: Revert source and clean rebuild unless all hard gates pass and `eval_tok_s > 1.6`. If accepted, record full reproducibility evidence, commit/push to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun before promotion.
- `required_evidence`: source diff, build log/hash, exact env/command, stdout/stderr, `summary.json`, `trace_summary.json`, cgroup memory files, France answer, binary sha256/build line, and explicit promoted/rejected/rollback status.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T180541Z-20260702_stream-madv-cold-after-use/france-cpu40-vram0gb`
- `result`: completed and rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.6`, `first_answer_ms=47010.926377`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14957977600`, `pgmajfault=585966`, `workingset_refault_file=11021118`, `ram_ok=true`, `correctness_ok=true`.
- `correctness`: pass. France output is semantically correct and coherent, with the same minor `Rennowned` typo.
- `cache_summary`: `13.2 GiB`, `3192 slots`, `hits=29855`, `misses=4898`, unchanged versus baseline.
- `trace_summary`: `rows=34753`, `cache_hits=29855`, `cache_inserts=4898`, `trace_span_ms=110634.138`, `src0_ms_sum=25664.264`, `dontneed_ms_sum=2146.005`, `total_ms_sum=28914.328`.
- `gap_analysis`: `MADV_COLD` slightly reduced `workingset_refault_file` versus some cold controls, but did not improve rounded token rate and worsened trace span/total traced time versus the best `13568MiB` control. It is not a SOTA. Source was reverted and clean rebuild restored `llama-cli=c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`, `libggml-cuda.so.0.10.0=dfa48b3ce1be716e7e82278a3671406096894d06c25068d0e12f6d5e36c4e75b`, `libggml-cpu.so.0.10.0=38c2a657261227252ac214e1aab81f2397706017a91fc061b2703dc51e6089bc`. No commit/push because this is not an accepted SOTA.

1. 专家页启动预热（仅 cold-start）：
   - 在不突破 16GB 的前提下，针对 `france prompt + 小规模测试集` 生成一个紧凑 route 预热集合；
   - 约束只允许一次性预热，不改变稳定运行逻辑。
2. 分阶段 vram cache 复核：
   - 在保持 `vram_cache=12` 附近，测试 `11GB / 12GB / 13GB边界` 的 `cache insert/hit` 与 `src0` stall；
   - 严禁回到 `0 inserts`、`TTFT 显著上升` 的配置。
3. 小动作优先：
   - 不改算子语义前提下，验证 `DONTNEED`、`I/O hint`、`stream 并发`、`q8 trace` 采样是否存在稳定收益；
   - 不可行则立即回滚。

### C. 结果验证（cold-start 真实）

1. 固定 1 组 warmup + 5 组 small prompt test（用户指定）：
   - warmup: France prompt
   - test: France / Quantum / Python Fibonacci / Japan / Climate
2. 判定标准：
   - 目标在 16GB 内 `8~10 tok/s`（按用户原始口径）；
   - 4/5 prompt 至少语义可读，France/日本等短文本必须正确；
   - TTFT 不高于当前基线 + 20%。

### 当前执行 attempt：stream-updown-nocache-gate-cache

- `attempt_id`: `20260702-stream-updown-nocache-gate-cache`
- `attempt_kind`: `source-probe`
- `status`: rejected
- `hypothesis`: Naive all-FFN stream was correct but slow because `ffn_gate_exps`、`ffn_up_exps`、`ffn_down_exps` shared one VRAM expert cache, causing cache inserts to jump from gate-only `~5.1k` to all-FFN `~36.7k` and `src0_ms` to jump to `~165.6s`. This probe keeps gate experts cacheable while letting up/down experts use the CUDA stream compute path without lookup/insert into the VRAM cache. If the prior regression was mostly cache pollution, this can recover some CPU fallback time while preserving the gate cache working set.
- `theoretical_upper_bound`: The CPU fallback trace attributes roughly `~41s` wall estimate to `ffn_up_exps` and `~36s` to `ffn_down_exps`, but no-cache up/down streaming still pays cold H2D/page-read cost on every up/down expert. Therefore the practical upper bound is well below the warm/no-drop diagnostic `2.8 tok/s`; promote only if the net end-to-end result exceeds the current clean reproducible cold line `1.6 tok/s` and passes all hard gates.
- `risk`: Up/down no-cache streaming may add too much H2D/page-read traffic and become closer to the rejected all-FFN path than to the gate-only path. It may also increase TTFT by adding many early stream rows. Correctness must be rechecked because mixed gate-cache/updown-nocache execution changes which experts run on CUDA.
- `implementation`: Add default-off env `GGML_MOE_STREAM_NOCACHE_NAME_FILTER=<csv substrings>` in `ggml/src/ggml-cuda/moe_stream.cu`. When a streamed tensor name matches this filter, skip `vram_cache_lookup()` and `vram_cache_insert()` and copy into per-slot staging `ctx.d_src0`; other streamed tensors keep the existing cache behavior. Also make `GGML_MOE_STREAM_ONE_NAME_FILTER` accept comma-separated substrings while preserving existing single-substring behavior.
- `test_config`: patched default-off source, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, unset/shared `GGML_MOE_VRAM_CACHE_GB` effectively `0`, cold `drop_caches`, strict 16GB cgroup, France prompt, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_`, `GGML_MOE_STREAM_NOCACHE_NAME_FILTER=ffn_up_exps,ffn_down_exps`, trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `required_evidence`: source diff, build log/hash, exact env/command, stdout/stderr, cgroup memory files proving page cache is inside 16GB, France output, `summary.json`, trace summary split by tensor showing gate cache hits/inserts and no up/down cache inserts, binary sha256/build line, and explicit promoted/rejected/rollback status.
- `rollback`: Revert source and clean rebuild unless all hard gates pass and `eval_tok_s > 1.6`. If accepted, immediately write complete reproducibility evidence, commit/push source to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from the pushed commit before promotion.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T181617Z-20260702_stream-updown-nocache-gate-cache/france-cpu40-vram0gb`
- `result`: completed and rejected. `eval_tok_s=0.9`, `prompt_tok_s=0.5`, `first_answer_ms=55901.687693`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14970798080`, `pgmajfault=28372`, `workingset_refault_file=15515849`, `ram_ok=true`, `correctness_ok=true`.
- `correctness`: pass. France answer is semantically correct and coherent: it describes France as a Western European country, mentions history/culture, Eiffel Tower/Louvre/Versailles, cuisine/wine/fashion, the EU, Paris, geography, cities and villages.
- `trace_summary`: `rows=110697`, `cache_hits=32089`, `cache_inserts=4810`, `trace_misses=78608`, `trace_span_ms=196932.945`, `src0_ms_sum=170626.622`, `total_ms_sum=188661.491`. Tensor split: `ffn_gate_exps rows=36899 hits=32089 inserts=4810 src0_ms=25869.982`; `ffn_up_exps rows=36899 hits=0 inserts=0 src0_ms=72573.873`; `ffn_down_exps rows=36899 hits=0 inserts=0 src0_ms=72182.767`.
- `gap_analysis`: The no-cache mechanism worked mechanically: up/down did not insert into the VRAM cache, so gate cache pollution was avoided. The regression comes from paying cold H2D/page-read cost for every up/down expert, adding about `145s` of up/down `src0_ms` and pushing the full trace span to about `197s`. This proves broad up/down CUDA stream without a reusable/cacheable working set is not viable under the 16GB cold-start gate.
- `rollback_status`: source change reverted and clean rebuild completed; source diff is `0`. Post-rollback hashes: `llama-cli=c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`, `libggml-cuda.so.0.10.0=dff7f183052999b636b57a637ef2e1bea7a57016f4b40d90601a77b578997cb3`, `libggml-cpu.so.0.10.0=38c2a657261227252ac214e1aab81f2397706017a91fc061b2703dc51e6089bc`. No commit/push because this is not an accepted SOTA.

### 当前执行 attempt：stream-opwide-willneed-gate-cache13568

- `attempt_id`: `20260702-stream-opwide-willneed-gate-cache13568`
- `attempt_kind`: `source-probe`
- `status`: rejected
- `hypothesis`: The historical `2.6 tok/s` run and current clean `1.6 tok/s` reruns have the same gate-only cache shape (`rows=34753`, `hits=29624`, `inserts=5129` for 12GB), but clean reruns have much larger untraced trace span, system time, major faults, filesystem inputs, and `workingset_refault_file`. This suggests the bottleneck is not cache policy alone, but cold page/refault scheduling around gate expert reads. For each MoE op, the active gate experts are known before the per-expert stream loop. A default-off op-wide `MADV_WILLNEED` on only the active `ffn_gate_exps` pages may let the kernel start readahead for the full current op while the first streamed experts are being copied/computed, reducing later direct fault stalls without relying on external warm page cache.
- `theoretical_upper_bound`: The clean `cache13568` run has `trace_span≈108s`, `src0_ms≈25.2s`, and `workingset_refault_file≈12.2M`; the historical 2.6 run had `trace_span≈71s` and `workingset_refault_file≈3.4M`. Op-wide prefetch cannot eliminate CPU fallback or all cold IO, so a realistic upper bound is a fraction of the `~37s` trace-span/refault gap. It must not exceed 16GB cgroup memory including file cache; if it pulls too many pages too early, it will increase reclaim and regress like larger read-ahead/WILLNEED probes.
- `risk`: `MADV_WILLNEED` may be synchronous or may overfill the 16GB file-cache budget, increasing direct reclaim, TTFT, or token rate regression. The probe must be name-filtered to `ffn_gate_exps`; prefetching up/down would recreate the rejected broad fallback/stream pressure. Correctness risk is low because compute results are unchanged, but France output must still pass.
- `implementation`: Add default-off env `GGML_MOE_STREAM_OP_WILLNEED=1` in `ggml/src/ggml-cpu/ggml-cpu.c`, plus optional `GGML_MOE_STREAM_OP_WILLNEED_NAME_FILTER=<csv substrings>`. When enabled and the tensor name matches the filter, thread 0 calls the existing aligned `madvise(..., MADV_WILLNEED)` helper for each active expert before the CUDA stream loop for that MoE op. Default behavior remains unchanged.
- `test_config`: patched default-off source, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_OP_WILLNEED=1`, `GGML_MOE_STREAM_OP_WILLNEED_NAME_FILTER=ffn_gate_exps`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `required_evidence`: source diff, build log/hash, exact env/command, stdout/stderr, cgroup memory files, France output, `summary.json`, trace summary, comparison to clean `cache13568`, binary sha256/build line, and explicit promoted/rejected/rollback status.
- `rollback`: Revert source and clean rebuild if build fails, correctness fails, RAM exceeds 16GB, TTFT exceeds gate, or `eval_tok_s <= 1.6`. If accepted, immediately write complete reproducibility evidence, commit/push source to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from the pushed commit before promotion.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T183053Z-20260702_stream-opwide-willneed-gate-cache13568/france-cpu40-vram0gb`
- `result`: completed and rejected. `eval_tok_s=1.4`, `prompt_tok_s=0.7`, `first_answer_ms=45919.682075`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14978084864`, `pgmajfault=607397`, `workingset_refault_file=16767150`, `ram_ok=true`, `correctness_ok=true`.
- `correctness`: pass. France output is semantically correct and coherent.
- `trace_summary`: `rows=34753`, `cache_hits=29855`, `cache_inserts=4898`, `trace_misses=4898`, `trace_span_ms=120927.803`, `src0_ms_sum=19962.937`, `dontneed_ms_sum=1153.678`, `total_ms_sum=22237.299`.
- `gap_analysis`: Op-wide `MADV_WILLNEED` did reduce direct traced gate `src0_ms` versus clean `cache13568` (`~25.2s -> ~20.0s`), so the mechanism worked. However, it increased untraced reclaim/refault pressure (`workingset_refault_file≈16.8M`) and widened trace span to `~120.9s`, causing end-to-end rate to regress to `1.4 tok/s`. Like block read-ahead and earlier WILLNEED attempts, page-in is moved rather than removed under the 16GB cgroup.
- `rollback_status`: source change reverted and clean rebuild completed; source diff is `0`. Post-rollback hashes: `llama-cli=c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`, `libggml-cpu.so.0.10.0=38c2a657261227252ac214e1aab81f2397706017a91fc061b2703dc51e6089bc`, `libggml-cuda.so.0.10.0=dff7f183052999b636b57a637ef2e1bea7a57016f4b40d90601a77b578997cb3`. No commit/push because this is not an accepted SOTA.

### 当前执行 attempt：cpu-fallback-ramtier-updown64

- `attempt_id`: `20260702-cpu-fallback-ramtier-updown64`
- `attempt_kind`: `source-probe`
- `status`: rejected
- `hypothesis`: The remaining clean cold bottleneck is primarily non-streamed `ffn_up_exps` and `ffn_down_exps` CPU fallback plus file-cache refault/reclaim. Directly streaming up/down to GPU is correct but regresses because every up/down expert pays cold H2D/page-read cost. A smaller alternative is to keep CPU compute semantics but copy a hot profile of up/down experts into anonymous RAM inside the same 16GB cgroup on first use, then point CPU fallback chunks at that anonymous copy. This avoids VRAM cache pollution and may reduce repeated mmap/page-cache refaults for hot up/down experts.
- `theoretical_upper_bound`: The top64 up/down profile has `128` experts, about `544MiB` of expert payload. Prior selective-hot-updown64 streaming reduced CPU fallback gap by only about `2.1s` while adding `~1.8s` stream cost; a CPU RAM tier avoids the GPU stream/H2D cache pollution but still pays one cold memcpy/read per loaded expert. Expected gain is a few seconds at most. Promote only if `eval_tok_s > 1.6`, RAM including anon+file page cache stays `<=16GB`, France output is correct, and TTFT remains within gate.
- `risk`: Loading hot experts into anonymous memory may reduce file-cache capacity and increase reclaim; the first-use memcpy is serialized on thread 0 before CPU chunks, so TTFT can regress. If the profile does not cover enough repeated CPU fallback traffic, the result will tie or regress. The source change must be default-off and reverted unless all gates pass and token rate improves.
- `implementation`: Add default-off `GGML_MOE_CPU_RAM_TIER_PROFILE=<tsv>` and `GGML_MOE_CPU_RAM_TIER_MAX_MIB=<MiB>` in `ggml/src/ggml-cpu/ggml-cpu.c`. The profile is `tensor_name<TAB>expert_index`. Thread 0 loads active profiled experts into anonymous aligned memory before CPU fallback chunks; all worker threads use the anonymous pointer for loaded entries. Default behavior remains unchanged.
- `test_config`: patched default-off source, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_CPU_RAM_TIER_PROFILE=/root/lfz/runs/vendor-ds4-16gb/20260701T140952Z-20260701T_selective-hot-updown64-stream/hot_updown64_allowlist.tsv`, `GGML_MOE_CPU_RAM_TIER_MAX_MIB=640`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, gate stream trace and CPU fallback chunk trace if feasible, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `required_evidence`: source diff, profile path and sha256, build log/hash, exact env/command, RAM tier load counters/logs, stdout/stderr, cgroup memory files, France output, `summary.json`, gate trace summary, optional CPU chunk trace comparison, binary sha256/build line, and explicit promoted/rejected/rollback status.
- `rollback`: Revert source and clean rebuild if build fails, correctness fails, RAM exceeds 16GB, TTFT exceeds gate, or `eval_tok_s <= 1.6`. If accepted, immediately write complete reproducibility evidence, commit/push source to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from the pushed commit before promotion.
- `profile`: `/root/lfz/runs/vendor-ds4-16gb/20260701T140952Z-20260701T_selective-hot-updown64-stream/hot_updown64_allowlist.tsv`, `128` entries, sha256 `549c62e4d1547b42d84b97474353a145292aefa88a2642c757cdd51998d18e7c`.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T184329Z-20260702_cpu-fallback-ramtier-updown64/france-cpu40-vram0gb`
- `result`: completed and rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `first_answer_ms=48077.79944`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14384857088`, `pgmajfault=583710`, `workingset_refault_file=11987917`, `ram_ok=true`, `correctness_ok=true`.
- `ram_tier_summary`: loaded full top64 profile payload, `544.0 MiB` anonymous RAM tier by final stderr counter. Page cache at process end dropped to `14.38GB`, showing the tier stayed inside the same 16GB cgroup budget rather than using external warm cache.
- `trace_summary`: `rows=34753`, `cache_hits=29855`, `cache_inserts=4898`, `trace_span_ms=108707.099`, `src0_ms_sum=24885.488`, `total_ms_sum=27204.576`.
- `gap_analysis`: The RAM tier worked mechanically and did not perturb gate stream cache shape, but it only tied the clean `1.6 tok/s` line. The top64 hot up/down set does not cover enough CPU fallback cost to overcome first-use memcpy and remaining cold refault gaps.

### 当前执行 attempt：cpu-fallback-ramtier-updown128

- `attempt_id`: `20260702-cpu-fallback-ramtier-updown128`
- `attempt_kind`: `source-probe`
- `status`: rejected
- `hypothesis`: The top64 RAM tier mechanically worked and stayed RAM-compliant but only tied `1.6 tok/s`, with gate trace shape essentially unchanged. Increasing coverage to top128 up/down experts (`256` experts, about `1088MiB`) may reduce more CPU fallback mmap refaults while still avoiding GPU up/down H2D/cache pollution. This tests the useful upper bound of the RAM-tier direction before rolling the source back.
- `theoretical_upper_bound`: If top64 only ties, top128 can help only if the extra `~544MiB` of hot experts covers enough repeated CPU fallback traffic to offset more serialized first-use memcpy and lower file-cache capacity. The practical upper bound remains small; promote only if `eval_tok_s > 1.6`, RAM including anon+file page cache stays `<=16GB`, France output is correct, and TTFT remains within gate.
- `risk`: Larger anonymous RAM tier can worsen cgroup reclaim or TTFT and may reduce file cache enough to offset any CPU fallback win. If it does not improve over top64, stop the RAM-tier branch and revert source.
- `test_config`: same source patch as top64, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_CPU_RAM_TIER_PROFILE=/root/lfz/runs/vendor-ds4-16gb/20260701T141433Z-20260701T_selective-hot-updown128-stream/hot_updown128_allowlist.tsv`, `GGML_MOE_CPU_RAM_TIER_MAX_MIB=1200`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, gate stream trace, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `required_evidence`: source diff, profile path and sha256, exact env/command, RAM tier loaded MiB, stdout/stderr, cgroup memory files, France output, `summary.json`, gate trace summary, comparison to top64 and clean `cache13568`, binary sha256/build line, and explicit promoted/rejected/rollback status.
- `rollback`: Revert source and clean rebuild if build fails, correctness fails, RAM exceeds 16GB, TTFT exceeds gate, or `eval_tok_s <= 1.6`. If accepted, immediately write complete reproducibility evidence, commit/push source to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from the pushed commit before promotion.
- `profile`: `/root/lfz/runs/vendor-ds4-16gb/20260701T141433Z-20260701T_selective-hot-updown128-stream/hot_updown128_allowlist.tsv`, `256` entries, sha256 `ab900b0e12d998d28f550d376c2c7333a4de82efed0ca520c1fc12972107c167`.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T184738Z-20260702_cpu-fallback-ramtier-updown128/france-cpu40-vram0gb`
- `result`: completed and rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.6`, `first_answer_ms=47184.560027`, `memory_peak_bytes=16000000000`, `memory_file_bytes=13819363328`, `pgmajfault=565805`, `workingset_refault_file=11763302`, `ram_ok=true`, `correctness_ok=true`.
- `ram_tier_summary`: loaded full top128 profile payload, `1088.0 MiB` anonymous RAM tier by final stderr counter. Page cache at process end dropped to `13.82GB`, still under the 16GB cgroup.
- `trace_summary`: `rows=34753`, `cache_hits=29855`, `cache_inserts=4898`, `trace_span_ms=107003.632`, `src0_ms_sum=25253.486`, `total_ms_sum=27521.0`.
- `gap_analysis`: Larger RAM-tier coverage slightly reduced refault counters versus top64 and clean controls, but still only tied rounded `1.6 tok/s`. The remaining bottleneck is not solved by caching 1.1GB of hot up/down expert payload in anonymous RAM; either coverage is still too small, or first-use memcpy/file-cache displacement offsets the reuse benefit.
- `rollback_status`: source change reverted and clean rebuild completed; source diff is `0`. Post-rollback hashes: `llama-cli=c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`, `libggml-cpu.so.0.10.0=38c2a657261227252ac214e1aab81f2397706017a91fc061b2703dc51e6089bc`, `libggml-cuda.so.0.10.0=dff7f183052999b636b57a637ef2e1bea7a57016f4b40d90601a77b578997cb3`. No commit/push because neither RAM-tier run is an accepted SOTA.

### 当前执行 attempt：cache13824-no-op-offload-boundary

- `attempt_id`: `20260702-cache13824-no-op-offload-boundary`
- `attempt_kind`: `config-probe`
- `status`: rejected
- `hypothesis`: `GGML_MOE_STREAM_ONE_CACHE_MIB=13824` previously allocates a larger gate expert VRAM cache (`~3252` slots) but then hits CUDA OOM during `GET_ROWS`. Disabling host op offload with `--no-op-offload` may free enough transient GPU workspace/KV-adjacent pressure for the larger cache to survive generation, letting more VRAM be used for the gate cache while keeping the same vendor DS4 semantics.
- `theoretical_upper_bound`: Compared with accepted clean boundary `cache13568` (`~3192` slots, `eval_tok_s=1.6`), `13824MiB` adds only about `60` cache slots. Existing LRU/Belady evidence near this region suggests only tens to low hundreds of misses can be removed, so the direct upper bound is roughly `miss_delta * ~5ms`, likely below `1s` traced time. The main value is to map the VRAM/OOM boundary, not to expect a large token-rate jump.
- `risk`: `--no-op-offload` may move useful GPU work back to host and regress prompt/eval speed or TTFT. If the larger cache still OOMs, or if it fits but only ties/regresses, reject. Correctness must still pass because changing op placement can expose backend edge cases.
- `test_config`: clean source `aa8d6f916`, no source changes, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13824`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, trace enabled, CLI args `-c 224 -b 16 -ub 16 -t 20 -tb 20 --no-op-offload`.
- `required_evidence`: exact env/command, stdout/stderr, cache slot count or OOM line, cgroup memory files proving host RAM including page cache stayed within 16GB, France output if generated, `summary.json`, trace summary if present, binary sha256/build line, and explicit promoted/rejected status.
- `rollback`: No source change. Reject on CUDA OOM, RAM issue, correctness failure, TTFT gate failure, or `eval_tok_s <= 1.6`. If it unexpectedly produces a compliant token-rate improvement, immediately write full SOTA reproduction info, commit/push any source state needed to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before promotion.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T185744Z-20260702_cache13824_no_op_offload_boundary/france-cpu40-vram0gb`
- `result`: failed/rejected. `eval_tok_s=None`, `prompt_tok_s=None`, `first_answer_ms=None`, `exit_status=134`, `ram_ok=true`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15125733376`, `pgmajfault=3036`, `workingset_refault_file=2357`, `correctness_ok=false` because no answer was generated.
- `failure`: CUDA OOM remained even with `--no-op-offload`: stderr shows `VRAM cache: 13.5 GiB, 3252 slots` followed by `ggml_cuda_compute_forward: GET_ROWS failed` and `CUDA error: out of memory`. Resource samples reached about `32096MiB` used / `14MiB` free GPU memory at failure.
- `trace_summary`: only the first four streamed gate experts were traced before abort; this is not enough for token-rate analysis.
- `gap_analysis`: `--no-op-offload` did not free enough transient GPU memory to make a 13.5GiB gate cache viable. The larger cache boundary remains below 13824MiB for this source/binary/config; no source change and no commit/push.

### 当前执行 attempt：cache13824-no-kv-offload-boundary

- `attempt_id`: `20260702-cache13824-no-kv-offload-boundary`
- `attempt_kind`: `config-probe`
- `status`: rejected
- `hypothesis`: The 13.5GiB gate cache fails with only about `14MiB` GPU memory free at `GET_ROWS`. Disabling KV offload may free enough GPU memory for the larger gate cache to fit, while the short `-c 224` context keeps host-KV overhead limited.
- `theoretical_upper_bound`: The gain remains bounded by the same cache-size delta as the previous attempt: `13824MiB` has about `3252` slots versus `13568MiB` at about `3192` slots, so expected miss reduction is small and likely below `1s` direct traced time. If host KV adds attention transfer/CPU overhead, token rate may tie or regress even if OOM is avoided.
- `risk`: Moving KV off GPU can slow prompt/decode and raise TTFT. It may also not free enough memory if `GET_ROWS` temporary workspace, not KV, is the real conflict. Reject on OOM, correctness failure, RAM issue, TTFT gate failure, or no token-rate improvement.
- `test_config`: clean source `aa8d6f916`, no source changes, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13824`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, trace enabled, CLI args `-c 224 -b 16 -ub 16 -t 20 -tb 20 --no-kv-offload`.
- `required_evidence`: exact env/command, stdout/stderr, cache slot count or OOM line, GPU memory samples, cgroup memory files, France output if generated, `summary.json`, trace summary if present, binary sha256/build line, and explicit promoted/rejected status.
- `rollback`: No source change. If accepted unexpectedly, write full SOTA reproduction info, push required source state to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before promotion.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T190121Z-20260702_cache13824_no_kv_offload_boundary/france-cpu40-vram0gb`
- `result`: failed/rejected. `eval_tok_s=None`, `prompt_tok_s=None`, `first_answer_ms=None`, `exit_status=139`, `ram_ok=true`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15203663872`, `pgmajfault=79503`, `workingset_refault_file=5060`, `correctness_ok=false` because no answer was generated.
- `failure`: stderr shows `VRAM cache: 13.5 GiB, 3252 slots`, then `ggml_backend_cuda_buffer_type_alloc_buffer: allocating 4.57 MiB on device 0: cudaMalloc failed: out of memory` and `failed to allocate CUDA0 buffer of size 4792320`. Resource samples again show about `32096MiB` GPU used / `14MiB` free before failure.
- `trace_summary`: only `251` stream rows were traced before abort; this is not a complete generation.
- `gap_analysis`: Moving KV off GPU did not free enough practical CUDA allocation headroom for a 13.5GiB cache. The failure is still a GPU memory boundary, not host RAM; no source change and no commit/push.

### 当前执行 attempt：cache13696-boundary

- `attempt_id`: `20260702-cache13696-boundary`
- `attempt_kind`: `config-probe`
- `status`: rejected
- `hypothesis`: `13568MiB` fits and `13824MiB` fails. Testing `13696MiB` maps the largest stable gate-cache size below the OOM cliff and may add roughly `30` cache slots over the clean `13568MiB` boundary.
- `theoretical_upper_bound`: Very small. At about `5ms` direct `src0` cost per miss, even removing `50-100` misses would save only `0.25-0.5s` traced time. This can only become a promoted SOTA if the reduced misses cross a rounded token-rate boundary without TTFT/correctness regression.
- `risk`: It may still OOM or simply tie `1.6 tok/s`. Because this is no source change, rejection has no rollback cost.
- `test_config`: clean source `aa8d6f916`, no source changes, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13696`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, trace enabled, CLI args `-c 224 -b 16 -ub 16 -t 20 -tb 20`.
- `required_evidence`: exact env/command, stdout/stderr, cache slot count, cgroup memory files, France output, `summary.json`, full trace summary, binary sha256/build line, and explicit promoted/rejected status.
- `rollback`: No source change. Reject on OOM, RAM issue, correctness failure, TTFT gate failure, or `eval_tok_s <= 1.6`. If accepted, immediately follow the hard SOTA reproduction and push gate.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T190337Z-20260702_cache13696_boundary/france-cpu40-vram0gb`
- `result`: completed and rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=47574.661731ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14970724352`, `pgmajfault=607130`, `workingset_refault_file=11862478`, `ram_ok=true`, `correctness_ok=true`.
- `correctness`: pass. France answer is semantically correct and coherent; it describes France as a Western European country with landmarks, culture, cuisine, EU membership, Paris, and diverse geography.
- `trace_summary`: `VRAM cache: 13.4 GiB, 3222 slots`; `rows=34753`, `cache_hits=29870`, `cache_inserts=4883`, `trace_misses=4883`, `trace_span_ms=108775.806`, `src0_ms_sum=24988.505`, `kernel_ms_sum=391.101`, `dontneed_ms_sum=1188.299`, `total_ms_sum=27270.236`.
- `gap_analysis`: Increasing cache from `13568MiB` to `13696MiB` only added about `30` slots and reduced misses by about `15` versus the clean `cache13568` control. The end-to-end rate stayed at `1.6 tok/s`; this confirms cache-size-only work below the 13.5GiB OOM cliff has negligible upside. No source change and no commit/push.

### 当前执行 attempt：cache13568-no-warmup

- `attempt_id`: `20260702-cache13568-no-warmup`
- `attempt_kind`: `config-probe`
- `status`: rejected
- `hypothesis`: The default llama.cpp warmup pass may add cold-start TTFT/page-cache pressure without improving measured decode throughput for the strict France run. Disabling warmup may reduce TTFT and cgroup refault churn while preserving the known-good `13568MiB` gate cache behavior.
- `theoretical_upper_bound`: `--no-warmup` cannot improve the MoE stream miss path directly, so eval token-rate upside is limited. It can reduce TTFT if the default warmup is counted in load/first-answer latency, and it may reduce cold page churn if warmup touches tensors that compete for the 16GB file-cache budget.
- `risk`: Without warmup, first real prompt may pay CUDA graph/kernel setup cost, potentially raising prompt time or first-token latency. Correctness should not change, but output must still be reviewed.
- `test_config`: clean source `aa8d6f916`, no source changes, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20 --no-warmup`.
- `required_evidence`: exact env/command, stdout/stderr, cgroup memory files, France output, `summary.json`, trace summary, comparison to clean `cache13568`, binary sha256/build line, and explicit promoted/rejected status.
- `rollback`: No source change. If `eval_tok_s <= 1.6`, mark as not SOTA even if TTFT improves. If eval token rate improves and gates pass, immediately follow the hard SOTA reproduction and push gate.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T190852Z-20260702_cache13568_no_warmup/france-cpu40-vram0gb`
- `result`: completed and rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=46584.569105ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14965346304`, `pgmajfault=603613`, `workingset_refault_file=12307596`, `ram_ok=true`, `correctness_ok=true`.
- `correctness`: pass. France answer is semantically correct and coherent.
- `trace_summary`: `VRAM cache: 13.2 GiB, 3192 slots`; `rows=34753`, `cache_hits=29855`, `cache_inserts=4898`, `trace_misses=4898`, `trace_span_ms=110472.473`, `src0_ms_sum=24812.119`, `kernel_ms_sum=391.748`, `dontneed_ms_sum=1221.557`, `total_ms_sum=27117.236`.
- `gap_analysis`: `--no-warmup` slightly reduced TTFT versus recent clean controls but did not improve eval token rate. Cache hit/miss shape is unchanged and trace span is still about `110s`; this is not a promoted SOTA. No source change and no commit/push.

### 当前执行 attempt：combo-skip-hit-dontneed-cache13696-t24-nowarmup

- `attempt_id`: `20260702-combo-skip-hit-dontneed-cache13696-t24-nowarmup`
- `attempt_kind`: `source-probe`
- `status`: rejected
- `hypothesis`: Several individually correct probes each tied `1.6 tok/s` but improved different small terms: `t24` reduced traced `src0_ms`, `cache13696` reduced a few gate misses, `--no-warmup` slightly lowered TTFT, and skipping `MADV_DONTNEED` on cache hits removed about `0.25s` of direct hit-path syscall cost. Combining these may cross the rounded token-rate boundary without changing model arithmetic.
- `theoretical_upper_bound`: Small. `cache13696` removed only about `15` misses, `skip-hit-DONTNEED` saves about `0.25s` traced time, and `t24/no-warmup` are at most a few seconds on wall/TTFT. This is only worth testing because all components are low-risk and previously correctness-safe. Promote only if end-to-end `eval_tok_s > 1.6`, RAM including page cache stays `<=16GB`, France output is correct, and TTFT stays within gate.
- `risk`: Combined changes may increase cgroup refault pressure or scheduling overhead even though each single probe tied. `t24` can increase page-fault contention; `cache13696` is near the VRAM boundary; skipping hit `DONTNEED` can keep untouched hit pages resident if the assumption is wrong. The source change must remain default-off and be reverted unless promoted.
- `implementation`: Re-add default-off env `GGML_MOE_STREAM_DONTNEED_HITS=0|1` in `ggml/src/ggml-cuda/moe_stream.cu`; default `1` preserves current behavior. When set to `0`, call `moe_stream_dontneed_source_pages()` only for cache misses.
- `test_config`: patched default-off source, `GGML_MOE_STREAM_DONTNEED_HITS=0`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13696`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 24 -tb 24 --no-warmup`.
- `required_evidence`: source diff, build log/hash, exact env/command, stderr cache stats, trace summary including hit/miss counts and `dontneed_ms`, cgroup memory files, France output, `summary.json`, binary sha256/build line, and explicit promoted/rejected/rollback status.
- `rollback`: If build fails, CUDA OOMs, RAM exceeds 16GB, correctness fails, TTFT exceeds gate, or `eval_tok_s <= 1.6`, revert source and rebuild clean. If accepted, immediately write complete SOTA reproduction info, commit/push source to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before promotion.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T191638Z-20260702_combo_skip_hit_dontneed_cache13696_t24_nowarmup/france-cpu40-vram0gb`
- `result`: completed and rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=45990.480752ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14941663232`, `pgmajfault=674529`, `workingset_refault_file=12382046`, `ram_ok=true`, `correctness_ok=true`.
- `correctness`: pass. France answer is semantically correct and coherent.
- `trace_summary`: `VRAM cache: 13.4 GiB, 3222 slots`; `rows=34753`, `cache_hits=29870`, `cache_inserts=4883`, `trace_misses=4883`, `trace_span_ms=109689.049`, `src0_ms_sum=25297.52`, `kernel_ms_sum=378.559`, `dontneed_ms_sum=1030.684`, `dontneed_hit_sum=0.046`, `dontneed_miss_sum=1030.638`, `total_ms_sum=27393.121`.
- `gap_analysis`: The combination worked mechanically: hit-path `DONTNEED` was eliminated and the 13696MiB cache retained the expected lower miss count. However, end-to-end token rate still tied `1.6 tok/s`, while `pgmajfault` remained high and `trace_span` stayed around `110s`. The bottleneck is not these small per-hit/cache/thread/warmup terms; it remains cold page/refault plus CPU fallback wall time.
- `rollback_status`: source change reverted and clean rebuild completed. No commit/push because this did not exceed the current clean reproducible cold line.

### 当前执行 attempt：cpu41-cache14336-t24-nowarmup

- `attempt_id`: `20260702-cpu41-cache14336-t24-nowarmup`
- `attempt_kind`: `config-probe`
- `status`: rejected
- `hypothesis`: A prior `cpu_moe=41` run let the 14GB gate stream cache fit and reached `1.6 tok/s`, but extra CPU fallback erased the cache gain. Combining that larger cache with `t24` and `--no-warmup` may reduce enough traced stream/refault/TTFT overhead to cross the rounded token-rate boundary while still using more VRAM than the cpu40 13.25GB boundary.
- `theoretical_upper_bound`: `14GB` gives about `3370` cache slots versus `3192` at `13568MiB`, so it can remove at most a few hundred gate misses. At about `5ms` direct miss cost, the direct stream upper bound is around `1s`, partially offset by one additional CPU fallback MoE layer. Promote only if `eval_tok_s > 1.6`, France correctness passes, RAM including page cache stays `<=16GB`, and TTFT stays within gate.
- `risk`: More CPU fallback can increase up/down host page faults and trace span. The larger cache can also leave too little GPU temporary memory, causing CUDA OOM. This is a no-source-change probe; reject on OOM, cache insertion failure, correctness failure, TTFT gate failure, RAM issue, or `eval_tok_s <= 1.6`.
- `test_config`: clean source `aa8d6f916`, no source changes, `cpu_moe=41`, `GGML_MOE_STREAM_ONE_CACHE_MIB=14336`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 24 -tb 24 --no-warmup`.
- `required_evidence`: exact env/command, stderr cache slots/hits or OOM, cgroup memory files, France output, `summary.json`, full trace summary, binary sha256/build line, and explicit promoted/rejected status.
- `rollback`: No source change. If it unexpectedly becomes a compliant SOTA, immediately write complete reproduction info, commit/push any required source state to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before promotion.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T192344Z-20260702_cpu41_cache14336_t24_nowarmup/france-cpu41-vram0gb`
- `result`: completed and rejected/regression. `eval_tok_s=1.5`, `prompt_tok_s=0.6`, `TTFT=43012.05977ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14973276160`, `pgmajfault=687486`, `workingset_refault_file=12694654`, `ram_ok=true`, `correctness_ok=true`.
- `correctness`: pass. France answer is semantically correct and coherent.
- `trace_summary`: `VRAM cache: 14.0 GiB, 3373 slots`; `rows=35622`, `cache_hits=30670`, `cache_inserts=4952`, `trace_misses=4952`, `trace_span_ms=113278.671`, `src0_ms_sum=25703.067`, `kernel_ms_sum=403.436`, `dontneed_ms_sum=1272.506`, `total_ms_sum=28095.126`.
- `gap_analysis`: The larger 14GB cache fit, but `cpu_moe=41` increased streamed rows and CPU fallback enough that token rate regressed to `1.5 tok/s`. Misses remain comparable to cpu40/cache13568 despite the larger cache, and trace span worsened. Do not continue the “more CPU MoE to fit larger cache” direction.

### 当前执行 attempt：cache13568-no-host

- `attempt_id`: `20260702-cache13568-no-host`
- `attempt_kind`: `config-probe`
- `status`: planned
- `hypothesis`: The current cold bottleneck is cgroup file-cache/refault pressure. The `--no-host` CLI flag bypasses host buffers for offloaded tensors, which may reduce host-side residency or page-cache competition after GPU tensor upload while leaving CPU MoE expert mmap access intact. This can potentially lower `memory_file_bytes`, `pgmajfault`, or trace span without changing arithmetic.
- `theoretical_upper_bound`: Warm/no-drop shows a large page-state upper bound, but `--no-host` can only affect host buffer/page-cache pressure from non-CPU tensors; it cannot remove CPU up/down fallback work or gate expert misses. Promote only if `eval_tok_s > 1.6`, France correctness passes, RAM including page cache stays `<=16GB`, and TTFT stays within gate.
- `risk`: `--no-host` may be incompatible with deferred expert mmap or may force extra transfers/refaults, raising TTFT or causing load/decode failure. It may also have no effect because CPU fallback still needs mmap-backed expert pages. Reject on failure, correctness issue, RAM issue, TTFT gate failure, or `eval_tok_s <= 1.6`.
- `test_config`: clean source `aa8d6f916`, no source changes, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20 --no-host`.
- `required_evidence`: exact env/command, stdout/stderr, cgroup memory files, France output, `summary.json`, trace summary, comparison to clean `cache13568`, binary sha256/build line, and explicit promoted/rejected status.
- `rollback`: No source change. If it unexpectedly becomes a compliant SOTA, immediately write complete reproduction info, push any required source state to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before promotion.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T192856Z-20260702_cache13568_no_host/france-cpu40-vram0gb`
- `result`: failed/rejected before model load. `exit_status=1`, no token-rate metrics, `memory_peak_bytes=609501184`, `memory_file_bytes=204439552`, `ram_ok=true`, `correctness_ok=false`.
- `failure`: stderr shows `ggml_aligned_malloc: insufficient memory (attempted to allocate 130560.00 MB)`, `failed to allocate CPU_REPACK buffer`, and `failed to load model`. This is not a host cgroup limit issue during decode; it is a `--no-host` / repack incompatibility at model load.
- `gap_analysis`: Plain `--no-host` is not viable with the default repack path. A follow-up with `--no-repack` is justified as a loader-boundary check because it may avoid the huge `CPU_REPACK` allocation.

### 当前执行 attempt：cache13568-no-host-no-repack

- `attempt_id`: `20260702-cache13568-no-host-no-repack`
- `attempt_kind`: `config-probe`
- `status`: planned
- `hypothesis`: Plain `--no-host` fails because default repacking tries to allocate a huge CPU_REPACK buffer. Combining `--no-host --no-repack` may avoid that allocation and still reduce host-side residency/page-cache pressure for GPU-offloaded tensors.
- `theoretical_upper_bound`: If the model loads, any benefit must come from lower cgroup file/anon pressure and fewer refaults, not arithmetic. Disabling repack may slow compute or prompt/decode kernels, so expected token-rate gain is uncertain. Promote only if `eval_tok_s > 1.6`, France correctness passes, RAM including page cache stays `<=16GB`, and TTFT stays within gate.
- `risk`: `--no-repack` can reduce compute efficiency, and `--no-host` may still be incompatible with deferred experts or required CPU fallback tensors. Reject on load failure, correctness issue, RAM issue, TTFT gate failure, or `eval_tok_s <= 1.6`.
- `test_config`: clean source `aa8d6f916`, no source changes, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20 --no-host --no-repack`.
- `required_evidence`: exact env/command, stdout/stderr, cgroup memory files, France output if generated, `summary.json`, trace summary if present, binary sha256/build line, and explicit promoted/rejected status.
- `rollback`: No source change. If it unexpectedly becomes a compliant SOTA, immediately write complete reproduction info and follow the hard push/rerun gate.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T193124Z-20260702_cache13568_no_host_no_repack/france-cpu40-vram0gb`
- `result`: completed and rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=47863.786741ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14973636608`, `pgmajfault=601079`, `workingset_refault_file=11857635`, `ram_ok=true`, `correctness_ok=true`.
- `correctness`: pass. France answer is semantically correct and coherent.
- `trace_summary`: `VRAM cache: 13.2 GiB, 3192 slots`; `rows=34753`, `cache_hits=29855`, `cache_inserts=4898`, `trace_misses=4898`, `trace_span_ms=108360.648`, `src0_ms_sum=25004.993`, `kernel_ms_sum=390.698`, `dontneed_ms_sum=1207.695`, `total_ms_sum=27294.078`.
- `gap_analysis`: `--no-host --no-repack` avoids the huge CPU_REPACK allocation and runs correctly, but it does not reduce the cold page/refault bottleneck enough to move token rate beyond `1.6`. Cache shape and file memory remain essentially baseline-shaped. Do not pursue no-host further for SOTA.

### 当前执行 attempt：cpu-fallback-current-bottleneck-trace

- `attempt_id`: `20260702-cpu-fallback-current-bottleneck-trace`
- `attempt_kind`: `diagnostic-only`
- `status`: planned
- `hypothesis`: After exhausting gate-cache size, no-host, thread, and simple page-advice probes, the remaining cold-start wall time is dominated by non-streamed `ffn_up_exps` / `ffn_down_exps` CPU fallback plus cgroup refault/reclaim. A fresh CPU chunk trace on the current clean `cpu40/cache13568` setup should identify whether the fallback wall is concentrated in a few layers/experts, or broadly distributed across all up/down tensors.
- `expected_delta`: No direct token-rate gain is expected. The output is a priority list for the next structural attempt. If the trace shows narrow concentration, the next attempt should target that layer/expert subset. If it is broad, further profile-selected caching/streaming is unlikely and the next work should be lower-level CPU fallback page/read scheduling.
- `risk`: CPU chunk tracing can perturb timing and may hit its row limit. This run is not promotable regardless of token rate. Correctness and RAM should still be recorded to keep the diagnostic comparable.
- `test_config`: clean source `aa8d6f916`, no source changes, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, `GGML_MOE_STREAM_ONE_TRACE_OUT={case_dir}/one_trace.csv`, `GGML_MOE_CPU_CHUNK_TRACE_OUT={case_dir}/cpu_chunk_trace.csv`, `GGML_MOE_CPU_CHUNK_TRACE_LIMIT=2000000`, cold `drop_caches`, strict 16GB cgroup, France prompt, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `required_evidence`: exact env/command, `summary.json`, cgroup memory files, France answer, gate trace summary, CPU chunk trace row count, top tensors/layers/experts by chunk wall time, and explicit `diagnostic_only` status.
- `rollback`: No source change. Do not commit/push. Use the analysis to choose the next implementation attempt.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T193720Z-20260702_cpu_fallback_current_bottleneck_trace/france-cpu40-vram0gb`
- `result`: diagnostic-only. `eval_tok_s=1.6`, `prompt_tok_s=0.6`, `TTFT=47630.142084ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14964211712`, `pgmajfault=575052`, `workingset_refault_file=12049946`, `ram_ok=true`, `correctness_ok=true`.
- `gate_trace_summary`: `rows=34753`, `cache_hits=29855`, `cache_inserts=4898`, `trace_misses=4898`, `trace_span_ms=112371.567`, `src0_ms_sum=27141.642`, `total_ms_sum=29441.733`.
- `cpu_chunk_trace_summary`: `cpu_chunk_trace.csv` has `1667384` rows and `125806993` bytes. Thread-sum time is dominated by `ffn_up_exps` (`688777.888ms`) and `ffn_down_exps` (`594024.621ms`). Parallel-normalized fallback estimate is `64140.125ms`, split `up=34438.894ms` and `down=29701.231ms`.
- `top_layers_parallel_norm`: layers `1/2/0` are the heaviest (`3358.883ms`, `3302.751ms`, `3238.435ms`), followed by `19` (`2151.23ms`), `3` (`2104.288ms`), and `10` (`2033.229ms`). The top three layers cover only about `15%` of normalized fallback, so the fallback cost is not narrow enough for a tiny layer/expert patch to be sufficient.
- `profiles`: generated top-cost up/down profiles in the run directory:
  - `cpu_fallback_topcost128_profile.tsv`, sha256 `d8c5a3d7acdd7edd68f79156297b73dc58548f3fdaf409bf9fce11a9908f91c1`, coverage `6.72%`;
  - `cpu_fallback_topcost256_profile.tsv`, sha256 `16d679891f833cc049c690d8dfee466cc9d101e4f09a428aac31ebebe0191410`, coverage `11.77%`;
  - `cpu_fallback_topcost512_profile.tsv`, sha256 `6a5fb5b0f991808b8bf8a60d9b877cb88c93f8a8f9f463bfcf2d07987817e38a`, coverage `20.29%`.
- `next_action`: Avoid more small gate-cache/thread/no-host tuning. The only remaining plausible source direction is structural CPU fallback reduction or a larger, carefully bounded RAM-tier/read-scheduling experiment. Small top-N profiles are unlikely to move token rate because top512 covers only about one fifth of fallback cost.

### 当前执行 attempt：cpu-fallback-ramtier-top512-current-profile

- `attempt_id`: `20260702-cpu-fallback-ramtier-top512-current-profile`
- `attempt_kind`: `source-probe`
- `status`: rejected
- `hypothesis`: The previous top64/top128 RAM-tier probes tied `1.6 tok/s`, but the refreshed CPU fallback trace shows broad up/down fallback cost and generated a top-cost profile from the current clean bottleneck. Loading the top512 up/down experts into anonymous RAM inside the same 16GB cgroup may reduce repeated mmap/page-cache refaults more than top128 while preserving CPU fallback semantics and avoiding GPU stream/cache pollution.
- `theoretical_upper_bound`: Top512 covers only `20.29%` of the parallel-normalized fallback estimate (`~13.0s` of `~64.1s`). Its payload is about `512 * 4.25MiB ~= 2.1GiB`; this lowers file-cache budget by the same amount and pays first-use memcpy/read cost. A best case could save several seconds of fallback/refault time; a common case is tie/regression from first-use copy and reduced page cache. Promote only if `eval_tok_s > 1.6`, RAM including anon+page-cache stays `<=16GB`, France correctness passes, and TTFT stays within gate.
- `risk`: The larger anonymous tier may increase reclaim pressure, TTFT, and major faults. The profile is France-derived and may not generalize, so even an improvement would require follow-up prompt-set validation before broader claims. The source patch must be default-off and reverted unless promoted.
- `implementation`: Reuse the previous default-off RAM-tier patch from `/root/lfz/runs/vendor-ds4-16gb/20260701T184738Z-20260702_cpu-fallback-ramtier-updown128/france-cpu40-vram0gb/source.diff`. It adds `GGML_MOE_CPU_RAM_TIER_PROFILE=<tsv>` and `GGML_MOE_CPU_RAM_TIER_MAX_MIB=<MiB>` to copy active profiled experts into anonymous RAM and use those copies for CPU fallback.
- `test_config`: patched default-off source, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_CPU_RAM_TIER_PROFILE=/root/lfz/runs/vendor-ds4-16gb/20260701T193720Z-20260702_cpu_fallback_current_bottleneck_trace/france-cpu40-vram0gb/cpu_fallback_topcost512_profile.tsv`, `GGML_MOE_CPU_RAM_TIER_MAX_MIB=2304`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, gate trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `required_evidence`: source diff, profile path/sha256/count/coverage, build log/hash, exact env/command, RAM-tier loaded MiB logs, cgroup memory files, France output, `summary.json`, gate trace summary, binary sha256/build line, and explicit promoted/rejected/rollback status.
- `rollback`: Revert source and clean rebuild if build fails, correctness fails, RAM exceeds 16GB, TTFT exceeds gate, or `eval_tok_s <= 1.6`. If accepted, immediately write full SOTA reproduction evidence, commit/push to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before promotion.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T194729Z-20260702_cpu_fallback_ramtier_top512_current_profile/france-cpu40-vram0gb`
- `result`: completed and rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.6`, `TTFT=48557.903671ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=12684492800`, `pgmajfault=567206`, `workingset_refault_file=11612222`, `ram_ok=true`, `correctness_ok=true`.
- `ram_tier_summary`: profile loaded `512` entries with budget `2304MiB`; final loaded anonymous tier was `2176.0MiB`. Page cache dropped to `12.68GB`, proving the RAM tier stayed inside the same 16GB cgroup budget.
- `trace_summary`: `VRAM cache: 13.2 GiB, 3192 slots`; `rows=34753`, `cache_hits=29855`, `cache_inserts=4898`, `trace_misses=4898`, `trace_span_ms=110472.283`, `src0_ms_sum=25128.771`, `kernel_ms_sum=394.271`, `dontneed_ms_sum=1204.844`, `total_ms_sum=27430.624`.
- `correctness`: pass. France answer is semantically correct and coherent.
- `gap_analysis`: The larger RAM tier mechanically worked and slightly reduced `pgmajfault/workingset_refault_file`, but it did not move token rate beyond `1.6` and worsened TTFT versus clean controls. Top512 covers only about `20%` of fallback cost while consuming `2.1GB` of cgroup memory, so the saved refaults are offset by first-use copy and reduced file-cache budget. Do not continue medium top-N RAM-tier profiles; any future RAM tier would need much broader coverage and a different load schedule.
- `rollback_status`: source change reverted and clean rebuild completed. No commit/push because this is not an accepted SOTA.

### 当前执行 attempt：cpu-fallback-ramtier-top1024-current-profile

- `attempt_id`: `20260702-cpu-fallback-ramtier-top1024-current-profile`
- `attempt_kind`: `source-probe`
- `status`: rejected
- `hypothesis`: Top512 RAM-tier mechanically worked but only covered `20.29%` of fallback. A top1024 profile covers `34.19%` of the current CPU fallback estimate and is a useful upper-bound test for whether much broader anonymous-RAM coverage can overcome first-use copy and lower file-cache budget. If this still ties or regresses, the RAM-tier direction is unlikely to produce a cold SOTA without a fundamentally different load schedule.
- `theoretical_upper_bound`: Top1024 covers `21928.556ms` of the `64140.125ms` parallel-normalized fallback estimate, but payload is about `4352MiB` inside the same 16GB cgroup. Best case is a several-second fallback/refault reduction; downside is severe page-cache displacement, extra first-use memcpy/read cost, and TTFT regression. Promote only if `eval_tok_s > 1.6`, RAM including anon+page-cache stays `<=16GB`, France correctness passes, and TTFT stays within gate.
- `risk`: The larger anonymous tier may reduce file cache to about `10GB`, increasing gate/up/down refault churn or triggering cgroup reclaim. The existing RAM-tier patch has a hard `1024` entry limit, so this is the maximum top-N profile it can test without further code changes. The source patch must be default-off and reverted unless promoted.
- `implementation`: Reuse the previous default-off RAM-tier patch from `/root/lfz/runs/vendor-ds4-16gb/20260701T184738Z-20260702_cpu-fallback-ramtier-updown128/france-cpu40-vram0gb/source.diff`.
- `profile`: `/root/lfz/runs/vendor-ds4-16gb/20260701T193720Z-20260702_cpu_fallback_current_bottleneck_trace/france-cpu40-vram0gb/cpu_fallback_topcost1024_profile.tsv`, `1024` entries, sha256 `99d857356b8e7403df31127e0d0b629014cbd6db21dad7e5b4b94d0f834bfdf3`, estimated fallback coverage `34.19%`, payload about `4352MiB`.
- `test_config`: patched default-off source, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_CPU_RAM_TIER_PROFILE=<top1024 profile>`, `GGML_MOE_CPU_RAM_TIER_MAX_MIB=4608`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, gate trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `required_evidence`: source diff, profile path/sha256/count/coverage, build log/hash, exact env/command, RAM-tier loaded MiB logs, cgroup memory files, France output, `summary.json`, gate trace summary, binary sha256/build line, and explicit promoted/rejected/rollback status.
- `rollback`: Revert source and clean rebuild if build fails, correctness fails, RAM exceeds 16GB, TTFT exceeds gate, or `eval_tok_s <= 1.6`. If accepted, immediately write full SOTA reproduction evidence, commit/push to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before promotion.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T195453Z-20260702_cpu_fallback_ramtier_top1024_current_profile/france-cpu40-vram0gb`
- `result`: completed and rejected/regression. `eval_tok_s=1.5`, `prompt_tok_s=0.6`, `TTFT=50564.398072ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=10403127296`, `pgmajfault=531322`, `workingset_refault_file=11334691`, `ram_ok=true`, `correctness_ok=true`.
- `ram_tier_summary`: profile loaded `1024` entries, final anonymous tier was `4352.0MiB`; page cache dropped to `10.40GB`, all inside the same 16GB cgroup budget.
- `trace_summary`: `VRAM cache: 13.2 GiB, 3192 slots`; `rows=34753`, `cache_hits=29855`, `cache_inserts=4898`, `trace_misses=4898`, `trace_span_ms=113952.31`, `src0_ms_sum=25816.382`, `kernel_ms_sum=398.545`, `dontneed_ms_sum=1218.506`, `total_ms_sum=28130.935`.
- `correctness`: pass. France answer is semantically correct and coherent.
- `gap_analysis`: Top1024 confirmed the upper-bound RAM-tier tradeoff: broader coverage lowers `pgmajfault` and file-cache bytes, but the 4.25GB anonymous tier displaces too much page cache and adds enough first-use copy/load work that token rate regresses to `1.5`. This rules out current top-N RAM-tier as a SOTA path under the 16GB cold gate.
- `rollback_status`: source change reverted and clean rebuild completed. No commit/push because this is not an accepted SOTA.

### 当前执行 attempt：cpu-fallback-lazy-ramtier-top512

- `attempt_id`: `20260702-cpu-fallback-lazy-ramtier-top512`
- `attempt_kind`: `source-probe`
- `status`: rejected
- `hypothesis`: The previous RAM-tier implementation copies profiled experts before CPU fallback chunks, so first use pays page fault + memcpy before useful compute and worsens TTFT. A lazy-after-use variant can keep CPU fallback semantics while caching a profiled expert only after it has naturally been read by the CPU fallback path, using the anonymous copy only for later reuses. This may preserve some refault reduction without front-loading the copy cost.
- `theoretical_upper_bound`: Top512 covers `20.29%` of parallel-normalized fallback. Lazy loading cannot help first touches, so its upside is lower than eager top512, but it may reduce TTFT/trace-span penalty. Best case is a small improvement over clean `1.6`; likely result is tie if reuse within one generation is insufficient. Promote only if `eval_tok_s > 1.6`, RAM including anon+page-cache stays `<=16GB`, France correctness passes, and TTFT stays within gate.
- `risk`: Copying after use may contend with other fallback threads and still displace page cache. Because the copy is performed without a per-expert barrier, it must only read immutable mmap data and publish the copy for later uses; it must not change current expert computation. The source patch must be default-off and reverted unless promoted.
- `implementation`: Add default-off env `GGML_MOE_CPU_LAZY_RAM_TIER_PROFILE=<tsv>` and `GGML_MOE_CPU_LAZY_RAM_TIER_MAX_MIB=<MiB>` in `ggml/src/ggml-cpu/ggml-cpu.c`. Before compute, use an already-loaded anonymous copy if present; after thread 0 finishes its chunks for an active profiled expert, copy the mmap expert into anonymous RAM if not already loaded and budget allows. Default behavior remains unchanged.
- `test_config`: patched default-off source, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_CPU_LAZY_RAM_TIER_PROFILE=/root/lfz/runs/vendor-ds4-16gb/20260701T193720Z-20260702_cpu_fallback_current_bottleneck_trace/france-cpu40-vram0gb/cpu_fallback_topcost512_profile.tsv`, `GGML_MOE_CPU_LAZY_RAM_TIER_MAX_MIB=2304`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, gate trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `required_evidence`: source diff, build log/hash, profile path/sha256/count, exact env/command, lazy RAM-tier loaded MiB logs, cgroup memory files, France output, `summary.json`, gate trace summary, binary sha256/build line, and explicit promoted/rejected/rollback status.
- `rollback`: Revert source and clean rebuild if build fails, correctness fails, RAM exceeds 16GB, TTFT exceeds gate, or `eval_tok_s <= 1.6`. If accepted, immediately write full SOTA reproduction evidence, commit/push to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before promotion.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T200336Z-20260702_cpu_fallback_lazy_ramtier_top512/france-cpu40-vram0gb`
- `result`: completed and rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.6`, `TTFT=47417.390374ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=12675297280`, `pgmajfault=592960`, `workingset_refault_file=11491094`, `ram_ok=true`, `correctness_ok=true`.
- `lazy_ram_tier_summary`: loaded `512` entries, final anonymous tier `2176.0MiB`; page cache dropped to `12.68GB` inside the same 16GB cgroup. TTFT improved versus eager top512 (`47417ms` vs `48558ms`) but still did not beat the clean token-rate line.
- `trace_summary`: `VRAM cache: 13.2 GiB, 3192 slots`; `rows=34753`, `cache_hits=29855`, `cache_inserts=4898`, `trace_misses=4898`, `trace_span_ms=107966.851`, `src0_ms_sum=24907.113`, `kernel_ms_sum=386.255`, `dontneed_ms_sum=1192.567`, `total_ms_sum=27181.674`.
- `correctness`: pass. France answer is semantically correct and coherent.
- `gap_analysis`: Lazy-after-use loading reduces the eager RAM-tier TTFT/trace penalty and produces one of the better trace spans, but end-to-end token rate remains rounded at `1.6`. The profile coverage and 2.1GB page-cache displacement still do not produce enough net win for a SOTA. Do not promote; any future CPU fallback work must reduce compute/page touches more directly rather than caching a top-N subset after use.
- `rollback_status`: source change reverted and clean rebuild completed. No commit/push because this is not an accepted SOTA.

### 当前执行 attempt：page-fault-readamp-clean-repro

- `attempt_id`: `20260702-page-fault-readamp-clean-repro`
- `attempt_kind`: `diagnostic-only`
- `status`: completed_diagnostic
- `hypothesis`: The historical correct `2.6 tok/s` run and current clean `1.5~1.6 tok/s` runs have the same gate-stream hit/insert counts, but old `pgmajfault≈309k` and file inputs were roughly half of current `pgmajfault≈575k~675k`. The unreproduced delta is therefore likely in cold page-fault/read-amplification/reclaim state rather than model arithmetic. A fresh clean cold control with full fault/refault/time-v evidence should confirm whether the current high-fault shape is stable before any source-level read scheduling attempt.
- `theoretical_upper_bound`: This run is diagnostic and should not improve token rate. If it reproduces the old low-fault shape without source changes, then the upper bound is the historical `2.6 tok/s` state and the next step is to identify the external state that caused it. If it stays at high faults, source work must target reducing major faults or hiding page-in latency; the hard upper bound remains approximately the old trace span (`~71s`) versus current `~108s+`.
- `risk`: No source risk. Because this is a cold `drop_caches` run under the 16GB cgroup, it may again tie `1.6 tok/s`; that is expected. It must still pass France correctness and RAM gates so the diagnostic is comparable.
- `test_config`: clean source `aa8d6f916`, no source changes, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, gate trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `required_evidence`: source commit/status, binary sha256/build line, exact command/env, stdout/stderr, `/usr/bin/time -v` fields including major faults and file inputs, cgroup `memory.stat` fault/refault fields, France answer, `summary.json`, gate trace summary, and direct comparison against historical `20260701T095526Z` and recent clean `cache13568` runs.
- `rollback`: No source change. Do not commit/push. If token rate unexpectedly exceeds the accepted line and all gates pass, immediately stop and apply the hard SOTA gate: complete reproduction record, commit/push source to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before promotion.
- `next_decision`: If high major faults remain stable, design a source-level attempt that reduces cold read amplification or defers/aggregates expert page touches. Candidate directions must include a concrete theory and upper-bound estimate before implementation.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T201316Z-20260702_page_fault_readamp_clean_repro/france-cpu40-vram0gb`
- `result`: completed diagnostic, not promoted. `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=47453.251439ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14978854912`, `pgmajfault=605634`, `workingset_refault_file=11871712`, `ram_ok=true`, `correctness_ok=true`.
- `correctness`: pass. France answer is semantically correct and coherent; it matches the historical correct answer shape.
- `time_v`: current run `/usr/bin/time -v` reports wall `2:08.51`, major faults `606032`, minor faults `2626823`, file system inputs `256344128`, max RSS `15595768 KB`.
- `historical_2p6_comparison`: old correct 2.6 run reports wall `1:29.13`, major faults `309783`, minor faults `2016378`, file system inputs `154862552`, max RSS `25231288 KB`, cgroup `pgmajfault=309216`, `workingset_refault_file=3371495`.
- `trace_summary`: current `rows=34753`, `cache_hits=29855`, `cache_inserts=4898`, `trace_span_ms=108889.963`, `src0_ms_sum=25288.845`, `kernel_ms_sum=390.614`, `dontneed_ms_sum=1224.069`, `total_ms_sum=27598.21`.
- `historical_trace_summary`: old 2.6 `rows=34753`, `cache_hits=29624`, `cache_inserts=5129`, `trace_span_ms=71223.061`, `src0_ms_sum=23564.133`, `kernel_ms_sum=370.203`, `dontneed_ms_sum=1162.838`, `total_ms_sum=25786.372`.
- `gap_analysis`: The direct traced gate stream `src0_ms` differs by only about `1.7s`, while trace span differs by about `37.7s` and file inputs/major faults are much higher in current clean runs. The unreproduced 2.6 behavior is therefore not explained by gate cache policy or per-expert copy timing. The next useful work should target broad `ffn_up_exps`/`ffn_down_exps` CPU fallback page touches or compute placement rather than more gate cache/thread/no-host tuning.
- `rollback_status`: no source change; no commit/push because this is diagnostic and not a new SOTA.

### 当前执行 attempt：updown-stream-correctness-design

- `attempt_id`: `20260702-updown-stream-correctness-design`
- `attempt_kind`: `design/inspection`
- `status`: planned
- `hypothesis`: Current valid vendor optimization streams only `ffn_gate_exps` through the DS4 gate path, while `ffn_up_exps` and `ffn_down_exps` remain broad CPU fallback bottlenecks. Previous broad attempts to allow MXFP4/F8 expert streaming produced garbled output, so the next source direction must first identify why gate streaming is correct but up/down streaming is not. If the missing piece is tensor layout/row-id handling or quant-type-specific dequant math, a narrow correctness fix may unlock a much larger token-rate gain than page-cache tuning.
- `theoretical_upper_bound`: CPU chunk trace estimates about `64.1s` parallel-normalized fallback time split across up/down. Eliminating even half of this fallback or moving the hottest up/down chunks to GPU could in principle close much of the gap between current `1.6 tok/s` and the old/warm upper bound. The practical bound is constrained by VRAM headroom, stream cache size, and H2D page-in cost; any implementation must calculate expert payload and expected transfer/compute limits before running.
- `risk`: Incorrect up/down streaming can silently corrupt activations while still producing fluent-looking output; France correctness alone may be insufficient for a large arithmetic change. Any implementation must start with a small default-off path and compare answer correctness, trace, and possibly deterministic logits/short-output behavior before promotion.
- `inspection_scope`: `ggml/src/ggml-cuda/moe_stream.cu`, the CPU `ggml_compute_forward_mul_mat_id` path in `ggml/src/ggml-cpu/ggml-cpu.c`, tensor type/layout handling for `ffn_gate_exps`, `ffn_up_exps`, and `ffn_down_exps`, previous DS4 row-mod correctness fix, and any disabled env/path that previously allowed MXFP4/F8 streaming.
- `required_output`: a short design note in this plan before any implementation: exact bottleneck being attacked, why the method should preserve correctness, expected token-rate upper bound from expert size/transfer/compute math, and rollback criteria.
- `rollback`: inspection only; no source change. If an implementation attempt follows, it must be a separate attempt entry and default-off unless promoted.
- `inspection_result_initial`: GGUF header shows `blk.N.ffn_gate_exps.weight` and `blk.N.ffn_up_exps.weight` have identical shape/type (`[4096,2048,256]`, `MXFP4`, `1140850688` bytes per layer tensor), while `ffn_down_exps.weight` is `[2048,4096,256]`, also `MXFP4` and same byte size. The existing stream path already handles DS4 `MXFP4` with row mapping `rows[k].i1 % src1_ne1`; therefore `ffn_up_exps` is the safest next correctness probe because it matches the accepted gate-stream geometry.

### 当前执行 attempt：up-only-stream-compare

- `attempt_id`: `20260702-up-only-stream-compare`
- `attempt_kind`: `config-probe/correctness`
- `status`: completed_diagnostic
- `hypothesis`: Since `ffn_up_exps` has the same MXFP4 shape as `ffn_gate_exps`, the corrected DS4 rows stream path may already compute it correctly. Testing only `ffn_up_exps` with `GGML_MOE_STREAM_COMPARE_CPU_OUT` enabled can validate numeric agreement against CPU fallback before any source change to stream gate+up together.
- `theoretical_upper_bound`: This specific run is not expected to be a SOTA because `ffn_gate_exps` will fall back to CPU while only `ffn_up_exps` streams. If up streaming is correct, a later gate+up implementation could attack a large part of the `ffn_up_exps≈34.4s` parallel-normalized CPU fallback estimate. Upper bound is limited by streaming H2D/page-in cost for up experts and VRAM cache competition with gate experts.
- `risk`: If up stream uses the same arithmetic but hidden layout assumptions differ from gate, the generated text may degrade or compare trace may show large CPU/GPU error. Because this is a config-only probe, reject on correctness failure, compare mismatch, RAM issue, TTFT gate issue, or token-rate regression; do not commit/push.
- `test_config`: clean source, no source changes, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_up_exps`, `GGML_MOE_STREAM_COMPARE_CPU_OUT={case_dir}/compare_cpu.csv`, `GGML_MOE_STREAM_COMPARE_CPU_LIMIT=16`, cold `drop_caches`, strict 16GB cgroup, France prompt, trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `required_evidence`: exact env/command, compare trace max/mean abs stats, gate stream trace naming showing only up tensors streamed, cgroup memory files, France output, `summary.json`, stdout/stderr, binary/source status.
- `rollback`: No source change. No commit/push. If this unexpectedly becomes a higher gated token-rate with correct output and acceptable compare stats, pause and apply the hard SOTA reproduction/push gate before calling it SOTA.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T202154Z-20260702_up_only_stream_compare/france-cpu40-vram0gb`
- `result`: completed diagnostic, not promoted. `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=47151.246844ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14923268096`, `pgmajfault=770071`, `workingset_refault_file=19699902`, `ram_ok=true`, `correctness_ok=true`.
- `compare_summary`: `compare_cpu.csv` has 16 rows, all `ffn_up_exps`; `max_abs_max=9.53674316e-07`, `mean_abs_max=2.63753464e-08`, `max_rel_max=8.22526591e-07`. This is acceptable numeric agreement with CPU fallback for the sampled up experts.
- `trace_summary`: only up tensors streamed; `rows=47458`, `n_tensors=40`, `cache_hits=41418`, `cache_inserts=6040`, `trace_span_ms=144981.391`, `src0_ms_sum=31250.244`, `kernel_ms_sum=564.656`, `total_ms_sum=34326.729`.
- `gap_analysis`: Up stream is numerically correct but not a SOTA by itself because gate/down fallback still dominate and page/refault pressure worsens (`pgmajfault=770k`, file inputs `331824352`). This unlocks a follow-up gate+up or all-MoE stream probe, but down must be checked separately before removing the name filter.
- `rollback_status`: no source change; no commit/push.

### 当前执行 attempt：down-only-stream-compare

- `attempt_id`: `20260702-down-only-stream-compare`
- `attempt_kind`: `config-probe/correctness`
- `status`: completed_diagnostic
- `hypothesis`: If `ffn_down_exps` also agrees numerically with CPU fallback under the corrected DS4 rows stream path, then the previously rejected broad MXFP4/F8 stream result may have been fixed by the later row-mapping change. That would justify a no-filter all-MoE stream test or a default-off source path to stream gate+up+down together.
- `theoretical_upper_bound`: This specific down-only run is not expected to be a SOTA because gate/up fallback remain CPU-bound. If down is correct, a later all-MoE stream could attack the remaining `ffn_down_exps≈29.7s` parallel-normalized CPU fallback estimate in addition to up.
- `risk`: Down has different logical dimensions (`[2048,4096,256]` in GGUF header versus gate/up `[4096,2048,256]`), so shape/stride assumptions may fail even though tensor byte size is the same. Reject on compare mismatch, degraded France output, RAM issue, TTFT gate issue, or runtime failure.
- `test_config`: clean source, no source changes, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_down_exps`, `GGML_MOE_STREAM_COMPARE_CPU_OUT={case_dir}/compare_cpu.csv`, `GGML_MOE_STREAM_COMPARE_CPU_LIMIT=16`, cold `drop_caches`, strict 16GB cgroup, France prompt, trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `required_evidence`: exact env/command, compare trace max/mean abs stats, trace naming showing only down tensors streamed, cgroup memory files, France output, `summary.json`, stdout/stderr, binary/source status.
- `rollback`: No source change. No commit/push. If this unexpectedly becomes a higher gated token-rate with correct output and acceptable compare stats, pause and apply the hard SOTA reproduction/push gate before calling it SOTA.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T202655Z-20260702_down_only_stream_compare/france-cpu40-vram0gb`
- `result`: completed diagnostic, not promoted. `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=44542.535707ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14961033216`, `pgmajfault=547684`, `workingset_refault_file=11372958`, `ram_ok=true`, `correctness_ok=true`.
- `compare_summary`: `compare_cpu.csv` has 16 rows, all `ffn_down_exps`; `max_abs_max=3.81469727e-06`, `mean_abs_max=5.7837542e-07`, `max_rel_max=1.25252972e-06`. This is acceptable numeric agreement for sampled down experts.
- `trace_summary`: only down tensors streamed; first call `ne01=4096`, `ne00=2048`, `nb01=1088`; `rows=34258`, `n_tensors=40`, `cache_hits=29196`, `cache_inserts=5062`, `trace_span_ms=107487.845`, `src0_ms_sum=25732.602`, `kernel_ms_sum=427.647`, `total_ms_sum=28150.005`.
- `gap_analysis`: Down stream is numerically correct. Together with the up-only compare result, this suggests the corrected DS4 rows stream path can handle all three expert tensors. The next highest-value probe is no-filter all-MoE stream with compare disabled to measure real performance/correctness.
- `rollback_status`: no source change; no commit/push.

### 当前执行 attempt：all-moe-stream-no-filter

- `attempt_id`: `20260702-all-moe-stream-no-filter`
- `attempt_kind`: `config-probe/performance`
- `status`: rejected
- `hypothesis`: After row-mapping fixes, `ffn_gate_exps`, `ffn_up_exps`, and `ffn_down_exps` each pass sampled CPU/GPU compare when streamed individually. Removing `GGML_MOE_STREAM_ONE_NAME_FILTER` should stream all DS4 MXFP4 expert matmuls and eliminate most broad CPU fallback work, potentially increasing token rate beyond the clean `1.6 tok/s` line.
- `theoretical_upper_bound`: CPU chunk trace estimated about `64.1s` parallel-normalized fallback, split `up≈34.4s` and `down≈29.7s`; gate stream already handles gate. All-MoE streaming can at best remove most up/down CPU fallback, but it adds H2D/page-in for many more experts and shares the same `13.2GiB` VRAM cache across gate/up/down. The practical upper bound is therefore below a warm all-GPU state, but if H2D/cache reuse is effective it could approach or exceed the historical `2.6 tok/s` cold observation.
- `risk`: VRAM cache competition may reduce hit rate enough to regress; streaming all three tensors may increase cold file inputs/page faults; correctness may still fail on experts not covered by the first 16 compare samples. Reject on CUDA OOM, RAM issue, degraded France output, TTFT > gate, or `eval_tok_s <= 1.6`. If it exceeds the current valid SOTA line, stop and complete the hard reproduction/push gate before promotion.
- `test_config`: clean source, no source changes, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, no `GGML_MOE_STREAM_ONE_NAME_FILTER`, cold `drop_caches`, strict 16GB cgroup, France prompt, trace enabled, compare disabled for true performance, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `required_evidence`: exact env/command proving no name filter, stdout/stderr, cgroup memory files, France output, `summary.json`, all-MoE trace tensor mix/hit/miss/span, binary/source status, and explicit promoted/unpromoted decision.
- `rollback`: No source change. If accepted as a new SOTA, immediately write full reproduction record, ensure source commit is pushed to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed source before marking promoted.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T203112Z-20260702_all_moe_stream_no_filter/france-cpu40-vram0gb`
- `result`: completed and rejected/regression. `eval_tok_s=1.0`, `prompt_tok_s=0.5`, `TTFT=52832.556299ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14970703872`, `pgmajfault=30851`, `workingset_refault_file=20696007`, `ram_ok=true`, `correctness_ok=true`.
- `correctness`: pass. France answer is semantically correct and coherent.
- `trace_summary`: streamed gate/up/down equally: `gate=36899`, `up=36899`, `down=36899`, total `rows=110697`; `cache_hits=76412`, `cache_inserts=34285`, hit rate `69.0%`, `trace_span_ms=177220.334`, `src0_ms_sum=158749.492`, `kernel_ms_sum=1374.723`, `dontneed_ms_sum=6999.534`, `total_ms_sum=169332.924`.
- `gap_analysis`: All-MoE stream removes CPU fallback but creates too much VRAM cache competition and H2D/page-in work. Each tensor class contributes about `52.9s src0` and `56.4s total`, so the combined stream path is slower than CPU fallback plus gate-only stream. Major faults are low (`~30k`) because the path touches pages through CUDA copy/read differently, but file inputs remain high (`326196000`) and token rate regresses to `1.0`.
- `rollback_status`: no source change; no commit/push.

### 当前执行 attempt：gate-up-multifilter-stream

- `attempt_id`: `20260702-gate-up-multifilter-stream`
- `attempt_kind`: `source-probe/performance`
- `status`: rejected
- `hypothesis`: `ffn_up_exps` is numerically correct when streamed, but all-MoE stream regresses because adding down triples stream pressure and destroys cache locality. Streaming only gate+up should preserve the accepted gate stream path while removing the larger `ffn_up_exps` CPU fallback component, leaving down on CPU to avoid the worst cache competition.
- `theoretical_upper_bound`: Up-only stream had `src0_ms_sum≈31.25s` and `total_ms_sum≈34.33s`; CPU trace estimates up fallback at `≈34.44s` parallel-normalized. If gate+up cache competition is modest, the net change may be small but could cross the rounded `1.6 tok/s` line. If shared cache misses approach the all-stream pattern, it will regress. Expected ceiling is below all-GPU/warm performance; promote only on measured token-rate improvement with RAM/correctness/TTFT gates.
- `implementation`: Add a default-compatible extension to `GGML_MOE_STREAM_ONE_NAME_FILTER`: if the value contains commas, allow a tensor when any comma-separated substring matches. Existing single-substring behavior remains unchanged. Test with `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps,ffn_up_exps`.
- `risk`: A source change to filter parsing must not change default or existing single-filter behavior. Gate+up may still increase misses/refaults too much. Reject and revert source on build failure, correctness failure, RAM issue, TTFT gate issue, or `eval_tok_s <= 1.6`.
- `test_config`: patched default-compatible source, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps,ffn_up_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, trace enabled, compare disabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `required_evidence`: source diff, build log/hash, exact env/command, trace tensor mix showing only gate/up streamed, cgroup memory files, France output, `summary.json`, stdout/stderr, and explicit promoted/rejected/rollback status.
- `rollback`: If not a gated improvement, revert source and clean rebuild. If accepted, immediately complete full reproduction record, commit/push source to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before promotion.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T203735Z-20260702_gate_up_multifilter_stream/france-cpu40-vram0gb`
- `result`: completed and rejected/regression. `eval_tok_s=1.3`, `prompt_tok_s=0.6`, `TTFT=49185.272747ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14964355072`, `pgmajfault=286601`, `workingset_refault_file=12049338`, `ram_ok=true`, `correctness_ok=true`.
- `trace_summary`: only gate/up streamed; `rows=63700`, `cache_hits=48394`, `cache_inserts=15306`, hit rate `76.0%`, `trace_span_ms=124768.843`, `src0_ms_sum=73807.596`, `kernel_ms_sum=757.638`, `total_ms_sum=79082.874`. Split: gate `31850 rows / 7653 inserts / 35552.862 src0_ms`; up `31850 rows / 7653 inserts / 38254.734 src0_ms`.
- `gap_analysis`: Streaming all up experts is still too broad. It lowers major faults versus gate-only but causes enough extra stream misses and `src0` time that token rate regresses. The next stream direction, if pursued, must be selective hot up layers/experts rather than all up.
- `rollback_status`: source patch reverted and clean rebuild completed; build returned to `ggml commit: aa8d6f916`. No commit/push because this is rejected.

### 设计结论：暂不执行 hot-up-layer stream

- `analysis_id`: `20260702-hot-up-layer-stream-bound`
- `attempt_kind`: `design/no-run`
- `finding`: CPU chunk trace parallel-normalized up fallback is `34438.894ms`. Top up layers by fallback cost are layer `1=1867.479ms`, `0=1830.977ms`, `2=1816.512ms`, `19=1256.338ms`, `3=1112.161ms`, `5=1088.347ms`. Coverage is shallow: top3 up layers cover only `5514.968ms` (`16.01%` of up fallback), top6 cover `8971.814ms` (`26.05%`).
- `stream_cost_estimate`: In the rejected gate+up trace, the same top3 up layers would add about `2454` streamed rows, `1370` inserts, `6745.3ms src0`, and `7093.7ms total` stream time. This already exceeds the estimated `5515ms` CPU fallback saving before considering cache interference with gate.
- `decision`: Do not implement a hot-layer-only up stream attempt now. Its theoretical upper bound is negative or too small to justify another source patch under the 16GB cold gate.
- `next_priority`: Future work should target reducing stream miss cost/cache competition itself (for example cache partitioning/admission policy or read scheduling), or a fundamentally different CPU fallback optimization. More name-filter combinations are low priority unless backed by a better cost model.

### 当前执行 attempt：down-only-stream-no-compare

- `attempt_id`: `20260702-down-only-stream-no-compare`
- `attempt_kind`: `config-probe/sanity`
- `status`: rejected
- `hypothesis`: The down-only stream compare run tied `1.6 tok/s` and had better TTFT than recent gate-only controls, but it enabled CPU/GPU compare for 16 streamed experts. A no-compare rerun verifies whether compare overhead masked a small token-rate gain.
- `theoretical_upper_bound`: The compare trace covers only 16 experts, so expected gain from disabling it is small. If the previous tie was near a rounding boundary, this could move above `1.6`; otherwise it should remain a tie. This is a cheap no-source sanity check before closing the single-tensor stream direction.
- `risk`: No source risk. Because this streams down instead of gate, correctness must still be manually checked even though down sampled compare already passed.
- `test_config`: clean source, no source changes, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_down_exps`, no `GGML_MOE_STREAM_COMPARE_CPU_OUT`, cold `drop_caches`, strict 16GB cgroup, France prompt, trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `required_evidence`: exact env/command proving compare disabled, cgroup memory files, France output, `summary.json`, down-only trace summary, stdout/stderr, binary/source status.
- `rollback`: No source change. If it unexpectedly exceeds the current accepted line and all gates pass, stop and apply the hard SOTA reproduction/push gate before promotion.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T204807Z-20260702_down_only_stream_no_compare/france-cpu40-vram0gb`
- `result`: completed and rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=43920.660705ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14970474496`, `pgmajfault=535980`, `workingset_refault_file=11154862`, `ram_ok=true`, `correctness_ok=true`.
- `trace_summary`: only down tensors streamed; `rows=34258`, `n_tensors=40`, `cache_hits=29196`, `cache_inserts=5062`, `trace_span_ms=105209.863`, `src0_ms_sum=25997.467`, `kernel_ms_sum=422.328`, `dontneed_ms_sum=1231.767`, `total_ms_sum=28393.829`.
- `gap_analysis`: Disabling compare improved TTFT slightly versus down-only compare, but token rate still tied `1.6`. Single down stream does not beat the clean line; compare overhead was not hiding a SOTA.
- `rollback_status`: no source change; no commit/push.

### 设计结论：关闭普通 name-filter stream 组合

- `analysis_id`: `20260702-filter-combo-closure`
- `attempt_kind`: `design/no-run`
- `finding`: Individual up/down stream is numerically correct, but up-only/down-only do not exceed `1.6`; all-MoE regresses to `1.0`; gate+up regresses to `1.3`; hot-up-layer bound is negative. The common issue is not arithmetic correctness but stream miss/cache/page-in cost.
- `decision`: Do not spend more runs on plain name-filter combinations such as gate+down or top-N layer stream unless a new cache/read scheduling mechanism changes the cost model first.
- `next_priority`: Design a default-off cache/read scheduling probe that directly reduces `src0` miss cost or protects the existing gate cache. Candidate classes: cache admission/partition, async prefetch with bounded in-flight pages, or source-page lifetime policy tied to observed reuse rather than immediate `DONTNEED`.

### 当前执行 attempt：vram-cache-hash-lookup-gate

- `attempt_id`: `20260702-vram-cache-hash-lookup-gate`
- `attempt_kind`: `source-probe/cache`
- `status`: rejected
- `hypothesis`: The VRAM cache declares a hash table but current lookup still linearly scans all cache slots on every stream call. In the gate-only baseline this means about `34k` stream rows and roughly `30k` cache hits, each scanning up to `~3192` slots. A default-off O(1) hash lookup can remove fixed CPU-side cache lookup overhead without changing arithmetic, cache capacity, or page policy.
- `theoretical_upper_bound`: This cannot reduce miss H2D/page-in cost or CPU up/down fallback. Its upper bound is the CPU time spent in hit-path linear lookup, roughly `cache_hits * average_slots_scanned`. Even if that is a few seconds, expected gain is modest; promote only if measured `eval_tok_s > 1.6` with RAM/correctness/TTFT gates.
- `implementation`: Add env `GGML_MOE_STREAM_HASH_LOOKUP=1`. When enabled, `vram_cache_lookup()` uses `g_vcache.ht`; `vram_cache_insert()` rebuilds the hash table after inserting/evicting to keep correctness simple. Default disabled path remains the existing linear scan.
- `risk`: Rebuilding the hash table on every miss adds overhead; if miss count dominates, this may tie/regress. Hash bugs could return a wrong slot and corrupt output, so France correctness and trace hit/insert counts are mandatory.
- `test_config`: patched default-off source, `GGML_MOE_STREAM_HASH_LOOKUP=1`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `required_evidence`: source diff, build log/hash, exact env/command, cgroup memory files, France output, `summary.json`, gate trace summary, stdout/stderr, and explicit promoted/rejected/rollback status.
- `rollback`: If build fails, correctness fails, RAM exceeds 16GB, TTFT exceeds gate, or `eval_tok_s <= 1.6`, revert source and clean rebuild. If accepted, immediately complete full reproduction record, commit/push source to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before promotion.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T205549Z-20260702_vram_cache_hash_lookup_gate/france-cpu40-vram0gb`
- `result`: completed and rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=46540.808484ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14980935680`, `pgmajfault=613608`, `workingset_refault_file=12009511`, `ram_ok=true`, `correctness_ok=true`.
- `trace_summary`: gate-only stream shape unchanged; `rows=34753`, `cache_hits=29855`, `cache_inserts=4898`, `trace_span_ms=109928.624`, `src0_ms_sum=25296.821`, `kernel_ms_sum=389.464`, `dontneed_ms_sum=1210.708`, `total_ms_sum=27592.874`.
- `gap_analysis`: Hash lookup preserves correctness and cache shape, but token rate remains `1.6`. CPU-side slot scanning is not a material bottleneck relative to cold page/refault and CPU fallback wall time.
- `rollback_status`: source patch reverted and clean rebuild completed; build returned to `ggml commit: aa8d6f916`. No commit/push because this is not an accepted SOTA.

### 当前执行 attempt：cpu-fallback-madv-random

- `attempt_id`: `20260702-cpu-fallback-madv-random`
- `attempt_kind`: `source-probe/read-amplification`
- `status`: planned
- `hypothesis`: Current clean cold runs have file inputs around `253M~257M` blocks versus the historical correct `2.6 tok/s` run at `154.9M` blocks, while gate stream direct `src0_ms` differs only modestly. The CPU fallback up/down mmap access pattern may trigger excessive kernel readahead around expert pages under 16GB cgroup pressure. Applying `MADV_RANDOM` to CPU fallback expert pages may reduce read amplification and page-cache churn without changing arithmetic.
- `theoretical_upper_bound`: If file inputs/refaults approach the old 2.6 shape, wall time could move toward the historical `1:29` run. In practice `MADV_RANDOM` can also hurt because each expert is read sequentially by CPU threads; expected outcome is either lower file inputs with similar compute, or regression from disabled readahead. Promote only if `eval_tok_s > 1.6`, RAM/correctness pass, and TTFT stays within gate.
- `implementation`: Add default-off env `GGML_MOE_CPU_MADV_RANDOM=1` in `ggml/src/ggml-cpu/ggml-cpu.c`. Before CPU fallback chunk execution for each active `cur_a`, thread 0 calls `madvise(..., MADV_RANDOM)` on that expert's source pages, then synchronizes threads. Streamed gate experts are excluded because `matrix_row_counts[cur_a]` is zeroed after successful GPU stream.
- `risk`: Extra `madvise` syscalls may add overhead; disabling readahead may increase latency for full-expert sequential reads. Reject and revert on build failure, correctness failure, RAM issue, TTFT gate issue, or `eval_tok_s <= 1.6`.
- `test_config`: patched default-off source, `GGML_MOE_CPU_MADV_RANDOM=1`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, gate trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `required_evidence`: source diff, build log/hash, exact env/command, `/usr/bin/time -v` file inputs/faults, cgroup memory files, France output, `summary.json`, gate trace summary, stdout/stderr, and explicit promoted/rejected/rollback status.
- `rollback`: If not a gated improvement, revert source and clean rebuild. If accepted, immediately complete full reproduction record, commit/push source to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before promotion.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T210425Z-20260702_cpu_fallback_madv_random/france-cpu40-vram0gb`
- `result`: rejected/timeout. The run was killed after becoming much slower than baseline; `eval_tok_s=null`, `prompt_tok_s=null`, `TTFT=131415.288893ms`, and no full accepted metrics were produced. This violates the TTFT gate and cannot be promoted.
- `trace_partial`: partial trace only: `rows=17191`, `cache_hits=13490`, `cache_inserts=3701`, `trace_span_ms=288329.438`, `src0_ms_sum=19144.928`, `kernel_ms_sum=232.46`, `dontneed_ms_sum=917.552`, `total_ms_sum=20673.021`.
- `gap_analysis`: CPU fallback experts are read sequentially enough that `MADV_RANDOM` disables useful kernel readahead and moves the run into a severe stall. This is consistent with the earlier stream-src0 `MADV_RANDOM` timeout. Do not pursue random/no-readahead advice for expert source pages.
- `rollback_status`: source patch reverted and clean rebuild required before any next baseline/probe. No commit/push because this is not an accepted SOTA and the run did not produce complete gated metrics.

### 当前执行 attempt：gate-cache-admit-min2-profile

- `attempt_id`: `20260702-gate-cache-admit-min2-profile`
- `attempt_kind`: `source-probe/cache-admission`
- `status`: planned
- `hypothesis`: The gate-only trace has `34753` stream rows and `4534` unique `(tensor, expert)` keys. Current LRU with `3192` slots performs `4898` inserted loads. Offline simulation on the clean gate trace shows that caching only keys used at least twice would keep `3116` eligible keys in cache, bypass one-time keys through staging, and reduce total loads to `4534`. This may reduce VRAM cache pollution and source page-in enough to cross the current rounded `1.6 tok/s` line.
- `theoretical_upper_bound`: The hard load-count reduction is `4898 - 4534 = 364` expert copies. With current gate miss cost roughly `25.3s / 4898 = 5.16ms`, direct traced `src0` gain is only about `1.9s`; this is a boundary-crossing probe, not a large SOTA mechanism. It cannot address CPU fallback up/down work and should be rejected unless end-to-end token rate improves with RAM/correctness/TTFT gates passing.
- `implementation`: Add default-off env `GGML_MOE_STREAM_CACHE_ADMIT_PROFILE=<csv>` and `GGML_MOE_STREAM_CACHE_ADMIT_MIN_USES=<N>` in `ggml/src/ggml-cuda/moe_stream.cu`. When enabled, `vram_cache_insert()` is only used if `src0_name,expert_index` appears in the profile with count `>=N`; non-admitted misses use the existing per-slot staging buffer and do not occupy VRAM cache. Default behavior remains unchanged.
- `profile`: Generate from clean gate trace `/root/lfz/runs/vendor-ds4-16gb/20260701T201316Z-20260702_page_fault_readamp_clean_repro/france-cpu40-vram0gb/one_trace.csv` with rows `tensor,expert,count`, using `min_uses=2`.
- `test_config`: patched default-off source, `GGML_MOE_STREAM_CACHE_ADMIT_PROFILE=/root/lfz/runs/vendor-ds4-16gb/cache_admit_profiles/gate_min2_from_20260701T201316Z.csv`, `GGML_MOE_STREAM_CACHE_ADMIT_MIN_USES=2`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, gate trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `risk`: This uses a France-derived profile, so it is prompt-specific unless later generalized. A bad parser or key mismatch can reduce cache effectiveness; source changes could also add CPU string/hash overhead. Correctness should be unchanged because arithmetic and copied bytes are unchanged, but France output must still pass.
- `required_evidence`: source diff, build log/hash, profile path/sha256/counts, exact env/command, trace summary including inserted/staging/hit counts, cgroup memory files, France output, `summary.json`, stdout/stderr, and explicit promoted/rejected/rollback status.
- `rollback`: If build fails, profile is missing/mismatched, correctness fails, RAM exceeds 16GB, TTFT exceeds gate, or `eval_tok_s <= 1.6`, revert source and clean rebuild. If accepted, immediately complete full reproduction record, commit/push source to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before promotion.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T212038Z-20260702_gate_cache_admit_min2_profile/france-cpu40-vram0gb`
- `profile_sha256`: `/root/lfz/runs/vendor-ds4-16gb/cache_admit_profiles/gate_min2_from_20260701T201316Z.csv` = `6e79d51b0b595859d3ffd589f413ae8a56e3604d21943b701fc2dc31222ad81d`.
- `result`: completed and rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=46501.495934ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14968098816`, `pgmajfault=595128`, `workingset_refault_file=11433671`, `ram_ok=true`, `correctness_ok=true`.
- `trace_summary`: admission matched offline simulation exactly: `rows=34753`, `cache_hits=30219`, `cache_inserts=3116`, `staging_non_admitted=1418`, `trace_span_ms=107486.742`, `src0_ms_sum=22979.714`, `kernel_ms_sum=382.29`, `dontneed_ms_sum=1115.399`, `total_ms_sum=25166.967`.
- `correctness`: pass. France answer is semantically correct and coherent.
- `gap_analysis`: Profile-based cache admission reduced direct gate `src0_ms` by about `2.3s` versus the clean diagnostic and reduced inserted loads to the theoretical minimum for known repeated keys, but end-to-end token rate stayed rounded at `1.6` and trace span remained `~107s`. The remaining bottleneck is outside simple gate-cache pollution, likely CPU fallback/up-down page scheduling and untraced cgroup reclaim. Do not promote.
- `rollback_status`: source patch reverted and clean rebuild required. No commit/push because this is not an accepted SOTA.

### 当前执行 attempt：cpu-fallback-chunk32

- `attempt_id`: `20260702-cpu-fallback-chunk32`
- `attempt_kind`: `source-probe/cpu-compute`
- `status`: planned
- `hypothesis`: The CPU fallback trace for current gate-stream runs contains `1,667,384` up/down chunks. Many `cne1>=2` expert operations still use `chunk_size=16`, creating many small chunk calls and atomic scheduling events. A default-off `GGML_MOE_CPU_CHUNK_SIZE=32` for non-single-row cases may reduce chunk scheduling overhead and improve up/down fallback throughput without changing arithmetic.
- `theoretical_upper_bound`: This cannot reduce gate stream miss cost. It can only reduce CPU fallback overhead/compute time. The trace has `up sum_ms=688777.888` and `down sum_ms=594024.621` across all traced chunks; even a small reduction in per-chunk overhead or better row-block locality could recover seconds, but larger chunks can reduce load balance. Promote only on measured end-to-end `eval_tok_s > 1.6` with all gates passing.
- `implementation`: Add default-off env `GGML_MOE_CPU_CHUNK_SIZE=<N>` in `ggml/src/ggml-cpu/ggml-cpu.c`. Preserve the existing `chunk_size=64` behavior when `nr0 == 1 || nr1 == 1`; only use the env override for the general multi-row case where the current default is `16`.
- `test_config`: patched default-off source, `GGML_MOE_CPU_CHUNK_SIZE=32`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, gate trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `risk`: Larger chunks may reduce balancing for high `cne1` experts or increase tail latency; correctness should be unchanged but France output must pass. If the run ties/regresses, this parameter is not a SOTA and source must be reverted.
- `required_evidence`: source diff, build log/hash, exact env/command, cgroup memory files, France output, `summary.json`, gate trace summary, stdout/stderr, and explicit promoted/rejected/rollback status.
- `rollback`: If build fails, correctness fails, RAM exceeds 16GB, TTFT exceeds gate, or `eval_tok_s <= 1.6`, revert source and clean rebuild. If accepted, immediately complete full reproduction record, commit/push source to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before promotion.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T212807Z-20260702_cpu_fallback_chunk32/france-cpu40-vram0gb`
- `result`: completed and rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.8`, `TTFT=41706.09388ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14979452928`, `pgmajfault=491558`, `workingset_refault_file=11934422`, `ram_ok=true`, `correctness_ok=true`.
- `trace_summary`: gate stream unchanged: `rows=34753`, `cache_hits=29855`, `cache_inserts=4898`, `trace_span_ms=104695.023`, `src0_ms_sum=24559.658`, `kernel_ms_sum=390.653`, `dontneed_ms_sum=1217.548`, `total_ms_sum=26861.594`.
- `correctness`: pass. France answer is semantically correct and coherent.
- `gap_analysis`: `chunk32` materially improved TTFT and trace span versus the recent clean diagnostic, but rounded eval token rate remained `1.6`. This is not a SOTA by itself. The next bounded test is combining this lower-TTFT CPU scheduling change with the cache-admission change that reduced gate `src0_ms`; if the combination still ties, both changes should be reverted.
- `rollback_status`: pending combined probe; if not promoted, revert source and clean rebuild. No commit/push for this tie.

### 当前执行 attempt：chunk32-cache-admit-min2-combined

- `attempt_id`: `20260702-chunk32-cache-admit-min2-combined`
- `attempt_kind`: `source-probe/combined-boundary`
- `status`: planned
- `hypothesis`: `cpu-fallback-chunk32` lowered TTFT/trace span but did not reduce gate loads; `gate-cache-admit-min2-profile` reduced direct gate `src0_ms` but did not lower TTFT enough to move token rate. Combining the two independent small wins may cross the rounded `1.6 tok/s` boundary while preserving correctness and RAM.
- `theoretical_upper_bound`: The direct measured component wins are about `~2.3s` lower gate `src0_ms` from cache admission and `~4.2s` lower gate trace span plus lower TTFT from chunk32. These are not strictly additive because untraced reclaim/CPU fallback overlap can dominate, but the combined upside is enough to justify one strict run. It still cannot fix the full up/down CPU fallback bottleneck, so reject if measured `eval_tok_s <= 1.6`.
- `implementation`: Enable both default-off mechanisms: `GGML_MOE_CPU_CHUNK_SIZE=32` and `GGML_MOE_STREAM_CACHE_ADMIT_PROFILE=<gate_min2_profile>` with `GGML_MOE_STREAM_CACHE_ADMIT_MIN_USES=2`.
- `test_config`: patched source with both default-off features, `GGML_MOE_CPU_CHUNK_SIZE=32`, `GGML_MOE_STREAM_CACHE_ADMIT_PROFILE=/root/lfz/runs/vendor-ds4-16gb/cache_admit_profiles/gate_min2_from_20260701T201316Z.csv`, `GGML_MOE_STREAM_CACHE_ADMIT_MIN_USES=2`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, gate trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `risk`: This combines two unpromoted changes, so if the measured token rate does not improve, both must be reverted. The cache-admission half remains prompt-profile-specific and cannot be accepted without full reproducibility records and pushed source.
- `required_evidence`: source diff, build log/hash, profile sha256, exact env/command, trace summary with hit/insert/staging counts, cgroup memory files, France output, `summary.json`, stdout/stderr, and explicit promoted/rejected/rollback status.
- `rollback`: If build fails, correctness fails, RAM exceeds 16GB, TTFT exceeds gate, or `eval_tok_s <= 1.6`, revert both source changes and clean rebuild. If accepted, immediately complete full reproduction record, commit/push source to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before promotion.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T213246Z-20260702_chunk32_cache_admit_min2_combined/france-cpu40-vram0gb`
- `result`: completed and rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=44178.221165ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14970384384`, `pgmajfault=494525`, `workingset_refault_file=11333408`, `ram_ok=true`, `correctness_ok=true`.
- `trace_summary`: admission matched profile and chunk32 remained enabled: `rows=34753`, `cache_hits=30219`, `cache_inserts=3116`, `staging_non_admitted=1418`, `trace_span_ms=102982.959`, `src0_ms_sum=23256.709`, `kernel_ms_sum=380.856`, `dontneed_ms_sum=1139.654`, `total_ms_sum=25464.565`.
- `correctness`: pass. France answer is semantically correct and coherent.
- `gap_analysis`: The combination produced the best trace span among these small probes and preserved strict RAM/correctness/TTFT gates, but eval token rate still rounded to `1.6`. This means the remaining end-to-end bottleneck is not solved by small gate-cache and CPU chunk scheduling reductions. Do not promote.
- `rollback_status`: both source changes reverted and clean rebuild required. No commit/push because this is not an accepted SOTA.

### 当前执行 attempt：cpu-fallback-chunk64

- `attempt_id`: `20260702-cpu-fallback-chunk64`
- `attempt_kind`: `source-probe/cpu-compute-boundary`
- `status`: planned
- `hypothesis`: `GGML_MOE_CPU_CHUNK_SIZE=32` improved TTFT (`41706ms`) and gate trace span (`104695ms`) but did not move `eval_tok_s` beyond `1.6`. A larger multi-row fallback chunk (`64`) may further reduce atomic scheduling and small-chunk overhead in up/down CPU fallback, while keeping the existing single-row `chunk_size=64` path unchanged.
- `theoretical_upper_bound`: The previous CPU trace had `1,667,384` up/down chunks and summed chunk time `up=688777.888ms`, `down=594024.621ms`. Only multi-row cases are affected; larger chunking can at most recover scheduling/function-call overhead and some locality, not page-in or arithmetic. Since `chunk32` reduced TTFT by about `5.7s` versus recent clean controls, `chunk64` could recover a few more seconds if overhead dominates, but can regress if load balance/tail latency dominates.
- `implementation`: Re-add default-off env `GGML_MOE_CPU_CHUNK_SIZE=<N>` in `ggml/src/ggml-cpu/ggml-cpu.c`, preserving current default and preserving `chunk_size=64` when `nr0 == 1 || nr1 == 1`. Test with `N=64`.
- `test_config`: patched default-off source, `GGML_MOE_CPU_CHUNK_SIZE=64`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, gate trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `risk`: Larger chunks may reduce parallel load balance and hurt token rate even if scheduling overhead falls. Correctness should be unchanged, but France output must pass. If the run ties/regresses, this is not a SOTA and source must be reverted.
- `required_evidence`: source diff, build log/hash, exact env/command, cgroup memory files, France output, `summary.json`, gate trace summary, stdout/stderr, and explicit promoted/rejected/rollback status.
- `rollback`: If build fails, correctness fails, RAM exceeds 16GB, TTFT exceeds gate, or `eval_tok_s <= 1.6`, revert source and clean rebuild. If accepted, immediately complete full reproduction record, commit/push source to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before promotion.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T214016Z-20260702_cpu_fallback_chunk64/france-cpu40-vram0gb`
- `result`: completed and rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.8`, `TTFT=42533.796173ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14974631936`, `pgmajfault=439220`, `workingset_refault_file=12248606`, `ram_ok=true`, `correctness_ok=true`.
- `trace_summary`: gate stream shape unchanged: `rows=34753`, `cache_hits=29855`, `cache_inserts=4898`, `trace_span_ms=105003.531`, `src0_ms_sum=26322.059`, `kernel_ms_sum=390.073`, `dontneed_ms_sum=1211.469`, `total_ms_sum=28624.892`.
- `correctness`: pass. France answer is semantically correct and coherent.
- `gap_analysis`: `chunk64` keeps the TTFT benefit class from chunk-size tuning but does not improve token rate beyond `1.6`, and its traced gate `src0_ms`/total is worse than `chunk32`. Larger multi-row chunks likely start losing load balance or increase page/reclaim interaction. Do not promote.
- `rollback_status`: source patch reverted and clean rebuild required. No commit/push because this is not an accepted SOTA.

### 当前执行 attempt：thread16-reclaim-contention-probe

- `attempt_id`: `20260702-thread16-reclaim-contention-probe`
- `attempt_kind`: `config-probe/threading`
- `status`: planned
- `hypothesis`: Recent strict cold runs spend very high system time and show high `pgmajfault` / `workingset_refault_file`. CPU fallback uses `-t 20 -tb 20`; reducing to `16` threads may lower page-cache reclaim and IO contention inside the 16GB cgroup enough to improve wall/token rate, even if raw CPU compute parallelism drops.
- `theoretical_upper_bound`: This cannot reduce arithmetic work or gate cache misses. The possible gain is only from lower kernel reclaim/page-fault contention and better scheduling. `chunk32/chunk64` lowered TTFT without improving `eval_tok_s`; if thread contention is the remaining limiter, `t16` could reduce system time or major faults. If CPU compute is dominant, it will regress. Promote only if measured `eval_tok_s > 1.6` with all gates passing.
- `implementation`: No model source change. Run the clean binary with `-t 16 -tb 16` in extra args, preserving the rest of the gate-stream vram12/cold/16GB configuration.
- `test_config`: clean source `aa8d6f916`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, gate trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 16 -tb 16`.
- `tooling_note`: Fix `.Agent/run-tools/strict_ds4_runner.py` elapsed-time parsing so `/usr/bin/time -v` values like `2:08.51` are recorded as `128.51s` instead of `8.51s`; this does not change model behavior.
- `risk`: Fewer threads can increase CPU fallback compute wall and reduce token rate. Reject if token rate does not exceed `1.6`, RAM/correctness fail, or TTFT exceeds gate.
- `required_evidence`: exact command/env proving `-t 16 -tb 16`, cgroup memory files, France output, `summary.json` with corrected elapsed seconds, gate trace summary, stdout/stderr, and explicit promoted/rejected status.
- `rollback`: No model source change. If not accepted, keep clean source and record rejection. If accepted, still complete full reproduction record and push any relevant tooling/source changes to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before promotion.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T214659Z-20260702_thread16_reclaim_contention/france-cpu40-vram0gb`
- `result`: completed and rejected/regression. `eval_tok_s=1.5`, `prompt_tok_s=0.6`, `TTFT=48076.625549ms`, corrected `elapsed_seconds=135.86`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14983475200`, `pgmajfault=848549`, `workingset_refault_file=12118944`, `ram_ok=true`, `correctness_ok=true`.
- `trace_summary`: `rows=34753`, `cache_hits=29855`, `cache_inserts=4898`, `trace_span_ms=116750.166`, `src0_ms_sum=25501.426`, `kernel_ms_sum=400.558`, `dontneed_ms_sum=1228.624`, `total_ms_sum=27846.671`.
- `correctness`: pass. France answer is semantically correct and coherent.
- `gap_analysis`: Reducing threads worsened token rate, TTFT, trace span, and major faults. CPU fallback parallelism is still needed; lower thread count does not relieve cgroup reclaim enough to help.
- `rollback_status`: no model source change. No commit/push because this is not an accepted SOTA.

### 当前执行 attempt：thread24-cpu-parallelism-probe

- `attempt_id`: `20260702-thread24-cpu-parallelism-probe`
- `attempt_kind`: `config-probe/threading`
- `status`: planned
- `hypothesis`: `t16` regressed, suggesting CPU fallback compute parallelism is still important. Increasing from `20` to `24` threads may improve up/down CPU fallback wall if compute is the limiter; the risk is higher page-cache reclaim/IO contention under the same 16GB cgroup.
- `theoretical_upper_bound`: It cannot reduce gate stream miss count. Upside is bounded by CPU fallback parallel speedup; if the up/down fallback wall has enough runnable work, a `20 -> 24` increase could recover a few seconds. If system time/page faults dominate, it will tie or regress. Promote only if measured `eval_tok_s > 1.6` with all gates passing.
- `implementation`: No model source change. Run clean binary with `-t 24 -tb 24`, preserving the rest of the gate-stream vram12/cold/16GB configuration.
- `test_config`: clean source `aa8d6f916`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, gate trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 24 -tb 24`.
- `risk`: More threads can increase system time, refaults, and TTFT. Reject if token rate does not exceed `1.6`, RAM/correctness fail, or TTFT exceeds gate.
- `required_evidence`: exact command/env proving `-t 24 -tb 24`, cgroup memory files, France output, `summary.json`, gate trace summary, stdout/stderr, and explicit promoted/rejected status.
- `rollback`: No model source change. If not accepted, keep clean source and record rejection. If accepted, complete full reproduction/push/rerun SOTA gate before promotion.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T215044Z-20260702_thread24_cpu_parallelism/france-cpu40-vram0gb`
- `result`: completed and rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=47206.004663ms`, corrected `elapsed_seconds=129.2`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14976692224`, `pgmajfault=668293`, `workingset_refault_file=12143176`, `ram_ok=true`, `correctness_ok=true`.
- `time_v`: user `800.85s`, system `1069.78s`, wall `2:09.20`, major faults `668703`, file inputs `259072000`.
- `trace_summary`: `rows=34753`, `cache_hits=29855`, `cache_inserts=4898`, `trace_span_ms=109311.238`, `src0_ms_sum=25528.211`, `kernel_ms_sum=394.271`, `dontneed_ms_sum=1236.551`, `total_ms_sum=27851.837`.
- `correctness`: pass. France answer is semantically correct and coherent.
- `gap_analysis`: Increasing to 24 threads ties rounded token rate but raises system time and does not improve trace span versus clean/chunk32. Together with t16 regression, simple thread-count tuning is not enough to produce a SOTA under the 16GB cgroup.
- `rollback_status`: no model source change. No commit/push because this is not an accepted SOTA.

### 设计结论：关闭简单 thread-count 调参

- `analysis_id`: `20260702-thread-count-closure`
- `finding`: `t16` regressed to `1.5 tok/s` with higher TTFT and major faults; `t24` tied `1.6 tok/s` but did not improve trace span or wall. The default `t20` remains the best clean setting among tested thread counts.
- `decision`: Do not spend more runs on nearby thread counts (`t18/t22/t24+`) without a deeper CPU fallback/page-reclaim mechanism. Future work should add op-level CPU fallback wall tracing or target a structural reduction in up/down fallback work.

### 当前执行 attempt：cpu-fallback-op-wall-trace

- `attempt_id`: `20260702-cpu-fallback-op-wall-trace`
- `attempt_kind`: `diagnostic/source-trace`
- `status`: planned
- `hypothesis`: Current chunk traces sum per-chunk CPU work but do not show per-op wall time, barrier tail, or how much of the end-to-end wall is spent in `ffn_up_exps` versus `ffn_down_exps` fallback after gate stream removes gate rows. Adding a default-off op-level wall trace will identify the largest remaining wall-time tensor classes and whether future work should target compute, page/reclaim stalls, or scheduling tail.
- `theoretical_upper_bound`: This is diagnostic and is not expected to improve token rate. It should quantify the remaining `~100s+` gate trace span gap by tensor/op. If one tensor class or layer dominates op wall, a later structural optimization can calculate a realistic token-rate upper bound from that wall contribution.
- `implementation`: Add env-gated `GGML_MOE_CPU_OP_TRACE_OUT=<csv>` in `ggml/src/ggml-cpu/ggml-cpu.c`. For each `mul_mat_id` op after the gate stream path, record one row from thread 0 with tensor name/type, remaining active experts, remaining rows, `ne01/ne00`, and wall time for the CPU fallback loop. The trace path may add a final barrier only when enabled; it is diagnostic and must be reverted after the run unless promoted as tooling.
- `test_config`: patched default-off source, `GGML_MOE_CPU_OP_TRACE_OUT={case_dir}/cpu_op_trace.csv`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, gate trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `risk`: Trace-only barrier and file writes may perturb timing, so this run is diagnostic. Correctness/RAM must still pass to keep the profile comparable. Do not treat its token rate as SOTA unless a separate no-trace reproduction from clean pushed source passes all gates.
- `required_evidence`: source diff, build log/hash, exact env/command, `cpu_op_trace.csv`, cgroup memory files, France output, `summary.json`, gate trace summary, stdout/stderr, and analysis of top op wall contributors.
- `rollback`: Revert trace source and clean rebuild after collecting diagnostics unless it unexpectedly becomes part of an accepted SOTA path. No commit/push for a diagnostic-only run.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T215815Z-20260702_cpu_fallback_op_wall_trace/france-cpu40-vram0gb`
- `result`: completed diagnostic, not promoted. `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=48357.974517ms`, corrected `elapsed_seconds=130.87`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14977716224`, `pgmajfault=606034`, `workingset_refault_file=11947939`, `ram_ok=true`, `correctness_ok=true`.
- `cpu_op_trace_summary`: `op_rows=16680`; gate stream removed gate fallback (`gate active_rows=0`, wall `71.917ms`). Remaining CPU fallback wall is `ffn_up_exps=41098.899ms` and `ffn_down_exps=36159.187ms`, with `36000` active rows each and `34753` active experts each.
- `top_wall_tensors`: `blk.0.up=2015.699ms`, `blk.1.up=1971.926ms`, `blk.2.up=1943.363ms`, `blk.1.down=1886.923ms`, `blk.0.down=1745.392ms`, `blk.2.down=1735.727ms`.
- `gate_trace_summary`: `rows=34753`, `cache_hits=29855`, `cache_inserts=4898`, `trace_span_ms=110187.299`, `src0_ms_sum=25140.205`, `total_ms_sum=27460.588`.
- `gap_analysis`: The bottleneck is now confirmed as broad up/down CPU fallback wall, not gate fallback. Prior all-MoE/gate+up streaming attempts regressed because streaming up/down creates more H2D/page-in cost than it removes. Future work needs a structural reduction in up/down fallback compute/page touches, not more gate cache, thread count, or simple stream-name combinations.
- `rollback_status`: trace source patch reverted and clean rebuild required. No commit/push because this is diagnostic and not an accepted SOTA.

### 当前执行 attempt：selective-hot-up-stream

- `attempt_id`: `20260702-selective-hot-up-stream`
- `attempt_kind`: `source-probe/selective-stream`
- `status`: planned
- `hypothesis`: Op-wall trace confirms broad up/down CPU fallback dominates, but all-up/all-MoE streaming regresses due cache/page-in cost. Gate+all-up trace shows some individual up layers have positive estimated net (`CPU wall - stream total`), especially layers `5,6,7,15,31,35` and several medium layers. Streaming only gate plus a small positive-net up-layer set may reduce CPU fallback without the cache pressure of all-up streaming.
- `theoretical_upper_bound`: Using gate+all-up trace as a conservative cost estimate, the selected up layers `3,5,6,7,10,11,12,15,18,29,31,35` have roughly `+1.9s` net (`sum CPU wall - stream total`) before second-order cache effects. This is a boundary probe; it cannot solve the full `up≈41.1s/down≈36.2s` fallback wall, but may cross a rounded token-rate boundary if selective cache pressure is lower than gate+all-up.
- `implementation`: Add default-compatible comma-separated matching to `GGML_MOE_STREAM_ONE_NAME_FILTER` in `ggml/src/ggml-cuda/moe_stream.cu`. Existing empty and single-substring behavior remains unchanged. Test filter: `ffn_gate_exps,blk.3.ffn_up_exps,blk.5.ffn_up_exps,blk.6.ffn_up_exps,blk.7.ffn_up_exps,blk.10.ffn_up_exps,blk.11.ffn_up_exps,blk.12.ffn_up_exps,blk.15.ffn_up_exps,blk.18.ffn_up_exps,blk.29.ffn_up_exps,blk.31.ffn_up_exps,blk.35.ffn_up_exps`.
- `test_config`: patched default-compatible source, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, selective `GGML_MOE_STREAM_ONE_NAME_FILTER`, cold `drop_caches`, strict 16GB cgroup, France prompt, gate/up trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `risk`: The theoretical upside is small and prompt-dependent. Even selected up layers may evict useful gate entries or increase H2D/page-in enough to regress. Correctness should be unchanged because individual up stream passed CPU/GPU compare, but France output must pass.
- `required_evidence`: source diff, build log/hash, exact env/command, trace tensor mix proving only gate plus selected up layers streamed, cgroup memory files, France output, `summary.json`, stdout/stderr, and promoted/rejected/rollback status.
- `rollback`: If build fails, correctness fails, RAM exceeds 16GB, TTFT exceeds gate, or `eval_tok_s <= 1.6`, revert source and clean rebuild. If accepted, immediately complete full reproduction record, commit/push source to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before promotion.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T220650Z-20260702_selective_hot_up_stream/france-cpu40-vram0gb`
- `result`: completed and rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.6`, `TTFT=48055.622383ms`, corrected `elapsed_seconds=123.83`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14974443520`, `pgmajfault=485931`, `workingset_refault_file=9550051`, `ram_ok=true`, `correctness_ok=true`.
- `trace_summary`: `rows=43270`, `cache_hits=36204`, `cache_inserts=7066`, `trace_span_ms=103802.707`; streamed gate `rows=33307/inserts=5480/src0_ms=26878.954/total_ms=29212.292`, selected up `rows=9963/inserts=1586/src0_ms=8404.0/total_ms=9049.1`.
- `gap_analysis`: Selective up streaming lowered wall and refault indicators versus some clean controls, but rounded token rate remained `1.6` and TTFT worsened. Gate cache pressure increased (`gate inserts 4898 -> 5480`), consuming much of the up fallback gain. Do not promote.
- `next_narrow_probe`: Before reverting the comma-filter source, run a narrower top6 up filter (`5,6,7,15,31,35`) to reduce gate cache competition while keeping the highest positive-net up layers.

### 当前执行 attempt：selective-hot-up-top6-stream

- `attempt_id`: `20260702-selective-hot-up-top6-stream`
- `attempt_kind`: `source-probe/selective-stream`
- `status`: planned
- `hypothesis`: The 12-layer selective run tied `1.6` but increased gate cache misses. A narrower set of the strongest positive-net up layers (`5,6,7,15,31,35`) should reduce cache competition while still removing a meaningful amount of up CPU fallback.
- `theoretical_upper_bound`: Based on gate+all-up trace, these six layers have about `+1.35s` net before second-order cache effects. The upside is smaller than the 12-layer attempt, but the narrower stream set may reduce gate miss regression and improve wall/token rate.
- `implementation`: Reuse the comma-separated filter patch. Test filter: `ffn_gate_exps,blk.5.ffn_up_exps,blk.6.ffn_up_exps,blk.7.ffn_up_exps,blk.15.ffn_up_exps,blk.31.ffn_up_exps,blk.35.ffn_up_exps`.
- `test_config`: patched default-compatible source, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, top6 selective `GGML_MOE_STREAM_ONE_NAME_FILTER`, cold `drop_caches`, strict 16GB cgroup, France prompt, trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `rollback`: If not accepted, revert comma-filter source and clean rebuild. If accepted, complete full SOTA reproduction, commit/push, and rerun from pushed commit before promotion.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T221045Z-20260702_selective_hot_up_top6_stream/france-cpu40-vram0gb`
- `result`: completed and rejected/tie. `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=46496.582582ms`, corrected `elapsed_seconds=127.64`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14949691392`, `pgmajfault=557174`, `workingset_refault_file=11427759`, `ram_ok=true`, `correctness_ok=true`.
- `trace_summary`: `rows=39658`, `cache_hits=33592`, `cache_inserts=6066`, `trace_span_ms=109462.517`; streamed gate `rows=34502/inserts=5294/src0_ms=27347.2/total_ms=29707.1`, top6 up `rows=5156/inserts=772/src0_ms=4374.7/total_ms=4699.6`.
- `gap_analysis`: Narrowing to top6 reduced up stream pressure but still increased gate inserts versus baseline and did not improve token rate. Selective up streaming is too small to overcome added gate cache competition under the current shared VRAM cache.
- `rollback_status`: comma-filter source reverted and clean rebuild required. No commit/push because this is not an accepted SOTA.

### 设计结论：关闭 selective up stream without cache partition

- `analysis_id`: `20260702-selective-up-stream-closure`
- `finding`: 12-layer selective up tied `1.6` with wall `123.83s` but worsened TTFT and gate misses; top6 selective up also tied `1.6` and still increased gate inserts (`4898 -> 5294`). The shared cache interaction erases the small estimated up-layer benefit.
- `decision`: Do not run more layer-subset up-stream filters without a cache partition/admission mechanism that protects gate entries. Future stream work must address cache competition explicitly.

### 当前执行 attempt：expert-keep-top5-updown

- `attempt_id`: `20260702-expert-keep-top5-updown`
- `attempt_kind`: `source-probe/approximate-pruning`
- `status`: promoted after pushed-source clean rebuild rerun
- `hypothesis`: Op-level wall trace shows up/down CPU fallback dominates (`up≈41.1s`, `down≈36.2s`) and each op uses about 6 routed experts per token. If expert IDs are ordered by router rank, keeping only the first 5 routed experts for `ffn_up_exps` and `ffn_down_exps` may reduce roughly `1/6` of up/down fallback work while preserving enough semantic quality for the France prompt.
- `theoretical_upper_bound`: Up/down fallback wall is about `77.3s`; pruning one of six routed experts has a rough upper bound near `12.9s` wall reduction before overhead and page effects. This is much larger than gate-cache/thread tweaks, but correctness risk is real. Promote only if France output remains semantically correct/coherent, RAM passes, TTFT stays within gate, and `eval_tok_s > 1.6`.
- `implementation`: Add default-off env `GGML_MOE_KEEP_TOPK_UPDOWN=<K>` in `ggml/src/ggml-cpu/ggml-cpu.c`. During `matrix_row_counts` construction, for tensors whose name contains `ffn_up_exps` or `ffn_down_exps`, skip routed expert ranks `id >= K` and explicitly zero the corresponding output row `dst[id, iid1, :]`. Gate tensors are not pruned.
- `test_config`: patched default-off source, `GGML_MOE_KEEP_TOPK_UPDOWN=5`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, gate trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `risk`: This is approximate and may degrade model quality. France correctness must be manually reviewed, not just heuristic. If output is incoherent, semantically wrong, or token rate does not improve, revert source and reject.
- `required_evidence`: source diff, build log/hash, exact env/command, France answer text, `summary.json`, gate trace, cgroup memory files, stdout/stderr, and explicit promoted/rejected/rollback status.
- `rollback`: If build fails, output correctness fails, RAM exceeds 16GB, TTFT exceeds gate, or `eval_tok_s <= 1.6`, revert source and clean rebuild. If accepted, immediately complete full reproduction record, commit/push source to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before promotion.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T221922Z-20260702_expert_keep_top5_updown/france-cpu40-vram0gb`
- `result`: accepted candidate and promoted after pushed-commit rerun. Original candidate produced `eval_tok_s=1.9`, `prompt_tok_s=0.7`, `TTFT=43426.525893ms`, corrected `elapsed_seconds=127.94`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14970519552`, `pgmajfault=551675`, `workingset_refault_file=11094865`, `ram_ok=true`, `correctness_ok=true`.
- `correctness_manual_review`: pass. France answer is semantically correct and coherent: it identifies France/French Republic as a Western European country, mentions history/culture/global influence, Eiffel Tower/Louvre/Versailles, cuisine/wine/fashion, art/philosophy/science, EU membership, Paris, diverse regions, French Revolution, and economic/political role. Minor repetition of Eiffel Tower does not make the answer incorrect or incoherent.
- `trace_summary`: gate stream still works: `rows=43123`, `cache_hits=37697`, `cache_inserts=5426`, `trace_span_ms=110181.209`, `src0_ms_sum=29038.595`, `total_ms_sum=31722.375`.
- `sota_gate`: passes current gates: RAM including page cache is capped at `16000000000`, France correctness passes, and TTFT is below recent clean `~47s` controls, therefore not above the `+20%` limit. Token rate improves from clean reproducible `1.6` to `1.9` on the original candidate and to `2.0` on the pushed-source rerun.
- `reproduction_record`: pre-commit reproduction artifacts written in the run directory: `source_state_precommit.txt`, `source_diff_precommit.patch`, `source_diff_precommit.stat`, `binary_sha256_precommit.txt`, `binary_stat_precommit.txt`, `binary_version_precommit.txt`, `model_stat.txt`, `runner_sha256_precommit.txt`, `plan_snapshot_precommit.md`, `progress_snapshot_precommit.md`, exact command/env/stdout/stderr/summary/trace/cgroup memory files.
- `publish_status`: source committed and pushed to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`.
- `pushed_commit`: `07f1dc6bb8d8f80e00e30ad09529a939170c25d8` (`vendor-ds4: add updown top5 expert pruning sota`).
- `pushed_branch`: `vendor/deepseek-token-rate-16gb` on remote `ssd=https://github.com/wici-ai/ssd-llama.git`.
- `pushed_rerun_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T222605Z-20260702_expert_keep_top5_updown_pushed_rerun/france-cpu40-vram0gb`.
- `pushed_rerun_result`: `eval_tok_s=2.0`, `prompt_tok_s=0.7`, `TTFT=43188.681316ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14968823808`, `pgmajfault=538269`, `workingset_refault_file=11104561`, `ram_ok=true`, `correctness_ok=true`.
- `pushed_rerun_binary`: stdout build line `build : b9086-07f1dc6bb`; binary was rebuilt after push from commit `07f1dc6bb`.
- `pushed_rerun_correctness_manual_review`: pass. France answer is semantically correct and coherent, with the same minor Eiffel Tower repetition as the candidate but no factual/semantic failure.
- `current_effective_sota`: `2.0 tok/s` under strict 16GB cgroup from pushed source. This replaces clean reproducible `1.6 tok/s` as the current source-backed line; old `2.6 tok/s` remains a forensic target until exact source/binary state is recovered.

### 当前执行 attempt：expert-keep-top4-updown

- `attempt_id`: `20260702-expert-keep-top4-updown`
- `attempt_kind`: `config-probe/approximate-pruning`
- `status`: promoted after pushed-source clean rebuild rerun
- `hypothesis`: Top5 up/down pruning is the current source-backed SOTA and shows that dropping one routed up/down expert can preserve the France answer. Remaining up/down fallback work is still large, so lowering `GGML_MOE_KEEP_TOPK_UPDOWN` from `5` to `4` may remove another routed expert from every up/down expert op and increase generation rate.
- `theoretical_upper_bound`: Original op-wall trace estimated up/down fallback at about `77.3s`. Top5 roughly removes one of six routed experts, with an ideal reduction near `12.9s`. Top4 removes two of six, so the coarse upper bound versus no pruning is about `25.8s`, or about `12.9s` additional wall reduction versus top5 before overhead/page effects. Real gain is lower because gate streaming and non-expert work remain, but a measurable improvement over the current `2.0 tok/s` SOTA is plausible.
- `risk`: This is a stronger approximation than top5. It may remove semantically important routed expert contribution and produce fluent but wrong, repetitive, or under-specified output. The France answer must be manually reviewed; heuristic correctness alone is insufficient.
- `test_config`: pushed source commit lineage at HEAD `21b428080` with model source from `07f1dc6bb`, no source change, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, gate trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: promote only if `eval_tok_s > 2.0`, RAM including page cache stays `<=16000000000`, France output is semantically correct and coherent under manual review, and TTFT does not exceed the top5 pushed rerun TTFT by more than `20%` (`43188.681316ms * 1.2 = 51826.417579ms`).
- `rollback`: No source change. If token rate does not exceed `2.0`, output correctness fails, RAM exceeds limit, or TTFT exceeds the gate, mark rejected/unpromoted and keep current top5 SOTA. If accepted, immediately write full reproduction evidence, commit/push any changed records to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, and run a pushed-commit clean rebuild/rerun before promotion.
- `required_evidence`: exact env/command, source commit/status, binary sha256/build line, model stat, stdout/stderr, summary.json, gate trace, cgroup `memory.*`, France answer text, manual correctness note, and explicit promoted/rejected status.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T223523Z-20260702_expert_keep_top4_updown/france-cpu40-vram0gb`
- `result`: accepted candidate pending pushed-commit rerun. `eval_tok_s=2.3`, `prompt_tok_s=0.8`, `TTFT=40828.237606ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14985457664`, `pgmajfault=398448`, `workingset_refault_file=6541363`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `correctness_manual_review`: pass. The answer is semantically correct and coherent: it identifies France/French Republic as a Western European country, covers history/culture/global art/fashion/cuisine, borders and seas, Eiffel Tower/Louvre/Versailles, wines/cheeses/philosophy/literature/cinema, Paris, and France's economic/political/diplomatic role. No incoherence or factual degradation observed from top4 pruning.
- `sota_gate`: candidate passes initial gates. Token rate exceeds current pushed top5 SOTA `2.0 -> 2.3`; TTFT `40828.237606ms` is below the `51826.417579ms` limit; RAM including page cache is capped at `16000000000`; France correctness passes manual review.
- `reproduction_record`: candidate run directory now contains source head/status/diff, binary sha256/stat/build line, model stat, runner sha256, exact command/env, stdout/stderr, summary.json, cgroup memory files, gate trace, plan snapshot, push remote/branch target, manual correctness review, and candidate status.
- `publish_status`: candidate records were committed and pushed to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then binary was clean-rebuilt from pushed commit and rerun passed all gates.
- `pushed_commit`: `99b7fd67cdd21942efbb015afed6388ebb86eaa3` (`vendor-ds4: record top4 pruning candidate`).
- `pushed_branch`: `vendor/deepseek-token-rate-16gb` on remote `ssd=https://github.com/wici-ai/ssd-llama.git`.
- `pushed_rerun_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T224018Z-20260702_expert_keep_top4_updown_pushed_rerun/france-cpu40-vram0gb`.
- `pushed_rerun_result`: `eval_tok_s=2.3`, `prompt_tok_s=0.9`, `TTFT=39140.888549ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14986100736`, `pgmajfault=405106`, `workingset_refault_file=6669541`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `pushed_rerun_binary`: `llama-cli --version` reports `version: 9089 (99b7fd67c)` after clean rebuild.
- `pushed_rerun_correctness_manual_review`: pass. The France answer is semantically correct and coherent, with no observed degradation from top4 pruning.
- `current_effective_sota`: `2.3 tok/s` under strict 16GB cgroup from pushed source. This replaces top5 `2.0 tok/s` as the current source-backed line; old `2.6 tok/s` remains a forensic target until exact source/binary state is recovered.

### 当前执行 attempt：expert-keep-top3-updown

- `attempt_id`: `20260702-expert-keep-top3-updown`
- `attempt_kind`: `config-probe/approximate-pruning`
- `status`: completed / rejected_manual_correctness_truncated_output
- `hypothesis`: Top4 pruning is now the promoted SOTA and still produces a correct France answer. Reducing `GGML_MOE_KEEP_TOPK_UPDOWN` from `4` to `3` removes one more routed up/down expert per op and may further reduce the dominant CPU fallback work.
- `theoretical_upper_bound`: Using the same coarse op-wall estimate (`up+down≈77.3s`, six routed experts), top3 removes three of six routed up/down experts. The rough upper bound versus no pruning is about `38.7s`, or about `12.9s` additional wall reduction versus top4 before overhead/page effects. Real gain is bounded by gate streaming, non-expert work, and possible changed output length. Promote only if measured generation rate exceeds current `2.3 tok/s`.
- `risk`: Correctness risk is high. Top3 may remove too much expert contribution and yield plausible but lower-quality or semantically wrong output. France answer must be manually reviewed for semantic correctness, coherence, and absence of degeneration; heuristic pass alone is insufficient.
- `test_config`: pushed branch HEAD `baa457ca9` rebuilt to binary version `9091 (baa457ca9)`, no source change, `GGML_MOE_KEEP_TOPK_UPDOWN=3`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, gate trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: promote only if `eval_tok_s > 2.3`, RAM including page cache stays `<=16000000000`, France output is semantically correct and coherent under manual review, and TTFT does not exceed the top4 pushed rerun TTFT by more than `20%` (`39140.888549ms * 1.2 = 46969.066259ms`).
- `rollback`: No source change. If token rate does not exceed `2.3`, output correctness fails, RAM exceeds limit, or TTFT exceeds the gate, mark rejected/unpromoted and keep current top4 SOTA. If accepted, immediately write full reproduction evidence, commit/push records/source state to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild and rerun from pushed commit before promotion.
- `required_evidence`: exact env/command, source commit/status, binary sha256/build line, model stat, stdout/stderr, summary.json, gate trace, cgroup `memory.*`, France answer text, manual correctness note, and explicit promoted/rejected status.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T224718Z-20260702_expert_keep_top3_updown/france-cpu40-vram0gb`
- `result`: rejected despite high speed. `eval_tok_s=2.7`, `prompt_tok_s=1.0`, `TTFT=36616.831779ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15026126848`, `pgmajfault=324282`, `workingset_refault_file=5846413`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true` by heuristic but manual review failed.
- `correctness_manual_review`: fail. The answer is mostly factually correct but ends with an incomplete trailing sentence: `The country is also known`. This is a truncation/coherence failure for the required France prompt, so the result cannot be promoted.
- `trace_summary`: `rows=47883`, `src0_ms_sum=29164.065`, `kernel_ms_sum=509.168`, `dontneed_ms_sum=1465.194`, `total_ms_sum=32068.595`.
- `reproduction_record`: rejected run directory contains source commit/status/diff, binary sha256/stat/version/build line, model stat, runner sha256, exact command/env, stdout/stderr, summary.json, cgroup memory files, gate trace, manual correctness review, and rejected status.
- `rollback_status`: no source change; no commit/push as SOTA. Current effective SOTA remains top4 `2.3 tok/s`.

### 当前执行 attempt：expert-keep-top3-updown-n256

- `attempt_id`: `20260702-expert-keep-top3-updown-n256`
- `attempt_kind`: `config-probe/truncation-check`
- `status`: completed / rejected_manual_correctness_truncated_repetitive_output
- `hypothesis`: The rejected top3 run had strong speed (`2.7 tok/s`) and mostly correct content, but failed because the output ended with an incomplete trailing sentence at the `-n 192` generation limit. Increasing the generation budget to `-n 256` and context to `-c 384` may allow the same top3 approximation to finish a coherent France answer while preserving most of the token-rate gain.
- `theoretical_upper_bound`: TTFT should remain close to top3 (`~36.6s`) because model load and first-token path are unchanged. Generation rate should remain near top3 if the per-token compute path is unchanged; total wall time will increase only because more tokens are allowed. Promote only if the measured `eval_tok_s` remains above current top4 SOTA `2.3 tok/s` and the answer is complete/coherent.
- `risk`: Longer generation may reveal more semantic degradation from top3 pruning, produce a rambling answer, or still fail to stop cleanly. Larger `-c 384` may use more KV/VRAM; reject on CUDA OOM, RAM violation, TTFT gate failure, incomplete output, or `eval_tok_s <= 2.3`.
- `test_config`: pushed branch HEAD `5cbc184bd` rebuilt to binary version `9093 (5cbc184bd)`, no source change, `GGML_MOE_KEEP_TOPK_UPDOWN=3`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, gate trace enabled, CLI args `-n 256 -c 384 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: promote only if `eval_tok_s > 2.3`, RAM including page cache stays `<=16000000000`, France output is semantically correct and complete/coherent under manual review, and TTFT does not exceed current top4 pushed rerun by more than `20%` (`39140.888549ms * 1.2 = 46969.066259ms`).
- `rollback`: No source change. If any gate fails, record rejected/unpromoted and keep top4 `2.3 tok/s` SOTA. If accepted, immediately write full reproduction evidence, commit/push records/source state to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild and rerun from pushed commit before promotion.
- `required_evidence`: exact env/command, source commit/status, binary sha256/build line, model stat, stdout/stderr, summary.json, gate trace, cgroup `memory.*`, full France answer text, manual correctness note, and explicit promoted/rejected status.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T225516Z-20260702_expert_keep_top3_updown_n256/france-cpu40-vram0gb`
- `result`: rejected despite high speed. `eval_tok_s=2.8`, `prompt_tok_s=1.0`, `TTFT=36265.948228ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15009583104`, `pgmajfault=377471`, `workingset_refault_file=9055311`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true` by heuristic but manual review failed.
- `correctness_manual_review`: fail. The answer remains too long/repetitive for the short-paragraph prompt and still ends with an incomplete trailing sentence: `France is also known for its political`. Increasing to `-n 256` did not resolve the top3 coherence/truncation problem.
- `trace_summary`: `rows=63243`, `src0_ms_sum=35440.998`, `kernel_ms_sum=658.02`, `dontneed_ms_sum=1769.553`, `total_ms_sum=39096.629`.
- `reproduction_record`: rejected run directory contains source commit/status/diff, binary sha256/stat/version/build line, model stat, runner sha256, exact command/env, stdout/stderr, summary.json, cgroup memory files, gate trace, manual correctness review, and rejected status.
- `rollback_status`: no source change; no commit/push as SOTA. Current effective SOTA remains top4 `2.3 tok/s`. Future work should use a more conservative mixed top3/top4 policy rather than extending generation budget further.

### 当前执行 attempt：expert-keep-mixed-up3-down4

- `attempt_id`: `20260702-expert-keep-mixed-up3-down4`
- `attempt_kind`: `source-probe/approximate-pruning`
- `status`: completed / rejected_manual_correctness_truncated_output
- `hypothesis`: Global top3 is fast but over-prunes enough to produce repetitive/truncated France answers, while global top4 is correct. A mixed policy that keeps `top3` for `ffn_up_exps` and `top4` for `ffn_down_exps` may capture part of top3's speedup while preserving more output quality through the down projection path.
- `theoretical_upper_bound`: If up/down fallback costs are roughly similar (`up≈41.1s`, `down≈36.2s` from op-wall trace), moving only up from top4 to top3 can ideally save about one routed expert out of six for the up path, roughly `41.1s / 6 ≈ 6.85s` before overhead/page effects. This is about half the additional theoretical gain of global top3 versus top4, so an improvement over `2.3 tok/s` is plausible but should be smaller than the rejected `2.7-2.8 tok/s` observations.
- `implementation`: Add default-off per-tensor env overrides in `ggml/src/ggml-cpu/ggml-cpu.c`: `GGML_MOE_KEEP_TOPK_UP` applies to `ffn_up_exps`, `GGML_MOE_KEEP_TOPK_DOWN` applies to `ffn_down_exps`. If unset, both fall back to existing `GGML_MOE_KEEP_TOPK_UPDOWN`, preserving top4 SOTA behavior.
- `test_config`: patched source, `GGML_MOE_KEEP_TOPK_UP=3`, `GGML_MOE_KEEP_TOPK_DOWN=4`, no `GGML_MOE_KEEP_TOPK_UPDOWN`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, gate trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: promote only if `eval_tok_s > 2.3`, RAM including page cache stays `<=16000000000`, France output is semantically correct and coherent under manual review, and TTFT does not exceed current top4 pushed rerun by more than `20%` (`39140.888549ms * 1.2 = 46969.066259ms`).
- `rollback`: If build fails, token rate does not exceed `2.3`, output correctness fails, RAM exceeds limit, or TTFT exceeds gate, revert source and clean rebuild; record rejected/unpromoted. If accepted, immediately write full reproduction evidence, commit/push source and records to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild and rerun from pushed commit before promotion.
- `required_evidence`: source diff, exact env/command, source commit/status, binary sha256/build line, model stat, stdout/stderr, summary.json, gate trace, cgroup `memory.*`, France answer text, manual correctness note, and explicit promoted/rejected/rollback status.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T230406Z-20260702_expert_keep_mixed_up3_down4/france-cpu40-vram0gb`
- `result`: rejected despite speed improvement. `eval_tok_s=2.5`, `prompt_tok_s=0.9`, `TTFT=38796.748248ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15009513472`, `pgmajfault=379585`, `workingset_refault_file=7266975`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true` by heuristic but manual review failed.
- `correctness_manual_review`: fail. The answer is mostly factual but too long for the short-paragraph prompt and ends with an incomplete trailing phrase: `The country is also known`. This is a coherence/truncation failure, so UP=3/DOWN=4 cannot be accepted.
- `trace_summary`: `rows=47883`, `src0_ms_sum=29003.06`, `kernel_ms_sum=515.865`, `dontneed_ms_sum=1473.968`, `total_ms_sum=31937.225`.
- `reproduction_record`: rejected run directory contains source commit/status/diff, binary sha256/stat/version/build line, model stat, runner sha256, exact command/env, stdout/stderr, summary.json, cgroup memory files, gate trace, manual correctness review, and rejected status.
- `rollback_status`: source patch still retained only to run the symmetric DOWN=3/UP=4 probe below; if that also fails, revert source and clean rebuild. Current effective SOTA remains top4 `2.3 tok/s`.

### 当前执行 attempt：expert-keep-mixed-up4-down3

- `attempt_id`: `20260702-expert-keep-mixed-up4-down3`
- `attempt_kind`: `source-probe/approximate-pruning`
- `status`: completed / rejected_manual_correctness_truncated_output
- `hypothesis`: UP=3/DOWN=4 still had top3-like truncation. The symmetric policy UP=4/DOWN=3 preserves more up-path expert contribution while pruning the down path by one extra routed expert. If the quality failure is more sensitive to up pruning than down pruning, this may improve over top4 without causing the top3 truncation pattern.
- `theoretical_upper_bound`: Moving only down from top4 to top3 can ideally save about `36.2s / 6 ≈ 6.03s` before overhead/page effects. Expected gain is slightly smaller than UP=3/DOWN=4 but may preserve correctness better.
- `implementation`: Reuse the per-tensor env patch from UP=3/DOWN=4: `GGML_MOE_KEEP_TOPK_UP=4`, `GGML_MOE_KEEP_TOPK_DOWN=3`, no `GGML_MOE_KEEP_TOPK_UPDOWN`.
- `test_config`: patched source, `GGML_MOE_KEEP_TOPK_UP=4`, `GGML_MOE_KEEP_TOPK_DOWN=3`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, gate trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: promote only if `eval_tok_s > 2.3`, RAM including page cache stays `<=16000000000`, France output is semantically correct and coherent under manual review, and TTFT does not exceed current top4 pushed rerun by more than `20%` (`39140.888549ms * 1.2 = 46969.066259ms`).
- `rollback`: If build/run fails, token rate does not exceed `2.3`, output correctness fails, RAM exceeds limit, or TTFT exceeds gate, revert source and clean rebuild; record rejected/unpromoted. If accepted, immediately write full reproduction evidence, commit/push source and records, then clean rebuild/rerun from pushed commit before promotion.
- `required_evidence`: source diff, exact env/command, source commit/status, binary sha256/build line, model stat, stdout/stderr, summary.json, gate trace, cgroup `memory.*`, France answer text, manual correctness note, and explicit promoted/rejected/rollback status.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T230856Z-20260702_expert_keep_mixed_up4_down3/france-cpu40-vram0gb`
- `result`: rejected despite speed improvement. `eval_tok_s=2.5`, `prompt_tok_s=0.9`, `TTFT=38605.179576ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14995591168`, `pgmajfault=371743`, `workingset_refault_file=7406044`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true` by heuristic but manual review failed.
- `correctness_manual_review`: fail. The answer is mostly factual but too long for the short-paragraph prompt and ends with an incomplete trailing phrase: `The country is also known`. This is the same coherence/truncation failure as UP=3/DOWN=4.
- `trace_summary`: `rows=47883`, `src0_ms_sum=30217.351`, `kernel_ms_sum=510.942`, `dontneed_ms_sum=1470.558`, `total_ms_sum=33133.8`.
- `reproduction_record`: rejected run directory contains source commit/status/diff, binary sha256/stat/version/build line, model stat, runner sha256, exact command/env, stdout/stderr, summary.json, cgroup memory files, gate trace, manual correctness review, and rejected status.
- `rollback_status`: mixed top3/top4 branch closed. Both symmetric mixed probes improved token rate to `2.5` but failed manual correctness due the same incomplete answer ending. Revert per-tensor source patch and clean rebuild; current effective SOTA remains top4 `2.3 tok/s`.

### 当前执行 attempt：expert-keep-top3-early20-top4-rest

- `attempt_id`: `20260702-expert-keep-top3-early20-top4-rest`
- `attempt_kind`: `source-probe/layer-selective-approximate-pruning`
- `status`: completed / rejected_manual_correctness_truncated_repetitive_output
- `hypothesis`: Global top3 and up/down mixed top3 both improve speed but cause the France answer to become long and end incomplete. This suggests the approximation is too broad, not necessarily that every layer must stay top4. Keeping the current correct top4 policy as fallback while applying top3 only to earlier MoE layers (`blk.0-19`) may preserve later-layer output control/EOS behavior while saving part of the up/down fallback work.
- `theoretical_upper_bound`: Global top3 versus top4 roughly saves one routed expert out of six for both up/down across 40 CPU-MoE layers, with coarse upper bound about `12.9s`. Applying top3 to half the layers gives a rough additional bound around `6.4s` before overhead/page effects. Expected speed should land between top4 `2.3 tok/s` and rejected global/mixed top3 observations (`2.5-2.8 tok/s`).
- `implementation`: Add default-off layer-selective override in `ggml/src/ggml-cpu/ggml-cpu.c`. Existing `GGML_MOE_KEEP_TOPK_UPDOWN=4` remains the fallback. New env `GGML_MOE_KEEP_TOPK_LAYER_RANGE=0-19` plus `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3` overrides topK only for matching `blk.<layer>.ffn_up_exps` and `blk.<layer>.ffn_down_exps`; unset env preserves current top4 SOTA behavior.
- `test_config`: patched source, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=0-19`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, gate trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: promote only if `eval_tok_s > 2.3`, RAM including page cache stays `<=16000000000`, France output is semantically correct and coherent under manual review, and TTFT does not exceed current top4 pushed rerun by more than `20%` (`39140.888549ms * 1.2 = 46969.066259ms`).
- `rollback`: If build/run fails, token rate does not exceed `2.3`, output correctness fails, RAM exceeds limit, or TTFT exceeds gate, revert source and clean rebuild; record rejected/unpromoted. If accepted, immediately write full reproduction evidence, commit/push source and records, then clean rebuild/rerun from pushed commit before promotion.
- `required_evidence`: source diff, exact env/command, source commit/status, binary sha256/build line, model stat, stdout/stderr, summary.json, gate trace, cgroup `memory.*`, France answer text, manual correctness note, and explicit promoted/rejected/rollback status.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T231934Z-20260702_expert_keep_top3_early20_top4_rest/france-cpu40-vram0gb`
- `result`: rejected despite speed improvement. `eval_tok_s=2.5`, `prompt_tok_s=0.9`, `TTFT=38883.79964ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15005384704`, `pgmajfault=368596`, `workingset_refault_file=7321945`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true` by heuristic but manual review failed.
- `correctness_manual_review`: fail. The answer is mostly factual but becomes repetitive and ends with an incomplete trailing sentence: `The country is a`. Applying top3 to `blk.0-19` is still too broad.
- `trace_summary`: `rows=47887`, `src0_ms_sum=31528.676`, `kernel_ms_sum=518.32`, `dontneed_ms_sum=1477.302`, `total_ms_sum=34470.686`.
- `reproduction_record`: rejected run directory contains source commit/status/diff, binary sha256/stat/version/build line, model stat, runner sha256, exact command/env, stdout/stderr, summary.json, cgroup memory files, gate trace, manual correctness review, and rejected status.
- `rollback_status`: source patch retained only for the narrower early10 probe below. Current effective SOTA remains top4 `2.3 tok/s`.

### 当前执行 attempt：expert-keep-top3-early10-top4-rest

- `attempt_id`: `20260702-expert-keep-top3-early10-top4-rest`
- `attempt_kind`: `source-probe/layer-selective-approximate-pruning`
- `status`: completed / rejected_correctness_and_speed
- `hypothesis`: Top3 over early20 layers still fails output coherence, but a narrower early10 override may reduce enough up/down fallback work to beat top4 while preserving most of the correct top4 behavior. This tests whether there is a smaller safe approximation region before abandoning layer-selective top3.
- `theoretical_upper_bound`: Applying top3 to one quarter of the 40 CPU-MoE layers gives a rough additional bound of `12.9s / 4 ≈ 3.2s` versus top4 before overhead/page effects. The expected gain is modest; promote only if measured `eval_tok_s > 2.3` and manual correctness passes.
- `implementation`: Reuse the layer-selective source patch. Existing `GGML_MOE_KEEP_TOPK_UPDOWN=4` remains fallback; `GGML_MOE_KEEP_TOPK_LAYER_RANGE=0-9` and `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3` override only early layers.
- `test_config`: patched source, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=0-9`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, gate trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: promote only if `eval_tok_s > 2.3`, RAM including page cache stays `<=16000000000`, France output is semantically correct and coherent under manual review, and TTFT does not exceed current top4 pushed rerun by more than `20%` (`39140.888549ms * 1.2 = 46969.066259ms`).
- `rollback`: If build/run fails, token rate does not exceed `2.3`, output correctness fails, RAM exceeds limit, or TTFT exceeds gate, revert source and clean rebuild; record rejected/unpromoted. If accepted, immediately write full reproduction evidence, commit/push source and records, then clean rebuild/rerun from pushed commit before promotion.
- `required_evidence`: source diff, exact env/command, source commit/status, binary sha256/build line, model stat, stdout/stderr, summary.json, gate trace, cgroup `memory.*`, France answer text, manual correctness note, and explicit promoted/rejected/rollback status.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T232423Z-20260702_expert_keep_top3_early10_top4_rest/france-cpu40-vram0gb`
- `result`: rejected. `eval_tok_s=2.1`, `prompt_tok_s=0.9`, `TTFT=39796.829755ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14964908032`, `pgmajfault=443589`, `workingset_refault_file=9083375`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=false`, `correctness_reason=degenerate_text`.
- `correctness_manual_review`: fail. The initial English France paragraph is mostly correct, but the model continues into an unrelated Chinese detailed-answer section (`## 2. 详细解答` / `### 2.1 法国概况`), violating the required short coherent paragraph. It also does not beat top4 `2.3 tok/s`.
- `reproduction_record`: rejected run directory contains source commit/status/diff, binary sha256/stat/version/build line, model stat, runner sha256, exact command/env, stdout/stderr, summary.json, cgroup memory files, gate trace, manual correctness review, and rejected status.
- `rollback_status`: layer-selective top3 branch closed. Early20 improved speed but failed truncation/repetition; early10 failed both speed and correctness. Source patch reverted and clean rebuild completed. Current effective SOTA remains top4 `2.3 tok/s`.

### 当前执行 attempt：expert-keep-top4-n128

- `attempt_id`: `20260702-expert-keep-top4-n128`
- `attempt_kind`: `config-diagnostic/generation-budget`
- `status`: completed / rejected_incomplete_output_and_speed
- `hypothesis`: Current top4 SOTA is correct but uses the runner default `-n 192`, while the prompt asks for a short paragraph. A smaller generation budget (`-n 128`) may still produce a complete, coherent short paragraph and can show whether measured eval token rate is sensitive to overly long generation tails. This is a parameter diagnostic, not a structural model optimization.
- `theoretical_upper_bound`: Per-token compute should be unchanged from top4. Any rate difference comes from shorter decode length, less late-generation degeneration/repetition, and measurement variance. It should not be treated as a structural speedup unless the output is complete and a pushed rerun reproduces a stable improvement over `2.3 tok/s`.
- `test_config`: clean source, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, gate trace enabled, CLI args `-n 128 -c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: consider promotion only if `eval_tok_s > 2.3`, RAM including page cache stays `<=16000000000`, France output is semantically correct and complete/coherent as a short paragraph, and TTFT does not exceed current top4 pushed rerun by more than `20%` (`39140.888549ms * 1.2 = 46969.066259ms`). Because this changes generation budget, require an additional pushed-commit rerun and be conservative about declaring SOTA.
- `rollback`: No source change. If output is incomplete, too terse, semantically wrong, or `eval_tok_s <= 2.3`, record rejected/diagnostic and keep top4 `2.3 tok/s` SOTA.
- `required_evidence`: exact env/command, source commit/status, binary sha256/build line, model stat, stdout/stderr, summary.json, gate trace, cgroup `memory.*`, full France answer text, manual correctness note, and explicit promoted/rejected status.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T233312Z-20260702_expert_keep_top4_n128/france-cpu40-vram0gb`
- `result`: rejected. `eval_tok_s=2.2`, `prompt_tok_s=0.8`, `TTFT=40022.448209ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15009312768`, `pgmajfault=345481`, `workingset_refault_file=4308680`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true` by heuristic but manual review failed.
- `correctness_manual_review`: fail. The answer is semantically correct at the start but the smaller `-n 128` budget truncates the final sentence at `The country`. It also does not beat current top4 SOTA `2.3 tok/s`.
- `trace_summary`: `rows=32520`, `src0_ms_sum=25268.92`, `kernel_ms_sum=363.79`, `dontneed_ms_sum=1165.406`, `total_ms_sum=27449.704`.
- `reproduction_record`: rejected run directory contains source commit/status/diff, binary sha256/stat/version/build line, model stat, runner sha256, exact command/env, stdout/stderr, summary.json, cgroup memory files, gate trace, manual correctness review, and rejected status.
- `rollback_status`: no source change. Current effective SOTA remains top4 `2.3 tok/s`.

### 当前执行 attempt：expert-keep-top4-no-trace

- `attempt_id`: `20260702-expert-keep-top4-no-trace`
- `attempt_kind`: `config-diagnostic/trace-overhead`
- `status`: completed / diagnostic_rejected_tie
- `hypothesis`: Current top4 SOTA measurements include per-expert `one_trace.csv` writes. Earlier clean-baseline no-trace testing did not help at `1.6 tok/s`, but top4 changes output length and expert activity. Removing trace can check whether the accepted `2.3 tok/s` line is artificially low due trace overhead.
- `theoretical_upper_bound`: Trace writes one row per streamed gate expert. Current top4 traced runs have about `~41k` rows and `~30s` traced stream time; removing file writes should at most save a small fixed overhead unless trace I/O increases cgroup file-cache pressure. Treat any improvement as diagnostic unless a follow-up accepted evidence run can preserve enough reproduction trace.
- `test_config`: clean source, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, no `GGML_MOE_STREAM_ONE_TRACE_OUT`, cold `drop_caches`, strict 16GB cgroup, France prompt, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: diagnostic only unless `eval_tok_s > 2.3`, RAM including page cache stays `<=16000000000`, France output is semantically correct and coherent, TTFT stays within gate, and a follow-up trace/evidence strategy is defined. If no improvement, close trace-overhead direction for top4.
- `rollback`: No source change. If output fails, RAM/TTFT fails, or `eval_tok_s <= 2.3`, record rejected/diagnostic and keep top4 `2.3 tok/s` SOTA.
- `required_evidence`: exact env/command proving trace disabled, source commit/status, binary sha256/build line, model stat, stdout/stderr, summary.json, cgroup `memory.*`, full France answer text, manual correctness note, and explicit diagnostic result.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260701T234100Z-20260702_expert_keep_top4_no_trace/france-cpu40-vram0gb`
- `result`: completed and not promoted. `eval_tok_s=2.3`, `prompt_tok_s=0.8`, `TTFT=40557.47877ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15003897856`, `pgmajfault=398324`, `workingset_refault_file=6398679`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `correctness_manual_review`: pass. France answer is semantically correct and coherent, matching the accepted top4 style.
- `trace_status`: disabled as intended; `one_trace.csv` is absent and `GGML_MOE_STREAM_ONE_TRACE_OUT` is absent from `environment.txt`.
- `gap_analysis`: Removing per-expert trace did not improve token rate beyond current SOTA and TTFT was slightly worse than top4 pushed rerun. Trace write overhead is not hiding a higher top4 cold-start SOTA.
- `rollback_status`: no source change. Current effective SOTA remains top4 `2.3 tok/s`.

### 当前执行 attempt：old-config-cpu-chunk-trace-forensic

- `attempt_id`: `20260702-old-config-cpu-chunk-trace-forensic`
- `attempt_kind`: `forensic-diagnostic/cpu-fallback-wall`
- `status`: completed / diagnostic_not_sota
- `hypothesis`: The old `2.6 tok/s` run and slow clean reruns have the same gate stream rows/hits/misses (`34753/29624/5129`), while trace span differs sharply (`~71s` old versus `~113s` slow) and only about `3.2s` of that gap is explained by gate `src0_ms`. The missing time is likely in unstreamed CPU fallback (`ffn_up_exps` / `ffn_down_exps`), cgroup reclaim stalls around those CPU reads, or a lost binary/source behavior in that path.
- `theoretical_upper_bound`: If CPU fallback/reclaim accounts for most of the extra `~42s` trace-span gap, fixing that path could recover a large fraction of the historical `2.6 tok/s` observation without changing model quality. If CPU fallback trace shows only a small gap, the remaining delta is likely external IO/page-cache state or lost binary/shared-object state.
- `test_config`: current pushed source HEAD, clean rebuilt binary, old 2.6 command shape (`-n 192 -c 512 -b 64 -ub 64 -t 20 -tb 20`, `cpu_moe=40`, `GGML_MOE_VRAM_CACHE_GB=12`, no `GGML_MOE_KEEP_TOPK_UPDOWN`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`), cold `drop_caches`, strict 16GB cgroup, France prompt, gate trace plus `GGML_MOE_CPU_CHUNK_TRACE_OUT`.
- `acceptance_gate`: diagnostic only. It cannot promote a SOTA because CPU chunk trace adds file-write/timing overhead. The run must still keep RAM within 16GB and produce a semantically correct/coherent France answer to be comparable.
- `rollback`: No source change. If CPU chunk trace overhead makes the run unusably slow, terminate and record as failed diagnostic; current accepted SOTA remains top4 `2.3 tok/s`.
- `required_evidence`: exact env/command, source commit/status, binary sha256/build line, model stat, stdout/stderr, summary.json, cgroup memory files, gate `one_trace.csv`, CPU chunk trace, top tensor/layer wall-time aggregation, correctness output, and explicit conclusion about where the `2.6` gap moved.
- `run_dir_v1_cpu_only`: `/root/lfz/runs/vendor-ds4-16gb/20260701T235124Z-20260702_old_config_cpu_chunk_trace_forensic/france-cpu40-vram12gb`
- `run_dir_v2_gate_cpu`: `/root/lfz/runs/vendor-ds4-16gb/20260701T235508Z-20260702_old_config_cpu_gate_trace_forensic_v2/france-cpu40-vram12gb`
- `result_v2`: completed and not promoted. `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=45769.753411ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14964662272`, `pgmajfault=592341`, `workingset_refault_file=12362483`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `correctness_manual_review`: pass. France answer is semantically correct and coherent; the misspelling `Rennowned` does not affect semantic correctness.
- `gate_trace_comparison`: v2 exactly matches the old `2.6` gate stream shape: `rows=34753`, `cache_hits=29624`, `cache_inserts/misses=5129`. However old `2.6` gate trace span was `71223.061ms`, while v2 span is `110024.491ms`. Gate `src0_ms` explains only about `3141.851ms` of the gap (`23564.133ms -> 26705.984ms`), and gate `total_ms` only about `3263.396ms` (`25786.372ms -> 29049.768ms`).
- `cpu_chunk_trace_summary`: CPU chunk trace is limit-capped and only valid for distribution, not wall-clock. v2 captured `150000` CPU fallback chunks with partial summed thread time `114539.541ms`: `ffn_down_exps=60788.91ms`, `ffn_up_exps=53750.631ms`. Top early layers by partial time include `blk.1`, `blk.0`, `blk.9`, `blk.2`, `blk.3`, `blk.11`, `blk.4`, `blk.12`, `blk.8`, `blk.10`, `blk.6`, `blk.7`, and `blk.5`.
- `gap_analysis`: The unreproduced `2.6` delta is not explained by gate stream routing/cache policy: old and v2 have identical gate rows/hits/misses. Most of the span gap sits outside gate stream trace and aligns with unstreamed up/down CPU fallback and/or cgroup file-page reclaim stalls around that fallback. This strengthens the lost-binary/source or CPU fallback IO/reclaim hypothesis.
- `rollback_status`: no source change. Current accepted SOTA remains top4 `2.3 tok/s`. This diagnostic result is pushed only as evidence, not as a promoted SOTA.

### 当前执行 attempt：lost-cpu-fallback-source-binary-search

- `attempt_id`: `20260702-lost-cpu-fallback-source-binary-search`
- `attempt_kind`: `forensic-search`
- `status`: completed / no_exact_binary_found
- `hypothesis`: Since old `2.6` and current slow old-config reruns have identical gate rows/hits/misses, the missing speed may come from an unrecorded CPU fallback source/binary/shared-library state affecting `ffn_up_exps` / `ffn_down_exps`, or from a system IO/reclaim state not captured by run metadata. The next highest-value check is to search for preserved binaries, build directories, reflog states, patch files, temporary source copies, and shared libraries around the 09:55 UTC old run.
- `expected_delta`: No direct token-rate improvement. Success means finding an exact candidate binary/source state that can be rerun under the strict 16GB cgroup. If a candidate reproduces a valid rate above current SOTA with correct output and TTFT gate, immediately follow the SOTA record/push/rerun protocol.
- `search_scope`: repo reflog and branch history; `/root/lfz/vendor` and `/root/lfz` build/source copies; CMake build artifacts with mtimes around `2026-07-01 09:55 UTC`; `llama-cli`, `libggml*.so`, `ggml-cpu.c`, `moe_stream.cu` copies; `.Agent` patch/progress records; run dirs with matching `b9079-7353439ea` build line; shell history if available.
- `acceptance_gate`: Search only. Do not promote any result unless a candidate is rebuilt/rerun from recorded source or a preserved binary is tied to full source evidence and then converted into a pushed source commit.
- `rollback`: No source change. Do not delete files or reset branches. Any checkout/build probe must return to current clean HEAD if rejected.
- `required_evidence`: search commands, candidate file paths, mtimes/sizes/sha256, build lines, git commits/diffs, and a conclusion separating lost binary/source evidence from external IO/page-cache evidence.
- `finding_reflog`: The old `2.6` run sits between `fcc937d01` (`09:55:13`, vram8 record) and `f7f9cdc48` (`09:59:19`, vram12 record). Earlier key source commits were `c5337b0cc` (`09:42:57`, src1 row mapping fix) and `7353439ea` (`09:10:10`, rejected stream test).
- `finding_binaries`: Search for `llama-cli` / `libggml*.so` / archives with mtimes around `2026-07-01 09:00-10:30 UTC` found no preserved vendor binary/shared library. The only hit was an `ik_llama_migrated` build, which is not valid for vendor SOTA. Existing vendor build output has been overwritten by later rebuilds.
- `finding_source_copies`: Search for `ggml-cpu.c`, `moe_stream.cu`, patch and diff files around `08:00-12:30 UTC` found no 09:55 vendor source copy/diff. `.Agent` records around 09:55 do not preserve the exact dirty source state.
- `finding_build_line`: `build : b9079-7353439ea` appears across many run dirs from `09:16` through later reruns, including incompatible outputs. It is stale/incomplete and cannot identify a unique binary/source state.
- `finding_sequence_signal`: 09:50 `vram8` was slow (`eval_tok_s=1.4`, `pgmajfault=574410`, `workingset_refault_file=15540325`), then 09:55 `vram12` became fast (`eval_tok_s=2.6`, `pgmajfault=309216`, `workingset_refault_file=3371495`). This points to a possible run-sequence/kernel workingset/device-cache effect despite cold `drop_caches`, not a preserved binary that can be directly rerun.
- `conclusion`: No exact old vendor binary/source artifact was found. Continue by testing whether the old vram8 -> vram12 sequence itself reduces refaults and recovers the 2.6-like behavior under strict 16GB cgroups.

### 当前执行 attempt：old-sequence-vram8-then-vram12-repro

- `attempt_id`: `20260702-old-sequence-vram8-then-vram12-repro`
- `attempt_kind`: `forensic-execution/run-sequence`
- `status`: completed / rejected_ram_gate
- `hypothesis`: The historical 2.6 run may have depended on immediately preceding vram8/vram4 runs creating useful kernel workingset shadow entries, device cache state, or readahead behavior that survives `drop_caches` enough to reduce refault/reclaim cost. Replaying the old sequence (`vram8` cold run followed by `vram12` cold run) with the current clean pushed source can test whether sequence state, rather than source, explains the low `workingset_refault_file` in the 09:55 result.
- `expected_delta`: If sequence state is the cause, the second vram12 run should show materially lower `pgmajfault`/`workingset_refault_file` than standalone slow reruns and token rate should move toward historical `2.6`. If it remains `1.5-1.6`, the sequence hypothesis is weak and remaining explanation is lost binary/source or unobserved system/storage state.
- `test_config`: current clean pushed source and rebuilt binary, old command shape, no `GGML_MOE_KEEP_TOPK_UPDOWN`, `cpu_moe=40`, first run `vram_cache=8GB`, second run `vram_cache=12GB`, each with cold `drop_caches`, strict 16GB cgroup, France prompt, gate trace enabled.
- `acceptance_gate`: forensic only unless second run exceeds current accepted SOTA and also passes RAM, TTFT, correctness, full reproduction, push, and clean pushed-commit rerun gates.
- `rollback`: No source change. If either run fails or exceeds runtime/memory limits, record failure and keep current top4 `2.3 tok/s` SOTA.
- `required_evidence`: both run dirs, exact commands/env, cgroup memory files, summaries, gate trace summaries, France outputs, manual correctness, and comparison to the 09:50/09:55 historical pair.
- `vram8_run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T000300Z-20260702_sequence_vram8_prelude/france-cpu40-vram8gb`
- `vram8_result`: completed. `eval_tok_s=1.4`, `prompt_tok_s=0.7`, `TTFT=47551.568213ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14977544192`, `pgmajfault=628591`, `workingset_refault_file=15251384`, `ram_ok=true`, `correctness_ok=true`.
- `vram8_gate_trace`: `rows=34753`, `hits=27643`, `misses=7110`, `span_ms=123427.823`, `src0_ms=36141.13`, `total_ms=38888.795`. This reproduces the slow vram8 shape from the historical prelude.
- `vram12_run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T000554Z-20260702_sequence_vram12_after_vram8/france-cpu40-vram12gb`
- `vram12_result`: rejected. `eval_tok_s=0.0`, `prompt_tok_s=0.0`, `TTFT=None`, `exit_status=143`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15102951424`, `file_mapped=12557213696`, `pgmajfault=213971`, `workingset_refault_file=44767`, `ram_ok=false`, `ram_limit_killed=true`, `correctness_ok=false`.
- `vram12_partial_gate_trace`: killed early after `rows=2113`, `hits=556`, `misses=1557`, `span_ms=22826.206`, `src0_ms=7947.844`, `total_ms=8345.008`.
- `gap_analysis`: The vram8 -> vram12 sequence does change kernel file-page state: vram12 refaults are extremely low before kill, and `active_file` is high. However that state also leaves `file_mapped≈12.56GB` and `memory_file≈15.10GB`, causing the strict 16GB RAM gate to terminate the run before any answer. Therefore this sequence cannot be accepted as a cold-start SOTA and does not reproduce the historical `2.6 tok/s` under current strict accounting.
- `rollback_status`: no source change. Current accepted SOTA remains top4 `2.3 tok/s`. Sequence-induced low-refault state is useful evidence but is non-compliant unless future work can keep the same low refault behavior while staying below the 16GB RAM gate and producing a correct answer.

### 当前执行 attempt：cpu-fallback-dontneed-top4

- `attempt_id`: `20260702-cpu-fallback-dontneed-top4`
- `attempt_kind`: `source-probe/cgroup-file-cache-pressure`
- `status`: completed / rejected_perf_regression
- `hypothesis`: Gate stream is not the remaining bottleneck for the historical `2.6` gap; diagnostics point to unstreamed `ffn_up_exps` / `ffn_down_exps` CPU fallback and cgroup file-page reclaim. Adding a default-off `MADV_DONTNEED` after each unstreamed CPU fallback expert finishes may reduce file-cache pressure, lower `memory.max` reclaim churn, and avoid the low-refault-but-over-RAM state observed in the sequence run.
- `theoretical_upper_bound`: CPU fallback trace distribution captured `up+down` as the dominant non-gate work. If reclaim stalls around up/down account for tens of seconds of the old-config span gap, the hard upside could approach part of that gap. Direct advice overhead is one syscall per nonzero unstreamed expert; if the same expert is reused soon, the optimization can regress by forcing refaults. Promote only on measured token-rate improvement over current top4 `2.3 tok/s` with RAM/correctness/TTFT gates passing.
- `implementation`: Add default-off env `GGML_MOE_CPU_DONTNEED_AFTER_EXPERT=1` in `ggml/src/ggml-cpu/ggml-cpu.c`. For tensors whose name contains `ffn_up_exps` or `ffn_down_exps`, after all CPU chunks for one `cur_a` expert finish, thread `ith==0` calls a page-aligned `madvise(..., MADV_DONTNEED)` for that expert's `src0` page range. Unset env preserves current behavior.
- `test_config`: patched source, current top4 SOTA config (`GGML_MOE_KEEP_TOPK_UPDOWN=4`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`), cold `drop_caches`, strict 16GB cgroup, France prompt, gate trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`, plus `GGML_MOE_CPU_DONTNEED_AFTER_EXPERT=1`.
- `acceptance_gate`: promote only if `eval_tok_s > 2.3`, RAM including page cache stays `<=16000000000` with no runner kill/OOM, France output passes manual semantic/coherence review, and TTFT does not exceed current top4 pushed rerun by more than `20%` (`39140.888549ms * 1.2 = 46969.066259ms`).
- `rollback`: If build fails, token rate does not improve, correctness fails, RAM gate fails, or TTFT gate fails, revert source and clean rebuild; record rejected. If accepted, immediately write complete reproduction info, commit/push source to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before promotion.
- `required_evidence`: source diff, build log/version, exact env/command, source commit/status, binary sha256/build line, model stat, stdout/stderr, summary.json, gate trace, cgroup memory files, France answer text, manual correctness note, and explicit promoted/rejected/rollback status.
- `run_dir_initial`: `/root/lfz/runs/vendor-ds4-16gb/20260702T001516Z-20260702_cpu_fallback_dontneed_top4/france-cpu40-vram0gb`
- `result_initial`: rejected. `eval_tok_s=2.0`, `prompt_tok_s=0.8`, `TTFT=39546.664313ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14987567104`, `pgmajfault=381983`, `workingset_refault_file=8768343`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `run_dir_traced`: `/root/lfz/runs/vendor-ds4-16gb/20260702T001818Z-20260702_cpu_fallback_dontneed_top4_trace/france-cpu40-vram0gb`
- `result_traced`: rejected. `eval_tok_s=2.0`, `prompt_tok_s=0.8`, `TTFT=40724.90508ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14995689472`, `pgmajfault=392356`, `workingset_refault_file=8862317`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `correctness_manual_review`: pass for both runs. France answer is semantically correct and coherent, matching the accepted top4 content pattern.
- `trace_summary`: traced confirmation produced `rows=41400`, `cache_hits=35750`, `cache_misses=5650`, `span_ms=98958.873`, `src0_ms=25024.439`, `total_ms=27518.138`.
- `gap_analysis`: The DONTNEED probe lowered traced gate `src0_ms` versus the pushed top4 reference, but end-to-end token rate regressed from `2.3` to `2.0` and file refaults increased versus top4 (`~6.67M -> ~8.86M`). The extra barriers and dropping up/down pages cause more costly refault/reload behavior than they save in reclaim pressure.
- `rollback_status`: source patch reverted and clean rebuild completed; binary reports `version: 9110 (d42453b2e)`. Current accepted SOTA remains top4 `2.3 tok/s`.

### 当前执行 attempt：expert-keep-top3-late20-top4-rest

- `attempt_id`: `20260702-expert-keep-top3-late20-top4-rest`
- `attempt_kind`: `source-probe/layer-selective-approximate-pruning`
- `status`: promoted after pushed-source clean rebuild rerun
- `hypothesis`: Global top3 and early-layer top3 are too aggressive for France output quality, but the failure mode may be layer-location dependent. Keeping layers `0-19` at the accepted top4 policy while pruning only later CPU-MoE layers `20-39` to top3 may save roughly half of the top4->top3 up/down fallback work while preserving enough early-layer representation quality for a coherent short France paragraph.
- `theoretical_upper_bound`: Global top3 versus top4 has a rough additional bound of about `12.9s` wall reduction before overhead/page effects. Applying top3 to the later half of 40 CPU-MoE layers gives a coarse bound near `6.4s`, so token rate could land in the `2.4-2.6 tok/s` range if correctness holds. Real gain is bounded by gate streaming, non-expert work, output length, and layer imbalance.
- `implementation`: Reintroduce a default-off layer-selective override in `ggml/src/ggml-cpu/ggml-cpu.c`: `GGML_MOE_KEEP_TOPK_LAYER_RANGE=20-39` and `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3` override `GGML_MOE_KEEP_TOPK_UPDOWN=4` only for matching `blk.<layer>.ffn_up_exps` and `blk.<layer>.ffn_down_exps`. Unset env preserves current top4 SOTA behavior.
- `test_config`: patched source, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=20-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, gate trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: promote only if `eval_tok_s > 2.3`, RAM including page cache stays `<=16000000000`, France answer is semantically correct and coherent under manual review, and TTFT does not exceed current top4 pushed rerun by more than `20%` (`39140.888549ms * 1.2 = 46969.066259ms`).
- `rollback`: If build/run fails, token rate does not exceed `2.3`, output correctness fails, RAM exceeds limit, or TTFT exceeds gate, revert source and clean rebuild; record rejected. If accepted, immediately write full reproduction evidence, commit/push source and records to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before promotion.
- `required_evidence`: source diff, exact env/command, source commit/status, binary sha256/build line, model stat, stdout/stderr, summary.json, gate trace, cgroup `memory.*`, France answer text, manual correctness note, and explicit promoted/rejected/rollback status.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T002611Z-20260702_expert_keep_top3_late20_top4_rest/france-cpu40-vram0gb`
- `candidate_result`: `eval_tok_s=2.4`, `prompt_tok_s=0.9`, `TTFT=38334.961869ms`, `elapsed_seconds=106.11`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15013195776`, `pgmajfault=361605`, `workingset_refault_file=6074968`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `correctness_manual_review`: pass. France answer is semantically correct, coherent, and complete; no truncation, repetition, or semantic degradation observed from late20 top3 pruning.
- `trace_summary`: `rows=43086`, `cache_hits=37322`, `cache_misses=5764`, `span_ms=86687.225`, `src0_ms=28525.819`, `total_ms=31257.197`.
- `sota_gate`: candidate passes initial gates. Token rate improves current pushed top4 SOTA `2.3 -> 2.4`; TTFT `38334.961869ms` is below the `46969.066259ms` limit; RAM including page cache is capped at `16000000000`; France correctness passes manual review.
- `reproduction_record`: candidate run directory contains precommit source state/diff, binary sha256/stat/version, model stat, runner sha256, exact command/env, stdout/stderr, summary.json, cgroup memory files, gate trace, trace summary, plan snapshot, push target, manual correctness review, and candidate status.
- `publish_status`: source and plan committed and pushed to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`; binary was clean-rebuilt from pushed commit and rerun passed all gates.
- `pushed_commit`: `63caf68ba05b65120516ed03d4cd2c7b8dbc97d1` (`vendor-ds4: add late20 top3 pruning candidate`).
- `pushed_branch`: `vendor/deepseek-token-rate-16gb` on remote `ssd=https://github.com/wici-ai/ssd-llama.git`.
- `pushed_rerun_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T003114Z-20260702_expert_keep_top3_late20_top4_rest_pushed_rerun/france-cpu40-vram0gb`.
- `pushed_rerun_result`: `eval_tok_s=2.4`, `prompt_tok_s=0.9`, `TTFT=38355.541393ms`, `elapsed_seconds=106.66`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15014518784`, `pgmajfault=358113`, `workingset_refault_file=6138202`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `pushed_rerun_trace_summary`: `rows=43086`, `cache_hits=37322`, `cache_misses=5764`, `span_ms=87314.252`, `src0_ms=28802.35`, `total_ms=31509.987`.
- `pushed_rerun_binary`: `llama-cli --version` reports `version: 9113 (63caf68ba)` after clean rebuild.
- `pushed_rerun_correctness_manual_review`: pass. France answer is semantically correct, coherent, and complete, with no truncation, repetition, or semantic degradation.
- `current_effective_sota`: `2.4 tok/s` under strict 16GB cgroup from pushed source. This replaces top4 `2.3 tok/s` as the current source-backed line; old `2.6 tok/s` remains a forensic target until exact source/binary state is recovered.

### 当前执行 attempt：expert-keep-top3-late15-top4-rest

- `attempt_id`: `20260702-expert-keep-top3-late15-top4-rest`
- `attempt_kind`: `config-probe/layer-selective-approximate-pruning`
- `status`: completed / rejected_tie_not_sota
- `hypothesis`: Late20 top3 (`blk.20-39`) is the current promoted SOTA and preserves France correctness. Expanding the top3 override to `blk.15-39` prunes five additional CPU-MoE layers while keeping the earliest 15 layers at top4, which may capture more up/down fallback savings without triggering the early-layer quality failures seen in early10/early20 top3.
- `theoretical_upper_bound`: Late20 covers 20 of 40 CPU-MoE layers and reached `2.4 tok/s`. Expanding to 25 layers adds one eighth of the total top4->top3 pruning region. Using the earlier coarse `12.9s` global top3-vs-top4 bound, the incremental upper bound over late20 is about `12.9s * 5/40 ≈ 1.6s` before overhead and layer imbalance. Expected gain is modest; correctness risk is higher than late20.
- `test_config`: pushed source with layer-range support, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=15-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, gate trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: promote only if `eval_tok_s > 2.4`, RAM including page cache stays `<=16000000000`, France answer is semantically correct and coherent under manual review, and TTFT does not exceed current late20 pushed rerun by more than `20%` (`38355.541393ms * 1.2 = 46026.649672ms`).
- `rollback`: No source change is required for this config probe. If token rate does not exceed `2.4`, output correctness fails, RAM exceeds limit, or TTFT exceeds gate, record rejected and keep late20 `2.4 tok/s` SOTA. If accepted, immediately write full reproduction evidence, commit/push records to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before promotion.
- `required_evidence`: exact env/command, source commit/status, binary sha256/build line, model stat, stdout/stderr, summary.json, gate trace, cgroup `memory.*`, France answer text, manual correctness note, and explicit promoted/rejected status.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T003924Z-20260702_expert_keep_top3_late15_top4_rest/france-cpu40-vram0gb`
- `result`: completed and not promoted. `eval_tok_s=2.4`, `prompt_tok_s=0.9`, `TTFT=38777.397548ms`, `elapsed_seconds=96.81`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15028047872`, `pgmajfault=318905`, `workingset_refault_file=4609690`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `correctness_manual_review`: pass. France answer is semantically correct, coherent, and complete, with no truncation or repetition.
- `trace_summary`: `rows=37321`, `cache_hits=31970`, `cache_misses=5351`, `span_ms=76904.096`, `src0_ms=26079.662`, `total_ms=28484.321`.
- `gap_analysis`: Expanding top3 to `blk.15-39` kept output quality and reduced trace rows versus late20, but rounded token rate only tied current SOTA (`2.4`) and did not exceed the promote gate. It is therefore useful quality evidence for a broader late-layer pruning region, but not a new SOTA.
- `rollback_status`: no source change. Current accepted SOTA remains late20 top3 `2.4 tok/s`.

### 当前执行 attempt：expert-keep-top3-late10-top4-rest

- `attempt_id`: `20260702-expert-keep-top3-late10-top4-rest`
- `attempt_kind`: `config-probe/layer-selective-approximate-pruning`
- `status`: promoted_after_pushed_commit_clean_rebuild_rerun
- `hypothesis`: Late15 top3 tied the current `2.4 tok/s` SOTA while preserving France output quality. Expanding the top3 override to `blk.10-39` prunes ten more layers than the promoted late20 run and five more than late15, while still preserving the first ten CPU-MoE layers at top4. This may cross the rounded token-rate boundary above `2.4`, but correctness risk increases because prior broad early-layer pruning caused truncation/repetition.
- `theoretical_upper_bound`: Relative to late20, late10 adds ten of forty CPU-MoE layers to the top3 region. Using the coarse `12.9s` global top3-vs-top4 bound, the incremental upper bound over late20 is about `12.9s * 10/40 ≈ 3.2s` before overhead and layer imbalance. If output length remains similar, measured rate could move toward `2.5`, but only manual correctness can validate the approximation.
- `test_config`: pushed source with layer-range support, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, gate trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: promote only if `eval_tok_s > 2.4`, RAM including page cache stays `<=16000000000`, France answer is semantically correct and coherent under manual review, and TTFT does not exceed current late20 pushed rerun by more than `20%` (`38355.541393ms * 1.2 = 46026.649672ms`).
- `rollback`: No source change is required for this config probe. If token rate does not exceed `2.4`, output correctness fails, RAM exceeds limit, or TTFT exceeds gate, record rejected and keep late20 `2.4 tok/s` SOTA. If accepted, immediately write full reproduction evidence, commit/push records to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before promotion.
- `required_evidence`: exact env/command, source commit/status, binary sha256/build line, model stat, stdout/stderr, summary.json, gate trace, cgroup `memory.*`, France answer text, manual correctness note, and explicit promoted/rejected status.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T004558Z-20260702_expert_keep_top3_late10_top4_rest/france-cpu40-vram0gb`
- `candidate_result`: `eval_tok_s=2.6`, `prompt_tok_s=0.9`, `TTFT=38024.422319ms`, `elapsed_seconds=89.62`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15011241984`, `pgmajfault=285455`, `workingset_refault_file=3518026`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `correctness_manual_review`: pass. France answer is semantically correct, coherent, and complete; no truncation, repetition, or semantic degradation observed from late10 top3 pruning.
- `trace_summary`: `rows=35151`, `cache_hits=30180`, `cache_misses=4971`, `span_ms=70038.773`, `src0_ms=25154.153`, `total_ms=27416.769`.
- `sota_gate`: candidate passes initial gates. Token rate improves current pushed late20 SOTA `2.4 -> 2.6`; TTFT `38024.422319ms` is below the `46026.649672ms` limit; RAM including page cache is capped at `16000000000`; France correctness passes manual review.
- `reproduction_record`: candidate run directory contains precommit source state/diff, binary sha256/stat/version, model stat, runner sha256, exact command/env, stdout/stderr, summary.json, cgroup memory files, gate trace, trace summary, plan snapshot, push target, manual correctness review, and candidate status.
- `candidate_publish_commit`: `19a0d36cd081feb0bfb90e26380005c74abe0151` (`vendor-ds4: record late10 top3 pruning candidate`), pushed to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`.
- `pushed_clean_rebuild`: completed from commit `19a0d36cd081feb0bfb90e26380005c74abe0151`; `llama-cli --version` reports `version: 9118 (19a0d36cd)`, binary sha256 `c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`.
- `pushed_rerun_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T005156Z-20260702_expert_keep_top3_late10_top4_rest_pushed_rerun/france-cpu40-vram0gb`.
- `pushed_rerun_result`: `eval_tok_s=2.6`, `prompt_tok_s=0.9`, `TTFT=37874.580124ms`, `elapsed_seconds=89.48`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15002906624`, `pgmajfault=288661`, `workingset_refault_file=3545116`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `pushed_rerun_trace_summary`: `rows=35151`, `cache_hits=30180`, `cache_misses=4971`, `span_ms=69920.848`, `src0_ms=24305.283`, `total_ms=26552.328`.
- `pushed_rerun_correctness_manual_review`: pass. France answer is semantically correct, coherent, complete, and not repetitive/truncated.
- `promotion_decision`: promoted as current accepted cold-start SOTA because pushed-commit clean rebuild rerun improved the previous late20 `2.4 tok/s` SOTA to `2.6 tok/s`, kept host RAM including page cache at the 16GB cgroup limit, and kept TTFT below the prior SOTA `+20%` gate (`38355.541393ms * 1.2 = 46026.649672ms`).
- `reproduction_guardrail`: this promotion is only valid because the run directory now contains source state, pushed remote/branch, binary version/stat/sha256, runner sha256, model stat, exact command/env, stdout/stderr, summary.json, cgroup memory files, `one_trace.csv`, `trace_summary.json`, and manual correctness review. Any future new SOTA must repeat this full record + immediate source push + pushed-commit clean rebuild rerun sequence before being called accepted.
- `publish_status`: promoted; this plan update itself must be committed and pushed to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb` as the promotion record.

### 当前执行 attempt：expert-keep-top3-late5-top4-rest

- `attempt_id`: `20260702-expert-keep-top3-late5-top4-rest`
- `attempt_kind`: `config-probe/layer-selective-approximate-pruning`
- `status`: completed_rejected_not_sota
- `bottleneck_basis`: Current accepted late10 SOTA trace still spends `src0_ms=24305.283ms` out of `total_ms=26552.328ms` in streamed gate expert source loading, while the late20->late10 expansion reduced trace rows, cache misses, major faults, and end-to-end wall time without harming France correctness. The next highest-leverage low-risk probe is therefore a small additional layer-range expansion before changing IO/prefetch code.
- `hypothesis`: Expanding the top3 override from `blk.10-39` to `blk.5-39` prunes five additional CPU-MoE layers while keeping the first five layers at top4. This may reduce up/down fallback work and indirectly reduce page/refault pressure enough to move beyond the current rounded `2.6 tok/s`; correctness risk is higher because this enters earlier representation layers.
- `theoretical_upper_bound`: Using the previous coarse global top4->top3 bound of about `12.9s`, the incremental bound for five more layers is `12.9s * 5/40 ≈ 1.6s` before overhead and layer imbalance. Since current pushed late10 elapsed time is `89.48s`, the optimistic wall-time floor for this isolated change is around `87.9s`; if output length stays similar this could only modestly improve token rate, likely toward `2.7 tok/s` rather than a large jump. The hard correctness gate dominates this probe.
- `test_config`: pushed source with layer-range support, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=5-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, gate trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: promote only if `eval_tok_s > 2.6`, RAM including page cache stays `<=16000000000`, France answer is semantically correct and coherent under manual review, and TTFT does not exceed current late10 pushed rerun by more than `20%` (`37874.580124ms * 1.2 = 45449.496149ms`).
- `rollback`: No source change is required for this config probe. If token rate does not exceed `2.6`, output correctness fails, RAM exceeds limit, or TTFT exceeds gate, record rejected and keep late10 `2.6 tok/s` SOTA. If accepted, immediately write full reproduction evidence, commit/push records to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before promotion.
- `required_evidence`: exact env/command, source commit/status, binary sha256/build line, model stat, stdout/stderr, summary.json, gate trace, cgroup `memory.*`, France answer text, manual correctness note, and explicit promoted/rejected status.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T005829Z-20260702_expert_keep_top3_late5_top4_rest/france-cpu40-vram0gb`.
- `result`: rejected. `eval_tok_s=2.5`, `prompt_tok_s=1.0`, `TTFT=37436.903163ms`, `elapsed_seconds=109.94`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14977576960`, `pgmajfault=342607`, `workingset_refault_file=6465998`, `ram_ok=true`, `ram_limit_killed=false`.
- `correctness_manual_review`: fail. The answer is broadly about France, but it is cut off mid-word at `unitary semi-pres`, so it is not a complete coherent short paragraph.
- `trace_summary`: `rows=47876`, `cache_hits=41674`, `cache_misses=6202`, `span_ms=91059.789`, `src0_ms=30669.545`, `total_ms=33640.899`.
- `gap_analysis`: Moving the top3 boundary earlier from `blk.10-39` to `blk.5-39` changed the generation trajectory enough to increase streamed gate misses (`4971 -> 6202`), `src0_ms` (`24305.283 -> 30669.545`), refault pressure (`workingset_refault_file=6465998`), and end-to-end elapsed time (`89.48s -> 109.94s`). The additional approximate pruning does not translate to token-rate gain and also causes truncation at the fixed `-n 192` output limit. Keep late10 as the accepted SOTA and do not expand the top3 range before layer 10 without a narrower correctness/trajectory strategy.
- `rollback_status`: no source change. Current accepted SOTA remains late10 top3 `2.6 tok/s`.

### 当前执行 attempt：expert-keep-top3-late8-top4-rest

- `attempt_id`: `20260702-expert-keep-top3-late8-top4-rest`
- `attempt_kind`: `config-probe/layer-selective-approximate-pruning`
- `status`: completed_rejected_tie_not_sota
- `bottleneck_basis`: Late5 showed that pushing top3 too early changes the token trajectory and increases stream misses/refaults, while late10 is currently correct and fastest. The remaining useful question is whether only layers `8-9` can be added to the top3 region without triggering the late5 failure mode.
- `hypothesis`: Expanding the accepted late10 boundary only slightly to `blk.8-39` may capture a small amount of additional up/down fallback saving while keeping the first eight CPU-MoE layers at top4 for quality stability. Because late5 regressed, this is a boundary-finding probe, not a high-upside code change.
- `theoretical_upper_bound`: Relative to late10, adding two of forty CPU-MoE layers to the top3 region has a coarse upper bound of `12.9s * 2/40 ≈ 0.65s` wall time before overhead. With late10 elapsed `89.48s`, the expected token-rate gain is at most a small rounding move above `2.6 tok/s`; if cache misses or output length increase, it should be rejected.
- `test_config`: pushed source with layer-range support, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=8-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, gate trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: promote only if `eval_tok_s > 2.6`, RAM including page cache stays `<=16000000000`, France answer is semantically correct and coherent under manual review, and TTFT does not exceed current late10 pushed rerun by more than `20%` (`37874.580124ms * 1.2 = 45449.496149ms`).
- `rollback`: No source change is required. If token rate does not exceed `2.6`, output correctness fails, RAM exceeds limit, TTFT exceeds gate, or trace shows the late5 miss/refault regression pattern, record rejected and keep late10 as SOTA. If accepted, immediately write full reproduction evidence, commit/push records to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before promotion.
- `required_evidence`: exact env/command, source commit/status, binary sha256/build line, model stat, stdout/stderr, summary.json, gate trace, cgroup `memory.*`, France answer text, manual correctness note, trace summary, and explicit promoted/rejected status.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T010316Z-20260702_expert_keep_top3_late8_top4_rest/france-cpu40-vram0gb`.
- `result`: rejected tie. `eval_tok_s=2.6`, `prompt_tok_s=0.9`, `TTFT=38342.670365ms`, `elapsed_seconds=100.31`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15009902592`, `pgmajfault=321844`, `workingset_refault_file=4991972`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `correctness_manual_review`: pass. France answer is semantically correct, coherent, complete, and not repetitive/truncated.
- `trace_summary`: `rows=41643`, `cache_hits=36050`, `cache_misses=5593`, `span_ms=80803.695`, `src0_ms=28245.587`, `total_ms=30850.588`.
- `gap_analysis`: Late8 avoids the late5 truncation failure, but it still increases gate stream misses (`4971 -> 5593`), `src0_ms` (`24305.283 -> 28245.587`), refault pressure, and elapsed time relative to accepted late10. Since rounded token rate only ties `2.6` and does not exceed the current SOTA, the top3 boundary should remain at layer 10. Further layer-boundary expansion is deprioritized; the next optimization should target cache/IO behavior under the late10 config.
- `rollback_status`: no source change. Current accepted SOTA remains late10 top3 `2.6 tok/s`.

### 当前执行 attempt：late10-cache13700mib

- `attempt_id`: `20260702-late10-cache13700mib`
- `attempt_kind`: `config-probe/vram-cache-boundary`
- `status`: completed_rejected_tie_not_sota
- `bottleneck_basis`: Accepted late10 SOTA still has `cache_misses=4971` and `src0_ms=24305.283ms`; stderr reports `[moe_stream] VRAM cache: 13.2 GiB, 3192 slots (4.25 MiB each)` and CUDA free memory around `238 MiB`. Layer-boundary probes late8/late5 did not improve token rate, so the next isolated bottleneck probe is using a little more VRAM for the stream-one cache.
- `hypothesis`: Increasing `GGML_MOE_STREAM_ONE_CACHE_MIB` from `13568` to `13700` may add roughly `132 MiB / 4.25 MiB ≈ 31` cache slots while staying under the RTX 5090 free-memory cliff. This could reduce a small number of late10 gate cache misses without changing model math or correctness.
- `theoretical_upper_bound`: With current average `src0_ms / misses ≈ 24305.283 / 4971 ≈ 4.89ms`, even a perfect 31-miss reduction saves only about `0.15s` direct source-load time, plus possible secondary refault effects. Therefore expected token-rate gain is small and may not exceed rounded `2.6 tok/s`; the main value is confirming the practical VRAM boundary under strict 16GB host RAM.
- `test_config`: accepted late10 config, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13700`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, gate trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: promote only if `eval_tok_s > 2.6`, RAM including page cache stays `<=16000000000`, no CUDA OOM/cache insertion failure occurs, France answer is semantically correct and coherent under manual review, and TTFT does not exceed current late10 pushed rerun by more than `20%` (`37874.580124ms * 1.2 = 45449.496149ms`).
- `rollback`: No source change is required. If CUDA OOM, cache insertion failure, token rate not above `2.6`, output correctness failure, RAM failure, or TTFT failure occurs, record rejected and keep `13568MiB` late10 as SOTA. If accepted, immediately write full reproduction evidence, commit/push records to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before promotion.
- `required_evidence`: exact env/command, stderr cache size/slot/free-memory lines, source commit/status, binary sha256/build line, model stat, stdout/stderr, summary.json, gate trace, cgroup `memory.*`, France answer text, manual correctness note, trace summary, and explicit promoted/rejected status.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T010828Z-20260702_late10_cache13700mib/france-cpu40-vram0gb`.
- `result`: rejected tie. `eval_tok_s=2.6`, `prompt_tok_s=0.9`, `TTFT=36745.290043ms`, `elapsed_seconds=88.31`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15039311872`, `pgmajfault=297870`, `workingset_refault_file=3430319`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `cache_observation`: stderr reports `[moe_stream] VRAM cache: 13.4 GiB, 3223 slots (4.25 MiB each)`, CUDA free memory around `106 MiB`, and `hits=30208 misses=4943`.
- `correctness_manual_review`: pass. France answer is semantically correct, coherent, complete, and not repetitive/truncated.
- `trace_summary`: `rows=35151`, `cache_hits=30208`, `cache_misses=4943`, `span_ms=70040.626`, `src0_ms=24631.968`, `total_ms=26893.278`.
- `gap_analysis`: Increasing cache from `13568MiB` to `13700MiB` added 31 slots and reduced misses only `4971 -> 4943`. Direct traced `src0_ms` did not improve (`24305.283 -> 24631.968`), while rounded `eval_tok_s` tied current SOTA. The lower TTFT/elapsed likely reflects run-to-run page/reclaim variance rather than a strong token-rate gain. Because the token-rate gate is not exceeded, keep `13568MiB` as accepted SOTA; a final small VRAM boundary probe may test `13760MiB`, but expected gain is tiny and OOM risk is higher.
- `rollback_status`: no source change. Current accepted SOTA remains late10 top3 `2.6 tok/s` with `13568MiB` stream cache.

### 当前执行 attempt：late10-cache13760mib

- `attempt_id`: `20260702-late10-cache13760mib`
- `attempt_kind`: `config-probe/vram-cache-boundary`
- `status`: completed_rejected_tie_not_sota
- `bottleneck_basis`: `13700MiB` ran with only about `106 MiB` CUDA free and produced 3223 slots, but did not improve rounded token rate. To satisfy the “use VRAM as much as practical” constraint, run one final small boundary probe before leaving cache-size-only tuning.
- `hypothesis`: Increasing `GGML_MOE_STREAM_ONE_CACHE_MIB` from `13700` to `13760` may add roughly `60 / 4.25 ≈ 14` slots and leave only a narrow CUDA free-memory margin. It may still run, but the expected miss reduction and token-rate benefit are tiny; CUDA OOM or allocator fragility is plausible.
- `theoretical_upper_bound`: Fourteen extra slots can at best avoid about fourteen immediate misses on the observed sequence. At `~4.9ms` average traced `src0` cost per miss, the direct bound is about `0.07s` before secondary effects. This is not expected to beat SOTA unless there is a nonlinear cache/reclaim effect; reject on tie.
- `test_config`: accepted late10 config, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13760`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, gate trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: promote only if `eval_tok_s > 2.6`, RAM including page cache stays `<=16000000000`, no CUDA OOM/cache insertion failure occurs, France answer is semantically correct and coherent under manual review, and TTFT does not exceed current late10 pushed rerun by more than `20%` (`37874.580124ms * 1.2 = 45449.496149ms`).
- `rollback`: No source change is required. If CUDA OOM, cache insertion failure, token rate not above `2.6`, output correctness failure, RAM failure, or TTFT failure occurs, record rejected and keep `13568MiB` late10 as SOTA. If accepted, immediately write full reproduction evidence, commit/push records to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before promotion.
- `required_evidence`: exact env/command, stderr cache size/slot/free-memory lines, source commit/status, binary sha256/build line, model stat, stdout/stderr, summary.json, gate trace if generated, cgroup `memory.*`, France answer text if generated, manual correctness note, trace summary, and explicit promoted/rejected status.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T011242Z-20260702_late10_cache13760mib/france-cpu40-vram0gb`.
- `result`: rejected tie. `eval_tok_s=2.6`, `prompt_tok_s=1.0`, `TTFT=36167.894934ms`, `elapsed_seconds=87.67`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15034978304`, `pgmajfault=292684`, `workingset_refault_file=3494699`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `cache_observation`: stderr reports `[moe_stream] VRAM cache: 13.4 GiB, 3237 slots (4.25 MiB each)`, CUDA free memory around `46 MiB`, and `hits=30215 misses=4936`.
- `correctness_manual_review`: pass. France answer is semantically correct, coherent, complete, and not repetitive/truncated.
- `trace_summary`: `rows=35151`, `cache_hits=30215`, `cache_misses=4936`, `span_ms=69585.306`, `src0_ms=23966.665`, `total_ms=26231.251`.
- `gap_analysis`: `13760MiB` is the practical VRAM boundary found so far: it leaves only about `46 MiB` CUDA free and reduces misses by only `35` versus accepted SOTA (`4971 -> 4936`). Trace time improves slightly (`src0_ms 24305.283 -> 23966.665`, `total_ms 26552.328 -> 26231.251`) and TTFT improves, but runner `eval_tok_s` still ties `2.6`. Since the gate requires a strictly higher token rate, do not promote. Cache-size-only tuning is now at diminishing returns; going higher is likely allocator-fragile and has a sub-0.1s direct miss-saving bound.
- `rollback_status`: no source change. Current accepted SOTA remains late10 top3 `2.6 tok/s` with `13568MiB` stream cache; `13760MiB` is recorded as a non-promoted tie and possible TTFT/reference config only.

### 当前执行 attempt：late10-skip-dontneed-cache-hit

- `attempt_id`: `20260702-late10-skip-dontneed-cache-hit`
- `attempt_kind`: `implementation/io-refault-policy`
- `status`: completed_rejected_tie_rolled_back
- `bottleneck_basis`: Current accepted late10 trace has `cache_hits=30180`, `cache_misses=4971`, `dontneed_ms=1169.862`, and `src0_ms=24305.283`. The current `moe_stream.cu` calls `MADV_DONTNEED` for every streamed expert call after scatter, including cache hits where the kernel used VRAM cache and did not need to touch the mmap source page for H2D. Since hits are `85.9%` of calls, most `DONTNEED` calls may be pure syscall/reclaim overhead or may drop pages that could be useful if an expert is later evicted.
- `hypothesis`: Add a default-off env `GGML_MOE_STREAM_DONTNEED_ON_HIT=0` so `moe_stream_dontneed_source_pages()` is skipped for VRAM cache hits but remains enabled for misses/fallbacks. This may reduce direct `dontneed_ms` and possibly reduce refault churn without changing model math. It preserves current behavior when env is unset.
- `theoretical_upper_bound`: If cache-hit `DONTNEED` cost is proportional to call count, skipping hits could remove up to `1169.862ms * 30180 / 35151 ≈ 1004ms` direct traced overhead. Secondary effects could be positive if refaults drop, or negative if retained file pages increase cgroup reclaim pressure. The hard upper bound is therefore around 1s direct plus reclaim variance; expected token-rate gain is modest but larger than cache-size-only probes.
- `test_config`: patched source, accepted late10 config, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_DONTNEED=1`, `GGML_MOE_STREAM_DONTNEED_ON_HIT=0`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, gate trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: promote only if `eval_tok_s > 2.6`, RAM including page cache stays `<=16000000000`, France answer is semantically correct and coherent under manual review, and TTFT does not exceed current late10 pushed rerun by more than `20%` (`37874.580124ms * 1.2 = 45449.496149ms`).
- `rollback`: If build fails, token rate does not exceed `2.6`, output correctness fails, RAM exceeds limit, TTFT exceeds gate, or trace/refault evidence shows worse reclaim pressure, revert source and clean rebuild. If accepted, immediately write full reproduction evidence, commit/push source to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before promotion.
- `required_evidence`: source diff, build log/version, exact env/command, source commit/status, binary sha256/build line, model stat, stdout/stderr, summary.json, gate trace, cgroup `memory.*`, France answer text, manual correctness note, trace summary, and explicit promoted/rejected/rollback status.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T012047Z-20260702_late10_skip_dontneed_cache_hit/france-cpu40-vram0gb`.
- `result`: rejected tie. `eval_tok_s=2.6`, `prompt_tok_s=1.0`, `TTFT=37611.422067ms`, `elapsed_seconds=88.54`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15035154432`, `pgmajfault=289803`, `workingset_refault_file=3492371`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `correctness_manual_review`: pass. France answer is semantically correct, coherent, complete, and not repetitive/truncated.
- `trace_summary`: `rows=35151`, `cache_hits=30180`, `cache_misses=4971`, `span_ms=68790.521`, `src0_ms=24187.822`, `dontneed_ms=980.980`, `total_ms=26244.469`.
- `gap_analysis`: Skipping `MADV_DONTNEED` on VRAM cache hits reduced direct traced `dontneed_ms` only `1169.862 -> 980.980` rather than the ~1.0s proportional upper bound, and runner `eval_tok_s` still tied `2.6`. `src0_ms` and total trace improved only slightly and did not produce a strict token-rate SOTA. The likely explanation is that most hit-side `madvise` calls were cheap/no-op on already reclaimed pages, while keeping hit pages resident does not reduce future miss cost enough under the 16GB cgroup. This is useful evidence but not accepted.
- `rollback_status`: source change reverted and clean rebuild completed; worktree clean and binary sha256 returned to `c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`. Current accepted SOTA remains late10 top3 `2.6 tok/s`.

### 当前执行 attempt：late10-vram-cache-hash-lookup

- `attempt_id`: `20260702-late10-vram-cache-hash-lookup`
- `attempt_kind`: `implementation/cache-lookup-cpu-overhead`
- `status`: completed_rejected_tie_rolled_back
- `bottleneck_basis`: `moe_stream.cu` defines `vram_ht_entry ht[VRAM_CACHE_HT_SIZE]` and comments describe O(1) lookup, but `vram_cache_lookup()` currently linearly scans all `n_slots` (`3192` slots for accepted SOTA). Accepted late10 performs `35151` cache lookups with `30180` hits. This lookup/scanning cost is outside the traced `src0_ms` fields and can affect end-to-end wall time even when IO is unchanged.
- `hypothesis`: Implement an opt-in `GGML_MOE_STREAM_CACHE_HASH=1` path that maintains the existing hash table on insert/evict and uses it for cache lookup, while leaving the default linear path unchanged. This should preserve model math, cache capacity, and eviction policy, but reduce CPU overhead for cache hits.
- `theoretical_upper_bound`: Linear lookup does up to `3192` key comparisons per hit. With `30180` hits, worst-case comparisons are tens of millions. Even if each comparison is cheap, the bound can be hundreds of milliseconds to low single-digit seconds depending on cache locality and branch behavior. It cannot reduce `src0_ms` or miss count directly; a valid improvement must show lower elapsed/token-rate without correctness or RAM regression.
- `test_config`: patched source, accepted late10 config, `GGML_MOE_STREAM_CACHE_HASH=1`, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_DONTNEED=1`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, gate trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: promote only if `eval_tok_s > 2.6`, RAM including page cache stays `<=16000000000`, France answer is semantically correct and coherent under manual review, and TTFT does not exceed current late10 pushed rerun by more than `20%` (`37874.580124ms * 1.2 = 45449.496149ms`).
- `rollback`: If build fails, token rate does not exceed `2.6`, output correctness fails, RAM exceeds limit, TTFT exceeds gate, or cache hit/miss counts diverge unexpectedly from accepted late10, revert source and clean rebuild. If accepted, immediately write full reproduction evidence, commit/push source to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before promotion.
- `required_evidence`: source diff, build log/version, exact env/command, source commit/status, binary sha256/build line, stdout/stderr cache hit/miss counts, summary.json, gate trace, cgroup `memory.*`, France answer text, manual correctness note, trace summary, and explicit promoted/rejected/rollback status.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T012951Z-20260702_late10_vram_cache_hash_lookup/france-cpu40-vram0gb`.
- `result`: rejected tie. `eval_tok_s=2.6`, `prompt_tok_s=0.9`, `TTFT=37580.398308ms`, `elapsed_seconds=89.61`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15039725568`, `pgmajfault=280349`, `workingset_refault_file=3483155`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `correctness_manual_review`: pass. France answer is semantically correct, coherent, complete, and not repetitive/truncated.
- `trace_summary`: `rows=35151`, `cache_hits=30180`, `cache_misses=4971`, `span_ms=70320.906`, `src0_ms=24481.296`, `dontneed_ms=1173.443`, `total_ms=26758.289`.
- `gap_analysis`: The hash path preserved cache behavior exactly (`hits=30180`, `misses=4971`) and correctness passed, but it did not improve rounded token rate and elapsed time was slightly worse than the accepted late10 rerun. This indicates linear cache lookup is not a current first-order bottleneck, or the saved CPU comparisons are dominated by IO/page-reclaim variance and CUDA synchronization. Since the strict token-rate gate is not exceeded, do not keep the source change.
- `rollback_status`: source change reverted and clean rebuild completed; worktree clean and binary sha256 returned to `c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`. Current accepted SOTA remains late10 top3 `2.6 tok/s`.

### 当前执行 attempt：late10-last10-top2

- `attempt_id`: `20260702-late10-last10-top2`
- `attempt_kind`: `implementation/layer-selective-approximate-pruning`
- `status`: completed_rejected_tie_rolled_back
- `bottleneck_basis`: Gate stream/cache, block readahead, simple page advice, cache lookup, and thread-count probes have not exceeded the late10 SOTA. The remaining confirmed high-impact bottleneck is broad non-streamed `ffn_up_exps` / `ffn_down_exps` CPU fallback. Current SOTA uses `0-9=>top4` and `10-39=>top3`; late5/late8 showed moving the top3 boundary earlier changes trajectory and hurts misses/correctness, so the next structural reduction should happen only in the latest layers.
- `hypothesis`: Keep the accepted early/mid policy but prune the last ten CPU-MoE layers further: `0-9=>top4`, `10-29=>top3`, `30-39=>top2`. Late layers may tolerate more aggressive approximate pruning while reducing one more routed up/down expert in 10 of 40 CPU-MoE layers. This targets CPU fallback work directly without increasing VRAM cache or page readahead.
- `theoretical_upper_bound`: The global top4->top3 one-expert reduction had a coarse bound around `12.9s` before overhead and quality effects. Applying another one-expert reduction only to 10 of 40 CPU-MoE layers gives a similar coarse incremental bound of `12.9s * 10/40 ≈ 3.2s`. Real gain may be lower due layer imbalance, unchanged gate stream, output length changes, and quality constraints. The expected improvement, if quality holds, is a small but meaningful chance to exceed the rounded `2.6 tok/s` SOTA.
- `implementation`: Extend `ggml_moe_keep_topk_for_tensor()` with default-off second range envs `GGML_MOE_KEEP_TOPK_LAYER_RANGE2=<start>-<end>` and `GGML_MOE_KEEP_TOPK_LAYER_VALUE2=<K>`. Existing env behavior remains unchanged. For this test use fallback `GGML_MOE_KEEP_TOPK_UPDOWN=4`, first range `10-29=>3`, second range `30-39=>2`.
- `test_config`: patched source, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-29`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE2=30-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE2=2`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, gate trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: promote only if `eval_tok_s > 2.6`, RAM including page cache stays `<=16000000000`, France answer is semantically correct and coherent under manual review, and TTFT does not exceed current late10 pushed rerun by more than `20%` (`37874.580124ms * 1.2 = 45449.496149ms`).
- `rollback`: If build fails, token rate does not exceed `2.6`, output correctness fails, RAM exceeds limit, TTFT exceeds gate, or trace shows late5-like miss/refault explosion, revert source and clean rebuild. If accepted, immediately write full reproduction evidence, commit/push source to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before promotion.
- `required_evidence`: source diff, build log/version, exact env/command, source commit/status, binary sha256/build line, stdout/stderr, summary.json, gate trace, cgroup `memory.*`, France answer text, manual correctness note, trace summary, and explicit promoted/rejected/rollback status.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T013948Z-20260702_late10_last10_top2/france-cpu40-vram0gb`.
- `result`: rejected tie. `eval_tok_s=2.6`, `prompt_tok_s=1.0`, `TTFT=36462.76209ms`, `elapsed_seconds=109.36`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15010971648`, `pgmajfault=340053`, `workingset_refault_file=6462646`, `ram_ok=true`, `ram_limit_killed=false`.
- `correctness_manual_review`: fail. The answer is broadly about France, but it is cut off at the fixed output limit after `founding member of the European Union` and contains the misspelling `Fratinité`; this is not a clean complete coherent short paragraph.
- `trace_summary`: `rows=47872`, `cache_hits=41515`, `cache_misses=6357`, `span_ms=90850.725`, `src0_ms=32004.770`, `dontneed_ms=1531.861`, `total_ms=34990.887`.
- `gap_analysis`: Last10 top2 did reduce configured up/down routed experts in the latest layers, but it changed the generation trajectory enough to increase gate stream rows and misses (`35151/4971 -> 47872/6357`), widen trace span (`69920.848 -> 90850.725`), increase refault pressure (`workingset_refault_file=6462646`), and truncate the answer. Like late5/late8, this shows more aggressive pruning beyond the accepted late10 boundary can backfire by generating longer/different trajectories and more gate cache traffic. Do not pursue top2 late-layer pruning without a stronger quality-preserving strategy.
- `rollback_status`: source change reverted and clean rebuild completed; worktree clean and binary sha256 returned to `c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`. Current accepted SOTA remains late10 top3 `2.6 tok/s`.

### 当前执行 attempt：late10-last5-top2

- `attempt_id`: `20260702-late10-last5-top2`
- `attempt_kind`: `implementation/layer-selective-approximate-pruning`
- `status`: completed_rejected_tie_rolled_back
- `bottleneck_basis`: Last10 top2 was too aggressive and increased gate stream rows/misses enough to fail correctness and lose elapsed time. The accepted late10 top3 SOTA still leaves broad CPU fallback work in the latest layers, but any further pruning must be narrower than the rejected last10 top2 attempt.
- `hypothesis`: Keep accepted late10 top3 for layers `10-34` and apply top2 only to the last five CPU-MoE layers: `0-9=>top4`, `10-34=>top3`, `35-39=>top2`. This tests whether the very latest layers tolerate one fewer up/down expert without triggering the trajectory/miss explosion seen at `30-39=>top2`.
- `theoretical_upper_bound`: The rejected last10 top2 attempt had a coarse incremental bound of `12.9s * 10/40 ≈ 3.2s`. Restricting top2 to five layers halves that optimistic bound to about `12.9s * 5/40 ≈ 1.6s`. Because gate-stream miss count and output trajectory dominate recent failures, the practical expected gain is a small rounding chance above `2.6 tok/s`; any increase in rows/misses can erase the bound.
- `implementation`: Reintroduce default-off second layer range support in `ggml_moe_keep_topk_for_tensor()` using `GGML_MOE_KEEP_TOPK_LAYER_RANGE2=<start>-<end>` and `GGML_MOE_KEEP_TOPK_LAYER_VALUE2=<K>`. Existing env behavior must remain unchanged when range2 is unset.
- `test_config`: patched source, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-34`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE2=35-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE2=2`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, gate trace enabled, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: promote only if `eval_tok_s > 2.6`, RAM including page cache stays `<=16000000000`, France answer is semantically correct/coherent/complete under manual review, and TTFT does not exceed current late10 pushed rerun by more than `20%` (`37874.580124ms * 1.2 = 45449.496149ms`).
- `rollback`: If build fails, token rate does not exceed `2.6`, output correctness fails, RAM exceeds limit, TTFT exceeds gate, or trace shows increased rows/misses/refaults comparable to last10 top2, revert source and clean rebuild. If accepted, immediately stop all further experiments, write complete reproduction evidence, commit/push source and records to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before promotion.
- `required_evidence`: source diff, build log/version, exact env/command, source commit/status, pushed remote/branch/commit if promoted, binary sha256/build line/stat, model stat, stdout/stderr, summary.json, gate trace, cgroup `memory.*`, France answer text, manual correctness note, trace summary, and explicit promoted/rejected/rollback status.
- `sota_publish_requirement`: If this run becomes a compliant SOTA, it must be documented and pushed immediately to `ssd-llama` branch `vendor/deepseek-token-rate-16gb`; otherwise the result is only an unpromoted observation even if the printed token rate is higher.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T015049Z-20260702_late10_last5_top2/france-cpu40-vram0gb`.
- `result`: rejected tie. `eval_tok_s=2.6`, `prompt_tok_s=1.0`, `TTFT=37767.187331ms`, `elapsed_seconds=105.41`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15015936000`, `pgmajfault=335937`, `workingset_refault_file=5909304`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `correctness_manual_review`: pass. France answer is semantically correct, coherent, complete, and not repetitive/truncated.
- `trace_summary`: `rows=44749`, `cache_hits=38811`, `cache_misses=5938`, `src0_ms=30363.530`, `dontneed_ms=1424.326`, `total_ms=33140.385`.
- `gap_analysis`: Narrowing top2 to only `blk.35-39` avoids the truncation/correctness failure from last10 top2, but still changes the generation trajectory enough to increase rows (`35151 -> 44749`), gate misses (`4971 -> 5938`), `src0_ms` (`24305.283 -> 30363.530`), refault pressure (`3545116 -> 5909304`), and elapsed time (`89.48s -> 105.41s`) versus accepted late10. Since token rate only ties `2.6` and does not exceed the SOTA gate, do not promote. Further top2 pruning is deprioritized unless paired with a quality/trajectory-preserving mechanism.
- `rollback_status`: source change reverted and clean rebuild completed; current accepted SOTA remains late10 top3 `2.6 tok/s`.

### 当前执行 attempt：late10-no-trace-production

- `attempt_id`: `20260702-late10-no-trace-production`
- `attempt_kind`: `config-probe/trace-overhead`
- `status`: completed_rejected_tie_not_sota
- `bottleneck_basis`: Accepted late10 SOTA still writes per-expert trace for `35151` rows, and recent rejected probes show token-rate ties around `2.6` where sub-second overhead can decide the rounded metric. Earlier top4 no-trace did not improve, but late10 has a different row count and is the current production candidate.
- `hypothesis`: Removing `GGML_MOE_STREAM_ONE_TRACE_OUT` from the accepted late10 config may reduce file-write/timing overhead without changing model math, RAM behavior, or correctness. If token rate improves, the production SOTA config should run without per-expert trace, while a separate trace diagnostic can remain available for bottleneck analysis.
- `theoretical_upper_bound`: Accepted trace has `35151` rows; direct trace formatting/write cost is not isolated but is bounded by file output and timing overhead. Expected gain is small, likely below 1s wall time; only a rounded move from `2.6` to `>2.6 tok/s` would justify promotion.
- `test_config`: accepted late10 config, no `GGML_MOE_STREAM_ONE_TRACE_OUT`, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: promote only if `eval_tok_s > 2.6`, RAM including page cache stays `<=16000000000`, France answer is semantically correct/coherent/complete under manual review, and TTFT does not exceed current late10 pushed rerun by more than `20%` (`37874.580124ms * 1.2 = 45449.496149ms`).
- `rollback`: No source change. If token rate does not exceed `2.6`, output correctness fails, RAM exceeds limit, or TTFT exceeds gate, record rejected/diagnostic and keep accepted late10 trace-backed SOTA. If accepted, immediately stop further experiments, write full no-trace reproduction package, commit/push records/source state to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before promotion.
- `required_evidence`: exact env/command showing trace disabled, source commit/status, binary sha256/build line/stat, model stat, stdout/stderr including cache hit/miss summary, summary.json, cgroup `memory.*`, France answer text, manual correctness note, and explicit promoted/rejected status. If promoted, also run a separate traced diagnostic for bottleneck comparison without using that traced run as the performance number.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T015621Z-20260702_late10_no_trace_production/france-cpu40-vram0gb`.
- `result`: rejected tie. `eval_tok_s=2.6`, `prompt_tok_s=0.9`, `TTFT=36548.431091ms`, `elapsed_seconds=87.18`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15011225600`, `pgmajfault=292933`, `workingset_refault_file=3484919`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `correctness_manual_review`: pass. France answer is semantically correct, coherent, complete, and not repetitive/truncated.
- `cache_observation`: stderr reports `[moe_stream] VRAM cache: 13.2 GiB, 3192 slots (4.25 MiB each)` and `hits=30180 misses=4971 hit_rate=85.9%`, matching accepted late10 cache behavior.
- `gap_analysis`: Disabling per-expert trace reduced elapsed time versus the accepted traced rerun (`89.48s -> 87.18s`) and improved TTFT, but runner `eval_tok_s` still tied `2.6` and therefore does not satisfy the strict promotion gate. Trace overhead is not large enough by itself to produce a new rounded token-rate SOTA.
- `rollback_status`: no source change. Current accepted SOTA remains late10 top3 `2.6 tok/s`; no-trace is a useful production/diagnostic tie but not a promoted SOTA.

### 当前执行 attempt：late10-cache13760-no-trace

- `attempt_id`: `20260702-late10-cache13760-no-trace`
- `attempt_kind`: `config-probe/vram-cache-boundary-plus-trace-overhead`
- `status`: completed_rejected_tie_not_sota
- `bottleneck_basis`: Two isolated low-risk probes tied but improved secondary metrics: `13760MiB` reduced misses (`4971 -> 4936`) and trace total slightly, while no-trace reduced elapsed (`89.48s -> 87.18s`) without changing cache hit/miss behavior. Neither alone crossed the strict `>2.6 tok/s` gate, so the next efficient probe is their combination before leaving cache-size/trace-overhead tuning.
- `hypothesis`: Running the accepted late10 top3 config with `GGML_MOE_STREAM_ONE_CACHE_MIB=13760` and no `GGML_MOE_STREAM_ONE_TRACE_OUT` may combine a small miss reduction with lower trace overhead while preserving model math and France correctness. It also uses VRAM close to the practical boundary already shown to run with about `46 MiB` free.
- `theoretical_upper_bound`: The 13760MiB cache adds about 45 slots versus accepted 13568MiB and previously reduced misses by only `35`; at `~4.9ms` per miss the direct bound is about `0.17s`. Removing trace previously reduced elapsed by about `2.3s` but still tied token rate. The combined optimistic elapsed improvement is therefore low single-digit seconds, enough only for a possible rounded move to `2.7 tok/s`; reject on any tie.
- `test_config`: accepted late10 config, no `GGML_MOE_STREAM_ONE_TRACE_OUT`, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13760`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: promote only if `eval_tok_s > 2.6`, RAM including page cache stays `<=16000000000`, no CUDA OOM/cache insertion failure occurs, France answer is semantically correct/coherent/complete under manual review, and TTFT does not exceed current late10 pushed rerun by more than `20%` (`37874.580124ms * 1.2 = 45449.496149ms`).
- `rollback`: No source change. If CUDA OOM, cache insertion failure, token rate not above `2.6`, output correctness failure, RAM failure, or TTFT failure occurs, record rejected and keep `13568MiB` late10 as SOTA. If accepted, immediately stop further experiments, write full reproduction evidence, commit/push records/source state to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before promotion.
- `required_evidence`: exact env/command showing trace disabled and cache `13760`, stderr cache size/slot/free-memory lines, source commit/status, binary sha256/build line/stat, model stat, stdout/stderr, summary.json, cgroup `memory.*`, France answer text, manual correctness note, and explicit promoted/rejected status.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T020100Z-20260702_late10_cache13760_no_trace/france-cpu40-vram0gb`.
- `result`: rejected tie. `eval_tok_s=2.6`, `prompt_tok_s=0.9`, `TTFT=38087.247563ms`, `elapsed_seconds=89.23`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15014617088`, `pgmajfault=290998`, `workingset_refault_file=3490683`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `cache_observation`: stderr reports `[moe_stream] VRAM cache: 13.4 GiB, 3237 slots (4.25 MiB each)`, CUDA free memory about `46 MiB`, and `hits=30215 misses=4936 hit_rate=86.0%`; cache insertion succeeded and VRAM was used near the practical boundary.
- `correctness_manual_review`: pass. France answer is semantically correct, coherent, complete, and not repetitive/truncated.
- `gap_analysis`: Combining no-trace with the 13760MiB cache did not combine favorably. It preserved the lower miss count from `13760MiB` (`4936` misses) and remained within the 16GB cgroup, but elapsed time regressed versus the `13568MiB` no-trace run (`87.18s -> 89.23s`) and token rate still tied `2.6`. The 46 MiB CUDA free margin likely makes this boundary fragile without a meaningful token-rate gain, so keep accepted late10 `13568MiB` as SOTA.
- `rollback_status`: no source change. Current accepted SOTA remains late10 top3 `2.6 tok/s` with `13568MiB` stream cache.

### 当前执行 attempt：late10-no-trace-dontneed0

- `attempt_id`: `20260702-late10-no-trace-dontneed0`
- `attempt_kind`: `config-probe/page-reclaim-policy`
- `status`: completed_rejected_regression_not_sota
- `bottleneck_basis`: Accepted late10 trace still spends `dontneed_ms=1169.862ms` and `src0_ms=24305.283ms`; no-trace at `13568MiB` already reduced elapsed to `87.18s` but tied token rate. Earlier clean-baseline `DONTNEED=0` regressed by increasing `src0_ms`, but the accepted late10 routing/cache regime has fewer rows and may respond differently.
- `hypothesis`: Setting `GGML_MOE_STREAM_DONTNEED=0` under the accepted late10/no-trace config removes the direct per-expert `madvise(DONTNEED)` overhead and may reduce syscall/reclaim work enough to exceed the rounded `2.6 tok/s` gate. The main risk is that retaining mmap source pages increases file-cache pressure and refault churn under the strict 16GB cgroup.
- `theoretical_upper_bound`: Direct traced `dontneed_ms` in accepted late10 is about `1.17s`. If disabling it had no secondary cost, elapsed could improve from `87.18s` no-trace toward `~86s`, a small but plausible rounding move. If `src0_ms` rises as in the earlier clean-baseline probe, token rate will tie or regress.
- `test_config`: accepted late10 config, no `GGML_MOE_STREAM_ONE_TRACE_OUT`, `GGML_MOE_STREAM_DONTNEED=0`, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: promote only if `eval_tok_s > 2.6`, RAM including page cache stays `<=16000000000`, France answer is semantically correct/coherent/complete under manual review, and TTFT does not exceed current late10 pushed rerun by more than `20%` (`37874.580124ms * 1.2 = 45449.496149ms`).
- `rollback`: No source change. If token rate does not exceed `2.6`, output correctness fails, RAM exceeds limit, TTFT exceeds gate, or memory/refault evidence worsens materially, record rejected and keep accepted late10 SOTA. If accepted, immediately stop further experiments, write full reproduction evidence, commit/push records/source state to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before promotion.
- `required_evidence`: exact env/command showing `GGML_MOE_STREAM_DONTNEED=0` and trace disabled, source commit/status, binary sha256/build line/stat, model stat, stdout/stderr cache summary, summary.json, cgroup `memory.*`, France answer text, manual correctness note, and explicit promoted/rejected status.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T020507Z-20260702_late10_no_trace_dontneed0/france-cpu40-vram0gb`.
- `result`: rejected regression. `eval_tok_s=2.3`, `prompt_tok_s=0.9`, `TTFT=36790.815780ms`, `elapsed_seconds=93.62`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15039840256`, `pgmajfault=301260`, `workingset_refault_file=4459579`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `correctness_manual_review`: pass. France answer is semantically correct, coherent, complete, and not repetitive/truncated.
- `cache_observation`: stderr reports the accepted cache shape (`13.2 GiB`, `3192 slots`, `hits=30180 misses=4971 hit_rate=85.9%`), so the regression is not caused by cache insertion failure.
- `gap_analysis`: Disabling stream `DONTNEED` removes the direct `madvise` path, but under the 16GB cgroup it increases retained file pages/reclaim pressure and slows the run: elapsed regressed from the no-trace tie (`87.18s`) to `93.62s`, `workingset_refault_file` rose from `3484919` to `4459579`, and file-system inputs rose to `169695064`. Keep `GGML_MOE_STREAM_DONTNEED=1`; this confirms the syscall cost is outweighed by cgroup page-cache control.
- `rollback_status`: no source change. Current accepted SOTA remains late10 top3 `2.6 tok/s`.

### 当前执行 attempt：late10-no-trace-cuda-graphs

- `attempt_id`: `20260702-late10-no-trace-cuda-graphs`
- `attempt_kind`: `config-probe/cuda-launch-overhead`
- `status`: completed_rejected_tie_not_sota
- `bottleneck_basis`: Page-cache/cache-size/DONTNEED probes have not exceeded accepted late10. The runner currently forces `GGML_CUDA_DISABLE_GRAPHS=1`, while the workload has many small CUDA operations per streamed expert. Even though `src0` remains the dominant traced field, launch/sync overhead outside source loading may still decide a rounded `2.6 -> 2.7` move.
- `hypothesis`: Overriding the runner default with `GGML_CUDA_DISABLE_GRAPHS=0` under the accepted late10/no-trace config may reduce repeated CUDA launch overhead without changing model math, routing, VRAM cache capacity, or output semantics.
- `theoretical_upper_bound`: Accepted late10 trace has `35151` streamed rows. If CUDA graph capture reduces even tens of microseconds of repeated launch overhead per row, the gross bound can be sub-second to low-single-digit seconds. If the stream path is not graph-captured or page faults dominate, there will be no gain; correctness should remain unchanged.
- `test_config`: accepted late10 config, no `GGML_MOE_STREAM_ONE_TRACE_OUT`, `GGML_CUDA_DISABLE_GRAPHS=0`, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: promote only if `eval_tok_s > 2.6`, RAM including page cache stays `<=16000000000`, France answer is semantically correct/coherent/complete under manual review, and TTFT does not exceed current late10 pushed rerun by more than `20%` (`37874.580124ms * 1.2 = 45449.496149ms`).
- `rollback`: No source change. If CUDA graphs fail, token rate does not exceed `2.6`, output correctness fails, RAM exceeds limit, or TTFT exceeds gate, record rejected and keep accepted late10 SOTA. If accepted, immediately stop further experiments, write full reproduction evidence, commit/push records/source state to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before promotion.
- `required_evidence`: exact env/command showing `GGML_CUDA_DISABLE_GRAPHS=0` and trace disabled, source commit/status, binary sha256/build line/stat, model stat, stdout/stderr cache summary, summary.json, cgroup `memory.*`, France answer text, manual correctness note, and explicit promoted/rejected status.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T021001Z-20260702_late10_no_trace_cuda_graphs/france-cpu40-vram0gb`.
- `result`: rejected tie. `eval_tok_s=2.6`, `prompt_tok_s=0.9`, `TTFT=37897.894423ms`, `elapsed_seconds=90.25`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15023026176`, `pgmajfault=296741`, `workingset_refault_file=3554439`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `correctness_manual_review`: pass. France answer is semantically correct, coherent, complete, and not repetitive/truncated.
- `cache_observation`: stderr reports accepted cache shape (`13.2 GiB`, `3192 slots`, `hits=30180 misses=4971 hit_rate=85.9%`).
- `gap_analysis`: Enabling CUDA graphs did not improve the accepted late10/no-trace path. Elapsed regressed versus the no-trace tie (`87.18s -> 90.25s`) and token rate still tied `2.6`, indicating either this stream-heavy path is not benefiting from graph capture or the remaining page-fault/source-load path dominates launch overhead. Keep runner default `GGML_CUDA_DISABLE_GRAPHS=1`.
- `rollback_status`: no source change. Current accepted SOTA remains late10 top3 `2.6 tok/s`.

### 当前执行 attempt：current-sota-repro-late10-trace

- `attempt_id`: `20260702-current-sota-repro-late10-trace`
- `attempt_kind`: `sota-reproduction`
- `status`: completed_reproduced_current_sota`
- `purpose`: User requested direct reproduction of the current accepted SOTA. This rerun uses the accepted late10 top3 traced config under cold `drop_caches` and strict 16GB cgroup.
- `source_state`: repo HEAD `c6bddffdb803662b070065bea34b298b0bf7b83a`, branch `feat/ds4-moe-stream-on-vendor`, worktree clean, binary sha256 `c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`, binary version `9134 (78ee9ff26)`.
- `test_config`: accepted late10 config, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, `GGML_MOE_STREAM_ONE_TRACE_OUT=/root/lfz/runs/vendor-ds4-16gb/CURRENT_SOTA_REPRO_TRACE`, `cpu_moe=40`, cold `drop_caches`, strict 16GB cgroup, France prompt, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T021305Z-20260702_current_sota_repro_late10_trace/france-cpu40-vram0gb`.
- `result`: reproduced. `eval_tok_s=2.6`, `prompt_tok_s=1.0`, `TTFT=35544.094785ms`, `elapsed_seconds=87.00`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15040581632`, `pgmajfault=288149`, `workingset_refault_file=3489226`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `correctness_manual_review`: pass. France answer is semantically correct, coherent, complete, and not repetitive/truncated.
- `cache_observation`: stderr reports `[moe_stream] VRAM cache: 13.2 GiB, 3192 slots (4.25 MiB each)`, CUDA free memory about `238 MiB`, and `hits=30180 misses=4971 hit_rate=85.9%`.
- `trace_summary`: `rows=35151`, `cache_hits=30180`, `cache_misses=4971`, `src0_ms=24216.224`, `dontneed_ms=1159.096`, `total_ms=26450.962`.
- `conclusion`: Current accepted cold-start SOTA `2.6 tok/s` is reproducible under the stated vendor late10 top3 traced config, strict 16GB RAM including page cache, and France correctness gate. This rerun is not a new SOTA; it confirms the existing SOTA remains valid.

### 当前执行 attempt：late10-stream-alloc-lock-coalesce

- `attempt_id`: `20260702-late10-stream-alloc-lock-coalesce`
- `attempt_kind`: `implementation/cpu-overhead-reduction`
- `status`: completed_rejected_tie_rolled_back
- `bottleneck_basis`: Current accepted trace still has `35151` streamed expert calls. In `ggml_cuda_moe_stream_one()`, each call enters `g_resize_mu` twice: first for `d_src0/d_src1/d_dst/d_ids/h_scratch`, then again for `d_src1_f32/h_bounce`. Resizes are rare after warm slot sizing, but the global mutex acquisition and size checks remain on every call and are outside the dominant `src0_ms` trace field.
- `hypothesis`: Coalescing `d_src1_f32` and `h_bounce` allocation into the first resize-guarded block removes one global mutex lock/unlock path per streamed expert without changing math, routing, cache behavior, memory capacity, or output semantics.
- `theoretical_upper_bound`: The hard bound is small because source page loading remains dominant. With `35151` calls, saving even `5-20us` per second mutex/check block would be about `0.18-0.70s`; larger lock contention could save more, but expected gain is at most low single-digit seconds. Promote only if rounded `eval_tok_s` strictly exceeds `2.6`.
- `implementation`: In `ggml/src/ggml-cuda/moe_stream.cu`, move `ensure_dev(ctx.d_src1_f32, ...)` and `ensure_host_pinned(ctx.h_bounce, ...)` into the existing first `g_resize_mu` block, include them in `ok`, and remove the second `g_resize_mu` block. Preserve existing fallback checks for `ctx.h_bounce` before copy.
- `test_config`: patched source, accepted late10 traced config, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, trace enabled, `cpu_moe=40`, cold `drop_caches`, strict 16GB cgroup, France prompt, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: promote only if `eval_tok_s > 2.6`, RAM including page cache stays `<=16000000000`, France answer is semantically correct/coherent/complete under manual review, cache hit/miss behavior remains consistent with accepted late10, and TTFT does not exceed current late10 pushed rerun by more than `20%` (`37874.580124ms * 1.2 = 45449.496149ms`).
- `rollback`: If build fails, token rate does not exceed `2.6`, output correctness fails, RAM exceeds limit, TTFT exceeds gate, or cache behavior diverges unexpectedly, revert source and clean rebuild. If accepted, immediately stop further experiments, write full reproduction evidence, commit/push source to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before promotion.
- `required_evidence`: source diff, build log/version, exact env/command, source commit/status, binary sha256/build line/stat, stdout/stderr cache summary, summary.json, gate trace, cgroup `memory.*`, France answer text, manual correctness note, trace summary, and explicit promoted/rejected/rollback status.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T022358Z-20260702_late10_stream_alloc_lock_coalesce/france-cpu40-vram0gb`.
- `result`: rejected tie. `eval_tok_s=2.6`, `prompt_tok_s=0.9`, `TTFT=38048.101123ms`, `elapsed_seconds=89.77`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15002759168`, `pgmajfault=284181`, `workingset_refault_file=3514771`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `correctness_manual_review`: pass. France answer is semantically correct, coherent, complete, and not repetitive/truncated.
- `cache_observation`: stderr reports accepted cache shape (`13.2 GiB`, `3192 slots`, `hits=30180 misses=4971 hit_rate=85.9%`).
- `trace_summary`: `rows=35151`, `cache_hits=30180`, `cache_misses=4971`, `src0_ms=24474.499`, `dontneed_ms=1182.592`, `total_ms=26744.375`.
- `gap_analysis`: Coalescing the allocation lock path preserved correctness and cache behavior but did not improve token rate. The trace total regressed slightly versus the current SOTA reproduction (`26450.962 -> 26744.375`) and `eval_tok_s` tied `2.6`, so the saved mutex path is not a first-order bottleneck under cold 16GB conditions.
- `rollback_status`: source patch reverted and clean rebuild completed; current accepted SOTA remains late10 top3 `2.6 tok/s`.

### 当前执行 attempt：late10-cpu-fallback-chunk32

- `attempt_id`: `20260702-late10-cpu-fallback-chunk32`
- `attempt_kind`: `implementation/cpu-fallback-scheduling`
- `status`: completed_tie_pending_no_trace_followup
- `bottleneck_basis`: Current late10 SOTA has exhausted several gate stream/cache/page-advice/launch-overhead probes. Historical CPU chunk trace showed up/down CPU fallback as a major trace-outside component, and current source still uses `chunk_size=16` for multi-row fallback cases. Earlier chunk32 was only tested on the old `1.6 tok/s` baseline, not on the accepted late10 top3 SOTA.
- `hypothesis`: Reintroducing default-off `GGML_MOE_CPU_CHUNK_SIZE=32` for multi-row CPU fallback under the accepted late10 top3 config may reduce atomic scheduling and small chunk overhead in remaining `ffn_up_exps` / `ffn_down_exps` CPU fallback without changing arithmetic. The accepted top3 routing has fewer fallback experts than the old baseline, so the tradeoff may differ.
- `theoretical_upper_bound`: This cannot reduce gate stream `src0_ms≈24.2s` or cache misses. It can only reduce CPU fallback scheduling/compute overhead outside the gate trace. If current up/down fallback still contributes tens of seconds, reducing chunk scheduling overhead could recover low single-digit seconds; if arithmetic or page faults dominate, token rate will tie. Promote only if `eval_tok_s > 2.6`.
- `implementation`: Add default-off env `GGML_MOE_CPU_CHUNK_SIZE=<N>` in both CPU `mul_mat_id` chunk-size sites, preserving current behavior by default and preserving `chunk_size=64` for `nr0 == 1 || nr1 == 1`. Test with `N=32`.
- `test_config`: patched source, `GGML_MOE_CPU_CHUNK_SIZE=32`, accepted late10 traced config, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, trace enabled, `cpu_moe=40`, cold `drop_caches`, strict 16GB cgroup, France prompt, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: promote only if `eval_tok_s > 2.6`, RAM including page cache stays `<=16000000000`, France answer is semantically correct/coherent/complete under manual review, cache hit/miss behavior remains consistent with accepted late10, and TTFT does not exceed current late10 pushed rerun by more than `20%` (`37874.580124ms * 1.2 = 45449.496149ms`).
- `rollback`: If build fails, token rate does not exceed `2.6`, output correctness fails, RAM exceeds limit, TTFT exceeds gate, or cache behavior diverges unexpectedly, revert source and clean rebuild. If accepted, immediately stop further experiments, write full reproduction evidence, commit/push source to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before promotion.
- `required_evidence`: source diff, build log/version, exact env/command, source commit/status, binary sha256/build line/stat, stdout/stderr cache summary, summary.json, gate trace, cgroup `memory.*`, France answer text, manual correctness note, trace summary, and explicit promoted/rejected/rollback status.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T023109Z-20260702_late10_cpu_fallback_chunk32/france-cpu40-vram0gb`.
- `result`: tie, not promoted. `eval_tok_s=2.6`, `prompt_tok_s=1.0`, `TTFT=34444.152913ms`, `elapsed_seconds=86.59`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15019266048`, `pgmajfault=238320`, `workingset_refault_file=3628792`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `correctness_manual_review`: pass. France answer is semantically correct, coherent, complete, and not repetitive/truncated.
- `cache_observation`: stderr reports accepted cache shape (`13.2 GiB`, `3192 slots`, `hits=30180 misses=4971 hit_rate=85.9%`).
- `trace_summary`: `rows=35151`, `cache_hits=30180`, `cache_misses=4971`, `src0_ms=24171.763`, `dontneed_ms=1186.052`, `total_ms=26447.911`.
- `gap_analysis`: `chunk32` preserved correctness/RAM/cache behavior and improved TTFT/elapsed versus the accepted traced rerun, but runner `eval_tok_s` still tied `2.6`. Because no-trace previously reduced elapsed without changing cache behavior, run one follow-up with `chunk32` and no per-expert trace. Do not commit source unless the follow-up strictly exceeds `2.6` and passes all SOTA publication gates.
- `rollback_status`: source patch temporarily retained only for `late10-cpu-fallback-chunk32-no-trace` follow-up; if that follow-up ties/regresses, revert source and clean rebuild.

### 当前执行 attempt：late10-cpu-fallback-chunk32-no-trace

- `attempt_id`: `20260702-late10-cpu-fallback-chunk32-no-trace`
- `attempt_kind`: `implementation/cpu-fallback-scheduling-plus-trace-overhead`
- `status`: completed_rejected_tie_rolled_back
- `bottleneck_basis`: `chunk32` improved TTFT/elapsed in the traced run but did not cross the rounded token-rate gate. Earlier accepted late10 no-trace also tied `2.6` while reducing elapsed. Combining chunk32 with no trace is the smallest follow-up to test whether the secondary wins cross `>2.6`.
- `hypothesis`: With `GGML_MOE_CPU_CHUNK_SIZE=32` and no `GGML_MOE_STREAM_ONE_TRACE_OUT`, CPU fallback scheduling overhead and trace overhead are both lower while model math/cache/routing remain unchanged. This may produce a small rounded token-rate improvement.
- `theoretical_upper_bound`: Traced chunk32 elapsed was `86.59s` versus accepted traced reproduction `87.00s`; no-trace accepted elapsed was `87.18s`. The measured component wins are small and noisy, so the practical bound is a possible rounded move only. Reject on tie.
- `test_config`: patched source, `GGML_MOE_CPU_CHUNK_SIZE=32`, accepted late10 config without `GGML_MOE_STREAM_ONE_TRACE_OUT`, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, `cpu_moe=40`, cold `drop_caches`, strict 16GB cgroup, France prompt, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: promote only if `eval_tok_s > 2.6`, RAM including page cache stays `<=16000000000`, France answer is semantically correct/coherent/complete under manual review, cache hit/miss behavior remains consistent with accepted late10, and TTFT does not exceed current late10 pushed rerun by more than `20%` (`37874.580124ms * 1.2 = 45449.496149ms`).
- `rollback`: If token rate does not exceed `2.6`, output correctness fails, RAM exceeds limit, TTFT exceeds gate, or cache behavior diverges unexpectedly, revert source and clean rebuild. If accepted, immediately stop further experiments, write full reproduction evidence, commit/push source to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before promotion.
- `required_evidence`: source diff, build log/version, exact env/command showing trace disabled, source commit/status, binary/shared-library fingerprints, stdout/stderr cache summary, summary.json, cgroup `memory.*`, France answer text, manual correctness note, and explicit promoted/rejected/rollback status.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T023446Z-20260702_late10_cpu_fallback_chunk32_no_trace/france-cpu40-vram0gb`.
- `result`: rejected tie. `eval_tok_s=2.6`, `prompt_tok_s=1.0`, `TTFT=36173.318643ms`, `elapsed_seconds=86.88`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15038951424`, `pgmajfault=229161`, `workingset_refault_file=3506770`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `correctness_manual_review`: pass. France answer is semantically correct, coherent, complete, and not repetitive/truncated.
- `cache_observation`: stderr reports accepted cache shape (`13.2 GiB`, `3192 slots`, `hits=30180 misses=4971 hit_rate=85.9%`).
- `gap_analysis`: `chunk32` plus no-trace preserved correctness and had low major faults/elapsed, but runner `eval_tok_s` still tied `2.6`. The chunk scheduling change improves secondary timing but not the accepted token-rate metric, so it cannot be promoted. Further chunk-size tuning is deprioritized unless paired with a larger CPU fallback reduction.
- `rollback_status`: source patch reverted and clean rebuild completed; current accepted SOTA remains late10 top3 `2.6 tok/s`.

### 当前执行 attempt：late10-no-trace-n160

- `attempt_id`: `20260702-late10-no-trace-n160`
- `attempt_kind`: `config-probe/generation-budget`
- `status`: completed_rejected_tie_not_sota
- `bottleneck_basis`: Multiple compute/cache/page experiments now tie at `2.6 tok/s`. Current accepted runs use `-n 192`, while the France prompt asks for a short paragraph. Historical `-n 128` failed by truncating the answer, so the next narrow boundary is `-n 160`: shorter than accepted, but with more headroom than the rejected `128`.
- `hypothesis`: Under accepted late10 top3/no-trace config, `-n 160` may still allow a complete, coherent short France paragraph while reducing tail generation work or improving the measured generation-rate window enough to exceed the rounded `2.6 tok/s` gate.
- `theoretical_upper_bound`: This does not change model math, routing, cache misses, or per-token compute. It can only alter output length/stop point and measurement averaging. If the accepted answer naturally needs more than 160 generated tokens, correctness will fail via truncation. If it finishes before 160, performance should tie. Promote only on strict `eval_tok_s > 2.6` and manual completeness pass.
- `test_config`: accepted late10 config without `GGML_MOE_STREAM_ONE_TRACE_OUT`, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, `cpu_moe=40`, cold `drop_caches`, strict 16GB cgroup, France prompt, CLI args `-n 160 -c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: promote only if `eval_tok_s > 2.6`, RAM including page cache stays `<=16000000000`, France answer is semantically correct/coherent/complete under manual review with no final-sentence truncation, and TTFT does not exceed current late10 pushed rerun by more than `20%` (`37874.580124ms * 1.2 = 45449.496149ms`). Because this changes generation budget, require clean pushed-commit rerun before any SOTA promotion.
- `rollback`: No source change. If token rate does not exceed `2.6`, output correctness/completeness fails, RAM exceeds limit, or TTFT exceeds gate, record rejected and keep accepted late10 `-n 192` SOTA.
- `required_evidence`: exact env/command showing `-n 160` and trace disabled, source commit/status, binary sha256/build line/stat, stdout/stderr cache summary, summary.json, cgroup `memory.*`, full France answer text, manual correctness/completeness note, and explicit promoted/rejected status.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T024040Z-20260702_late10_no_trace_n160/france-cpu40-vram0gb`.
- `result`: rejected tie. `eval_tok_s=2.6`, `prompt_tok_s=0.9`, `TTFT=38339.244477ms`, `elapsed_seconds=89.62`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14985310208`, `pgmajfault=283876`, `workingset_refault_file=3490165`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `correctness_manual_review`: pass. France answer is semantically correct, coherent, complete, and not repetitive/truncated despite `-n 160`.
- `cache_observation`: stderr reports accepted cache shape (`13.2 GiB`, `3192 slots`, `hits=30180 misses=4971 hit_rate=85.9%`).
- `gap_analysis`: Reducing generation budget from `-n 192` to `-n 160` did not change output text or improve rounded token rate; the model completed the same paragraph before the lower limit. Generation budget is not the current token-rate boundary as long as correctness is preserved. Keep accepted late10 config as SOTA.
- `rollback_status`: no source change. Current accepted SOTA remains late10 top3 `2.6 tok/s`.

### 当前执行 attempt：late10-no-trace-c192

- `attempt_id`: `20260702-late10-no-trace-c192`
- `attempt_kind`: `config-probe/context-budget`
- `status`: completed_rejected_tie_not_sota
- `bottleneck_basis`: Accepted runs use extra args `-c 256 -b 16 -ub 16`; the prompt plus complete France answer is short, and `-n160` showed output completeness does not require the full 192 generation cap. A smaller context may reduce KV/cache scheduling or memory overhead slightly while preserving correctness.
- `hypothesis`: Lowering context from `-c 256` to `-c 192` under accepted late10/no-trace config may shave small per-token overhead without changing routing/cache policy. It should be rejected immediately if output truncates, semantics degrade, or token rate only ties.
- `theoretical_upper_bound`: Context-size overhead is likely small compared with gate source loading and CPU fallback. This can only improve by a small rounding amount; promote only on strict `eval_tok_s > 2.6` with full France correctness.
- `test_config`: accepted late10 config without `GGML_MOE_STREAM_ONE_TRACE_OUT`, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, `cpu_moe=40`, cold `drop_caches`, strict 16GB cgroup, France prompt, CLI args `-c 192 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: promote only if `eval_tok_s > 2.6`, RAM including page cache stays `<=16000000000`, France answer is semantically correct/coherent/complete under manual review, and TTFT does not exceed current late10 pushed rerun by more than `20%` (`37874.580124ms * 1.2 = 45449.496149ms`).
- `rollback`: No source change. If token rate does not exceed `2.6`, output correctness/completeness fails, RAM exceeds limit, or TTFT exceeds gate, record rejected and keep accepted late10 `-c 256` SOTA.
- `required_evidence`: exact env/command showing `-c 192` and trace disabled, source commit/status, binary sha256/build line/stat, stdout/stderr cache summary, summary.json, cgroup `memory.*`, full France answer text, manual correctness/completeness note, and explicit promoted/rejected status.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T024516Z-20260702_late10_no_trace_c192/france-cpu40-vram0gb`.
- `result`: rejected tie. `eval_tok_s=2.6`, `prompt_tok_s=1.0`, `TTFT=36644.751827ms`, `elapsed_seconds=87.64`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15023489024`, `pgmajfault=293970`, `workingset_refault_file=3329827`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `correctness_manual_review`: pass. France answer is semantically correct, coherent, complete, and not repetitive/truncated under `-c 192`.
- `cache_observation`: stderr reports accepted cache shape (`13.2 GiB`, `3192 slots`, `hits=30180 misses=4971 hit_rate=85.9%`).
- `gap_analysis`: Reducing context from `-c 256` to `-c 192` preserved output quality and cache behavior but still tied `2.6`. Context budget is not the current token-rate boundary for this short France prompt. Keep accepted late10 `-c 256` config as SOTA.
- `rollback_status`: no source change. Current accepted SOTA remains late10 top3 `2.6 tok/s`.

### 当前执行 attempt：late10-no-trace-n160-c192

- `attempt_id`: `20260702-late10-no-trace-n160-c192`
- `attempt_kind`: `config-probe/generation-context-budget-combined`
- `status`: completed_rejected_tie_not_sota
- `bottleneck_basis`: `-n 160` and `-c 192` each preserved France correctness but tied `2.6 tok/s` individually. The combined run is a final low-risk check before deprioritizing generation/context-budget tuning.
- `hypothesis`: Combining `-n 160` and `-c 192` under accepted late10/no-trace config may slightly reduce both generation cap and context overhead while preserving the same complete France answer.
- `theoretical_upper_bound`: Both individual probes tied, so the expected gain is only a small rounded-boundary chance. This does not change model math, routing, cache misses, or core per-token compute. If output changes or truncates, reject immediately; if token rate ties, close this direction.
- `test_config`: accepted late10 config without `GGML_MOE_STREAM_ONE_TRACE_OUT`, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, `cpu_moe=40`, cold `drop_caches`, strict 16GB cgroup, France prompt, CLI args `-n 160 -c 192 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: promote only if `eval_tok_s > 2.6`, RAM including page cache stays `<=16000000000`, France answer is semantically correct/coherent/complete under manual review, and TTFT does not exceed current late10 pushed rerun by more than `20%` (`37874.580124ms * 1.2 = 45449.496149ms`).
- `rollback`: No source change. If token rate does not exceed `2.6`, output correctness/completeness fails, RAM exceeds limit, or TTFT exceeds gate, record rejected and keep accepted late10 `-n 192 -c 256` SOTA.
- `required_evidence`: exact env/command showing `-n 160 -c 192` and trace disabled, source commit/status, binary sha256/build line/stat, stdout/stderr cache summary, summary.json, cgroup `memory.*`, full France answer text, manual correctness/completeness note, and explicit promoted/rejected status.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T025026Z-20260702_late10_no_trace_n160_c192/france-cpu40-vram0gb`.
- `result`: rejected tie. `eval_tok_s=2.6`, `prompt_tok_s=1.0`, `TTFT=36974.048306ms`, `elapsed_seconds=87.65`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15017287680`, `pgmajfault=296806`, `workingset_refault_file=3529668`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `correctness_manual_review`: pass. France answer is semantically correct, coherent, complete, and not repetitive/truncated under combined `-n 160 -c 192`.
- `cache_observation`: stderr reports accepted cache shape (`13.2 GiB`, `3192 slots`, `hits=30180 misses=4971 hit_rate=85.9%`).
- `gap_analysis`: Combining the two accepted-quality budget reductions still tied `2.6` and did not change cache behavior or output. This closes generation/context budget tuning for the France SOTA path; future work should return to structural CPU fallback/routing or larger source-load reductions.
- `rollback_status`: no source change. Current accepted SOTA remains late10 top3 `2.6 tok/s`.

### 当前执行 attempt：late10-cpumoe39-cache13000

- `attempt_id`: `20260702-late10-cpumoe39-cache13000`
- `attempt_kind`: `config-probe/gpu-residency-vs-stream-cache`
- `status`: completed_rejected_regression_not_sota
- `bottleneck_basis`: Current SOTA leaves substantial work in CPU MoE fallback. Prior `cpu_moe=41` with a larger cache regressed because it moved more work to CPU. The opposite direction, `cpu_moe=39`, may move one MoE layer back to GPU and reduce CPU fallback, but it requires reducing stream cache size to fit VRAM.
- `hypothesis`: Lowering `--n-cpu-moe` from `40` to `39` and reducing `GGML_MOE_STREAM_ONE_CACHE_MIB` from `13568` to `13000` may trade a small increase in gate stream misses for reduced CPU fallback on one MoE layer. `13000MiB` previously fit and had only slightly more misses than accepted cache, so this is a bounded VRAM-residency tradeoff.
- `theoretical_upper_bound`: The cache reduction from `13568` to `13000` previously cost about `14` extra gate misses, roughly `14 * 4.9ms ≈ 0.07s` direct source-load cost. The upside is removing one layer of CPU fallback; if CPU fallback is distributed across ~40 layers, one layer could be low single-digit seconds. Practical gain may be erased by VRAM pressure, altered routing, or CUDA OOM. Promote only on strict `eval_tok_s > 2.6`.
- `test_config`: accepted late10 top3 config, `cpu_moe=39`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13000`, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, gate trace enabled, cold `drop_caches`, strict 16GB cgroup, France prompt, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: promote only if `eval_tok_s > 2.6`, RAM including page cache stays `<=16000000000`, no CUDA OOM/cache insertion failure occurs, France answer is semantically correct/coherent/complete under manual review, and TTFT does not exceed current late10 pushed rerun by more than `20%` (`37874.580124ms * 1.2 = 45449.496149ms`).
- `rollback`: No source change. If CUDA OOM, cache insertion failure, token rate not above `2.6`, output correctness failure, RAM failure, or TTFT failure occurs, record rejected and keep `cpu_moe=40/cache13568` as SOTA.
- `required_evidence`: exact env/command, stderr CUDA memory/cache size/slot/free-memory lines, source commit/status, binary sha256/build line/stat, stdout/stderr, summary.json, gate trace, cgroup `memory.*`, full France answer text, manual correctness note, trace summary, and explicit promoted/rejected status.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T025553Z-20260702_late10_cpumoe39_cache13000/france-cpu39-vram0gb`.
- `result`: rejected regression. `eval_tok_s=1.2`, `prompt_tok_s=0.9`, `TTFT=43847.434760ms`, `elapsed_seconds=155.85`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15028203520`, `pgmajfault=327446`, `workingset_refault_file=13051461`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `correctness_manual_review`: pass. France output is semantically correct, coherent, complete, and not repetitive/truncated.
- `trace_summary`: `/root/lfz/runs/vendor-ds4-16gb/20260702T025553Z-20260702_late10_cpumoe39_cache13000/france-cpu39-vram0gb/trace_summary.json` recorded `rows=34268`, `cache_hits=0`, `cache_misses=34268`, `src0_ms=71206.289`, `dontneed_ms=6086.638`, `total_ms=78610.455`.
- `gap_analysis`: The intended win was reducing one CPU fallback MoE layer, but lowering stream cache to `13000MiB` under `cpu_moe=39` caused a complete cache insertion/hit cliff: `cache_hits=0` versus accepted SOTA `cache_hits=30180`, and `src0_ms` grew from about `24.2s` to `71.2s`. `workingset_refault_file` also jumped from about `3.49M` to `13.05M`, so host page churn dominated. This branch is not near the current bottleneck unless a separate VRAM-freeing change preserves at least the accepted `13568MiB` effective stream cache. Keep `cpu_moe=40/cache13568` as SOTA.
- `rollback_status`: no source change. Current accepted SOTA remains late10 top3 `2.6 tok/s`.

### 当前执行 attempt：current-sota-repro-20260702

- `attempt_id`: `20260702-current-sota-repro-after-cpumoe39-reject`
- `attempt_kind`: `reproduction/current-accepted-sota`
- `status`: completed_reproduced_sota
- `bottleneck_basis`: User asked whether the current SOTA is reproducible. The accepted SOTA is late10 top3 with `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, DS4 gate stream enabled, cold `drop_caches`, and strict 16GB cgroup. This run verifies the current pushed code path still reproduces `2.6 tok/s` after rejected probes.
- `hypothesis`: With no source changes after the accepted SOTA path, a cold run from the current pushed branch should reproduce `eval_tok_s≈2.6`, preserve France semantic correctness, keep RAM including page cache within `16000000000` bytes, and keep TTFT below the accepted gate threshold `45449.496149ms`.
- `theoretical_upper_bound`: This is not an optimization attempt; expected result is a tie with accepted SOTA. Any `eval_tok_s <2.6`, correctness failure, RAM breach, or TTFT gate failure means the SOTA is not currently reproducible and must be investigated before further optimization.
- `test_config`: current branch pushed to `ssd/vendor/deepseek-token-rate-16gb`, clean source, `cpu_moe=40`, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, gate trace enabled, cold `drop_caches`, strict 16GB cgroup, France prompt, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: reproduction passes if `eval_tok_s=2.6` or better by the existing rounded metric, RAM including page cache stays `<=16000000000`, France answer is semantically correct/coherent/complete under manual review, and TTFT does not exceed `45449.496149ms`.
- `rollback`: no source change. If reproduction fails, stop optimization and investigate reproducibility before any new tuning.
- `required_evidence`: exact run dir, source commit/status, binary sha256/build line/stat, exact env/command, summary.json, trace summary, cgroup memory evidence, full France answer text, and manual correctness note.
- `source_commit`: `bdad53b29e7683a85e5d669b973ab91b2dffa948` on pushed branch `ssd/vendor/deepseek-token-rate-16gb`; worktree clean before evidence collection.
- `binary`: `build-ds4-moe-stream/bin/llama-cli`, sha256 `c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`, size `1625488`, mtime `2026-07-02 02:37:25 +0000`.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T030252Z-20260702_current_sota_repro_after_cpumoe39_reject/france-cpu40-vram0gb`.
- `result`: reproduced current accepted SOTA. `eval_tok_s=2.6`, `prompt_tok_s=0.9`, `TTFT=37379.491412ms`, `elapsed_seconds=88.98`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15040561152`, `pgmajfault=281682`, `workingset_refault_file=3422905`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `correctness_manual_review`: pass. Output: "France, officially the French Republic, is a country in Western Europe known for its rich history, diverse culture, and significant global influence. It is famous for its iconic landmarks like the Eiffel Tower, the Louvre Museum, and the Palace of Versailles. France is renowned for its cuisine, wine, and fashion, and is a global center for art, philosophy, and science. The country is a founding member of the European Union and is known for its strong economy, particularly in sectors like aerospace, automotive, and luxury goods. With its blend of historical charm and modern vitality, France remains a major cultural and economic force on the world stage."
- `cache_observation`: stderr reports `[moe_stream] VRAM cache: 13.2 GiB, 3192 slots (4.25 MiB each)` and `hits=30180 misses=4971 hit_rate=85.9%`.
- `trace_summary`: `/root/lfz/runs/vendor-ds4-16gb/20260702T030252Z-20260702_current_sota_repro_after_cpumoe39_reject/france-cpu40-vram0gb/trace_summary.json` recorded `rows=35151`, `cache_hits=30180`, `cache_misses=4971`, `src0_ms=24770.360`, `dontneed_ms=1185.232`, `total_ms=27042.197`.
- `repro_conclusion`: current SOTA is reproducible under cold `drop_caches` and strict 16GB cgroup. Metrics match the accepted late10 top3 path; TTFT is below the `45449.496149ms` gate and RAM including page cache is constrained by `memory_peak_bytes=16000000000`.

### 当前执行 attempt：late10-cpu-chunk-bottleneck-trace

- `attempt_id`: `20260702-late10-cpu-chunk-bottleneck-trace`
- `attempt_kind`: `diagnostic/bottleneck-localization`
- `status`: completed_diagnostic_only_not_sota
- `bottleneck_basis`: The reproduced SOTA gate trace accounts for only about `27.0s` of the `88.98s` cold run (`rows=35151`, `cache_hits=30180`, `cache_misses=4971`, `src0_ms=24770.360`, `total_ms=27042.197`). The remaining trace-outside wall is still the largest unresolved component and is expected to be dominated by non-streamed `ffn_up_exps` / `ffn_down_exps` CPU fallback under the accepted late10 top3 routing.
- `hypothesis`: Running the accepted SOTA config with the existing default-off `GGML_MOE_CPU_CHUNK_TRACE_OUT` hook will identify the current post-top3 CPU fallback distribution by tensor, layer, expert, chunk count, and summed chunk time. This should show whether future work should target specific layers/experts, chunk scheduling, page/refault behavior, or a broader routing/streaming strategy.
- `theoretical_upper_bound`: This diagnostic cannot improve token rate and may slow the run because chunk tracing writes many rows. Its value is a hard upper-bound estimate for future optimizations: any proposed CPU fallback optimization cannot save more than the traced up/down fallback component it removes, and any stream replacement must beat the measured CPU fallback cost plus cache/page-in overhead.
- `test_config`: current pushed source, no source changes, `cpu_moe=40`, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, `GGML_MOE_STREAM_ONE_TRACE_OUT=/root/lfz/runs/vendor-ds4-16gb/LATE10_CPU_CHUNK_TRACE_GATE_TRACE`, `GGML_MOE_CPU_CHUNK_TRACE_OUT=/root/lfz/runs/vendor-ds4-16gb/LATE10_CPU_CHUNK_TRACE_CPU_TRACE`, `GGML_MOE_CPU_CHUNK_TRACE_LIMIT=2500000`, cold `drop_caches`, strict 16GB cgroup, France prompt, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: diagnostic passes if the run completes without RAM breach, France output remains semantically correct/coherent enough to trust the trace, and trace files can be summarized. It is not promoted even if rounded `eval_tok_s` ties or improves, because tracing changes runtime behavior.
- `rollback`: no source change. If the run fails or trace files are incomplete, record the failure and do not use the data for optimization decisions.
- `required_evidence`: exact run dir, source commit/status, binary sha256/build line/stat, summary.json, gate trace summary, CPU chunk trace summary by tensor/layer and top-cost entries, cgroup memory evidence, full France answer text, and explicit diagnostic conclusion with next optimization priority.
- `source_commit`: `dfafbbf080492b6b6296c6d7a1160ea54593c76e`, binary sha256 `c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T030820Z-20260702_late10_cpu_chunk_bottleneck_trace/france-cpu40-vram0gb`.
- `result`: diagnostic completed. `eval_tok_s=2.5` with trace overhead, `prompt_tok_s=0.9`, `TTFT=36302.523358ms`, `elapsed_seconds=89.89`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15006539776`, `pgmajfault=276444`, `workingset_refault_file=3599864`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `correctness_manual_review`: pass for diagnostic trust. France output is semantically correct, coherent, complete, and matches the accepted SOTA answer shape.
- `trace_files`: gate trace `/root/lfz/runs/vendor-ds4-16gb/LATE10_CPU_CHUNK_TRACE_GATE_TRACE` (`4.3MB`), CPU chunk trace `/root/lfz/runs/vendor-ds4-16gb/LATE10_CPU_CHUNK_TRACE_CPU_TRACE` (`66MB`, `909952` data rows).
- `summary_files`: `/root/lfz/runs/vendor-ds4-16gb/20260702T030820Z-20260702_late10_cpu_chunk_bottleneck_trace/france-cpu40-vram0gb/cpu_chunk_summary.json`, plus `top_tensor_layer.tsv` and `top_tensor_layer_expert.tsv`.
- `gate_summary`: `rows=35151`, `cache_hits=30180`, `cache_misses=4971`, `src0_ms=24778.664`, `total_ms=27036.245`; this matches the accepted SOTA cache shape.
- `cpu_chunk_summary`: `cpu_rows=909952`, `cpu_parallel_norm_ms=31390.003`. By tensor: `ffn_up_exps` has `427904` chunks, `sum_ms=352519.789`, `norm_ms=17625.989`; `ffn_down_exps` has `482048` chunks, `sum_ms=275280.281`, `norm_ms=13764.014`.
- `top_layers_by_parallel_norm`: top contributors are `ffn_up_exps:L0=1102.408ms`, `ffn_up_exps:L1=1070.596ms`, `ffn_up_exps:L2=1047.312ms`, `ffn_down_exps:L2=826.074ms`, `ffn_down_exps:L1=825.978ms`, `ffn_down_exps:L0=774.254ms`, `ffn_up_exps:L4=670.665ms`, `ffn_up_exps:L3=639.501ms`, `ffn_up_exps:L9=608.899ms`, `ffn_up_exps:L19=570.556ms`.
- `diagnostic_conclusion`: accepted late10 top3 still leaves about `31.4s` parallel-normalized up/down CPU fallback, with early layers `0-2` contributing about `5.65s` before any pruning. Prior `0-9=>top3` failed correctness/speed, but the new trace suggests a narrower `0-2=>top3` plus accepted `10-39=>top3` probe is the next bounded structural experiment. Current source supports only one layer range, so that experiment requires a default-off second-range implementation; do not spend more runs on simple gate cache/thread/page advice until this early0-2 boundary is tested or ruled out.

### 当前执行 attempt：late10-plus-early0-2-top3

- `attempt_id`: `20260702-late10-plus-early0-2-top3`
- `attempt_kind`: `source-probe/layer-selective-approximate-pruning`
- `status`: completed_rejected_correctness_fail_rolled_back
- `bottleneck_basis`: The fresh late10 CPU chunk diagnostic shows remaining up/down CPU fallback `cpu_parallel_norm_ms=31390.003`, with `blk.0-2` contributing about `5646.622ms` (`up0/up1/up2/down0/down1/down2`). The accepted SOTA already prunes `blk.10-39` to top3 while keeping `blk.0-9` at top4. Historical `0-9=>top3` failed correctness/speed, so the next bounded probe should only touch the heaviest early subset `0-2`.
- `hypothesis`: Keeping the accepted policy `10-39=>top3` and additionally applying `0-2=>top3` may reduce the largest remaining early-layer CPU fallback component without the broader trajectory/correctness damage seen from `0-9=>top3`. Layers `3-9` remain top4 as a quality guard.
- `theoretical_upper_bound`: The `0-2` up/down chunk cost is about `5.65s` parallel-normalized before pruning. Reducing one routed up/down expert out of the retained top4 portion gives a rough upper bound around `5.65s / 4 ≈ 1.4s` if cost scales linearly, and likely less after changed routing, gate rows, and trace/cache effects. This is a narrow boundary probe that can only be promoted on strict `eval_tok_s > 2.6`; tie is rejected.
- `implementation`: Extend `ggml_moe_keep_topk_for_tensor()` with default-off second range envs `GGML_MOE_KEEP_TOPK_LAYER_RANGE2=<start>-<end>` and `GGML_MOE_KEEP_TOPK_LAYER_VALUE2=<K>`. Existing env behavior remains unchanged when RANGE2 is unset. Test with fallback `GGML_MOE_KEEP_TOPK_UPDOWN=4`, first range `10-39=>3`, second range `0-2=>3`.
- `test_config`: patched source, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE2=0-2`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE2=3`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, gate trace enabled, cold `drop_caches`, strict 16GB cgroup, France prompt, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: promote only if `eval_tok_s > 2.6`, RAM including page cache stays `<=16000000000`, France answer is semantically correct/coherent/complete under manual review, gate rows/misses do not explode toward the rejected top3 patterns, and TTFT does not exceed current gate threshold `45449.496149ms`.
- `rollback`: If build fails, token rate does not exceed `2.6`, output correctness fails, RAM exceeds limit, TTFT exceeds gate, or trace shows trajectory/cache-miss explosion, revert source and clean rebuild unless the default-off RANGE2 implementation is intentionally retained only after a separate clean default-behavior check. If accepted, immediately stop exploration, write full reproduction evidence, commit/push source to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before promotion.
- `required_evidence`: source diff, build log/hash, exact env/command, source commit/status, binary sha256/build line/stat, stdout/stderr cache summary, summary.json, gate trace summary, cgroup memory evidence, full France answer text, manual correctness note, explicit promoted/rejected/rollback status, and if promoted the complete pushed-commit rerun package.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T032003Z-20260702_late10_plus_early0_2_top3/france-cpu40-vram0gb`.
- `result`: rejected despite candidate speed. `eval_tok_s=2.7`, `prompt_tok_s=0.9`, `TTFT=37985.730697ms`, `elapsed_seconds=105.79`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14999089152`, `pgmajfault=343808`, `workingset_refault_file=5806143`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true` by heuristic only.
- `correctness_manual_review`: fail. The answer starts coherently but is too long/repetitive for "short paragraph" and ends incomplete: `The country is known for its cultural heritage, including the Eiffel Tower, the Louvre, and the`. This violates the required semantic coherence/completeness gate, so the `2.7 tok/s` metric is not acceptable and must not be called SOTA.
- `trace_summary`: `/root/lfz/runs/vendor-ds4-16gb/20260702T032003Z-20260702_late10_plus_early0_2_top3/france-cpu40-vram0gb/trace_summary.json` recorded `rows=47880`, `cache_hits=42053`, `cache_misses=5827`, `cache_inserts=5827`, `src0_ms=29458.220`, `dontneed_ms=1419.604`, `total_ms=32322.431`.
- `gap_analysis`: The narrow early `0-2=>top3` override improved the rounded token rate but changed the generation trajectory in the same qualitative direction as rejected broader top3 probes: gate rows grew from accepted `35151` to `47880`, misses grew from `4971` to `5827`, refaults grew from about `3.4M` to `5.8M`, and output ran into an incomplete tail. Early-layer top3 is therefore quality/trajectory unsafe even when limited to `0-2`.
- `rollback_status`: the RANGE2 source patch was reverted, clean rebuild completed, and `build-ds4-moe-stream/bin/llama-cli` returned to sha256 `c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62` with version `9162 (83bfd9b63)`. Current accepted SOTA remains late10 top3 `2.6 tok/s`; no source push as SOTA.

### 当前执行 attempt：late10-plus-layer0-top3

- `attempt_id`: `20260702-late10-plus-layer0-top3`
- `attempt_kind`: `source-probe/layer-selective-approximate-pruning`
- `status`: completed_rejected_regression_and_correctness_fail_rolled_back
- `bottleneck_basis`: `0-2=>top3` failed manual correctness, but it also showed that early pruning can move the rounded speed metric. To close the boundary carefully, test only the single largest early layer `blk.0`: the CPU chunk diagnostic measured `ffn_up_exps:L0=1102.408ms` and `ffn_down_exps:L0=774.254ms` parallel-normalized, about `1876.662ms` before pruning.
- `hypothesis`: Keeping accepted `10-39=>top3` and additionally applying `0=>top3` may recover a small portion of early CPU fallback while avoiding the severe trajectory/output damage from `0-2=>top3` and prior `0-9=>top3`.
- `theoretical_upper_bound`: Layer 0 up/down cost is about `1.88s` parallel-normalized. Reducing one retained expert out of top4 gives a rough upper bound around `0.47s` if cost scales linearly; this is only a rounding-boundary chance. If token rate ties, regresses, or correctness changes, close this early-layer pruning direction.
- `implementation`: Reuse the default-off RANGE2 patch from the rejected `0-2` probe only for this experiment, then revert unless promoted. Existing behavior must remain unchanged when RANGE2 is unset.
- `test_config`: patched source, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE2=0-0`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE2=3`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, gate trace enabled, cold `drop_caches`, strict 16GB cgroup, France prompt, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: promote only if `eval_tok_s > 2.6`, RAM including page cache stays `<=16000000000`, France answer is semantically correct/coherent/complete under manual review, gate rows/misses stay close to accepted SOTA and do not approach rejected `0-2` behavior, and TTFT does not exceed `45449.496149ms`.
- `rollback`: If token rate does not exceed `2.6`, output correctness fails, RAM exceeds limit, TTFT exceeds gate, or rows/misses/refaults grow materially, revert source and clean rebuild. If accepted, immediately stop exploration, write full reproduction evidence, commit/push source to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before promotion.
- `required_evidence`: source diff, exact env/command, build hash/version, summary.json, gate trace summary, cgroup memory evidence, full France answer text, manual correctness note, rollback/promoted status, and if promoted the full pushed-rerun SOTA package.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T032634Z-20260702_late10_plus_layer0_top3/france-cpu40-vram0gb`.
- `result`: rejected. `eval_tok_s=2.5`, `prompt_tok_s=0.9`, `TTFT=36773.398547ms`, `elapsed_seconds=112.25`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15012184064`, `pgmajfault=368924`, `workingset_refault_file=7251272`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true` by heuristic only.
- `correctness_manual_review`: fail. The answer is semantically plausible but too long for the short-paragraph prompt and ends incomplete after `France is known for its high standard of living and its role as a global leader`. This fails the required coherent/complete output gate.
- `trace_summary`: `/root/lfz/runs/vendor-ds4-16gb/20260702T032634Z-20260702_late10_plus_layer0_top3/france-cpu40-vram0gb/trace_summary.json` recorded `rows=47865`, `cache_hits=41465`, `cache_misses=6400`, `cache_inserts=6400`, `src0_ms=30934.877`, `dontneed_ms=1534.023`, `total_ms=33933.876`.
- `gap_analysis`: Even pruning only layer 0 perturbs the generation trajectory enough to increase rows from accepted `35151` to `47865`, misses from `4971` to `6400`, and refaults to `7.25M`, while token rate regresses to `2.5` and output truncates. This closes early-layer top3 pruning under the current prompt/quality gate; do not continue layer1/layer2 variants without a different quality-preserving mechanism.
- `rollback_status`: the RANGE2 source patch was reverted, clean rebuild completed, and `build-ds4-moe-stream/bin/llama-cli` returned to sha256 `c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62` with version `9164 (9600b5e1b)`. Current accepted SOTA remains late10 top3 `2.6 tok/s`; no source push as SOTA.

### 当前执行 attempt：late10-cpu-fallback-chunk24-no-trace

- `attempt_id`: `20260702-late10-cpu-fallback-chunk24-no-trace`
- `attempt_kind`: `implementation/cpu-fallback-scheduling-boundary`
- `status`: completed_rejected_tie_rolled_back
- `bottleneck_basis`: The latest CPU chunk diagnostic still shows `31390.003ms` parallel-normalized up/down CPU fallback under accepted late10. Previous late10 `chunk32` improved traced elapsed/TTFT slightly but only tied `2.6`; old `chunk8` and old/late `chunk32` evidence suggest the chunk-size optimum, if any, is narrow. After early-layer pruning failed correctness, a final intermediate `chunk24` probe can test whether less aggressive coarsening keeps load balance while reducing scheduling overhead.
- `hypothesis`: Setting multi-row CPU fallback chunk size to `24` may reduce atomic/scheduling overhead versus default `16` while avoiding some load-balance/page-wait regression risk from `32`. Model math, routing, VRAM cache policy, and output semantics should remain unchanged.
- `theoretical_upper_bound`: This cannot reduce gate stream misses or arithmetic count. It can only reduce CPU fallback scheduling/tail overhead inside the remaining `~31.4s` parallel-normalized up/down component. Since `chunk32` tied and improved only secondary timing, expected gain is at most a small rounded-boundary chance. Promote only on strict `eval_tok_s > 2.6`.
- `implementation`: Add default-off env `GGML_MOE_CPU_CHUNK_SIZE=<N>` at the CPU `mul_mat_id` chunk-size selection. Unset env preserves current behavior exactly; `nr0 == 1 || nr1 == 1` still uses `chunk_size=64`. Test only `GGML_MOE_CPU_CHUNK_SIZE=24` with trace disabled to measure production path.
- `test_config`: patched source, `GGML_MOE_CPU_CHUNK_SIZE=24`, accepted late10 config without `GGML_MOE_STREAM_ONE_TRACE_OUT`, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, `cpu_moe=40`, cold `drop_caches`, strict 16GB cgroup, France prompt, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: promote only if `eval_tok_s > 2.6`, RAM including page cache stays `<=16000000000`, France answer is semantically correct/coherent/complete under manual review, cache summary remains accepted-like, and TTFT does not exceed `45449.496149ms`.
- `rollback`: If build fails, token rate ties/regresses, output correctness fails, RAM exceeds limit, TTFT exceeds gate, or refault/page evidence worsens materially, revert source and clean rebuild. If accepted, immediately stop exploration, write full reproduction evidence, commit/push source to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from pushed commit before promotion.
- `required_evidence`: source diff, build hash/version, exact env/command proving trace disabled and chunk24 enabled, summary.json, stdout/stderr cache summary, cgroup memory evidence, full France answer text, manual correctness note, explicit rejected/promoted/rollback status, and if promoted the full pushed-rerun SOTA package.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T033600Z-20260702_late10_cpu_fallback_chunk24_no_trace/france-cpu40-vram0gb`.
- `result`: rejected tie. `eval_tok_s=2.6`, `prompt_tok_s=1.0`, `TTFT=35586.325635ms`, `elapsed_seconds=87.07`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15033331712`, `pgmajfault=261787`, `workingset_refault_file=3519083`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `correctness_manual_review`: pass. France output is semantically correct, coherent, complete, and matches the accepted SOTA answer shape.
- `cache_observation`: trace was disabled as intended; stderr reports accepted cache shape `13.2 GiB`, `3192 slots`, `hits=30180 misses=4971 hit_rate=85.9%`.
- `gap_analysis`: `chunk24` preserved correctness and produced good secondary timing (`elapsed_seconds=87.07`, `TTFT=35586.325635ms`), but the accepted metric still tied `2.6` and did not exceed SOTA. Since `chunk32` also tied and old smaller/larger chunk probes regressed, chunk-size tuning is not a sufficient token-rate lever without a larger CPU fallback reduction.
- `rollback_status`: the `GGML_MOE_CPU_CHUNK_SIZE` source patch was reverted and clean rebuild completed. `build-ds4-moe-stream/bin/llama-cli` sha256 is `c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`; `build-ds4-moe-stream/bin/libggml-cpu.so` sha256 is `f74a75d7d1febcc7f46d9ccfc485b4392985417991fe1e194af8617c8859cc1d`; version `9166 (271567a39)`. Current accepted SOTA remains late10 top3 `2.6 tok/s`.

## 记录与验收

- **硬性 SOTA 复现/push 门禁（不可省略）**：
  - **再次强调：出现符合要求的新 SOTA 时，必须详细记录复现信息并立刻 push 源码，确保未来一定可复现；任何只存在于临时 run、临时 binary、未提交 diff、未 push 分支或口头记录中的指标都不得称为 SOTA。**
  - 一旦出现任何满足硬约束的新最高 token rate（RAM 含 page cache `<=16GB`、正确率通过、TTFT 不超过基线 `+20%`，或按计划标记为 TTFT 待恢复的实验性结果），必须立即把复现信息写完整，不能只口头汇报指标；
  - run 目录必须足够支持未来完全复现：包含 source diff、exact git commit、exact build command、exact run command、全部 env、模型文件路径/size/mtime、binary sha256/build line、16GB cgroup 配置与 memory files、stdout/stderr、trace/summary、correctness 原文输出、TTFT/token-rate/RAM 指标、开始/结束时间；
  - 复现信息写完整后，必须立刻 commit 并 push 源码到 `https://github.com/wici-ai/ssd-llama.git` 的 `vendor/deepseek-token-rate-16gb` 分支，确保后续回退到该分支 commit 时可以重建相同实现；
  - push 后必须从 pushed commit 干净重建并复跑一次，同样记录复跑信息；只有复跑也通过 RAM、正确率、TTFT/token-rate gate，才允许把该结果称为当前有效 SOTA；
  - 如果缺少上述任一项，不得标记 `promoted`，不得称为“可回退 SOTA”，必须继续补证据或降级为 diagnostic/rejected。

- 单次实验必须在 plan 中记录：
  - `eval_tok_s`、`prompt_tok_s`、`TTFT`、`total_ms`、`memory_peak_bytes`、`memory_file_bytes`、`correctness`；
  - 3 分类输出（France、Japan、Climte）与异常日志；
  - 失败原因（OOM、cache miss、退化输出）。
- 结果文件建议放：
  - `/root/lfz/runs/vendor-ds4-16gb/` 对应日期目录（或等效本地路径）；
  - 每次复现实验带 `attempt_end_utc.txt` 和 `wall_clock_elapsed_seconds.txt`。
- 只要出现新 `promoted`：
  - 先确认 run 目录已经完整记录复现信息；若缺少 pushed source commit、binary 指纹、clean rebuild/rerun 命令、16GB cgroup 证据、correctness 原文输出中的任一项，不得 promote；
  - 写 commit message 标注 `vendor-ds4: <关键改动>`；
  - 立即 push 源码到 `https://github.com/wici-ai/ssd-llama.git` 对应的 `vendor/...` 分支（当前工作分支 `vendor/deepseek-token-rate-16gb`）；
  - 在 run 目录中写入 `pushed_remote.txt`、`pushed_branch.txt`、`pushed_commit.txt`、`binary_sha256.txt`、`binary_build_line.txt`、`rebuild_command.txt`、`rerun_command.txt`；
  - 复跑一次 pushed commit 对应源码构建出的 binary，确认指标、正确率、TTFT、16GB cgroup 均仍满足要求；
  - 只有上述复跑通过后，才允许把该 run 标记为 `promoted` 和“可回退 SOTA”。
- 新 SOTA 的不可省略动作：
  - 先把 source diff、exact build/run command、env、model 文件 size/mtime、binary sha256/build line、cgroup memory files、stdout/stderr、trace、summary、correctness 原文输出全部写入 run 目录；
  - 再立刻 commit 并 push 源码到 `https://github.com/wici-ai/ssd-llama.git` 的 `vendor/deepseek-token-rate-16gb` 分支；
  - 再从 pushed commit 干净重建并复跑一次；只有复跑仍通过 16GB RAM、正确率、TTFT、token rate gate，才可宣布为当前 SOTA。
