# DeepSeek V4 Flash Vendor 冷启动 token rate 迭代计划（v16GB）

## 目标

- 在 `vendor` 实现上继续优化 DeepSeek V4 Flash cold-start 解码速度；最终结果必须体现在 `vendor`，`ik_llama` 只能作为参考。
- 任务背景：最终部署目标是一台只有 `16GB host RAM` 和 `32GB RTX 5090 VRAM` 的机器；真实用户会随机输入 prompt，因此优化目标是让随机/泛化 prompt 的输出稳定达到 `>5 tok/s`，而不是让某个固定 prompt 达到高 token rate。
- 最终验收口径：在 strict cold-start、16GB cgroup（含 page cache）、32GB 5090 VRAM 约束下，一个与调优 prompt 分离的 evaluation prompt set 必须稳定达到 `eval_tok_s > 5`。至少固定五 prompt baseline 中每个 prompt 都要接近或超过该线；若扩展为更大随机 prompt holdout，则以 per-prompt 明细和低分位数为准，不能只报平均值或 France 单项。
- 当前分支已回退到 accepted O_DIRECT expert-pack SOTA 记录点 `5d65239a7`，后续优化必须从该点重新设计和推进。
- 严格保持：
  - Host RAM（含 page cache、进程 RSS、cgroup 内所有 file/anon memory）`<= 16 GB`；
  - 尽可能用满 VRAM，但不能造成 CUDA OOM、cache 插入失败或正确率退化；
  - France prompt: `Please introduce France in a short paragraph.` 必须语义正确、连贯、非重复、非截断；
  - 优化目标必须面向泛化 prompt，而不是 prompt-specific。任何 accepted 新 SOTA 不得依赖目标评测 prompt 本身生成的专用 trace、pack、admission profile、专家热集或参数；如需 profile/pack，只能来自与评测 prompt 分离的 calibration set，或来自 prompt-agnostic 的静态模型/张量信息；
  - 在继续实现新的 token-rate 优化前，必须先跑出当前配置的 strict cold 泛化 baseline。baseline 至少覆盖固定五个 prompt：France、quantum computing、Python Fibonacci、Japan、climate change，并逐条记录 token rate、TTFT、16GB cgroup memory/page cache、输出文本和 correctness；
  - accepted SOTA 的 TTFT 不得高于当前 accepted baseline 的 `20%`；若 TTFT 超过 20% 但 token rate 有参考价值，只能标记为 `not accepted`，不得替代 SOTA。


## 2026-07-06 新增限制：先建立泛化 prompt baseline

- `constraint_id`: `20260706-general-prompt-first`
- `status`: active_blocker_before_more_optimization
- `reason`: 之前 accepted France-pack 路径使用 France-derived gate miss pack/profile，已证明对 France 单 prompt 可以达到 SOTA envelope，但不能代表泛化 prompt token rate。后续目标改为泛化 prompt 后，继续围绕 France trace 做优化会得到 prompt-specific 结果，不能作为 accepted SOTA。
- `generalization_rule`: 后续 accepted 优化必须提升跨 prompt 的稳定表现。禁止把单个评测 prompt 的 route trace、answer trace、miss order、expert hot set、prompt-specific O_DIRECT pack、prompt-specific cache admit profile 用作 promoted 配置。任何 prompt-derived artifact 必须来自固定 calibration set，且 evaluation prompt set 必须分离；否则只能标为 diagnostic/prompt-specific，不得替代 accepted generalized SOTA。
- `machine_target`: `16GB host RAM + 32GB RTX 5090 VRAM`; all optimization, profiling, caching, packing, prefetching, and kernel work serves the product requirement that arbitrary user prompts should generate at a stable `>5 tok/s` under this hardware envelope.
- `not_a_goal`: France-only SOTA、prompt-specific expert pack、单 prompt cache hotset、不可泛化 route trace、warm page-cache steady-state、或只在某个演示 prompt 上有效的 token-rate 提升，都不能算完成任务。
- `baseline_prompt_set`:
  1. `Please introduce France in a short paragraph.`
  2. `Explain quantum computing briefly.`
  3. `Write a short Python function for Fibonacci.`
  4. `Introduce Japan in a short paragraph.`
  5. `Summarize climate change in one paragraph.`
- `prompt_split_rule`: 为了避免继续过拟合少数 prompt，后续 prompt 必须分成 `calibration/dev` 和 `held-out test`。`held-out test` 在优化期间不能用于 trace、expert pack、cache admission profile、hotset 选择、参数 sweep、kernel shape 筛选或人工针对性调参；只能在候选方案冻结后用于最终验收。最终 SOTA 必须报告 `held-out test` 指标，不能用 calibration/dev 或 France 单项替代。
- `calibration_dev_set_v1`:
  1. `Please introduce France in a short paragraph.`
  2. `Explain quantum computing briefly.`
  3. `Write a short Python function for Fibonacci.`
  4. `Introduce Japan in a short paragraph.`
  5. `Summarize climate change in one paragraph.`
- `held_out_test_set_v1_locked`: locked before further optimization. Do not use these prompts for design, tracing, packing, profiling, or parameter search.
  1. `Describe photosynthesis in a short paragraph.`
  2. `Give three practical tips for organizing a small home office.`
  3. `Write a concise JavaScript function that checks whether a string is a palindrome.`
  4. `Explain why regular exercise is important in one paragraph.`
  5. `Introduce Brazil in a short paragraph.`
- `sota_metric_update`: A future accepted generalized SOTA requires `held_out_test_set_v1_locked` per-prompt metrics after the candidate is frozen. The acceptance summary must include min/mean eval tok/s, every prompt's TTFT/RAM/page-cache/correctness, and exact outputs. The product target is not met until held-out test prompts are stable at `>5 tok/s` under the 16GB RAM + 32GB 5090 envelope.
- `baseline_required_before_next_code_change`: run the current pushed source/config under strict cold `drop_caches`, `MemoryMax=16000000000`, `MemorySwapMax=0`, page-cache accounting inside cgroup, and record each prompt's `eval_tok_s`, `prompt_tok_s`, `TTFT`, elapsed time, `memory_peak_bytes`, `memory_file_bytes`, OOM/swap status, exact env/CLI, answer text, and manual/automatic correctness note.
- `promotion_update`: A future accepted generalized SOTA must beat the baseline on prompt-set aggregate and must not introduce a severe regression on any individual prompt. France correctness remains a mandatory sentinel, but France alone is no longer sufficient evidence for promotion.
- `next_action`: pause W2/W3 implementation until the generalized baseline artifact is produced, written into this plan, committed, and pushed to `ssd/vendor/deepseek-token-rate-16gb`.
- `baseline_result_20260706_dev_reference`: completed no-prompt-specific strict cold baseline on `calibration_dev_set_v1`, not held-out SOTA. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/general-prompt-baseline-no-prompt-specific-20260706.json`. Config excludes `GGML_MOE_STREAM_ONE_EXPERT_PACK`, `GGML_MOE_STREAM_CACHE_ADMIT_PROFILE`, `GGML_MOE_STREAM_ONE_PREFILL_PROFILE`, and prompt-derived packs/profiles. Results: France `2.7 tok/s`, quantum `1.8 tok/s`, Fibonacci `1.8 tok/s`, Japan `2.4 tok/s`, climate `2.2 tok/s`; mean `2.18 tok/s`, min `1.8 tok/s`, all runs `memory_peak_bytes=16000000000`, all `ram_ok=true`, no prompt-specific env detected. This is a dev/reference baseline only; final SOTA must be measured on `held_out_test_set_v1_locked`.
- `invalidated_result`: `/root/lfz/runs/vendor-ds4-16gb/20260706T105534Z-general-prompt-baseline-current-config-20260706` used France-derived gate pack/profile and is invalid as a generalized baseline. It must not be used for SOTA or target progress.

## 2026-07-06 下一步主线：泛化 CPU fallback 修正

- `next_focus`: CPU fallback 修正仍然是下一步主线。之前的 n96 profile 已显示 `ffn_up_exps`/`ffn_down_exps` CPU fallback 是最大瓶颈；新增泛化要求后，执行方式改为先在 `calibration_dev_set_v1` 上证明它是跨 prompt 共同瓶颈，再修通用 GPU path。
- `why_not_prompt_pack`: 当前目标是随机用户 prompt 稳定 `>5 tok/s`。France-derived gate pack/profile 可以提高 France 单 prompt，但不能泛化；因此后续不得用 prompt-specific pack/profile 作为主要优化方向，也不得用 held-out test prompt 生成任何 hotset。
- `step_1_dev_fallback_profile`:
  - 使用当前 no-prompt-specific baseline 配置；
  - 只使用 `calibration_dev_set_v1`，禁止使用 `held_out_test_set_v1_locked`；
  - 开启 default-off profile：fallback reason、CPU fallback time、name profile、component timing；
  - 每个 prompt 记录 `per-token total time`, `expert read/page fault`, `gate stream`, `up CPU fallback`, `down CPU fallback`, `H2D/D2H`, `CUDA kernel`, `memory_file_bytes`, `workingset_refault_file`；
  - 输出按 prompt 和 aggregate 汇总，确认 up/down fallback 是否是共同主瓶颈。
- `step_1_result_20260706`: completed on `calibration_dev_set_v1` only; held-out test was not used. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/dev-fallback-profile-no-prompt-specific-20260706.json`.
- `step_1_config`: no prompt-specific pack/profile; excluded `GGML_MOE_STREAM_ONE_EXPERT_PACK`, `GGML_MOE_STREAM_CACHE_ADMIT_PROFILE`, `GGML_MOE_STREAM_ONE_PREFILL_PROFILE`; strict cold `drop_caches`; `MemoryMax=16000000000`; `MemorySwapMax=0`; `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`; `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`; `KEEP_TOPK_UPDOWN=4`; `KEEP_TOPK_LAYER_RANGE=10-39`; `KEEP_TOPK_LAYER_VALUE=3`.
- `step_1_metrics`: France `2.6 tok/s`, TTFT `37516.648734ms`, fallback total `31233.213ms` (`up=17327.240ms`, `down=13905.973ms`); quantum `1.9 tok/s`, TTFT `36335.945769ms`, fallback total `56352.754ms` (`up=31869.984ms`, `down=24482.770ms`); Fibonacci `1.7 tok/s`, TTFT `39047.801183ms`, fallback total `56350.986ms` (`up=33298.269ms`, `down=23052.717ms`); Japan `2.4 tok/s`, TTFT `38760.496441ms`, fallback total `32981.767ms` (`up=18715.830ms`, `down=14265.937ms`); climate `2.3 tok/s`, TTFT `39247.277955ms`, fallback total `39616.585ms` (`up=22262.710ms`, `down=17353.875ms`).
- `step_1_memory`: all five runs had `memory_peak_bytes=16000000000`, `ram_ok=true`, no cgroup OOM/kill; file/page-cache was inside the same 16GB cgroup. Workingset file refaults ranged from `3459278` to `14017152`.
- `step_1_conclusion`: up/down CPU fallback is a prompt-general bottleneck. Across the five dev prompts, up fallback totals `123474.033ms`, down fallback totals `93061.272ms`, and gate fallback is `0ms`. All up/down fallback is still blocked by `batch_env_missing` plus `one_name_filter` under the no-prompt-specific config. This validates proceeding to `step_2_down_mxfp4_fallback_fix` before any held-out test run.
- `step_2_down_mxfp4_fallback_fix`:
  - 先修 `ffn_down_exps` 的明确 eligibility mismatch：CPU 侧 `ggml_cuda_moe_stream_supports_down_batch()` 允许 MXFP4/type39，但 CUDA batch wrapper 的 `moe_stream_type_supported()` / `launch_moe_mmvq_compact_batch()` 拒绝 MXFP4，导致 `unsupported_type` 后回到 CPU；
  - 实现必须 default-off，例如 `GGML_MOE_STREAM_DOWN_MXFP4_PROBE=1`；
  - 先做 compact-row CPU vs GPU parity/row-mapping 验证，记录 `max_abs`, `mean_abs`, `max_rel`, active experts, dst/token row mapping；
  - parity 通过后才跑 `calibration_dev_set_v1` performance probe；
  - promoted 条件：down fallback 时间下降、输出正确、16GB RAM/page cache 合规、TTFT 不超 gate、dev set token rate 有稳定提升。否则 revert source，只保留 rejected 记录。
- `step_2_implementation_20260706`: added default-off `GGML_MOE_STREAM_DOWN_MXFP4_PROBE` in `ggml/src/ggml-cuda/moe_stream_batch.cu`. Mode `1` is parity-only: it attempts MXFP4/type39 `ffn_down_exps` compact batch, writes CPU/GPU error rows, then returns `false` so CPU fallback remains the final output. Mode `perf` is reserved for later performance testing after parity passes. Default unset behavior is unchanged.
- `step_2_multirow_fix`: down batch route collection now expands `matrix_row_counts[e]` multirow experts into compact active rows, and sizes temporary dst rows as `max(max_dst_id+1,n_active)`. This fixes the previous `multirow_not_supported` blocker exposed by the MXFP4 probe.
- `step_2_parity_smoke`: diagnostic only, calibration France prompt, `n=16`, no held-out test. Clean run: `/root/lfz/runs/vendor-ds4-16gb/20260706T115617Z-down-mxfp4-probe-parity-clean-smoke-20260706/france-cpu40-vram0gb`. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/down-mxfp4-defaultoff-parity-smoke-20260706.json`.
- `step_2_parity_result`: clean smoke exited `0`, stayed in 16GB cgroup (`memory_peak_bytes=16000000000`, `ram_ok=true`), and generated `8` MXFP4 down parity rows. Error maxima: `max_abs=0.0451961701`, `mean_abs=0.0067439112`, `max_rel=0.0323935935`, `mean_rel=0.0711474475`. Output correctness is intentionally false because `n=16` truncates the answer; this run is not a performance or SOTA run.
- `step_2_vram_note`: with `GGML_MOE_STREAM_ONE_CACHE_MIB=13568` plus `GGML_MOE_VRAM_CACHE_MIB=512`, parity rows were produced but later CUDA allocation OOMed due VRAM pressure. Clean parity used a conservative diagnostic split (`GGML_MOE_STREAM_ONE_CACHE_MIB=8192`, `GGML_MOE_VRAM_CACHE_MIB=512`). The next perf probe must sweep VRAM split conservatively and may not promote a result unless correctness/RAM/TTFT/dev-set metrics all pass.
- `step_3_up_fallback_fix`:
  - `ffn_up_exps` 当前主要因 `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps` 被排除而回到 CPU；
  - 不能简单把 up 加进 France gate cache 或复用 France route hotset；
  - 需要设计 prompt-general DS4 up sparse/grouped GPU decode path，或 up/gate grouped path；
  - 第一版只在 `calibration_dev_set_v1` 上调试和验证，不碰 held-out test；
  - 只有 up fallback 下降且 dev set end-to-end 提升后，才冻结候选跑 held-out test。
- `step_4_candidate_freeze_and_heldout_test`:
  - 当 down/up fallback 修正候选在 calibration/dev set 上稳定后，冻结源码、env、profile/calibration artifacts 和参数；
  - 之后才运行 `held_out_test_set_v1_locked`；
  - final SOTA 必须报告 held-out test 的 per-prompt metrics、min/mean tok/s、TTFT、RAM/page-cache、正确性和完整输出；
  - 若 held-out test 未稳定 `>5 tok/s`，该结果只能算阶段性 dev improvement，不算最终任务完成。

## 2026-07-06 最新执行计划：消灭 up/down CPU fallback

- `basis`: 当前有效优化方向来自 2026-07-06 重新 profile 当前 SOTA 配置，以及 Wafer/GLM-5.2 blog 的方法论复盘。Wafer 的可移用结论不是照搬 AMD/sglang/MTP，而是系统性识别 MoE fp4 路径是否 silently fallback 到慢路径，并为具体 shape 做 kernel mapping/tuning。当前 DeepSeek vendor 的同构问题更直接：decode 阶段 `ffn_up_exps`/`ffn_down_exps` 仍主要落在 CPU fallback。注意：该 profile 来自 France/n96 诊断，只能指导瓶颈方向；W2/W3 的 promoted 目标必须在泛化 prompt baseline 上验证。
- `profile_run`: `/root/lfz/runs/vendor-ds4-16gb/20260706T084029Z-20260706-current-sota-n96-component-profile/france-n96-profile-cpu40-vram0gb`。
- `profile_artifacts`:
  - `.Agent/runs/20260705-vendor-ds4-coldstart/current-sota-n96-component-profile-bottleneck-20260706.json`;
  - `.Agent/runs/20260705-vendor-ds4-coldstart/current-sota-n96-cpu-chunk-analysis-20260706.json`;
  - `.Agent/runs/20260705-vendor-ds4-coldstart/current-sota-n96-name-profile-analysis-20260706.json`.
- `profile_status`: diagnostic only, not accepted SOTA。`n96` 截断 France 输出，结尾停在 `European Union and`，因此 `correctness_ok=false`。该 run 只用于瓶颈拆分，不替代当前 accepted SOTA。
- `profile_metrics`: `eval_tok_s=4.2`, `prompt_tok_s=1.8`, `TTFT=32484.00542ms`, `elapsed=54.80s`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15126450176`, `ram_ok=true`, VRAM peak 约 `31874MiB`。
- `decode_gap`: 首 token 后 decode window 约 `22315.995ms`。若 `n96` 达到 `10 tok/s`，decode 目标时间为 `9600ms`，差额约 `12715.995ms`。
- `component_breakdown`:
  - decode up/down CPU fallback：`13958.513ms`，其中 `up=7088.894ms`, `down=6869.619ms`；这是最大瓶颈，单项已经大于距离 `10 tok/s` 的差额。
  - gate one-stream excluding first prefill marker：`2445.603ms`；其中 gate expert read/cache handling `1506.058ms`，activation pinned staging/H2D `113.851ms`，gate kernel `225.747ms`，D2H/sync/scatter/dontneed 合计 `596.260ms`。
  - cold gate prefill：`4654.732ms`，主要影响 TTFT，不是 decode token rate 的主瓶颈。
  - gate pack/cache 状态：expert pack `hits=4308 misses=0 reads=4308 bytes=19198377984 direct_reads=4308`；VRAM cache `hits=23523 misses=1308 hit_rate=94.7%`。
  - CPU chunk trace：`sum_chunk_ms=398898.148`, 20 线程理想摊平 `19944.907ms`, `max_thread_ms=21156.170ms`，尾部不均衡约 `1.2s`，不是主要矛盾。
- `current_conclusion`: gate O_DIRECT pack、gate VRAM cache 和 activation H2D 已不是优先瓶颈。要接近 `10 tok/s`，必须正面解决 decode up/down CPU fallback；继续只优化 gate read、page cache 或 H2D 的理论收益不足。

### Wafer/GLM 方法可移用点

- `applicable`: 采用 Wafer 式的 MoE fp4 slow-path audit：逐条记录 up/down 为什么没有进入 GPU path，确认是否存在 silently fallback、guard 误判、kernel selection 缺失或 shape mapping 缺失。
- `not_directly_applicable`: AMD MI355X、ROCm preprocessor guard、sglang、TP/DP、allreduce fusion、FP8 KV cache、MTP/spec decode 不能直接移用到当前 CUDA/vendor/单卡/16GB host RAM 约束。MTP 必须排在 up/down GPU path 跑通之后，否则会放大 CPU fallback。
- `main_transfer`: 对 DeepSeek DS4 的 MXFP4/F8 up/down shape 做类似 GLM fp4 MoE kernel mapping/tuning，避免合法的 fp4 MoE 路径落回 CPU 或慢 kernel。

### Phase W1：up/down fallback reason profile（先做，禁止跳过）

- `attempt_id`: `20260706-wafer-style-updown-fallback-reason-profile`
- `attempt_kind`: `measurement-design`
- `hypothesis`: 当前 `13.958s/n96` decode up/down fallback 中，可能混有三类：必须 CPU 算的 fallback、本应走 GPU 但被 eligibility/guard 拒绝的 fallback、以及 GPU path 存在但由于 cache/shape/kernel selection 不满足而 silently fallback 的路径。只有先定量分类，后续代码优化才不会硬猜。
- `required_output`: 对每次 up/down fallback 记录并汇总：`tensor`, `layer`, `role`, `expert_id`, `src0_type`, `phase`, `cne1`, `expert_bytes`, `fallback_ms`, `GPU eligible?`, `decline_reason`, `cache status`, `kernel path`, `row mapping mode`。
- `minimum_summary`: 按 role/layer/reason 输出 calls、fallback_ms、bytes、decode/prompt split；标出 top fallback reasons 和 top tensors。
- `implementation_rule`: 所有 instrumentation 必须 default-off；trace run 不替代 SOTA。不得改变默认计算路径。
- `result_20260706`: implemented default-off `GGML_MOE_FALLBACK_REASON_PROFILE_OUT` in `ggml/src/ggml-cpu/ggml-cpu.c`. It records final CPU fallback rows by role/tensor/phase/expert/type, batch eligibility reason, one-stream reason, final reason, rows/calls/fallback time, and attempt counters. Default behavior is unchanged when the env is unset.
- `validation_short`: `/root/lfz/runs/vendor-ds4-16gb/20260706T103131Z-20260706-wafer-updown-fallback-reason-short/france-n32-fallback-reason-cpu40-vram0gb`; `eval_tok_s=3.5`, `TTFT=33651.457186ms`, `memory_peak_bytes=16000000000`, `ram_ok=true`, `correctness_ok=false` because `n32` truncates the answer. This run verified CSV generation.
- `validation_n96`: `/root/lfz/runs/vendor-ds4-16gb/20260706T103309Z-20260706-wafer-updown-fallback-reason-n96/france-n96-fallback-reason-cpu40-vram0gb`; `eval_tok_s=4.4`, `prompt_tok_s=1.8`, `TTFT=30986.922549ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15119908864`, `ram_ok=true`, `correctness_ok=false` because `n96` truncates the answer. Diagnostic only, not accepted SOTA.
- `result_artifact`: `.Agent/runs/20260705-vendor-ds4-coldstart/wafer-updown-fallback-reason-n96-20260706.json`.
- `reason_result`: all recorded up/down fallback in the n96 diagnostic is classified as `batch_env_missing + one_name_filter -> one_name_filter`. Decode split: `up=6348.759ms`, `down=6201.186ms`; prompt split: `up=3091.818ms`, `down=4490.175ms`. Total classified fallback: `20131.938ms`, `27042` calls, `112.235GiB` logical expert-call bytes.
- `interpretation`: `batch_env_missing` means `GGML_MOE_STREAM_DOWN_BATCH` is not enabled in the accepted SOTA env. `one_name_filter` means `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps` intentionally prevents `ffn_up_exps` and `ffn_down_exps` from entering the one-stream path. Therefore the current bottleneck is not an unknown page-cache issue; it is an explicit routing/policy gap: up/down have no accepted GPU path under the SOTA config. The next step is W2/W3: enable or implement a correct up/down GPU path with numerical validation, not broaden one-stream naively.

### Phase W2：修复可修的 eligibility / guard / kernel selection

- `attempt_id`: `20260706-updown-gpu-eligibility-fix`
- `attempt_kind`: `implementation-probe`
- `hypothesis`: 如果 Phase W1 显示大量 up/down 是因为 name filter、type allowlist、kernel switch、cache lookup、shape guard、Kimi/DeepSeek guard 或 row mapping 判定而 fallback，则先修这些明确原因。Wafer blog 的核心启发是不要接受 fp4 MoE silently 走慢路径。
- `theoretical_bound`: `n96` decode fallback 总计 `13958.513ms`。若修复 eligibility 后能把其中 `X ms` 迁到 GPU，decode tok/s 上界约为 `96 / ((22315.995 - X + gpu_overhead_ms)/1000)`。要达到 `10 tok/s`，净减少量需要约 `12716ms`，所以小于数秒的修复只能算阶段性收益。
- `safety`: 任何 allowlist/switch 改动都必须先做数值对齐，再做性能 run。不能只让 batch_accept 增加；必须证明 CPU fallback profile 下降、输出正确且 token rate 不退化。
- `correctness`: 对同一 expert/row 做 CPU vs GPU `max_abs/mean_abs/max_rel` 对齐；France 输出必须完整、语义正确、连贯。
- `rollback`: 正确性失败、token rate 退化、TTFT 超 gate、RAM 超 16GB、或 fallback 未下降，全部 revert source，只保留 rejected 记录。
- `result_20260706_down_batch_decline_probe`: ran a default-off diagnostic with batch build plus `GGML_MOE_STREAM_DOWN_BATCH=1`, `GGML_MOE_STREAM_DECLINE_DEBUG=1`, and `GGML_MOE_FALLBACK_REASON_PROFILE_OUT`. Run dir: `/root/lfz/runs/vendor-ds4-16gb/20260706T104039Z-20260706-w2-down-batch-decline-n16/france-n16-down-batch-decline-cpu40-vram0gb`. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/w2-down-batch-decline-n16-20260706.json`.
- `result_status`: diagnostic only, not accepted SOTA. `n16` truncates the France answer (`correctness_ok=false`) and is used only to classify the down-batch rejection path. Metrics: `eval_tok_s=3.6`, `prompt_tok_s=1.8`, `TTFT=32971.884727ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15161106432`, `ram_ok=true`, `oom_seen=false`.
- `decline_result`: stderr reported `720` CUDA batch declines with `reason=unsupported_type`, `src0_type=39` for `ffn_down_exps`. The fallback reason CSV classifies down fallback as `eligible + not_attempted_batch_path_selected + batch_declined_internal_no_single_retry`, with prompt down `4493.635ms` and decode down `1375.815ms` in the short n16 run.
- `w2_finding`: CPU-side eligibility and CUDA-side support disagree. `ggml-cpu.c` permits MXFP4/type39 for down batch through `ggml_cuda_moe_stream_supports_down_batch()`, but `ggml/src/ggml-cuda/moe_stream_batch.cu::moe_stream_type_supported()` and `launch_moe_mmvq_compact_batch()` exclude `GGML_TYPE_MXFP4`, so every attempted MXFP4 down batch is rejected internally before compute.
- `w2_next_action`: do not promote naive MXFP4 allowlist changes. Historical large down-pack/batch probes regressed token rate despite correctness fixes. The next implementation must be default-off and guarded: first enable an MXFP4 down-batch probe behind an env flag, then add CPU-vs-GPU numerical validation/row-mapping checks for compact rows before any strict performance run. Only if fallback time falls and end-to-end France correctness/RAM/TTFT gates pass can it be considered for SOTA promotion.


### 2026-07-06 up one-stream 诊断（rejected）

- `attempt_id`: `20260706-up-one-stream-gpuonly-diagnostic`
- `attempt_kind`: calibration-only diagnostic; held-out test set was not used.
- `artifact`: `.Agent/runs/20260705-vendor-ds4-coldstart/up-one-stream-diagnostics-20260706.json`.
- `question`: 普通 `ffn_up_exps` 是否可以简单加入 `GGML_MOE_STREAM_ONE_NAME_FILTER`，从而消灭 up CPU fallback。
- `gate_only_reference_n32`: `/root/lfz/runs/vendor-ds4-16gb/20260706T153652Z-20260706-upgate-decline-diagnostic-n32/france-upgate-decline-cpu40-vram0gb`; no prompt-specific pack/profile; `eval_tok_s=1.9`, `prompt_tok_s=0.9`, `TTFT=38349.266845ms`, `memory_peak_bytes=16000000000`, `ram_ok=true`, truncated answer so not correctness/SOTA; fallback aggregate: up decode `4583.853ms`, up prompt `3893.623ms`, down decode `2980.180ms`, down prompt `4347.132ms`; VRAM cache hit rate `69.0%`.
- `up_cached_gpuonly_n16`: `/root/lfz/runs/vendor-ds4-16gb/20260706T153940Z-20260706-up-one-gpuonly-diagnostic-n16/france-up-one-gpuonly-cpu40-vram0gb`; `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps,ffn_up_exps`, `GGML_MOE_STREAM_ONE_GPU_ONLY_FILTER=ffn_up_exps`; run exited `0`, proving up one-stream can take all up rows in this short calibration path; `eval_tok_s=1.6`, `prompt_tok_s=0.8`, `TTFT=38811.285754ms`, `memory_peak_bytes=16000000000`, `ram_ok=true`, truncated answer; remaining fallback is down only; VRAM cache hit rate fell to `56.7%`.
- `up_no_cache_admit_n32`: `/root/lfz/runs/vendor-ds4-16gb/20260706T154122Z-20260706-up-one-no-cache-admit-diagnostic-n32/france-up-one-nocache-cpu40-vram0gb`; same up GPU-only fail-fast, plus `GGML_MOE_STREAM_CACHE_ADMIT_NAME_FILTER=ffn_gate_exps` so up does not occupy cache slots; run exited `0`, `eval_tok_s=1.5`, `prompt_tok_s=0.8`, `TTFT=40705.368092ms`, `memory_peak_bytes=16000000000`, `ram_ok=true`, truncated answer; remaining fallback is down only; VRAM cache hit rate `44.6%`, showing no-cache up still pays too much per-expert H2D/sync cost.
- `conclusion`: simple widening of `GGML_MOE_STREAM_ONE_NAME_FILTER` to include `ffn_up_exps` is rejected. It eliminates up CPU fallback under fail-fast, but worsens end-to-end token rate because one-stream up performs many per-expert H2D/cuda sync/cache operations and either pollutes the gate cache or repeatedly copies up experts. This path must not be promoted and should not be repeated as a SOTA attempt.
- `next_plan_update`: move W3 priority from simple one-stream up to grouped/batched MXFP4 up decode or true fused/grouped up-gate path. Required next design must compute rows/expert bytes/H2D bytes/launch count and prove CPU-vs-GPU parity before performance runs. Down-only MXFP4 remains rejected unless paired with an up solution because previous down perf removed down fallback but did not improve token rate.

### Phase W3：DS4 shape-specific up/down sparse GPU decode path

- `attempt_id`: `20260706-ds4-updown-sparse-gpu-decode-path`
- `attempt_kind`: `kernel-design-then-implementation`
- `hypothesis`: 当前 gate 的 DS4 one-stream GPU path 已能数值对齐并高命中；up/down 需要按 DeepSeek DS4 的实际 decode shape 做专用 sparse MMV/grouped dispatch，而不是泛化地把全部 up/down expert 加进 gate stream cache。
- `scope_first`: 第一版只覆盖当前 SOTA decode 的跨 prompt 共同热路径：`cpu_moe=40`, `KEEP_TOPK_UPDOWN=4`, `KEEP_TOPK_LAYER_RANGE=10-39`, `KEEP_TOPK_LAYER_VALUE=3`, DS4 MXFP4/F8 native GGUF，以及 fixed prompt-set baseline 中共同出现的 up/down fallback shape。prompt 阶段可先保留 CPU fallback，但 TTFT 不得超过 gate。
- `must_not_repeat`: 不重复 rejected 的 naive gate+up/gate+down 全量 streaming；历史上该路径造成 cache inserts 暴涨、page/refault 恶化和 token rate 下跌。
- `design_requirements`: 写清 tensor size、per-expert bytes、active rows、row mapping、kernel choice、H2D/D2H bytes、workspace、sync 点、理论 IO/compute 上界后才能改代码。
- `success_metric`: decode up/down fallback 在 prompt set 上明显下降，France 正确且五个 baseline prompts 的语义/代码输出不退化，RAM 合规，TTFT gate 合规。若只提升单个 prompt 或 trace 局部但 prompt-set end-to-end token rate 不升，不能 promoted。


### 2026-07-06 up MXFP4 batch probe（rejected, source reverted）

- `attempt_id`: `20260706-up-mxfp4-batch-probe`
- `attempt_kind`: default-off implementation probe; calibration-only; held-out test set was not used.
- `artifact`: `.Agent/runs/20260705-vendor-ds4-coldstart/up-batch-probe-rejected-20260706.json`.
- `source_status`: rejected source diff was saved inside the artifact and then reverted. No promoted code path remains from this attempt.
- `design`: temporarily added `GGML_MOE_STREAM_UP_BATCH_PROBE` to allow ordinary `ffn_up_exps` MXFP4 tensors through the existing compact batch wrapper. `parity` mode ran GPU batch and returned `false` so CPU fallback remained final output; `perf` mode returned GPU output.
- `cache_get_finding`: without `GGML_MOE_VRAM_CACHE_MIB`, batch cache is disabled by runner `--vram-cache-gb 0`; up batch declined with `cache_get`. Batch cache is separate from one-stream `GGML_MOE_STREAM_ONE_CACHE_MIB`.
- `parity_vram512_n16`: `/root/lfz/runs/vendor-ds4-16gb/20260706T155054Z-20260706-up-batch-parity-probe-vram512-n16/france-up-batch-parity-vram512-cpu40-vram0gb`; `GGML_MOE_VRAM_CACHE_MIB=512`, `GGML_MOE_STREAM_ONE_CACHE_MIB=8192`, `GGML_MOE_STREAM_UP_BATCH_PROBE=parity`; first 8 up batch calls wrote parity rows with `max_abs` range about `0.0138201907..0.0305707154`, `mean_abs` about `0.00246..0.00362`; RAM stayed at `16000000000`, no cgroup kill. This validates the small sampled math path only, not promotion.
- `perf_vram512_n32`: `/root/lfz/runs/vendor-ds4-16gb/20260706T155257Z-20260706-up-batch-perf-probe-vram512-n32/france-up-batch-perf-vram512-cpu40-vram0gb`; `eval_tok_s=1.3`, `prompt_tok_s=0.8`, `TTFT=42110.430142ms`, `memory_peak_bytes=16000000000`, `ram_ok=true`, but `correctness_ok=false` with incoherent/repetitive output (`that: France is a nation of independent...`). Batch cache hit rate was only `10.6%` (`hits=621 misses=5229`); up fallback disappeared, but cache churn/H2D/sync and/or insufficient parity coverage made the result slower and incorrect.
- `conclusion`: removing up CPU fallback with this simple compact batch wrapper is not enough and is rejected. The next design must avoid small up batch cache churn and must not D2H/scatter intermediate up results if it can be fused with gate. Candidate directions are prompt-general up hot admission/ranking with a proven gate-cache budget, true fused up/gate grouped compute, or larger grouped schedule with expanded parity before any perf run.

### Phase W4：up/down hot cache / pack / grouped dispatch

- `attempt_id`: `20260706-updown-hot-cache-pack-grouped-dispatch`
- `attempt_kind`: `optimization-after-gpu-path`
- `precondition`: 只有 W2/W3 证明 up/down GPU 计算数值正确且能减少 fallback 后，才进入本阶段。
- `hypothesis`: 如果 GPU up/down path 跑通后被 IO/H2D/launch/sync 卡住，则用 profile-selected hot expert cache、O_DIRECT pack、pinned staging 和 grouped dispatch 降低搬运及 launch overhead。
- `ranking_rule`: up/down cache 不能按 calls 粗排，必须按 `fallback_ms_saved_per_byte` 排序，并验证不会破坏 gate cache hit rate。需要 sweep gate cache 让出 `0.5/1/2GiB` 给 up/down 的代价。
- `theoretical_check`: 每个 up/down expert 约 `4.25MiB`；若仍按每次 miss 读，`n96` decode up/down logical calls `24700` 对应 `~102.5GiB` logical expert bytes，单靠直接读取不可能接近 `10 tok/s`。必须依赖缓存、复用、grouped dispatch 或减少 CPU fallback 工作量。
- `acceptance`: 同 SOTA gate；accepted 新 SOTA 必须立即记录完整复现信息、commit、push 到 `ssd/vendor/deepseek-token-rate-16gb`，并从 pushed commit clean rebuild/rerun。

### 当前执行优先级覆盖

1. 先补跑当前 pushed source/config 的 fixed five-prompt strict cold 泛化 baseline，并写入本计划。
2. 已完成的 W1 fallback reason profile 保留为瓶颈方向证据，但不作为泛化 SOTA 证据。
3. 若 baseline 也显示 up/down fallback 是跨 prompt 共同瓶颈，则继续 W2，小步修复 guard/eligibility/kernel-selection 并做数值对齐。
4. 若 fallback 主要是缺少正确 GPU compute path，进入 W3，做 DS4 shape-specific sparse GPU decode path，但 promoted 验证必须基于 prompt set。
5. 只有 GPU path 正确并在 prompt set 上减 fallback 后，才做 W4 的 cache/pack/grouped dispatch。
6. 暂不优先做 MTP/spec decode、KV cache、TP/DP、allreduce、prompt-specific pack，除非 up/down fallback 已被压下且 prompt-set baseline 证明收益泛化。

## 二次回退状态（2026-07-02）

- 已执行回退：当前源码分支重置到 `5d65239a74c9512967eb557743dc3cb5d1cf6c76`（`vendor-ds4: record odirect pack sota`）。
- 回退目的：撤销后续 rejected plan/checkpoint commits (`760e7f99b`..`3fb1f351d`) 对当前开发基线的影响，重新从已接受的 4.2 tok/s O_DIRECT expert-pack SOTA 制定方案。
- 源码 SOTA commit：`771966944ddb204fe6ec3d321c1e597eee63664e`（`vendor-ds4: add direct one-pack reads`）。
- SOTA 记录 commit：`5d65239a74c9512967eb557743dc3cb5d1cf6c76`。
- push 目标固定：`https://github.com/wici-ai/ssd-llama.git` / `vendor/deepseek-token-rate-16gb`。
- Git identity 固定：`L-Ark <fliangae@connect.ust.hk>`。
- 新 accepted SOTA 出现时必须：
  - 详细记录 source commit、run dir、完整 env/CLI、build command、binary sha256、model/profile/pack sha256、cgroup memory peak/stat/events、TTFT、prompt/eval token rate、耗时、France 原文输出、trace/counters；
  - 立即 commit 并 push 到 `ssd/vendor/deepseek-token-rate-16gb`；
  - 从 pushed commit clean rebuild/rerun，确认未来回退可完全复现指标；
  - 若性能下降、正确率下降、RAM 超 16GB、TTFT 超过 accepted gate，则只可记录为 rejected/not_accepted，不得替代 SOTA。

## 当前 accepted SOTA（回退后基线）

- 范围：France single-prompt strict cold-start accepted SOTA。
- 当前最高 accepted token rate：`4.2 tok/s`（2026-07-02 one-pack O_DIRECT）。
- run dir：`/root/lfz/runs/vendor-ds4-16gb/20260702T133103Z-20260702_pushed_onepack_odirect_sota_rerun/france-cpu40-vram0gb`。
- 配置：
  - `vendor` 框架，`cpu_moe=40`；
  - `GGML_MOE_KEEP_TOPK_UPDOWN=4`；
  - `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`；
  - `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`；
  - `GGML_MOE_STREAM_CACHE_ADMIT_PROFILE=/root/lfz/vendor/llama.cpp-deepseek-v4/.Agent/profiles/vendor-ds4/current_sota_gate_freq_ge2.tsv`；
  - `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`；
  - `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`；
  - `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`；
  - `GGML_MOE_STREAM_ONE_EXPERT_PACK=/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-gate-miss-firstorder-20260702.pack`；
  - `GGML_MOE_STREAM_ONE_EXPERT_PACK_IO=direct`；
  - `--vram-cache-gb 0`；
  - extra args：`-c 256 -b 16 -ub 16 -t 20 -tb 20`；
  - strict cold `drop_caches`，16GB cgroup。
- 指标：
  - `eval_tok_s=4.2`；
  - `prompt_tok_s=1.5`；
  - `TTFT=29984.770823ms`；
  - `memory_peak_bytes=16000000000`；
  - `memory_file_bytes=15080456192`，page cache 计入 16GB cgroup；
  - `pgmajfault=272611`；
  - `workingset_refault_file=1635934`；
  - `ram_ok=true`，`ram_limit_killed=false`，`correctness_ok=true`。
- pack：
  - `/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-gate-miss-firstorder-20260702.pack`；
  - sha256：`7ad26d8b14c20dccd4106a8abbffc9f846eb2fedff4fd00a5af7060941204076`；
  - 4599 unique gate miss pairs，first-miss order，约 20GB。
- binary/source fingerprints：
  - `llama-cli` sha256：`c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`；
  - `libggml-cuda.so` sha256：`26d61a0a23eb0f3d9a64dde89cb2072787cc0c20032248743ce29fb30d841d45`；
  - profile sha256：`8134c320730e0ba236d103ba4a0b53505b3bab16e69d8bdc2a08607ecfcc274b`。
- 计数：
  - one expert pack `hits=4623 misses=0 reads=4623 bytes=20602159104 failures=0 entries=4599 direct_enabled=1 direct_reads=4623 direct_failures=0 direct_fallbacks=0`；
  - VRAM cache `hits=30528 misses=4623 hit_rate=86.8%`。

## 从 4.2 SOTA 重新制定的方案

### Phase 0：冻结、推送并复现 4.2 SOTA

- `attempt_id`: `20260702-rollback-to-odirect-sota-freeze`
- `attempt_kind`: `baseline-repeat`
- `hypothesis`: 回退到 `5d65239a7` 后，SOTA 配置应能再次复现 France cold-start `~4.2 tok/s`，且 RAM/page cache 全部在 16GB cgroup 内。
- `expected_delta`: 无提升预期；目标是确认回退点仍可作为后续优化基线。
- `actions`:
  - clean rebuild 当前 HEAD，确认 build-info 与 `5d65239a7` 一致；
  - force-with-lease push rollback 后的分支到 `ssd/vendor/deepseek-token-rate-16gb`；
  - strict cold `drop_caches` rerun 一次 France SOTA；
  - 如果复现低于 `4.0 tok/s`，先排查 binary/pack/profile/cgroup/direct I/O 差异，不进入新优化。
- `required_evidence`: pushed commit、dirty status、build command、binary sha256、model/pack/profile sha256、exact env/run command、cgroup `memory.peak`/`memory.stat`/`memory.events`、stdout/stderr、summary、trace/counters、France 原文输出。
- `rollback`: 不改源码；如果复现失败，仍停在 `5d65239a7`，只追加 rejected/repro-failed 记录。
- `result_1`: completed on 2026-07-02 after clean rebuild at `5324b90d3` with `GGML_CUDA_MOE_STREAM_BATCH=OFF`. Run dir `/root/lfz/runs/vendor-ds4-16gb/20260702T140255Z-20260702_pre_kimi_latest_merge_odirect_sota_repro/france-cpu40-vram0gb`; `eval_tok_s=4.0`, `prompt_tok_s=1.6`, `TTFT=29108.584955ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15089516544`, `pgmajfault=268771`, `workingset_refault_file=1723767`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `result_2`: second strict cold rerun completed on 2026-07-02. Run dir `/root/lfz/runs/vendor-ds4-16gb/20260702T140551Z-20260702_pre_kimi_latest_merge_odirect_sota_repro2/france-cpu40-vram0gb`; `eval_tok_s=4.1`, `prompt_tok_s=1.5`, `TTFT=30034.533955ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15101833216`, `pgmajfault=268416`, `workingset_refault_file=1638894`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `fingerprints`: `llama-cli sha256=c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`; `libggml-cuda.so sha256=a468b55ae19e7a79914b6b9e05ecf3da7597081de0595fb7d487e045a50e0b89`; `pack_sha256=7ad26d8b14c20dccd4106a8abbffc9f846eb2fedff4fd00a5af7060941204076`; `profile_sha256=8134c320730e0ba236d103ba4a0b53505b3bab16e69d8bdc2a08607ecfcc274b`.
- `counters`: both reruns used O_DIRECT expert pack with `hits=4623 misses=0 reads=4623 bytes=20602159104 failures=0 entries=4599 direct_enabled=1 direct_reads=4623 direct_failures=0 direct_fallbacks=0`.
- `answer`: France output is semantic and coherent; it identifies France/French Republic in Western Europe and mentions history, culture, global influence, landmarks, cuisine/wine/fashion/art/science, EU membership, economy, and modern vitality.
- `decision`: historical accepted SOTA remains `4.2 tok/s` from run `20260702T133103Z-20260702_pushed_onepack_odirect_sota_rerun`; the current clean rebuild reproduces the same accepted configuration at `4.0-4.1 tok/s`, under the 16GB cgroup with correct output. Proceed to merge latest Kimi changes, then require a post-merge SOTA guard in the same range before continuing optimization.

### Kimi latest merge guard（2026-07-02）

- `attempt_id`: `20260702-post-kimi-latest-merge-odirect-sota-guard`
- `merge_commit`: `86ac3c0dd` merged `ssd/vendor/kimi-moe-stream-on-vendor` through `58885e449` into `feat/ds4-moe-stream-on-vendor`.
- `kimi_delta`: from previous merged Kimi base `122b44dc6` to `58885e449`, the branch changed only `.Agent/plans/kimi-token-rate-16gb-optimization-plan.md` (`928` inserted lines). No Kimi source file changed in this merge, so Kimi runtime functionality remains unchanged.
- `build`: clean rebuild after merge with `GGML_CUDA_MOE_STREAM_BATCH=OFF`; CMake build-info commit `86ac3c0dd`; `llama-cli sha256=c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`; `libggml-cuda.so sha256=a468b55ae19e7a79914b6b9e05ecf3da7597081de0595fb7d487e045a50e0b89`.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T140859Z-20260702_post_kimi_latest_merge_odirect_sota_guard/france-cpu40-vram0gb`.
- `config`: same accepted O_DIRECT expert-pack SOTA config: `cpu_moe=40`, `--vram-cache-gb 0`, strict cold `drop_caches`, 16GB cgroup, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERT_PACK_IO=direct`, same pack/profile hashes.
- `result`: `eval_tok_s=4.2`, `prompt_tok_s=1.6`, `TTFT=29080.225912ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15099367424`, `pgmajfault=272003`, `workingset_refault_file=1672143`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `counters`: one expert pack `hits=4623 misses=0 reads=4623 bytes=20602159104 failures=0 entries=4599 direct_enabled=1 direct_reads=4623 direct_failures=0 direct_fallbacks=0`.
- `answer`: France output remains semantic and coherent; it identifies France/French Republic in Western Europe and mentions history, culture, global influence, landmarks, cuisine/wine/fashion/art/science, EU membership, economy, and modern vitality.
- `decision`: accepted guard. The latest Kimi branch is merged without changing Kimi runtime functionality, and the vendor DeepSeek O_DIRECT cold-start SOTA is maintained at `4.2 tok/s` under the 16GB cgroup. Commit and push this plan record to `ssd/vendor/deepseek-token-rate-16gb`, then continue Phase 1 bottleneck slicing from the merged head.

### Phase 1：重新定位 4.2 SOTA 的真实瓶颈

- `attempt_id`: `20260702-odirect-sota-bottleneck-slice`
- `attempt_kind`: `measurement-design`
- `hypothesis`: O_DIRECT expert pack 已把 gate miss source load 从 buffered 版本显著降低；当前主要瓶颈更可能是 residual up/down CPU fallback、H2D/sync 序列化、pack direct-read 与 GPU work 的 overlap 不足，而不是 admission profile 本身。
- `required_measurement`:
  - one-stream trace：`src0_ms`、H2D copy、kernel、sync、cache hit/miss/insert、direct read counters；
  - CPU fallback profile：按 prompt/decode、up/down、layer/expert 汇总；
  - wall-clock per-token breakdown：TTFT、decode elapsed、non-MoE gap；
  - cgroup：`memory.peak`、`memory.stat` 中 file/anon、`memory.events`、`pgmajfault`、`workingset_refault_file`；
  - correctness：France 原文输出必须记录。
- `priority_rule`: 只优先做硬上界最高的部分；若某部分占比低于 5%，先不改。
- `acceptance`: Phase 1 不应改变默认行为。若需要新增 trace，必须 default-off；trace run 不替代 SOTA。
- `result`: completed diagnostic on merged head `d68452865` with strict cold 16GB cgroup and the accepted O_DIRECT expert-pack SOTA config plus `GGML_MOE_STREAM_ONE_TRACE_OUT`, `GGML_KIMI_CPU_MOE_PROFILE=1`, `GGML_KIMI_CPU_MOE_NAME_PROFILE=1`, `GGML_KIMI_CPU_MOE_FALLBACK_PROFILE_OUT`.
- `diag_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T141200Z-odirect-post-kimi-bottleneck-diag`.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T141143Z-20260702_odirect_post_kimi_bottleneck_trace_profile/france-cpu40-vram0gb`.
- `run_result`: `eval_tok_s=4.1`, `prompt_tok_s=1.6`, `TTFT=30207.999303ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15089967104`, `pgmajfault=267869`, `workingset_refault_file=1690515`, `ram_ok=true`, `correctness_ok=true`.
- `one_stream_trace`: `rows=35151`, `hits=30528`, `misses=4623`, `inserts=3337`, `src0_ms=5291.874`, `miss_src0_ms=5269.225`, `hit_src0_ms=22.649`, `kernel_ms=345.971`, `sync_ms=1017.478`, `total_ms=7055.857`, `miss_total_ms=6128.377`, `hit_total_ms=927.480`.
- `cpu_fallback_profile`: `total_fallback_ms=26131.116`; decode fallback `up=9362.934ms`, `down=9376.150ms`; prompt fallback `up=3262.913ms`, `down=4129.119ms`.
- `interpretation`: After O_DIRECT, gate pack/source load is no longer the biggest bottleneck (`~7.1s` one-stream total, `~5.3s` miss source read). Residual up/down CPU fallback dominates (`~26.1s` total, `~18.7s` decode). With `138` decode tokens at about `4.1 tok/s`, eliminating all decode up/down fallback would bound decode speed near `138 / (138/4.1 - 18.739) ~= 9.2 tok/s` before GPU kernel, copy, sync, and scheduling overhead. This is only an upper bound; the next practical target is to prove that a portion of type-39 down/up fallback can really execute on GPU without correctness loss.

### Phase 2：按瓶颈设计候选优化

- `candidate A: residual CPU fallback / up-down`：
  - 先解释为什么会 CPU fallback：当前 gate 走 one-stream VRAM cache/pack；up/down 仍有大量路径没有命中 GPU-resident expert，未命中时落到 CPU fallback。
  - 理论上界必须由 fallback profile 计算：`fallback_ms_saved / decode_wall_time`，并区分 up/down、prompt/decode。
  - 实现方向优先检查 `GGML_TYPE_MXFP4(type=39)` 的 batch/down 支持是否完整。不能只把 type 加入 allowlist；必须确认 cache_get、pack layout、kernel、scatter、同步都真的执行并并行。
  - 如果开启 `GGML_CUDA_MOE_STREAM_BATCH=ON` 或 `GGML_MOE_STREAM_DOWN_BATCH=1`，必须先短跑正确性，再 strict 16GB full run。
- `candidate B: direct I/O overlap / async read`：
  - 理论上界来自 direct pack readbench 与 one-stream trace 的 `src0_ms`，如果剩余 read 时间已小，不能优先做。
  - 可研究 io_uring，但必须先证明当前 `pread(O_DIRECT)` 在 decode 中仍阻塞 GPU；否则 io_uring 只是复杂化。
  - 不接受任何额外 host RAM buffer 导致 cgroup file+anon 超 16GB 的实现。
- `candidate C: prompt-general pack/profile`：
  - France first-miss pack 只能证明 France cold-start SOTA。若要声明不同 prompt steady/cold 稳定，必须构建小 prompt set 的通用 profile/pack，并单独记录每个 prompt 的 correctness、TTFT、tok/s、RAM。
  - 该方向不优先于 cold-start France SOTA，除非用户要求泛化验证。
- `candidate D: VRAM allocation/cache tuning`：
  - 当前 12GB/one-cache 经验不能盲目增加；vram13/14/12800MiB 曾 rejected。新调参必须说明 cache slots、insert failures、CUDA OOM 风险和理论 hit-rate 上界。

### Phase 2A 执行计划：MXFP4(type=39) down-batch probe

- `attempt_id`: `20260702-mxfp4-down-batch-probe`
- `attempt_kind`: `implementation-probe`
- `why_this_first`: The current largest measured bottleneck is decode up/down CPU fallback (`~18.7s`). Previous `GGML_MOE_STREAM_DOWN_BATCH=1` probe declined every route with `unsupported_type tensor=... type=39`. In this codebase, type `39` is `GGML_TYPE_MXFP4`; generic CUDA MMVQ/MMQ code has MXFP4 support, but `moe_stream_batch.cu::moe_stream_type_supported()` excludes it.
- `theoretical_bound`: If down-batch could remove all decode down fallback (`~9.376s`), decode speed upper bound is about `138 / (138/4.1 - 9.376) ~= 5.7 tok/s`. If it also enables more up/gate handoff and later removes up fallback too, absolute decode bound is about `9.2 tok/s`. The first probe only targets down-batch acceptance, so any accepted gain above `4.2 tok/s` is useful; a no-gain acceptance is diagnostic only.
- `implementation_rule`: Add `GGML_TYPE_MXFP4` to the batch-supported type list only behind the already default-off build/runtime path: build with `-DGGML_CUDA_MOE_STREAM_BATCH=ON`, run with `GGML_MOE_STREAM_DOWN_BATCH=1`. Do not change default build behavior or Kimi runtime functionality.
- `safety_checks`: Before full run, do a short strict run (`-n` small if runner allows or direct systemd wrapper if not) with `GGML_MOE_STREAM_DECLINE_DEBUG=1` and batch/profile envs. Verify no `unsupported_type`, no `launch_moe_mmvq_compact_batch` failure, France text starts coherently, and RAM stays under 16GB.
- `full_run_gate`: Full accepted run must use strict cold `drop_caches`, 16GB cgroup, France correctness, TTFT within the 20% gate, and exact counters/profile. If batch accepts but token rate falls or correctness breaks, revert source and record rejected with the decline/failure reason.
- `gap_analysis`: If batch accepts but speed does not improve, inspect batch cache hits/misses, `stage_ms`, `kernel_ms`, `d2h_ms`, cache insert failures, and CPU fallback profile. A path that only accepts but still leaves fallback unchanged is not an optimization.
- `short_probe_result`: Built with `GGML_CUDA_MOE_STREAM_BATCH=ON` after adding `GGML_TYPE_MXFP4` to `moe_stream_type_supported()`. Short strict run (`-n 32`) at `/root/lfz/runs/vendor-ds4-16gb/20260702T142057Z-20260702_mxfp4_down_batch_short_probe/france-cpu40-vram0gb` had `eval_tok_s=2.9`, `prompt_tok_s=1.6`, `TTFT=29331.321816ms`, `memory_peak_bytes=16000000000`, `ram_ok=true`, and no crash. However it is not accepted: output was intentionally truncated by `-n 32`, and down-batch still did not execute. Decline reasons changed from `unsupported_type` to `multirow_not_supported=116` and `cache_get=1244`. The environment showed `GGML_MOE_VRAM_CACHE_GB=0`, so the batch cache was disabled by the accepted SOTA CLI config while one-stream cache remained active (`GGML_MOE_STREAM_ONE_CACHE_MIB=13568`).
- `next_micro_probe`: Run a second short probe with `GGML_MOE_VRAM_CACHE_MIB=512` while keeping one-stream cache at `13568MiB`, still under strict 16GB host RAM. Purpose: determine whether `cache_get` disappears and whether decode down-batch actually accepts. Reject if CUDA OOM, RAM violation, correctness failure, or no batch-profile calls.
- `cache512_probe_result`: short strict run at `/root/lfz/runs/vendor-ds4-16gb/20260702T142357Z-20260702_mxfp4_down_batch_cache512_short_probe/france-cpu40-vram0gb`; `eval_tok_s=2.2`, `prompt_tok_s=1.5`, `TTFT=30749.949228ms`, `memory_peak_bytes=16000000000`, `ram_ok=true`, heuristic correctness true but answer truncated by `-n 32`, so not accepted. `cache_get` disappeared. Batch cache requested `512MiB` but actual free VRAM allowed only `~0.1GiB / 34 slots`. Decline reasons became `launch_moe_mmvq_compact_batch=1244` and `multirow_not_supported=116`. Decode down fallback fell to `236.711ms`, but token rate regressed, so this is still rejected/diagnostic.
- `launch_gap`: `launch_moe_mmvq_compact_batch()` has a switch allowlist that still excludes `GGML_TYPE_MXFP4`, even though it then calls `ggml_cuda_moe_stream_mmvq_dev()` which has MXFP4 support. Next micro-probe: add MXFP4 to that launch switch, rebuild batch-on, rerun the same 512MiB short probe. Accept only as diagnostic unless full France output and full strict run beat current SOTA.
- `launch512_probe_result`: rejected. After adding MXFP4 to the launch switch, short strict run `/root/lfz/runs/vendor-ds4-16gb/20260702T142934Z-20260702_mxfp4_down_batch_launch512_short_probe/france-cpu40-vram0gb` produced `eval_tok_s=1.6`, `prompt_tok_s=1.6`, `TTFT=30942.125987ms`, `memory_peak_bytes=16000000000`, `ram_ok=true`, but `correctness_ok=false` with answer starting `and each paragraph should be approximately...`. The batch path really executed (`batch_accept=1243`, `batch_decline=117`, batch profile `stage=6.086ms/call`, `kernel=0.035ms/call`, `d2h=0.006ms/call`, `total=6.136ms/call`), but output correctness broke and speed regressed. Source changes for MXFP4 down-batch are reverted; do not promote or keep this implementation active.
- `next_after_reject`: The MXFP4 down-batch path requires a separate numerical correctness investigation before any performance run: compare one CPU fallback down row vs GPU `ggml_cuda_moe_stream_mmvq_dev` output for the same MXFP4 expert/src1 row, then reason about row mapping/multirow semantics. Until that proof exists, keep accepted build at `GGML_CUDA_MOE_STREAM_BATCH=OFF` and continue optimization from the O_DIRECT SOTA.
- `revert_guard`: After reverting MXFP4 down-batch source and rebuilding with `GGML_CUDA_MOE_STREAM_BATCH=OFF`, strict cold guard `/root/lfz/runs/vendor-ds4-16gb/20260702T143722Z-20260702_after_reject_revert_odirect_sota_guard/france-cpu40-vram0gb` produced `eval_tok_s=4.1`, `prompt_tok_s=1.5`, `TTFT=30600.945998ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15095304192`, `pgmajfault=275181`, `workingset_refault_file=1675221`, `ram_ok=true`, `correctness_ok=true`. O_DIRECT pack counters matched accepted SOTA: `hits=4623 misses=0 reads=4623 bytes=20602159104 failures=0 entries=4599 direct_enabled=1 direct_reads=4623 direct_failures=0 direct_fallbacks=0`. Current accepted SOTA remains the post-Kimi O_DIRECT run at `4.2 tok/s`; reverted head maintains the same configuration in the `4.1-4.2 tok/s` band.

### Phase 2B 执行计划：MXFP4 down 单流 GPU/CPU 数值对齐

- `attempt_id`: `20260702-mxfp4-down-single-compare`
- `attempt_kind`: `diagnostic-no-source-change`
- `why`: MXFP4 down-batch was rejected because it broke France output and slowed generation. Before any new batch or fusion implementation, verify whether the underlying single-expert CUDA MMVQ result for `ffn_down_exps` matches CPU fallback. The code already has `GGML_MOE_STREAM_COMPARE_CPU_OUT`, which recomputes accepted `ggml_cuda_moe_stream_one()` results with CPU `vec_dot` and writes max/mean error.
- `method`: Rebuild/keep default accepted source with `GGML_CUDA_MOE_STREAM_BATCH=OFF`. Run a short strict cold diagnostic with `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_down_exps`, `GGML_MOE_STREAM_COMPARE_CPU_OUT=<diag>/compare.csv`, `GGML_MOE_STREAM_COMPARE_CPU_LIMIT=16`, and no accepted-SOTA promotion. Keep host RAM under 16GB; use a short `-n` because this is numerical validation, not a correctness/token-rate claim.
- `expected_evidence`: `compare.csv` rows for type `39` down tensors with `max_abs`, `mean_abs`, `max_rel`, expert id, row/col of max difference, plus run output/cgroup metrics. If max errors are small but full batch was wrong, investigate batch row mapping/multirow/scatter. If single down already diverges, investigate MXFP4 CUDA vecdot vs CPU vecdot/dequant semantics before any performance work.
- `rollback`: no source changes. If the diagnostic run gives bad answer or low tok/s, only record it as diagnostic; it cannot affect accepted SOTA.
- `result`: completed on 2026-07-02 without source changes. Diagnostic run `/root/lfz/runs/vendor-ds4-16gb/20260702T144208Z-20260702_mxfp4_down_single_compare/france-cpu40-vram0gb`, diag dir `/root/lfz/runs/vendor-ds4-16gb/20260702T145000Z-mxfp4-down-single-compare`; short run `eval_tok_s=1.3`, `prompt_tok_s=1.1`, `TTFT=37542.067425ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15096692736`, `ram_ok=true`, `correctness_ok=true` (speed is diagnostic only because CPU compare was enabled).
- `compare_summary`: `compare.csv` had 16 rows, all `ffn_down_exps`, type `39`, `cne1=2`, `count=8192` each. `max_abs=3.81469727e-06`, `mean_of_mean_abs=1.05820362e-07`, `max_rel=1.25252972e-06`. Worst row: `blk.0.ffn_down_exps.weight`, expert `222`, CPU `-3.04559422`, GPU `-3.04559803`.
- `interpretation`: MXFP4 single-expert CUDA MMVQ is numerically aligned with CPU fallback for sampled down rows. The rejected batch failure is therefore more likely in batch row mapping, multirow support, compact-output scatter, or active/dst id semantics, not in the underlying MXFP4 vecdot. Next optimization work should debug batch scatter/multirow correctness with a tiny synthetic or traced batch before any performance run.
- `followup_design`: The aligned one-stream DS4 path used `ggml_cuda_moe_stream_mmvq_rows_dev()` because `GGML_MOE_STREAM_ONE_DS4_NONIDS` was unset. The rejected batch compact path used per-row `ggml_cuda_moe_stream_mmvq_dev()`. Next no-source diagnostic: rerun the same down-only compare with `GGML_MOE_STREAM_ONE_DS4_NONIDS=1` to force the single-row `mmvq_dev` path. If this diverges, fix batch by using the rows/id path or repairing `mmvq_dev`; if it remains aligned, return to batch scatter/mapping analysis.
- `nonids_result`: completed no-source diagnostic at `/root/lfz/runs/vendor-ds4-16gb/20260702T144745Z-20260702_mxfp4_down_nonids_compare/france-cpu40-vram0gb`, diag dir `/root/lfz/runs/vendor-ds4-16gb/20260702T145500Z-mxfp4-down-nonids-compare`. With `GGML_MOE_STREAM_ONE_DS4_NONIDS=1`, `compare.csv` again had 16 type-39 down rows with `max_abs=3.81469727e-06`, `mean_of_mean_abs=1.05820362e-07`, `max_rel=1.25252972e-06`; same worst row as the rows-dev run. Therefore both `ggml_cuda_moe_stream_mmvq_rows_dev()` and per-row `ggml_cuda_moe_stream_mmvq_dev()` are numerically aligned in the single-stream path.
- `next_design`: Add a default-off, diagnostic-only batch compare (`GGML_MOE_BATCH_COMPARE_CPU_OUT`) inside `ggml_cuda_moe_stream_batch()` after D2H/scatter for accepted batch calls. It should recompute the accepted batch rows using CPU `vec_dot` with the same `active_experts`, `dst_ids`, `token_ids`, and source rows, and write per-call max/mean error. Only after this localizes the batch error should we change batch behavior for performance.
- `implementation_adjustment`: Prefer reusing the existing `GGML_MOE_STREAM_COMPARE_CPU_OUT` path in `ggml-cpu.c` immediately after a successful `ggml_cuda_moe_stream_batch()` call and before `matrix_row_counts` is cleared. This compares the actual batch-written `dst` with CPU `vec_dot` using the same row mappings, avoids adding CPU quant logic to CUDA code, and remains default-off.

### Phase 3：执行和提交规则

- 每次实践前先在本 plan 中新增 attempt：写 `hypothesis`、理论上界、预期指标、rollback 条件。
- 每次实践后记录时间、run dir、完整指标、France 输出、counters、gap analysis。
- 出现符合要求的新 SOTA：立刻 commit source+plan+metadata，push 到 `ssd/vendor/deepseek-token-rate-16gb`，然后从 pushed commit clean rebuild/rerun。
- 若只是 TTFT 超过 20% 但 token rate 更高：可以 commit/push 为 `not_accepted` 诊断，但不得改写当前 accepted SOTA，并必须写下一步如何压回 TTFT。
- 若性能下降/准确率下降/RAM 超限：记录 rejected，回退源码到上一 accepted SOTA。

### Phase 1：重新定位 SOTA bottleneck

- `attempt_id`: `20260702-sota-bottleneck-slice`
- `attempt_kind`: `measurement-design`
- `hypothesis`: 当前 3.4 tok/s 的主要瓶颈不再是 gate pack miss（pack miss 为 0），而是在 16GB cgroup 下的 pack 读取、page-cache refault、H2D staging、以及未命中 VRAM cache 后的 CPU/GPU 串行调度。
- `expected_output`: 每 token 总耗时拆分：pack read/pread 或 mmap fault、H2D staging、CUDA kernel、D2H/scatter、CPU fallback、synchronization、sampler/非 MoE 部分。
- `priority_rule`: 只优先做硬上界最高的部分；如果某部分占比低于 5%，先不改。
- `rollback`: 只加默认关闭的 trace 或外部分析脚本；若 trace 改动影响性能或正确率，立即回退。

### Phase 1 result：post-Kimi SOTA bottleneck slice

- `attempt_id`: `20260702-phase1-sota-bottleneck-trace-post-kimi`
- `runs`:
  - one-stream + CPU chunk trace: `/root/lfz/runs/vendor-ds4-16gb/20260702T131703Z-20260702_phase1_sota_bottleneck_trace_post_kimi/france-cpu40-vram0gb`, `eval_tok_s=3.3`, `ram_ok=true`, `correctness_ok=true`.
  - CPU MoE profile: `/root/lfz/runs/vendor-ds4-16gb/20260702T131926Z-20260702_phase1_sota_cpu_profile_post_kimi/france-cpu40-vram0gb`, `eval_tok_s=3.3`, `ram_ok=true`, `correctness_ok=true`.
- `one_stream_trace`: `35151` rows, VRAM cache hits `30528`, misses `4623`; expert pack hits `4623`, misses `0`, bytes `20602159104`; summed one-stream `total_ms=15419.15`, `src0_ms=13394.225`, `sync_ms=1251.6`, `kernel_ms=360.769`.
- `cpu_fallback_profile`: residual CPU fallback is still larger than the one-stream gate path: total fallback `28091.121ms`, split into `decode=20304.189ms` and `prompt=7786.932ms`, logical fallback bytes `~158.6GiB`; top1024 unique fallback experts are `4.25GiB` and cover `49.77%` of fallback time.
- `bottleneck_priority`: current SOTA is not GPU kernel bound. Highest-value target is residual up/down CPU fallback and its page/refault behavior; second target is one-stream pack `src0/read` time; kernel fusion alone has low expected payoff.

### Phase 1 next attempt：hot up/down CPU fallback pack mmap

- `attempt_id`: `20260702-hot1024-cpu-fallback-pack-mmap`
- `attempt_kind`: `config/data-pack-probe`
- `hypothesis`: Kimi merge added default-off `GGML_MOE_CPU_FALLBACK_PACK_MMAP`. A 4.25GiB expert pack containing the top1024 residual up/down fallback experts may replace scattered GGUF mmap fallback reads with a compact pack mmap, reducing page-cache refault and CPU fallback wall time while staying under the 16GB cgroup.
- `theoretical_bound`: top1024 covers `13980.911ms` of the measured `28091.121ms` fallback time. Removing all covered fallback wait would be an unrealistic upper bound; a practical target is reducing several seconds if compact pack locality materially improves page-in/refault behavior.
- `pack_generation`: derive a fake trace from `fallback-profile.csv` top1024 rows, then use `.Agent/run-tools/create_ds4_gate_trace_pack.py` with the DeepSeek GGUF to create `/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-fallback-hot1024-20260702.pack`.
- `test_config`: keep accepted gate one-pack env unchanged; additionally set `GGML_MOE_EXPERT_PACK=<hot1024 pack>` and `GGML_MOE_CPU_FALLBACK_PACK_MMAP=1`; strict cold 16GB cgroup; France prompt.
- `rollback`: no source change. Reject if token rate does not exceed `3.4`, TTFT exceeds gate, RAM/correctness fail, or Kimi expert-pack mmap reports high misses/overhead. If accepted SOTA appears, immediately record full reproduction info, commit, push, and clean rerun.
- `result`: rejected on 2026-07-02. Pack generation succeeded from top1024 fallback rows but de-duplicated to 977 entries, size `4.1G`, sha256 `f8b0d537f33b2977b16046ace0c5d35def559a53a603ad0b2cb0a7b85a0cbab9`; run `/root/lfz/runs/vendor-ds4-16gb/20260702T132338Z-20260702_hot1024_cpu_fallback_pack_mmap/france-cpu40-vram0gb` produced `eval_tok_s=3.4`, `TTFT=33606.306413ms`, `memory_peak_bytes=16000000000`, `ram_ok=true`, `correctness_ok=true`, but no improvement over SOTA.
- `gap_analysis`: `kimi_cpu_fallback_pack_mmap` reported `hits=0 misses=35880 bytes=0 fallback_gguf=35880`. Inspection showed this DeepSeek build does not define `GGML_CUDA_MOE_STREAM_BATCH`, so `moe_stream_batch.cu` compiles the stub branch and `ggml_cuda_moe_expert_pack_mmap_ptr()` returns `nullptr`. Enabling the full Kimi batch TU in the DS4 SOTA build is too broad for this probe. The generated 4.1G rejected pack was deleted to restore disk space; it is reproducible from the recorded fake trace and fallback-profile if needed.
- `next_action`: do not continue CPU fallback mmap under the current DS4 build. Return to Phase 2 one-stream pack IO / page-cache slicing, where the active SOTA path already uses real DS4 gate pack reads.

### Phase 2：pack IO / page-cache 路径优化

- `attempt_id`: `20260702-pack-io-cold-path`
- `attempt_kind`: `design-then-execution`
- `hypothesis`: France SOTA 仍需从 20GB pack 中读取约 20.6GB expert 数据；在 16GB cgroup 内，page cache 容量不足以完整容纳 pack，冷启动受 `pread`/page-cache/refault 路径约束。若 direct IO、aligned pinned staging、或更小粒度的 first-order pack layout 能减少 page-cache 污染和 refault，token rate 可能提高。
- `theoretical_bound`: 以当前 reads `20.6GB` 计算，若有效读带宽为 `1.98 GiB/s`，纯 IO 下限约 `10.4s`；若实际 trace 中 pack/read/H2D 远高于该值，gap 就来自小读放大、reclaim、同步或 staging。
- `execution_order`: 先做外部读基准（buffered vs direct）和 trace 切片；只有 direct/aligned 路径显示硬收益，才实现默认关闭的 `GGML_MOE_STREAM_ONE_EXPERT_PACK_IO=direct`。
- `rollback`: direct IO 若 token rate 不升、TTFT 超 20%、RAM 证据不合规、或 correctness 失败，回退源码，只保留 rejected 记录。

### Phase 2 result：one-pack O_DIRECT promoted candidate

- `attempt_id`: `20260702-onepack-odirect-probe`
- `implementation`: added default-off `GGML_MOE_STREAM_ONE_EXPERT_PACK_IO=direct` in `ggml/src/ggml-cuda/moe_stream.cu`. Header/index still use buffered fd; aligned payload reads use an `O_DIRECT` fd when requested; any direct read failure increments counters and falls back to buffered read, preserving correctness. Default behavior is unchanged when env is unset.
- `readbench`: France gate pack sequence (`4623` reads, `19.187GiB`) improved from buffered `8.651684s` (`2.218GiB/s`, cgroup peak `16GB`) to direct `5.451716s` (`3.519GiB/s`, cgroup peak `6492160` bytes).
- `dirty_probe_run`: `/root/lfz/runs/vendor-ds4-16gb/20260702T132829Z-20260702_onepack_odirect_probe_dirty/france-cpu40-vram0gb`.
- `dirty_probe_result`: `eval_tok_s=4.1`, `prompt_tok_s=1.5`, `TTFT=30350.466436ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15092461568`, `pgmajfault=269973`, `workingset_refault_file=1684863`, `ram_ok=true`, `correctness_ok=true`.
- `direct_counters`: one expert pack `hits=4623 misses=0 reads=4623 bytes=20602159104 failures=0 entries=4599 direct_enabled=1 direct_reads=4623 direct_failures=0 direct_fallbacks=0`; VRAM cache `hits=30528 misses=4623 hit_rate=86.8%`.
- `trace_delta`: one-stream traced `src0_ms` dropped from `13394.225ms` to `5353.209ms`; traced `total_ms` dropped from `15419.15ms` to `7132.354ms`; kernel remained small (`342.57ms`).
- `promotion_status`: candidate only until source commit, push, clean rebuild, and pushed-commit strict cold rerun pass.
- `accepted_sota`: promoted on 2026-07-02 after pushed-commit clean rebuild/rerun from commit `771966944ddb204fe6ec3d321c1e597eee63664e` pushed to `https://github.com/wici-ai/ssd-llama.git` / `vendor/deepseek-token-rate-16gb`.
- `accepted_run`: `/root/lfz/runs/vendor-ds4-16gb/20260702T133103Z-20260702_pushed_onepack_odirect_sota_rerun/france-cpu40-vram0gb`.
- `accepted_metrics`: `eval_tok_s=4.2`, `prompt_tok_s=1.5`, `TTFT=29984.770823ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15080456192`, `pgmajfault=272611`, `workingset_refault_file=1635934`, `ram_ok=true`, `correctness_ok=true`.
- `accepted_counters`: one expert pack `hits=4623 misses=0 reads=4623 bytes=20602159104 failures=0 entries=4599 direct_enabled=1 direct_reads=4623 direct_failures=0 direct_fallbacks=0`; VRAM cache `hits=30528 misses=4623 hit_rate=86.8%`.
- `accepted_fingerprints`: `libggml-cuda.so` sha256 `26d61a0a23eb0f3d9a64dde89cb2072787cc0c20032248743ce29fb30d841d45`; `llama-cli` sha256 `c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`; gate pack sha256 `7ad26d8b14c20dccd4106a8abbffc9f846eb2fedff4fd00a5af7060941204076`; admission profile sha256 `8134c320730e0ba236d103ba4a0b53505b3bab16e69d8bdc2a08607ecfcc274b`.
- `accepted_command_delta`: same SOTA command as previous France baseline plus `GGML_MOE_STREAM_ONE_EXPERT_PACK_IO=direct`.

### Phase 3：减少 fallback 与搬运次数

- `attempt_id`: `20260702-miss-path-fusion-design`
- `attempt_kind`: `math-and-kernel-design`
- `hypothesis`: 当前未命中 VRAM expert cache 时仍存在串行 gate/up/down 路径和 CPU fallback/搬运开销；只有把同一 expert 的 gate/up 读取、量化 staging、H2D、kernel 调度真正合并或批量化，才可能超过 IO 上界。
- `required_design_before_code`: 先写清楚 gate/up/down 张量大小、每 token top-k expert 数、每 expert pack entry 字节数、H2D 字节数、GPU 计算 FLOP/带宽上界、CPU fallback 触发条件。
- `success_gate`: 大 token-rate 提升必须伴随 France 正确输出；如果出现“更快但重复点号/语义错误”，立即 reject。
- `rollback`: 算子融合必须默认关闭；任何正确率下降、TTFT gate 失败或 RAM 超限都回退。

### Phase 4：prompt-set 泛化仅作为诊断

- `attempt_id`: `20260702-promptset-diagnostic-only`
- `attempt_kind`: `generalization-diagnostic`
- `scope`: France、quantum、Fibonacci、Japan、climate prompt-set 用于判断方法是否只过拟合 France pack；不替代 France cold-start SOTA。
- `rule`: 若优化只提升 France 但 prompt-set 大幅退化，必须记录为 France-specific；若要作为更通用方案，需要单独建立 prompt-set accepted 标准。

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

## 历史实验记录（回退前，不作为当前执行计划）

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

### 当前执行 attempt：late10-cpu-willneed-no-trace

- `attempt_id`: `20260702-late10-cpu-willneed-no-trace`
- `attempt_kind`: `config-probe/cpu-fallback-page-prefetch`
- `status`: completed_rejected_regression_not_sota
- `bottleneck_basis`: Accepted late10 still spends about `31.4s` parallel-normalized in up/down CPU fallback and cold runs remain at the 16GB cgroup ceiling. Broad `GGML_MOE_CPU_WILLNEED=1` was rejected before late10 top3 pruning, but late10 reduces active up/down experts and refault pressure versus the older top4 path. A single no-source late10 retest can determine whether the changed active set makes existing page prefetch useful.
- `hypothesis`: Enabling existing `GGML_MOE_CPU_WILLNEED=1` may reduce CPU fallback major faults/page stalls by advising active up/down expert pages before compute. Because it prefetches many pages, it may also increase reclaim pressure and regress; this is a bounded one-run retest, not a direction to expand if it fails.
- `theoretical_upper_bound`: It cannot reduce CPU arithmetic or gate stream misses. The hard upside is limited to the page-fault/reclaim portion of the remaining `~31.4s` CPU fallback plus secondary TTFT/refault improvements. If compute dominates or prefetch displaces useful cache, token rate will tie/regress. Promote only on strict `eval_tok_s > 2.6`.
- `test_config`: clean source, no source changes, `GGML_MOE_CPU_WILLNEED=1`, accepted late10 config without `GGML_MOE_STREAM_ONE_TRACE_OUT`, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, `cpu_moe=40`, cold `drop_caches`, strict 16GB cgroup, France prompt, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: promote only if `eval_tok_s > 2.6`, RAM including page cache stays `<=16000000000`, France answer is semantically correct/coherent/complete under manual review, cache summary remains accepted-like, and TTFT does not exceed `45449.496149ms`.
- `rollback`: no source change. If token rate ties/regresses, output correctness fails, RAM exceeds limit, TTFT exceeds gate, or refault/page evidence worsens materially, record rejected and keep accepted late10 SOTA. Do not continue broader/larger CPU prefetch if this fails.
- `required_evidence`: exact env/command proving trace disabled and WILLNEED enabled, source commit/status, binary/shared-library hashes, summary.json, stdout/stderr cache summary, cgroup memory evidence, full France answer text, manual correctness note, and explicit rejected/promoted status.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T034123Z-20260702_late10_cpu_willneed_no_trace/france-cpu40-vram0gb`.
- `result`: rejected regression. `eval_tok_s=2.5`, `prompt_tok_s=1.0`, `TTFT=35387.340617ms`, `elapsed_seconds=88.07`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15029985280`, `pgmajfault=190937`, `workingset_refault_file=3289956`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `correctness_manual_review`: pass. France output is semantically correct, coherent, complete, and matches accepted SOTA answer shape.
- `cache_observation`: trace disabled as intended; stderr reports accepted cache shape `13.2 GiB`, `3192 slots`, `hits=30180 misses=4971 hit_rate=85.9%`.
- `gap_analysis`: WILLNEED reduced major faults versus accepted no-trace/SOTA-like runs (`~261k-296k -> 190937`) and slightly improved TTFT, but token rate regressed to `2.5`. This indicates page-fault count alone is not the current token-rate limiter; broad prefetch adds enough overhead/reclaim disturbance or leaves CPU compute dominant. Do not continue broader CPU fallback prefetch under late10 without a new mechanism.
- `rollback_status`: no source change. Binary/shared-library hashes remain `llama-cli=c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`, `libggml-cpu.so=f74a75d7d1febcc7f46d9ccfc485b4392985417991fe1e194af8617c8859cc1d`, version `9166 (271567a39)`. Current accepted SOTA remains late10 top3 `2.6 tok/s`.

### 当前执行 attempt：global-top3-n160-quality-boundary

- `attempt_id`: `20260702-global-top3-n160-quality-boundary`
- `attempt_kind`: `config-probe/quality-preserving-generation-budget`
- `status`: completed_rejected_correctness_fail_not_sota
- `bottleneck_basis`: Global `GGML_MOE_KEEP_TOPK_UPDOWN=3` produced high speed (`2.7-2.8 tok/s`) but failed manual correctness because the answer continued too long and ended in an incomplete trailing sentence. The first several sentences were semantically correct, so a shorter generation budget may convert the same fast path into a complete short paragraph if it stops after a natural sentence boundary. This is a quality gate experiment, not permission to accept truncation.
- `hypothesis`: Running global top3 with `-n 160` may stop before the known top3 tail degeneration while preserving a coherent short France paragraph. `-n 160` is chosen because accepted late10 remained complete at this budget and it is materially below the rejected top3 `-n 192`, while leaving more room than the historically truncated `-n 128` probe.
- `theoretical_upper_bound`: Compute path is global top3, so generation rate can remain near the rejected `2.7-2.8 tok/s` if output quality passes. Wall time should fall relative to `-n 192` if fewer tokens are generated. The run must be rejected if the final token limit cuts a sentence, if the answer is repetitive/too long, or if semantics degrade.
- `test_config`: clean source, no source changes, `GGML_MOE_KEEP_TOPK_UPDOWN=3`, no layer range override, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, no `GGML_MOE_STREAM_ONE_TRACE_OUT`, cold `drop_caches`, strict 16GB cgroup, France prompt, CLI args `-n 160 -c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: promote only if `eval_tok_s > 2.6`, RAM including page cache stays `<=16000000000`, TTFT does not exceed `45449.496149ms`, and manual review confirms the France answer is a semantically correct, coherent, complete short paragraph with no incomplete final sentence or repetitive tail. Heuristic correctness is insufficient.
- `rollback`: no source change. If token rate does not exceed `2.6`, output is incomplete/repetitive/too long, RAM exceeds limit, or TTFT exceeds gate, record rejected and close this top3 budget-stop direction unless a future stop criterion can end on semantic sentence boundaries rather than token count.
- `required_evidence`: exact env/command showing global top3 and `-n 160`, source commit/status, binary/shared-library hashes, summary.json, stdout/stderr cache summary, cgroup memory evidence, full France answer text, manual correctness/completeness note, and explicit rejected/promoted status.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T034655Z-20260702_global_top3_n160_quality_boundary/france-cpu40-vram0gb`.
- `result`: rejected despite candidate speed. `eval_tok_s=2.7`, `prompt_tok_s=1.0`, `TTFT=37264.537833ms`, `elapsed_seconds=92.41`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15002345472`, `pgmajfault=283387`, `workingset_refault_file=4158551`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true` by heuristic only.
- `correctness_manual_review`: fail. The answer is factually plausible at the start but ends incomplete at `particularly in the areas of art,`. This is still token-limit truncation, not a coherent complete short paragraph, so it cannot be accepted.
- `cache_observation`: trace disabled as intended; stderr reports `13.2 GiB`, `3192 slots`, `hits=34865 misses=5338 hit_rate=86.7%`. Gate rows/misses are higher than accepted late10, consistent with changed top3 trajectory.
- `gap_analysis`: Reducing global top3 budget from `-n 192` to `-n 160` does not solve the quality problem; it only moves the truncation point earlier. The high-speed global top3 path remains unusable without a semantic stop mechanism or a quality-preserving routing policy. Do not promote token-limit clipped top3 outputs as SOTA.
- `rollback_status`: no source change. Binary/shared-library hashes remain `llama-cli=c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`, `libggml-cpu.so=f74a75d7d1febcc7f46d9ccfc485b4392985417991fe1e194af8617c8859cc1d`, version `9166 (271567a39)`. Current accepted SOTA remains late10 top3 `2.6 tok/s`.

### 当前执行 attempt：global-top3-reverse-stop-first-sentence

- `attempt_id`: `20260702-global-top3-reverse-stop-first-sentence`
- `attempt_kind`: `config-probe/semantic-stop-boundary`
- `status`: completed_rejected_too_short_not_sota
- `bottleneck_basis`: Global top3 has the best observed generation rate (`2.7-2.8 tok/s`) and the first sentence is semantically correct, but prior attempts fail because generation continues into a repetitive/incomplete tail. Token-budget clipping (`-n160`) only moves the truncation point. `llama-cli` supports `--reverse-prompt`, so a stop at the first sentence boundary is a bounded way to test whether the fast path can produce a complete short paragraph without relying on token-limit truncation.
- `hypothesis`: With `GGML_MOE_KEEP_TOPK_UPDOWN=3` and `--reverse-prompt ". "`, the model should stop after the first complete sentence. A single complete sentence can be a valid short paragraph if it introduces France coherently. This must be rejected if the period is omitted, if the sentence is too terse to satisfy the prompt, or if the stop behavior creates an interactive/control artifact.
- `theoretical_upper_bound`: Compute path is global top3, so per-token generation rate can stay near `2.7-2.8 tok/s`. Total wall may be lower because fewer tokens are generated, but SOTA promotion is based on measured `eval_tok_s`, RAM, TTFT, and strict manual correctness. This is a stop-policy experiment; it does not reduce underlying per-token CPU fallback cost.
- `test_config`: clean source, no source changes, `GGML_MOE_KEEP_TOPK_UPDOWN=3`, no layer range override, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, no `GGML_MOE_STREAM_ONE_TRACE_OUT`, cold `drop_caches`, strict 16GB cgroup, France prompt, CLI args `-n 192 -c 256 -b 16 -ub 16 -t 20 -tb 20 --reverse-prompt ". "`.
- `acceptance_gate`: promote only if `eval_tok_s > 2.6`, RAM including page cache stays `<=16000000000`, TTFT does not exceed `45449.496149ms`, and manual review confirms the output is a complete, semantically correct, coherent short paragraph with no missing final punctuation, no truncation, and no interactive artifact. Heuristic correctness is insufficient.
- `rollback`: no source change. If token rate does not exceed `2.6`, output is incomplete/too terse/artifacted, RAM exceeds limit, or TTFT exceeds gate, record rejected. If accepted, immediately write full reproduction evidence, commit/push records to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then rerun the pushed config before promotion.
- `required_evidence`: exact env/command showing global top3 and `--reverse-prompt ". "`, source commit/status, binary/shared-library hashes, summary.json, stdout/stderr cache summary, cgroup memory evidence, full France answer text, manual correctness/completeness note, explicit rejected/promoted status, and if promoted the pushed-rerun package.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T035227Z-20260702_global_top3_reverse_stop_first_sentence/france-cpu40-vram0gb`.
- `result`: rejected. `eval_tok_s=1.8`, `prompt_tok_s=1.0`, `TTFT=38839.473918ms`, `elapsed_seconds=47.48`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15146000384`, `pgmajfault=148986`, `workingset_refault_file=90214`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=false`, `correctness_reason=too_short`.
- `correctness_manual_review`: fail. Output was `France is a Western European country known for its rich history, diverse culture, and iconic landmarks` with no final period. `--reverse-prompt ". "` removed the stop sequence from visible output and produced a sentence fragment/too-short answer, so it does not satisfy the required complete short paragraph.
- `cache_observation`: stderr reports `13.2 GiB`, `3192 slots`, `hits=4289 misses=2554 hit_rate=62.7%`; short output did not reach steady cache reuse and token-rate metric regressed.
- `gap_analysis`: CLI reverse prompt is not a viable quality fix in this form: it stops before emitting the sentence-ending punctuation and reduces generated tokens enough that measured token rate falls to `1.8`. A semantic stop would need to preserve the stop punctuation and still produce an adequate paragraph; token/substring stopping cannot be promoted here.
- `rollback_status`: no source change. Binary/shared-library hashes remain `llama-cli=c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`, `libggml-cpu.so=f74a75d7d1febcc7f46d9ccfc485b4392985417991fe1e194af8617c8859cc1d`, version `9166 (271567a39)`. Current accepted SOTA remains late10 top3 `2.6 tok/s`.

### 当前执行 attempt：global-top3-reverse-stop-two-sentence

- `attempt_id`: `20260702-global-top3-reverse-stop-two-sentence`
- `attempt_kind`: `config-probe/semantic-stop-boundary`
- `status`: completed_rejected_token_rate_regression_not_sota
- `bottleneck_basis`: The first reverse-stop probe stopped on `. ` and removed the period, making the output too short. Historical global top3 output has a stable first two-sentence prefix ending in `fields.` before the next sentence begins with `The country is home...`. Stopping on `The country` may preserve two complete sentences and avoid the repetitive/incomplete tail.
- `hypothesis`: With global top3 and `--reverse-prompt "The country"`, visible output should contain the first two complete sentences: a compact introduction of France plus key cultural strengths. This may satisfy the short-paragraph gate while retaining the fast top3 compute path.
- `theoretical_upper_bound`: Per-token compute remains global top3, but output is much shorter than accepted late10. Very short outputs can lower the measured generation-rate metric due fixed overhead, as seen in the first reverse-stop probe (`1.8 tok/s`). Promotion requires measured `eval_tok_s > 2.6`, not only shorter wall time.
- `test_config`: clean source, no source changes, `GGML_MOE_KEEP_TOPK_UPDOWN=3`, no layer range override, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, no `GGML_MOE_STREAM_ONE_TRACE_OUT`, cold `drop_caches`, strict 16GB cgroup, France prompt, CLI args `-n 192 -c 256 -b 16 -ub 16 -t 20 -tb 20 --reverse-prompt "The country"`.
- `acceptance_gate`: promote only if `eval_tok_s > 2.6`, RAM including page cache stays `<=16000000000`, TTFT does not exceed `45449.496149ms`, and manual review confirms the output is a complete, semantically correct, coherent short paragraph with final punctuation and no reverse-prompt artifact.
- `rollback`: no source change. If token rate does not exceed `2.6`, output is incomplete/too terse/artifacted, RAM exceeds limit, or TTFT exceeds gate, record rejected and close substring-stop top3 rescue for this prompt unless a non-fragile semantic stopping mechanism is implemented.
- `required_evidence`: exact env/command showing global top3 and `--reverse-prompt "The country"`, source commit/status, binary/shared-library hashes, summary.json, stdout/stderr cache summary, cgroup memory evidence, full France answer text, manual correctness/completeness note, and explicit rejected/promoted status.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T035635Z-20260702_global_top3_reverse_stop_two_sentence/france-cpu40-vram0gb`.
- `result`: rejected. `eval_tok_s=2.3`, `prompt_tok_s=1.0`, `TTFT=37966.771791ms`, `elapsed_seconds=56.03`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15116042240`, `pgmajfault=174208`, `workingset_refault_file=342049`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `correctness_manual_review`: pass for content. Output is a complete two-sentence short paragraph: `France is a Western European country known for its rich history, diverse culture, and iconic landmarks. It is famous for its cuisine, art, and fashion, with Paris serving as its capital and a global center for these fields.`
- `cache_observation`: stderr reports `13.2 GiB`, `3192 slots`, `hits=10070 misses=3253 hit_rate=75.6%`; the short stopped output does not reach the accepted SOTA generation/cache window.
- `gap_analysis`: This stop target fixes the visible truncation issue, but the measured generation rate falls to `2.3`, below the accepted `2.6`. Substring stop does not provide a valid token-rate improvement because the shorter output changes measurement behavior and cache reuse. Close prompt-specific substring-stop rescue for global top3 unless a future semantic stopping mechanism can preserve both quality and measured rate.
- `rollback_status`: no source change. Binary/shared-library hashes remain `llama-cli=c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`, `libggml-cpu.so=f74a75d7d1febcc7f46d9ccfc485b4392985417991fe1e194af8617c8859cc1d`, version `9166 (271567a39)`. Current accepted SOTA remains late10 top3 `2.6 tok/s`.

### 当前执行 attempt：current-sota-repro-check-20260702T040203Z

- `attempt_id`: `20260702-current-sota-repro-check-040203`
- `attempt_kind`: `repro-check/current-accepted-sota`
- `status`: completed_reproduced_current_sota_not_new_sota
- `bottleneck_basis`: User asked whether the current accepted SOTA is reproducible. This run checks the pushed `ssd/vendor/deepseek-token-rate-16gb` branch at `ae2183e49090dff7bb29eda26ddee2b554b4c4bf` using the accepted late10 top3 configuration, strict cold `drop_caches`, and a 16GB cgroup including page cache.
- `test_config`: clean source, no source changes, binary hashes `llama-cli=c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`, `libggml-cpu.so=f74a75d7d1febcc7f46d9ccfc485b4392985417991fe1e194af8617c8859cc1d`, `cpu_moe=40`, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, cold `drop_caches`, strict 16GB cgroup, France prompt, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: reproduction passes if `eval_tok_s=2.6` or better by the existing rounded metric, RAM including page cache stays `<=16000000000`, France answer is semantically correct/coherent/complete under manual review, and TTFT does not exceed `45449.496149ms`.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T040203Z-20260702_sota_repro_check_040203/france-cpu40-vram0gb`.
- `result`: reproduced current accepted SOTA. `eval_tok_s=2.6`, `prompt_tok_s=1.0`, `TTFT=36099.909549ms`, `elapsed_seconds=87.39`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15036350464`, `pgmajfault=291555`, `workingset_refault_file=3568217`, `memory_max_events=25829`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `correctness_manual_review`: pass. Output is a complete, coherent France paragraph: `Here is a short paragraph introducing France: France, officially the French Republic, is a country in Western Europe known for its rich history, diverse culture, and significant global influence. It is famous for its iconic landmarks like the Eiffel Tower, the Louvre Museum, and the Palace of Versailles. France is renowned for its cuisine, wine, and fashion, and is a global center for art, philosophy, and science. The country is a founding member of the European Union and is known for its strong economy, particularly in sectors like aerospace, automotive, and luxury goods. With its blend of historical charm and modern vitality, France remains a major cultural and economic force on the world stage.`
- `ram_evidence`: cgroup `memory.peak=16000000000`, `memory.stat file=15036350464`, `active_file=10401259520`, `inactive_file=4635029504`, `oom=0`, `oom_kill=0`. Page cache is included inside the cgroup limit.
- `trace_note`: `GGML_MOE_STREAM_ONE_TRACE_OUT` was passed, but the path did not pre-exist under the runner's timestamped run directory, so no new trace file was produced. This check is still valid as a SOTA reproduction because exact command/env, stdout, summary, cgroup memory files, and output correctness are recorded; previous accepted trace shape remains `/root/lfz/runs/vendor-ds4-16gb/20260702T030252Z-20260702_current_sota_repro_after_cpumoe39_reject/france-cpu40-vram0gb/trace_summary.json`.
- `conclusion`: Current SOTA remains reproducible at `2.6 tok/s` under the strict 16GB host RAM/page-cache constraint. This is not a new SOTA and does not require source changes.

### 当前执行 attempt：late10-cpu-phase-trace-current-sota

- `attempt_id`: `20260702-late10-cpu-phase-trace-current-sota`
- `attempt_kind`: `diagnostic/bottleneck-localization`
- `status`: planned
- `bottleneck_basis`: The current accepted late10 SOTA is reproducible at `2.6 tok/s`, but the known traces still leave a large unlocalized wall-time gap. The accepted gate trace accounts for about `27.0s` of work, and the CPU chunk trace accounts for about `31.4s` parallel-normalized up/down fallback. The remaining cold wall includes model load, row grouping, top-k pruning zero fills, stream synchronization, CPU fallback loop overhead outside chunk timing, and cgroup reclaim/page-cache effects. Recent source/config probes are mostly tied or rejected, so the next design step should split this residual time before another structural change.
- `hypothesis`: A default-off phase trace around `mul_mat_id` can separate per-op row grouping/pruned-zero bytes, stream phase wall, remaining CPU fallback wall, active expert/row counts, and total op wall under the exact accepted late10 config. If pruned zeroing or stream sync is material, a low-risk implementation target exists; if the residual is dominated by unavoidable load/reclaim or broad CPU loop wall, future work should avoid small syscall/cache tweaks and focus on larger CPU fallback math/page-touch reductions.
- `theoretical_upper_bound`: This diagnostic cannot improve token rate. It only bounds future optimizations. For example, a zero-fill optimization cannot save more than the measured prune zero phase; a stream-sync optimization cannot save more than the measured stream phase overhead beyond existing gate trace totals; CPU loop scheduling cannot save more than the measured loop wall that is not already explained by chunk compute. Any follow-up optimization must compare its theoretical saving against the `2.6 tok/s` rounded gate.
- `implementation`: Add default-off env `GGML_MOE_CPU_PHASE_TRACE_OUT=<path>` in `ggml/src/ggml-cpu/ggml-cpu.c`. When enabled, record one CSV row per `mul_mat_id` op from thread 0 with tensor name, layer, `n_ids`, selected keep-topk, pruned entries/bytes, remaining active experts/rows, stream phase wall, CPU fallback loop wall, and total op wall. The trace may add diagnostic-only barriers and timing calls; unset env must preserve existing behavior exactly.
- `test_config`: patched diagnostic source, `GGML_MOE_CPU_PHASE_TRACE_OUT=/root/lfz/runs/vendor-ds4-16gb/LATE10_CPU_PHASE_TRACE_CURRENT`, accepted late10 config without performance trace promotion, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, `cpu_moe=40`, cold `drop_caches`, strict 16GB cgroup, France prompt, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: diagnostic is usable only if RAM including page cache stays `<=16000000000`, the run is not cgroup-killed, France output remains semantically correct/coherent/complete enough to trust the trace, and the phase trace can be summarized. It is not promoted even if rounded `eval_tok_s` ties or improves, because tracing changes runtime behavior.
- `rollback`: After collecting the diagnostic, revert the phase-trace source patch and clean rebuild. If the run fails, record the failure and do not use incomplete phase data for optimization decisions. No source commit/push as SOTA unless a later non-traced clean run with a real optimization passes all SOTA gates.
- `required_evidence`: source diff, build log/version, exact env/command, summary.json, stdout/stderr, cgroup memory evidence, full France answer text, manual correctness note, phase trace CSV, summarized phase totals by tensor/layer/phase, explicit diagnostic conclusion, and rollback status.
- `sota_publish_requirement`: If any follow-up implementation based on this diagnostic produces a compliant new SOTA, immediately write complete reproduction information and push source to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from the pushed commit before declaring it current SOTA.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T041023Z-20260702_late10_cpu_phase_trace_current_041023/france-cpu40-vram0gb`.
- `result`: completed diagnostic only, not promoted. `eval_tok_s=2.5` with trace overhead, `prompt_tok_s=0.9`, `TTFT=38025.110244ms`, `elapsed_seconds=90.76`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15037526016`, `pgmajfault=296558`, `workingset_refault_file=3603558`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `correctness_manual_review`: pass for diagnostic trust. France output is semantically correct, coherent, complete, and matches the accepted SOTA answer shape.
- `trace_files`: `cpu_phase_trace.csv` copied into the run dir, `16920` rows, `1.4MB`; `cpu_phase_summary.json` copied into the run dir; source diff and rollback rebuild log are also stored in the run dir.
- `phase_summary`: summed phase trace shows `group_ms=82.008`, `stream_ms=27308.181`, `cpu_loop_ms=39000.853`, `total_ms=66450.030`. By kind: gate has `stream_ms=27282.562` and no remaining CPU rows after stream; up has `cpu_loop_ms=21591.384`; down has `cpu_loop_ms=17346.019`.
- `prune_zero_observation`: pruning zero-fill is not a useful SOTA lever under late10. The trace counted `33440` pruned entries and `410910720` zeroed bytes, but all grouping plus zero-fill time was only `82.008ms`; even a perfect zero-fill optimization cannot move the rounded `2.6 tok/s` metric.
- `next_bottleneck_priority`: continue targeting broad up/down CPU fallback first (`~39.0s` traced phase wall; chunk trace `~31.4s` parallel-normalized) or gate stream miss/source-load cost (`~27.3s`). Do not spend another run on memset/grouping/zero optimization unless a future trace contradicts this result.
- `rollback_status`: diagnostic source patch reverted with `git apply -R`, clean rebuild completed, and hashes returned to accepted SOTA binaries: `llama-cli=c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`, `libggml-cpu.so=f74a75d7d1febcc7f46d9ccfc485b4392985417991fe1e194af8617c8859cc1d`, version `9177 (a99478d5b)`. Worktree clean after rollback.

### 当前执行 attempt：late10-last10-up-only-top2

- `attempt_id`: `20260702-late10-last10-up-only-top2`
- `attempt_kind`: `implementation/tensor-selective-approximate-pruning`
- `status`: planned
- `bottleneck_basis`: Phase trace closed zero-fill/grouping as a useful target and confirmed broad up/down CPU loop remains the largest localized component (`up cpu_loop_ms=21591.384`, `down cpu_loop_ms=17346.019`). Prior `30-39=>top2` for both up and down changed the generation trajectory too much, causing rows/misses/refaults to grow and the answer to truncate. A narrower tensor-specific version may reduce some CPU fallback while perturbing model math less than pruning both projections.
- `hypothesis`: Keep accepted late10 top3 for both up/down (`10-39=>top3`) but apply top2 only to `ffn_up_exps` in layers `30-39`. This removes one routed up expert in the last ten CPU-MoE layers while leaving down projection at accepted top3. It may preserve the France trajectory better than the rejected up+down last10 top2 run and still save enough CPU fallback to cross the rounded `2.6 tok/s` gate.
- `theoretical_upper_bound`: Phase trace measured `ffn_up_exps` layers `30-39` at about `4656.601ms` CPU loop wall. Reducing top3 to top2 removes one of three retained rows in those layers, so the optimistic linear upper bound is `4656.601 / 3 ≈ 1552ms` before cache/trajectory effects. This is only a small rounded-boundary opportunity; if gate rows/misses increase like the rejected both-projection top2 attempt, the branch must be rejected.
- `implementation`: Extend `ggml_moe_keep_topk_for_tensor()` with default-off second-range envs `GGML_MOE_KEEP_TOPK_LAYER_RANGE2=<start>-<end>`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE2=<K>`, and optional `GGML_MOE_KEEP_TOPK_NAME_FILTER2=<substring>`. Range2 has priority over range1 only when set and when the optional name filter matches. Existing default, fallback, and range1 behavior must remain unchanged when range2 is unset.
- `test_config`: patched source, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE2=30-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE2=2`, `GGML_MOE_KEEP_TOPK_NAME_FILTER2=ffn_up_exps`, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, gate trace enabled, cold `drop_caches`, strict 16GB cgroup, France prompt, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: promote only if `eval_tok_s > 2.6`, RAM including page cache stays `<=16000000000`, France answer is semantically correct/coherent/complete under manual review, gate rows/misses do not explode toward the rejected top2/broad-top3 patterns, and TTFT does not exceed `45449.496149ms`.
- `rollback`: If build fails, token rate does not exceed `2.6`, output correctness fails, RAM exceeds limit, TTFT exceeds gate, or trace shows large trajectory/cache regression, revert source and clean rebuild. If accepted, immediately stop exploration, write full reproduction evidence, commit/push source to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from the pushed commit before promotion.
- `required_evidence`: source diff, build log/version, exact env/command, source commit/status, binary sha256/build line/stat, stdout/stderr, summary.json, gate trace and trace summary, cgroup `memory.*`, France answer text, manual correctness note, and explicit promoted/rejected/rollback status.
- `sota_publish_requirement`: If this run becomes a compliant SOTA, it must be documented and pushed immediately to `ssd-llama` branch `vendor/deepseek-token-rate-16gb`; otherwise it remains rejected/diagnostic even if some secondary timing improves.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T041644Z-20260702_late10_last10_up_only_top2_041644/france-cpu40-vram0gb`.
- `result`: rejected. `eval_tok_s=2.6`, `prompt_tok_s=1.0`, `TTFT=36400.835295ms`, `elapsed_seconds=108.98`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15007051776`, `pgmajfault=358293`, `workingset_refault_file=6784588`, `ram_ok=true`, `ram_limit_killed=false`, heuristic `correctness_ok=true` but manual correctness failed.
- `correctness_manual_review`: fail. The answer is broadly about France but ends incomplete at `a founding member of the European Union`, has no final punctuation, and contains the misspelling `Fratinité`; it is not a complete coherent short paragraph under the required gate.
- `trace_summary`: `/root/lfz/runs/vendor-ds4-16gb/20260702T041644Z-20260702_late10_last10_up_only_top2_041644/france-cpu40-vram0gb/trace_summary.json` recorded `rows=47872`, `cache_hits=41515`, `cache_misses=6357`, `cache_inserts=6357`, `src0_ms=31342.722`, `dontneed_ms=1520.689`, `total_ms=34332.266`.
- `gap_analysis`: Tensor-selective up-only top2 did not preserve the accepted late10 trajectory. It produced essentially the same gate rows/misses shape as rejected both-projection last10 top2 (`47872/6357`) and increased refault pressure (`~3.6M` in phase trace / `~3.4M` accepted repro -> `6.78M`). The theoretical `~1.55s` CPU-loop saving is erased by longer generation, higher gate miss cost, and quality failure. Do not continue plain top2 pruning variants (`down-only`, `last5-only`, or more late top2) without a new quality/trajectory-preserving mechanism.
- `rollback_status`: source patch reverted with `git apply -R`, clean rebuild completed, and hashes returned to accepted SOTA binaries: `llama-cli=c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`, `libggml-cpu.so=f74a75d7d1febcc7f46d9ccfc485b4392985417991fe1e194af8617c8859cc1d`, version `9179 (ccc0ec584)`. Worktree clean after rollback.

### 当前执行 attempt：late10-gate-cache-admit-current-min2

- `attempt_id`: `20260702-late10-gate-cache-admit-current-min2`
- `attempt_kind`: `implementation/profile-guided-cache-admission`
- `status`: planned
- `bottleneck_basis`: Phase trace shows gate stream remains a large localized component (`stream_ms≈27308ms`), and accepted SOTA gate trace has `4971` misses and `24747ms` miss `src0_ms`. Cache-size-only tuning reached the VRAM cliff and only reduced misses by `35`. A cache admission policy may reduce eviction pollution without using more VRAM.
- `profile_analysis`: Accepted SOTA trace has `35151` gate rows, `4599` unique `(tensor,expert)` keys, `1286` keys used once, and `3313` keys used at least twice versus `3192` cache slots. Simulating the current LRU with an admission profile that inserts only keys with frequency `>=2` gives `4623` misses versus current `4971`, an estimated `348` fewer misses and about `1756.6ms` less miss `src0_ms`. Higher thresholds regress because they stop caching too many reusable keys.
- `hypothesis`: Add a default-off admission profile for `moe_stream.cu` so cache misses still compute normally, but insertion occurs only when `(tensor,expert)` appears in `GGML_MOE_STREAM_CACHE_ADMIT_PROFILE`. Using the current SOTA `freq>=2` gate profile may reduce one-use cache pollution, lower gate miss/source-load time, and possibly cross the rounded `2.6 tok/s` boundary without changing model math or output trajectory.
- `theoretical_upper_bound`: The hard simulated `src0_ms` saving is only about `1.76s`, plus fewer evictions and less DONTNEED/reclaim churn. This is smaller than the remaining CPU fallback wall and may only tie. It must be rejected if token rate does not exceed `2.6`, if profile parsing overhead offsets the miss reduction, or if the trace does not show the expected miss drop.
- `implementation`: Add default-off env `GGML_MOE_STREAM_CACHE_ADMIT_PROFILE=<tsv>` in `ggml/src/ggml-cuda/moe_stream.cu`. The TSV contains `tensor<TAB>expert<TAB>count`; when unset, all current cache lookup/insert behavior is unchanged. When set, cache lookup still works for existing entries, but `vram_cache_insert()` is skipped for keys not in the profile. Test only gate stream with accepted late10 config.
- `test_config`: patched source, profile `/root/lfz/runs/vendor-ds4-16gb/cache_admit_profiles/current_sota_gate_freq_ge2.tsv` (`3313` entries from accepted SOTA trace), `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `GGML_MOE_STREAM_CACHE_ADMIT_PROFILE=<profile>`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, `cpu_moe=40`, gate trace enabled, cold `drop_caches`, strict 16GB cgroup, France prompt, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: promote only if `eval_tok_s > 2.6`, RAM including page cache stays `<=16000000000`, France answer is semantically correct/coherent/complete under manual review, TTFT does not exceed `45449.496149ms`, and trace confirms miss/source-load improvement without trajectory regression. Because this is profile-guided, the profile must be recorded and pushed with the reproducible SOTA package if accepted.
- `rollback`: If build fails, token rate ties/regresses, output correctness fails, RAM exceeds limit, TTFT exceeds gate, or trace does not show a meaningful miss/source-load reduction, revert source and clean rebuild. If accepted, immediately stop exploration, copy the profile into a repo-tracked reproduction location, write full reproduction evidence, commit/push source/profile to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`, then clean rebuild/rerun from the pushed commit before promotion.
- `required_evidence`: source diff, profile path/hash/entry count, build log/version, exact env/command, source commit/status, binary sha256/build line/stat, stdout/stderr, summary.json, gate trace and trace summary, cgroup `memory.*`, France answer text, manual correctness note, and explicit promoted/rejected/rollback status.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T042451Z-20260702_late10_gate_cache_admit_current_min2_042451/france-cpu40-vram0gb`.
- `status_update`: candidate_pending_pushed_rerun. This run passes first-run gates but is not yet promoted until source/profile are pushed and a clean rebuild from the pushed commit reproduces the result.
- `profile_repo_path`: `.Agent/profiles/vendor-ds4/current_sota_gate_freq_ge2.tsv`, `3313` entries, sha256 `8134c320730e0ba236d103ba4a0b53505b3bab16e69d8bdc2a08607ecfcc274b`.
- `result`: candidate improvement. `eval_tok_s=2.7`, `prompt_tok_s=0.9`, `TTFT=38312.537695ms`, `elapsed_seconds=88.43`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15020081152`, `pgmajfault=288061`, `workingset_refault_file=2947058`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `correctness_manual_review`: pass. France output is semantically correct, coherent, complete, and matches the accepted SOTA answer shape with no repetition or truncation.
- `trace_summary`: `/root/lfz/runs/vendor-ds4-16gb/20260702T042451Z-20260702_late10_gate_cache_admit_current_min2_042451/france-cpu40-vram0gb/trace_summary.json` recorded `rows=35151`, `cache_hits=30528`, `cache_misses=4623`, `cache_inserts=3337`, `src0_ms=22886.808`, `dontneed_ms=1126.618`, `total_ms=25110.875`.
- `gap_analysis`: The admission profile behaved close to simulation: misses dropped by `348` (`4971 -> 4623`) and gate `src0_ms` dropped by about `1.86s` versus the accepted SOTA trace (`24770.360 -> 22886.808`) without changing row count or output trajectory. This is a plausible real improvement, but it remains profile-guided and must be validated from a pushed clean rebuild before promotion.
- `next_required_action`: commit and push `ggml/src/ggml-cuda/moe_stream.cu`, `.Agent/profiles/vendor-ds4/current_sota_gate_freq_ge2.tsv`, and this plan to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`; then clean rebuild and rerun the same strict 16GB cold-start config using the repo-tracked profile path. Only if that rerun also passes may this become current SOTA.
- `pushed_commit`: `c01354cf2e8f15a2829188ddd5aeffaf088f7e91` (`vendor-ds4: add gate cache admission sota candidate`), pushed to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`.
- `pushed_rerun_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T042858Z-20260702_gate_cache_admit_min2_pushed_rerun_042858/france-cpu40-vram0gb`.
- `pushed_rerun_result`: promoted current SOTA. `eval_tok_s=2.7`, `prompt_tok_s=0.9`, `TTFT=37760.282814ms`, `elapsed_seconds=87.12`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15032840192`, `pgmajfault=289624`, `workingset_refault_file=2953216`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `pushed_rerun_correctness_manual_review`: pass. France output is semantically correct, coherent, complete, and not repetitive/truncated.
- `pushed_rerun_trace_summary`: `rows=35151`, `cache_hits=30528`, `cache_misses=4623`, `cache_inserts=3337`, `src0_ms=22756.581`, `dontneed_ms=1110.238`, `total_ms=24953.040`. The row count matches accepted SOTA while misses stay reduced by `348`.
- `pushed_rerun_repro_files`: run dir contains `pushed_commit.txt`, `pushed_remote.txt`, `pushed_branch.txt`, `binary_and_profile_sha256.txt`, `binary_build_line.txt`, `rebuild_command.txt`, `rerun_command.txt`, `gate_trace.csv`, `trace_summary.json`, `summary.json`, stdout/stderr, and cgroup `memory.*`.
- `promotion_decision`: promoted as current accepted vendor cold-start SOTA because the pushed-commit clean rebuild rerun improved the accepted `2.6 tok/s` to `2.7 tok/s`, kept host RAM including page cache inside the strict `16000000000` byte cgroup, kept TTFT below the `45449.496149ms` gate, preserved France correctness, and recorded/pushed the source plus profile required for reproduction.

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

### 当前执行 attempt：merge-kimi-split-pool-and-migrate-ds

- `attempt_id`: `20260702-merge-kimi-split-pool-and-migrate-ds`
- `attempt_kind`: `merge-and-port/kimi-split-pool`
- `status`: planned
- `source_branch`: `https://github.com/wici-ai/ssd-llama.git` branch `vendor/kimi-moe-stream-on-vendor`, fetched as `refs/remotes/ssd/vendor/kimi-moe-stream-on-vendor`, head `580aaa4fd8696009c46cebab9be0d29e492b104e`.
- `requested_scope`: merge all new Kimi branch changes without breaking either side's functionality, then migrate the Kimi split-pool optimization described in `docs/development/kimi-moe-split-pool-optimization.md` to further improve DeepSeek vendor token rate.
- `current_ds_sota_before_merge`: `2.7 tok/s` from pushed commit `c01354cf2e8f15a2829188ddd5aeffaf088f7e91` plus promotion record `907a5e6dc6319d0dd6f1779948a1bd383d6e729d`; config uses accepted late10 top3, gate-only `moe_stream.cu`, repo-tracked cache admission profile `.Agent/profiles/vendor-ds4/current_sota_gate_freq_ge2.tsv`, cold drop_caches, strict 16GB cgroup.
- `kimi_split_pool_summary`: Kimi split-pool is implemented in `ggml/src/ggml-cuda/moe_stream_batch.cu` and enabled by `GGML_MOE_VRAM_CACHE_SPLIT=1`, `GGML_MOE_VRAM_CACHE_SPLIT_MAX_MIB=6`, and `GGML_MOE_VRAM_CACHE_UPGATE_PCT=60`. It helps Kimi because small up/gate experts and larger down experts otherwise share a fixed-slot cache and waste VRAM.
- `deepseek_migration_hypothesis`: DeepSeek current SOTA uses the one-expert gate stream path in `moe_stream.cu`, not the Kimi batch path. A direct Kimi env copy is unlikely to help unless DeepSeek's active cached tensors have multiple size classes or unless the split-pool mechanism is adapted to the one-cache path. The first merge must preserve Kimi batch split-pool and DeepSeek gate admission, then measure whether DeepSeek can benefit from split pools or whether admission/profile tuning remains the correct mechanism.
- `merge_strategy`: perform a real merge from `ssd/vendor/kimi-moe-stream-on-vendor` into the current DeepSeek branch. Resolve deletes conservatively: keep existing DeepSeek `.Agent` plan/profile/run-tools and accepted SOTA evidence, add Kimi docs/plans/scripts, and merge code files so Kimi batch split-pool and DeepSeek DS4 gate/admission features coexist. Do not accept branch-side deletion of DeepSeek files.
- `validation_before_experiment`: after merge conflict resolution, build `build-ds4-moe-stream`; run a smoke or SOTA reproduction if feasible to ensure DeepSeek gate-admission path still works before testing new split-pool variants. Any compile failure or regression must be fixed before performance claims.
- `deepseek_split_pool_design_gate`: before running DeepSeek split-pool performance tests, inspect expert sizes and cache traces. If DeepSeek gate-only stream has one effective expert size, split-pool has no theoretical benefit for the current SOTA path and should not be promoted. If adapting split-pool to one-stream path, define explicit envs, cache budgets, expected miss reduction, and rollback conditions before executing.
- `acceptance_gate`: any post-merge DeepSeek improvement must exceed current `2.7 tok/s`, keep RAM including page cache `<=16000000000`, preserve France correctness under manual review, keep TTFT `<=45449.496149ms`, and be committed/pushed to `ssd/vendor/deepseek-token-rate-16gb` with a clean pushed-commit rerun before promotion.
- `rollback`: if the merged code breaks Kimi or DeepSeek build paths, resolve rather than dropping features. If a split-pool DeepSeek experiment fails performance/correctness/RAM/TTFT gates, keep the merge only if it preserves functionality; revert only the failing DeepSeek-specific experiment source/config and record rejection.

#### 2026-07-02 merge regression investigation

- `merge_status`: conflicts were resolved and the merged tree built, but it is not yet committed or pushed because DeepSeek SOTA reproduction regressed.
- `post_merge_control_run`: `/root/lfz/runs/vendor-ds4-16gb/20260702T045119Z-20260702_merge_kimi_ds_sota_repro_045119/france-cpu40-vram0gb`.
- `post_merge_control_result`: rejected. `eval_tok_s=2.1`, `prompt_tok_s=0.8`, `TTFT=29437.507562ms`, `elapsed_seconds=119.25`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15018037248`, `pgmajfault=421828`, `workingset_refault_file=9671728`, `ram_ok=true`, but manual correctness failed because the France answer was overlong/truncated.
- `post_merge_trace_gap`: gate trace changed from accepted SOTA `rows=35151`, `cache_misses=4623`, `src0_ms=22756.581` to `rows=51464`, `cache_misses=10866`, `src0_ms=38964.404`. This is a model trajectory/routing regression, not a small cache-policy variance.
- `current_hypothesis`: the highest-priority suspect is the Kimi graph change that switches `LLM_ARCH_DEEPSEEK2` MoE expert selection from `ggml_argsort_top_k()` to `ggml_top_k()`. This can change expert order/selection semantics and explains the row-count/output trajectory jump. The Kimi functionality should be preserved for `LLM_ARCH_KIMI_LINEAR`; DeepSeek V4 should keep the previously promoted selection behavior unless a dedicated DeepSeek correctness/performance test proves the new path is safe.
- `next_experiment`: patch `src/llama-graph.cpp` so only `LLM_ARCH_KIMI_LINEAR` and the Kimi-intended path use `ggml_top_k`, while `LLM_ARCH_DEEPSEEK2` returns to `ggml_argsort_top_k`. Rebuild and rerun the exact pushed SOTA config under cold `drop_caches` and 16GB cgroup. Acceptance for this repair is not a new SOTA; it must restore merge compatibility with `eval_tok_s` near `2.7`, `rows≈35151`, RAM ok, TTFT ok, and complete France output.

#### 2026-07-02 top-k repair result and bias-name hypothesis

- `topk_repair_patch`: `src/llama-graph.cpp` was patched to remove `LLM_ARCH_DEEPSEEK2` from the Kimi `ggml_top_k` condition while preserving `LLM_ARCH_KIMI_LINEAR` and `LLM_ARCH_MISTRAL4`.
- `topk_repair_run`: `/root/lfz/runs/vendor-ds4-16gb/20260702T051410Z-20260702_merge_kimi_ds_topk_repair_control/france-cpu40-vram0gb`.
- `topk_repair_result`: rejected as insufficient. `eval_tok_s=2.1`, `prompt_tok_s=0.8`, `TTFT=29944.17521ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15005294592`, `ram_ok=true`; output still ended incomplete at `Overall, France is a country with a` and gate rows stayed `51464`.
- `updated_hypothesis`: the next suspect is the Kimi merge change in `src/llama-model.cpp` that loads `ffn_exp_probs_b.bias` before the legacy `ffn_exp_probs_b` name in the `LLM_ARCH_DEEPSEEK2/MISTRAL4` tensor-loading case. DeepSeek routing bias directly affects expert selection and generated length; selecting a different optional bias tensor can explain the 35k -> 51k row jump and truncated output.
- `next_experiment_bias_order`: restore DeepSeek-compatible legacy bias-name priority: first load `tn(LLM_TENSOR_FFN_EXP_PROBS_B, i)`, then fall back to `tn(..., "bias", i)` only if legacy is absent. This preserves Kimi compatibility when only the `.bias` tensor exists, while keeping current DeepSeek V4 behavior. Rebuild and rerun the same cold 16GB SOTA control.

#### 2026-07-02 coarse frontend isolation plan

- `bias_order_result`: rejected as insufficient. `/root/lfz/runs/vendor-ds4-16gb/20260702T051938Z-20260702_merge_kimi_ds_bias_order_repair_control/france-cpu40-vram0gb` produced `eval_tok_s=2.2`, `prompt_tok_s=0.9`, `TTFT=28665.399538ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15017811968`, `ram_ok=true`, but output remained truncated and gate rows stayed `51464`.
- `updated_debug_strategy`: save the staged Kimi frontend/core patch for `src/llama-context.cpp`, `src/llama-graph.cpp`, `src/llama-model.cpp`, `src/models/deepseek2.cpp`, `ggml/include/ggml.h`, `ggml/src/ggml-backend*.cpp`, and `ggml/src/ggml.c`, then temporarily reverse it while leaving CUDA merge files intact. If DeepSeek rows/output recover, the regression is in this frontend/core set; if not, inspect CUDA/CPU MoE merge changes next.
- `coarse_frontend_acceptance`: diagnostic only. It does not preserve Kimi functionality and must not be committed as final. It is acceptable only as a narrowing experiment before reapplying the saved patch and adding targeted guards.

#### 2026-07-02 coarse frontend isolation result

- `coarse_frontend_saved_patch`: `/root/lfz/tmp/kimi_frontend_core_patch_20260702T0526.diff`.
- `coarse_frontend_run`: `/root/lfz/runs/vendor-ds4-16gb/20260702T053000Z-20260702_merge_kimi_frontend_reverted_control/france-cpu40-vram0gb`.
- `coarse_frontend_result`: diagnostic pass. With CUDA/Kimi split-pool files kept but `src/llama-context.cpp`, `src/llama-graph.cpp`, `src/llama-model.cpp`, and `src/models/deepseek2.cpp` temporarily restored to DeepSeek SOTA behavior, the control produced `eval_tok_s=2.8`, `prompt_tok_s=1.0`, `TTFT=36811.193769ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15002050560`, `ram_ok=true`, and complete France output.
- `coarse_frontend_trace`: `rows=35151`, `cache_hits=30528`, `cache_misses=4623`, `cache_inserts=3337`, `src0_ms=21908.727`, `dontneed_ms=1100.233`, `total_ms=24086.296`. This matches the promoted SOTA trajectory.
- `next_experiment_frontend_split`: reapply only `src/llama-model.cpp` and `src/models/deepseek2.cpp` from the saved Kimi patch and rerun the same control. If the 51k-row long-output regression returns, split those two files; otherwise test `src/llama-graph.cpp` next.

#### 2026-07-02 model/deepseek2 split result

- `model_deepseek2_run`: `/root/lfz/runs/vendor-ds4-16gb/20260702T053340Z-20260702_merge_kimi_model_deepseek2_only_control/france-cpu40-vram0gb`.
- `model_deepseek2_result`: rejected/regression reproduced. Reapplying only `src/llama-model.cpp` and `src/models/deepseek2.cpp` produced `eval_tok_s=2.2`, `prompt_tok_s=0.9`, `TTFT=28783.135053ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15024054272`, `ram_ok=true`, but the output/trajectory remained the long truncated France response.
- `next_experiment_model_vs_deepseek2`: reverse only `src/models/deepseek2.cpp` back to DeepSeek SOTA behavior while keeping the Kimi `src/llama-model.cpp` changes. If the SOTA trajectory returns, the regression is the DeepSeek2 YaRN formula change; otherwise the regression is in model loading/hparams behavior.
- `guard_strategy_if_deepseek2`: preserve Kimi branch behavior by making the new formula apply only where it is actually needed, while DeepSeek V4/Flash keeps the already validated old formula. Any such guard must be explicit and documented before merge commit.

#### 2026-07-02 hparams split plan

- `model_only_run`: `/root/lfz/runs/vendor-ds4-16gb/20260702T053649Z-20260702_merge_kimi_model_only_control/france-cpu40-vram0gb`.
- `model_only_result`: rejected/regression persists. Keeping only `src/llama-model.cpp` Kimi changes while `src/models/deepseek2.cpp` is back to SOTA behavior still produced `eval_tok_s=2.2`, `prompt_tok_s=0.9`, `TTFT=29095.128722ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15022125056`, `ram_ok=true`, with the same long truncated France output.
- `hparams_culprit_hypothesis`: the decisive hunk is the `LLM_ARCH_DEEPSEEK2/MISTRAL4` change that removed the historical `hparams.rope_yarn_log_mul /= 0.1f`. DeepSeek V4/Flash uses this architecture path and empirically requires the old scaled value to reproduce the accepted output trajectory.
- `next_experiment_hparams_restore`: restore the old `if (ml.get_key(...)) { hparams.rope_yarn_log_mul /= 0.1f; }` behavior while keeping the other Kimi `src/llama-model.cpp` changes. Rebuild and rerun the strict SOTA control. If rows/output recover, final merge must keep this DeepSeek-compatible hparams behavior and not adopt the Kimi branch hparams interpretation for DeepSeek V4.

#### 2026-07-02 hparams restore result

- `hparams_restore_run`: `/root/lfz/runs/vendor-ds4-16gb/20260702T054216Z-20260702_merge_kimi_hparams_restore_control/france-cpu40-vram0gb`.
- `hparams_restore_result`: rejected. Restoring the historical `/=0.1` hparams scaling while keeping the other `src/llama-model.cpp` Kimi changes still produced `eval_tok_s=2.1`, `prompt_tok_s=0.9`, `TTFT=29413.482979ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15015927808`, `ram_ok=true`, and the long truncated France trajectory.
- `updated_culprit`: the likely culprit is no longer hparams scaling; it is the early `defer_expert_mmap` tensor buffer override moved before tensor creation, or less likely the optional `.bias` fallback for `ffn_exp_probs_b`.
- `next_experiment_model_revert_then_minimal`: temporarily reverse all saved `src/llama-model.cpp` Kimi changes back to SOTA behavior, rebuild/rerun to reconfirm recovery, then add back only small compatible pieces. This is diagnostic and not the final merge state.

#### 2026-07-02 guarded model result

- `model_reverted_confirm_run`: `/root/lfz/runs/vendor-ds4-16gb/20260702T054628Z-20260702_merge_kimi_model_reverted_confirm/france-cpu40-vram0gb`, `eval_tok_s=2.7`, `TTFT=36888.895417ms`, `memory_peak_bytes=16000000000`, complete France output. This reconfirmed `src/llama-model.cpp` was the regression source.
- `model_guard_patch`: re-applied Kimi `src/llama-model.cpp` changes but kept DeepSeek-compatible YaRN hparams scaling and changed early deferred expert tensor CPU override from `defer_expert_mmap && !devices.empty()` to `defer_expert_mmap && arch == LLM_ARCH_KIMI_LINEAR && !devices.empty()`. This preserves the Kimi Linear deferred-expert path while preventing DeepSeek V4 from changing tensor placement/output trajectory.
- `model_guard_run`: `/root/lfz/runs/vendor-ds4-16gb/20260702T055131Z-20260702_merge_kimi_model_guarded_control/france-cpu40-vram0gb`.
- `model_guard_result`: accepted as merge repair, not new SOTA. `eval_tok_s=2.7`, `prompt_tok_s=1.0`, `TTFT=36771.871689ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14996398080`, `ram_ok=true`, France output complete/coherent.
- `next_experiment_reapply_graph_context`: reapply Kimi `src/llama-graph.cpp` and `src/llama-context.cpp` changes. Keep the prior DeepSeek guard that removes `LLM_ARCH_DEEPSEEK2` from the `ggml_top_k` condition. Do not reapply `src/models/deepseek2.cpp` formula change unless a Kimi-specific requirement proves it is needed; DeepSeek V4 SOTA requires the old formula/hparams pairing.

#### 2026-07-02 full guarded merge control result

- `full_guarded_patch`: Kimi CUDA/core/batch split-pool/docs/scripts plus Kimi graph/context/model changes are merged. DeepSeek guards kept: `LLM_ARCH_DEEPSEEK2` remains on `ggml_argsort_top_k`, DeepSeek YaRN hparams keeps historical `/=0.1`, `src/models/deepseek2.cpp` formula remains SOTA-compatible, and early deferred expert CPU override is limited to `arch == LLM_ARCH_KIMI_LINEAR`.
- `full_guarded_run`: `/root/lfz/runs/vendor-ds4-16gb/20260702T055445Z-20260702_merge_kimi_full_guarded_control/france-cpu40-vram0gb`.
- `full_guarded_result`: accepted as merge repair, not a new SOTA. `eval_tok_s=2.7`, `prompt_tok_s=1.0`, `TTFT=37205.423753ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15027433472`, `pgmajfault=289957`, `workingset_refault_file=2875841`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `full_guarded_trace`: `rows=35151`, `cache_hits=30528`, `cache_misses=4623`, `cache_inserts=3337`, `src0_ms=23479.610`, `dontneed_ms=1114.308`, `total_ms=25683.730`. The trajectory matches the promoted SOTA shape.
- `full_guarded_correctness_manual_review`: pass. France output is complete/coherent and matches the accepted answer shape.
- `merge_decision`: commit/push this guarded merge to `ssd/vendor/deepseek-token-rate-16gb` before continuing split-pool migration experiments. This checkpoint preserves current DeepSeek SOTA after merging Kimi changes; it does not yet claim a new DeepSeek token-rate improvement beyond `2.7 tok/s`.

#### 2026-07-02 pushed guarded merge reproduction

- `pushed_merge_commit`: `efd1aaee699b20a038041fe3e63880e0714da4fa` (`vendor-ds4: merge kimi split pool guarded`), pushed to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`.
- `pushed_clean_rebuild`: `cmake --build build-ds4-moe-stream -j 8`, clean build line `ggml commit: efd1aaee6`.
- `pushed_clean_run`: `/root/lfz/runs/vendor-ds4-16gb/20260702T055944Z-20260702_merge_kimi_full_guarded_pushed_clean/france-cpu40-vram0gb`.
- `pushed_clean_result`: accepted as reproducible merge checkpoint. `eval_tok_s=2.7`, `prompt_tok_s=0.9`, `TTFT=36333.418687ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15022682112`, `pgmajfault=287925`, `workingset_refault_file=2922099`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `pushed_clean_trace`: `rows=35151`, `cache_hits=30528`, `cache_misses=4623`, `cache_inserts=3337`, `src0_ms=23041.292`, `dontneed_ms=1108.210`, `total_ms=25233.510`.
- `pushed_clean_sha256`: `llama-cli=c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`, `libggml-cuda.so=347c3e94aa69e23bfcfba824c23f42e058e56506dcb6eda9e125d11bdcf37218`, profile `8134c320730e0ba236d103ba4a0b53505b3bab16e69d8bdc2a08607ecfcc274b`.

#### 2026-07-02 DeepSeek split-pool migration design result

- `analysis_status`: completed_design_rejected_for_current_sota_path; no source experiment is justified yet.
- `bottleneck_check`: parsed the pushed guarded SOTA trace and historical DeepSeek one-stream traces for gate-only, up-only, down-only, all-ffn, gate+up, and gate+down streaming runs. Every traced streamed expert had exactly one `src0_bytes` class: `4456448` bytes.
- `current_sota_size_evidence`: pushed guarded SOTA trace `/root/lfz/runs/vendor-ds4-16gb/20260702T055944Z-20260702_merge_kimi_full_guarded_pushed_clean/france-cpu40-vram0gb/gate_trace.csv` has `rows=35151`, `sizes={(4456448, 35151)}`, `kinds={ffn_gate_exps:35151}`, `hits=30528`, `misses=4623`, `inserts=3337`.
- `non_gate_size_evidence`: historical DeepSeek one-stream traces also show a single size class: up-only `47458` rows at `4456448` bytes, down-only `34258` rows at `4456448` bytes, all-ffn `110697` rows at `4456448` bytes, gate+up `63700` rows at `4456448` bytes, gate+down `81954` rows at `4456448` bytes. These runs were all slower (`~0.9-1.6 tok/s`) despite correctness/RAM passing, so they are not candidates for promotion.
- `kimi_split_pool_fit`: Kimi split-pool improves the batch path because small up/gate experts and larger down experts otherwise share fixed-size cache slots. Current DeepSeek SOTA uses the one-stream gate path (`moe_stream.cu`) and only one expert byte size. A size-based split pool would partition identical objects into smaller effective LRUs and cannot increase usable capacity; at best it is equivalent to the existing single pool, and at worst it reduces hit rate by isolating capacity.
- `theoretical_upper_bound`: for the current DeepSeek SOTA path, size-based split-pool upper bound is `0` additional cache slots and `0` avoidable misses because all cached objects are equal size. Directly enabling Kimi envs (`GGML_MOE_VRAM_CACHE_SPLIT`, `GGML_MOE_VRAM_CACHE_SPLIT_MAX_MIB`, `GGML_MOE_VRAM_CACHE_UPGATE_PCT`) would also not affect the SOTA path because those controls live in `moe_stream_batch.cu`, while SOTA uses `moe_stream.cu`.
- `admission_policy_bound`: current gate admission profile is already near the trace-level optimum. On the pushed guarded SOTA trace with capacity `3192`, current `freq>=2` admission gives `4623` misses; Belady optimal is `4599` misses. The remaining profile-policy bound is only `24` misses, roughly `~0.12s` at current miss cost, far below the noise/rounding needed for a new `>2.7 tok/s` SOTA.
- `decision`: do not port Kimi size split-pool into `moe_stream.cu` for the current DeepSeek SOTA path. It has no credible theoretical upside under the one-size gate stream trace and would add capacity-fragmentation risk. Keep the Kimi batch split-pool merge for Kimi compatibility, but DeepSeek optimization should continue on the real bottlenecks: broad up/down CPU fallback (`~39.0s` phase trace) and gate source-load cost (`~23-27s`) rather than size-pool partitioning.
- `next_action`: before any new code experiment, write a new attempt focused on CPU fallback math/page-touch or a larger stream/offload restructuring. Any future split-pool experiment is allowed only if a new DeepSeek path demonstrably caches multiple expert byte sizes or enters `moe_stream_batch.cu`; otherwise it must remain rejected at design stage.

### 当前执行 attempt：post-merge-ds-batch-split-pool-smoke

- `attempt_id`: `20260702-post-merge-ds-batch-split-pool-smoke`
- `attempt_kind`: `diagnostic/batch-path-split-pool-migration`
- `status`: planned
- `bottleneck_basis`: Current DeepSeek SOTA still spends about `~39.0s` phase-traced wall in non-streamed `ffn_up_exps`/`ffn_down_exps` CPU fallback, while Kimi split-pool lives in the CUDA batch path (`moe_stream_batch.cu`) that was not available in the original DeepSeek SOTA binary (`GGML_CUDA_MOE_STREAM_BATCH=OFF`). Earlier pre-merge batch compilation failed on a missing IQK header; after merging Kimi branch, `moe_stream_batch.cu` no longer includes that header and should be rechecked.
- `hypothesis`: If the post-merge batch path compiles and accepts DeepSeek down/up-gate calls, it may move part of the remaining up/down CPU fallback onto the CUDA streaming/staging path. Kimi split-pool envs are enabled in this smoke because they are native to the batch path; for DeepSeek the expected size-pool gain remains zero unless the batch path sees multiple size classes, so any speed gain must come from batch offload/staging rather than size partitioning.
- `theoretical_upper_bound`: The absolute upper bound is the remaining CPU fallback phase (`up cpu_loop_ms≈21591`, `down cpu_loop_ms≈17346`). The realistic bound is much lower because batch streaming adds H2D/page-cache traffic and may decline unsupported DeepSeek tensor types. If the batch path only declines, expected delta is `0`; if it accepts but thrashes cache, it can regress below `2.7 tok/s`.
- `build_plan`: configure a separate build directory `build-ds4-moe-stream-batch` with the same Release CUDA options as SOTA plus `-DGGML_CUDA_MOE_STREAM_BATCH=ON`; do not overwrite the accepted `build-ds4-moe-stream` binary.
- `test_config`: batch binary, current accepted DeepSeek SOTA env (`GGML_MOE_KEEP_TOPK_UPDOWN=4`, `10-39=>top3`, repo profile `.Agent/profiles/vendor-ds4/current_sota_gate_freq_ge2.tsv`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, gate one-stream filter), plus `GGML_MOE_STREAM_DOWN_BATCH=1`, `GGML_MOE_STREAM_DECLINE_DEBUG=1`, `GGML_MOE_VRAM_CACHE_SPLIT=1`, `GGML_MOE_VRAM_CACHE_SPLIT_MAX_MIB=6`, `GGML_MOE_VRAM_CACHE_UPGATE_PCT=60`; cold `drop_caches`, strict 16GB cgroup, France prompt, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: diagnostic accepts as useful if it compiles and records whether batch calls are accepted/declined. Promote only if `eval_tok_s > 2.7`, RAM including page cache stays `<=16000000000`, TTFT `<=45449.496149ms`, France output is semantically correct/coherent/complete, and a pushed-commit clean rerun from `ssd/vendor/deepseek-token-rate-16gb` reproduces the result.
- `rollback`: No source change is planned. If build fails, batch declines all relevant calls, token rate ties/regresses, correctness fails, TTFT fails, RAM fails, or CUDA OOM occurs, record rejected and keep current SOTA. Do not commit/push as SOTA unless all promotion gates pass.
- `required_evidence`: configure/build logs, build cache proving `GGML_CUDA_MOE_STREAM_BATCH=ON`, binary/library hashes, exact env/command, stdout/stderr with batch accept/decline/cache counters, summary.json, cgroup memory files, France answer, and explicit diagnostic conclusion.
- `configure_result`: initial separate configure without explicit `CMAKE_CUDA_COMPILER` failed to find nvcc; reconfigure with `/usr/local/cuda/bin/nvcc` succeeded and generated `build-ds4-moe-stream-batch` at commit `2b45af06e`.
- `build_result_initial`: failed in `ggml/src/ggml-cuda/moe_stream_batch.cu`, not because of the old missing `iqk/iqk_mul_mat.h`, but because Kimi batch source still had stale calls against current vendor CUDA helpers: `ggml_cuda_moe_stream_mmvq_dev` was called without `nb01`, `quantize_mmq_q8_1_cuda_id` no longer exists, `launch_moe_mmq_id_one` no longer exists, and `ne00_padded` was missing in the serial up/gate branch.
- `repair_hypothesis`: these are compile-time integration mismatches from merging the Kimi batch file into a newer vendor CUDA helper surface. A narrow default-off compile repair can preserve normal DeepSeek SOTA behavior while making the Kimi batch path available for the planned diagnostic.
- `repair_plan`: pass the original row stride `nb01` into `launch_moe_mmvq_compact_batch` and into `ggml_cuda_moe_stream_mmvq_dev`; replace the stale serial up/gate quantization call with `quantize_mmq_q8_1_cuda`; replace the stale one-expert MMQ launcher with the existing `launch_moe_mmq_slot_batch` using `n_active=1`. Rebuild batch target after patch.
- `repair_gate`: if the repair does not compile, revert it and record failure. If it compiles but runtime declines/regresses, keep or revert based on whether it is needed to preserve Kimi batch buildability; do not promote DeepSeek SOTA unless strict runtime gates pass.
- `build_result_repaired`: batch target compiled successfully after the narrow helper-signature repair. `build-ds4-moe-stream-batch/bin/llama-cli` was produced without touching the accepted SOTA `build-ds4-moe-stream` binary.
- `run_dir_absbin`: `/root/lfz/runs/vendor-ds4-16gb/20260702T064627Z-20260702_post_merge_ds_batch_split_pool_smoke_absbin/france-cpu40-vram0gb`.
- `run_result_absbin`: rejected/tie. `eval_tok_s=2.7`, `prompt_tok_s=1.0`, `TTFT=37047.112042ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15018090496`, `pgmajfault=286960`, `workingset_refault_file=2904364`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`. France output is complete and coherent.
- `trace_result_absbin`: gate trace stayed identical in shape to accepted SOTA: `rows=35151`, `cache_hits=30528`, `cache_misses=4623`, `cache_inserts=3337`, `src0_ms=23114.783`, `dontneed_ms=1110.366`, `total_ms=25323.246`, single size class `4456448`.
- `batch_decline_result`: every attempted down batch declined with `reason=unsupported_type`, `type=39`, tensor `blk.*.ffn_down_exps.weight`, `ne01=4096`, `ne00=2048`. `GGML_TYPE_MXFP4=39`, so the run did not actually offload DeepSeek down experts to batch; it fell back to the existing one-stream gate SOTA path and tied `2.7`.
- `decision`: the compile repair is useful for Kimi/batch buildability, but it is not a DeepSeek SOTA improvement by itself. The next real migration step must add a guarded MXFP4 down-batch path instead of merely enabling Kimi split-pool envs.

### 当前执行 attempt：post-merge-ds-mxfp4-down-batch

- `attempt_id`: `20260702-post-merge-ds-mxfp4-down-batch`
- `attempt_kind`: `implementation/DeepSeek-MXFP4-batch-path`
- `bottleneck_basis`: The previous batch smoke proved the post-merge batch path now builds but declines DeepSeek because `ffn_down_exps` are `GGML_TYPE_MXFP4`. One-stream already computes MXFP4 via `ggml_cuda_moe_stream_mmvq_rows_dev`; the batch path has a compact MMVQ helper that can call the same family via `ggml_cuda_moe_stream_mmvq_dev` if MXFP4 is admitted.
- `hypothesis`: Add `GGML_TYPE_MXFP4` to the batch-supported type set and to `launch_moe_mmvq_compact_batch`. With `GGML_MOE_STREAM_DOWN_BATCH=1`, DeepSeek down experts may be staged and computed in the batch path instead of CPU fallback. Kimi split-pool remains enabled, but because DeepSeek experts are one size, any gain should come from replacing CPU fallback/down staging, not size-pool capacity.
- `theoretical_upper_bound`: Down CPU fallback phase is about `17346ms` in the phase trace. If MXFP4 down batch accepted every eligible single-row route and its H2D/MMVQ/D2H cost were lower than CPU fallback, the hard upper bound is up to that component; realistic gain is smaller and may be negative if cache/page traffic or D2H scatter dominates. Promote only if strict cold `eval_tok_s > 2.7` with RAM/correctness/TTFT gates.
- `rollback`: If build fails, batch still declines, CUDA errors occur, correctness fails, RAM fails, TTFT fails, or token rate ties/regresses, mark rejected. Keep only buildability-preserving compile fixes if they do not alter default builds; revert any MXFP4 runtime admission if it is incorrect or harmful.
- `mxfp4_admission_build`: compiled successfully after adding `GGML_TYPE_MXFP4` to the batch-supported type set and MMVQ compact launcher.
- `mxfp4_run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T065121Z-20260702_post_merge_ds_mxfp4_down_batch/france-cpu40-vram0gb`.
- `mxfp4_result`: rejected/tie. `eval_tok_s=2.7`, `prompt_tok_s=0.9`, `TTFT=37136.714177ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15039926272`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`; France output is semantically correct and coherent.
- `mxfp4_gap`: runtime no longer declines with `unsupported_type`; it now declines with `reason=cache_get` for active down batches. The batch cache budget is separate from the one-stream gate cache: strict runner `--vram-cache-gb 0` sets `GGML_MOE_VRAM_CACHE_GB=0`, and no explicit `GGML_MOE_VRAM_CACHE_MIB` was provided, so the batch path effectively had zero admission budget.

### 当前执行 attempt：post-merge-ds-mxfp4-down-batch-512mib

- `attempt_id`: `20260702-post-merge-ds-mxfp4-down-batch-512mib`
- `hypothesis`: Set `GGML_MOE_VRAM_CACHE_MIB=512` for the batch cache while keeping the accepted one-stream gate cache (`GGML_MOE_STREAM_ONE_CACHE_MIB=13568`). At `4456448` bytes per DeepSeek expert, 512 MiB gives roughly 120 batch expert slots. This should prove whether MXFP4 down batch can actually admit experts and whether down CPU fallback can be reduced without violating the 16GB cgroup.
- `acceptance_gate`: promote only if strict cold `eval_tok_s > 2.7`, RAM including page cache remains `<=16000000000`, TTFT remains `<=45449.496149ms`, France output is correct/coherent/complete, stderr proves batch was accepted rather than declined, and the exact source/env/run metadata are committed and pushed to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb` with author `L-Ark <fliangae@connect.ust.hk>` before clean-rerun verification.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T065937Z-20260702_post_merge_ds_mxfp4_down_batch_512mib/france-cpu40-vram0gb`.
- `result`: rejected. `eval_tok_s=1.7`, `prompt_tok_s=1.0`, `TTFT=39409.797281ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14990073856`, `pgmajfault=261194`, `workingset_refault_file=9000118`, `ram_ok=true`, `ram_limit_killed=false`. The output discusses France but starts with an unrelated instruction fragment and is not a short paragraph, so it is not acceptable under the stricter France prompt gate.
- `stderr_gap`: explicit batch budget reached the batch cache code but only `0.1 GiB` / `34` slots could be allocated because VRAM was already mostly consumed by the accepted one-stream gate cache; all down calls still declined with `reason=multirow_not_supported`, and final batch cache counters were `hits=0 misses=24878`. This experiment did not replace CPU down fallback.
- `decision`: do not promote and do not push as SOTA. The next batch experiment must remove the `multirow_not_supported` blocker before cache-size sweeps can be meaningful.

### 当前执行 attempt：post-merge-ds-down-batch-multirow-flatten

- `attempt_id`: `20260702-post-merge-ds-down-batch-multirow-flatten`
- `bottleneck_basis`: The MXFP4 admission and explicit batch cache budget moved the failure from type/cache admission to `multirow_not_supported`. The down-batch code currently accepts only one route per expert, while the actual DeepSeek calls have `rows_stride=12/24/48` and active route counts up to the prompt/decode microbatch size.
- `hypothesis`: Flatten every `(expert, row)` route into the existing compact MMVQ batch arrays, mirroring the already-merged up/gate prompt-mode collection logic. This should allow DeepSeek MXFP4 down batches to actually execute instead of declining. The hard upper bound remains the prior down CPU fallback phase (`~17346ms`), but the practical bound is limited by VRAM cache capacity, H2D source staging, Q8 activation quantization, D2H, and scatter.
- `implementation_plan`: replace fixed `128` down arrays with `MOE_STREAM_MAX_ACTIVE`, iterate all `matrix_row_counts[e]` rows, validate route ids, set `dst_bytes` large enough for compact `n_active` MMVQ output, and keep the q8k/reference branch using its existing `dst_cols` behavior. Rebuild only `build-ds4-moe-stream-batch`.
- `test_config`: strict cold 16GB cgroup, France prompt, current accepted gate one-stream env/profile/cache, `GGML_MOE_STREAM_DOWN_BATCH=1`, `GGML_MOE_STREAM_DECLINE_DEBUG=1`, `GGML_MOE_VRAM_CACHE_MIB=512`, and batch split envs. If VRAM allocation still clamps the batch cache to `0.1 GiB`, record as a capacity gap rather than sweeping blindly.
- `acceptance_gate`: promote only if `eval_tok_s > 2.7`, TTFT `<=45449.496149ms`, RAM including page cache `<=16000000000`, France output is correct/coherent/complete and follows the short-paragraph request, stderr confirms real batch execution, and the exact source/env/run metadata are committed and pushed to `ssd/vendor/deepseek-token-rate-16gb` as `L-Ark` followed by a clean rerun from the pushed commit.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T071018Z-20260702_post_merge_ds_down_batch_multirow_flatten_512mib/france-cpu40-vram0gb`.
- `result`: rejected. The patch compiled and reached real batch execution (`trace call=0..39`, active routes `6-8`) but exited `134` after CUDA OOM. Metrics before abort: `TTFT=22194.119976ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15077339136`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=false` because no France answer was produced.
- `gap_analysis`: removing `multirow_not_supported` exposed the next bottleneck: VRAM capacity. With gate one-stream cache still at `13568MiB`, CUDA reported only about `226MiB` free; requested batch cache `512MiB` fell back to about `0.2GiB` / `45` slots and then later CUDA allocation failed. The route flattening direction is mechanically valid but cannot be assessed under the current VRAM split.

### 当前执行 attempt：post-merge-ds-down-batch-multirow-rebalance-13056-512

- `attempt_id`: `20260702-post-merge-ds-down-batch-multirow-rebalance-13056-512`
- `hypothesis`: Reduce `GGML_MOE_STREAM_ONE_CACHE_MIB` from `13568` to `13056`, freeing about `512MiB` of VRAM for down-batch cache and buffers while keeping the same 16GB host cgroup. The gate cache loses about `120` slots (`512MiB / 4.25MiB`); prior admission analysis showed gate policy is near optimal, so the gate miss penalty should be modest compared with the possible `~17346ms` down CPU fallback upper bound if batch down becomes correct and stable.
- `test_config`: strict cold 16GB cgroup, France prompt, `GGML_MOE_STREAM_ONE_CACHE_MIB=13056`, current accepted gate profile/top-k envs, `GGML_MOE_STREAM_DOWN_BATCH=1`, `GGML_MOE_BATCH_TRACE=1`, `GGML_MOE_VRAM_CACHE_MIB=512`, batch split envs, same CLI `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: same SOTA gate: `eval_tok_s > 2.7`, TTFT `<=45449.496149ms`, RAM including page cache `<=16000000000`, France output is correct/coherent/complete and short-paragraph shaped, no CUDA OOM, stderr shows real down-batch execution, then immediate detailed record + commit + push to `ssd/vendor/deepseek-token-rate-16gb` as `L-Ark`, followed by clean rerun from the pushed commit.

- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T071303Z-20260702_post_merge_ds_down_batch_multirow_rebalance_13056_512/france-cpu40-vram0gb`.
- `result`: rejected. `eval_tok_s=2.1`, `prompt_tok_s=0.9`, `TTFT=39745.587641ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15027212288`, `pgmajfault=150544`, `workingset_refault_file=4275322`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`. The France answer is semantically coherent and acceptable, but token rate is below the accepted `2.7 tok/s` SOTA.
- `batch_evidence`: real down-batch execution occurred. Batch cache allocated the requested `0.5GiB`, `120` slots. Final down batch counters were `hits=650`, `misses=21320`, `hit_rate=3.0%`; this is far too low to offset H2D/page-load/D2H/scatter overhead.
- `gate_tradeoff`: reducing gate one-stream cache to `13056MiB` reduced gate capacity to `3072` slots and produced `gate cache_hits=33448`, `gate cache_misses=5786`, `src0_ms=27276.107ms`, `total_ms=29820.9ms`, worse than accepted SOTA gate behavior. The VRAM rebalancing cost plus down-batch miss cost explains the regression.
- `decision`: do not promote, commit, or push as SOTA. Revert the experimental down-batch runtime changes; keep only this plan record. Further DeepSeek optimization should not spend more time on small down-batch cache under the current VRAM budget unless a fundamentally different admission/profile scheme can guarantee thousands of useful down slots, which conflicts with the accepted gate cache requirement.
- `next_bottleneck`: return to accepted SOTA path analysis. Remaining credible areas are CPU fallback math/thread scheduling and gate source-load reduction without extra VRAM, not Kimi size split-pool or small down-batch cache.


### 当前执行 attempt：ds-up-gate-fused-gpu-only

- `attempt_id`: `20260702-ds-up-gate-fused-gpu-only`
- `attempt_kind`: `implementation/DeepSeek-fused-up-gate-GPU-forced`
- `status`: planned
- `bottleneck_basis`: Accepted cold SOTA `2.7 tok/s` still uses the gate-first graph path: `ffn_gate_exps` is streamed/cached on CUDA first, then `ffn_up_exps` runs through the remaining non-fused path. Prior phase traces put the remaining broad CPU fallback work around tens of seconds, with up/down fallback dominating after gate cache source-load cost. Down-batch experiments were rejected because useful down cache hit rate was only `3.0%` under the VRAM budget; therefore the next credible experiment is to remove the separate up/gate CPU fallback path without adding a down cache requirement.
- `code_plan`: Enable the existing graph-level `GGML_OP_MOE_FUSED_UP_GATE` via `GGML_MOE_STREAM_FUSED_UP_GATE=1`, build the `GGML_CUDA_MOE_STREAM_BATCH=ON` binary so `ggml_cuda_moe_stream_up_gate_batch` is linked, and add an experimental guard `GGML_MOE_STREAM_FUSED_UP_GATE_GPU_ONLY=1` so any CUDA-helper decline or unavailable helper aborts instead of silently falling back to CPU. This is required to prove both `ffn_up_exps` and `ffn_gate_exps` are computed by the GPU batch helper.
- `theoretical_upper_bound`: The fused op computes `up(x) * silu(gate(x))`; up and gate have no data dependency before the final elementwise multiply, so their hard parallelism bound is the slower of the two matmuls plus staging/fuse/D2H rather than the sum. Since the current SOTA already optimizes gate via one-stream cache, the maximum realistic gain is capped by the avoidable `ffn_up_exps` CPU fallback portion and any gate scheduling overhead removed by fusion. It cannot remove the rejected down fallback bottleneck, and may regress if up/gate batch staging competes with the accepted `13568MiB` gate cache for VRAM or page-cache bandwidth.
- `initial_test_config`: strict cold 16GB cgroup, France prompt, batch binary `build-ds4-moe-stream-batch/bin/llama-cli`, accepted top-k/profile envs, `GGML_MOE_STREAM_ONE_CACHE_MIB=13056` plus `GGML_MOE_VRAM_CACHE_MIB=512` for up/gate batch staging, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, `GGML_MOE_STREAM_FUSED_UP_GATE=1`, `GGML_MOE_STREAM_FUSED_UP_GATE_GPU_ONLY=1`, `GGML_MOE_STREAM_DECLINE_DEBUG=1`, `GGML_MOE_BATCH_TRACE=1`, same CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`. If this OOMs or declines, try a smaller batch cache only if the failure log shows capacity rather than unsupported type/shape.
- `acceptance_gate`: promote only if `eval_tok_s > 2.7`, TTFT `<=45449.496149ms`, RAM including page cache `<=16000000000`, France output is semantically correct/coherent/complete and shaped as a short paragraph, stderr proves fused up/gate CUDA batch accepted without CPU fallback, and the result is immediately recorded with full reproduction info.
- `push_requirement`: Any compliant new SOTA must be committed and pushed immediately to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb` using `L-Ark <fliangae@connect.ust.hk>`. The record must include exact source commit, binary/library hashes, run directory, env, command, summary metrics, memory cgroup evidence including page cache, France answer, and a clean rerun from the pushed commit so future rollback can fully reproduce the metric.
- `rollback`: If CUDA up/gate declines, GPU-only aborts, correctness fails, TTFT exceeds the gate, RAM exceeds 16GB, CUDA OOMs, or token rate ties/regresses, mark the experiment rejected and do not promote as SOTA. Keep only default-off buildability/diagnostic guards that do not change accepted SOTA behavior.

- `build_result`: compiled successfully after wiring DeepSeek4 decode to `ggml_moe_up_gate(..., limit)`, adding `GGML_MOE_STREAM_FUSED_UP_GATE_GPU_ONLY`, passing `swiglu_limit` through `op_params`, and repairing the batch helper build for current vendor CUDA signatures.
- `pre_deepseek4_hook_run_no_mixed`: `/root/lfz/runs/vendor-ds4-16gb/20260702T082749Z-20260702_ds_up_gate_fused_gpu_only_13056_512/france-cpu40-vram0gb`. Result: rejected/tie. `eval_tok_s=2.7`, `prompt_tok_s=0.9`, `TTFT=38484.465533ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15037485056`, `ram_ok=true`, `correctness_ok=true`. Stderr had no up/gate/fused evidence because the generic Kimi-style graph hook does not execute for DeepSeek4.
- `pre_deepseek4_hook_run_mixed`: `/root/lfz/runs/vendor-ds4-16gb/20260702T083044Z-20260702_ds_up_gate_fused_gpu_only_mixed_13056_512/france-cpu40-vram0gb`. Result: rejected/tie. `eval_tok_s=2.7`, `prompt_tok_s=0.9`, `TTFT=36966.874951ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15040040960`, `ram_ok=true`, `correctness_ok=true`. Adding `GGML_MOE_STREAM_FUSED_UP_GATE_MIXED_TYPES=1` still produced no fused evidence, proving the DeepSeek4 model-specific graph path had to be patched.
- `deepseek4_fused_run`: `/root/lfz/runs/vendor-ds4-16gb/20260702T084212Z-20260702_ds_deepseek4_up_gate_fused_gpu_only_13056_512/france-cpu40-vram0gb`.
- `deepseek4_fused_result`: rejected. `eval_tok_s=0.5`, `prompt_tok_s=0.9`, `TTFT=47854.061177ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=12678258688`, `pgmajfault=392880`, `workingset_refault_file=33202993`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`. France output is semantically correct but truncated by the 192-token generation cap and the run is far below token-rate SOTA; TTFT exceeds the accepted pushed SOTA `36333.418687ms` by about `31.7%`, above the allowed `20%` gate.
- `deepseek4_fused_evidence`: stderr contains `[deepseek4] fused up/gate decode path enabled limit=10`, so the DeepSeek4-specific fused graph hook executed. The GPU-only guard did not abort, proving CUDA `ggml_cuda_moe_stream_up_gate_batch` accepted enough work instead of silently falling back to CPU.
- `deepseek4_fused_gap`: batch cache counters show `hits=0 misses=98556 preloads=0 hit_rate=0.0%`; every up/gate expert was effectively staged cold through the batch cache. The accepted gate one-stream trace collapsed to only `rows=2031`, `hits=553`, `misses=1478`, `src0_ms=7076.600`, `total_ms=7452.173`, so this path threw away the high-hit one-stream gate-cache advantage and replaced it with no-hit up/gate batch staging. The many `up/gate prompt mode disabled` logs are diagnostic noise, but the decisive bottleneck is zero cache hit rate and repeated host/page-cache source loads.
- `decision`: do not promote as SOTA. Current accepted cold SOTA remains pushed commit `7aa4e0f709ff020783c58c6f9a2d45dfec6790f6` at `2.7 tok/s` on branch `vendor/deepseek-token-rate-16gb`. The fused up/gate code is useful only as a rejected diagnostic unless a follow-up adds a real up/gate admission/profile or reuses the existing one-stream gate cache while computing up on GPU.
- `next_action`: do not rerun the same fused config. The next design step must target the zero-hit up/gate batch cache gap: either build an admission profile for up/gate with enough useful slots, make the fused helper reuse the one-stream gate cache for `ffn_gate_exps` and only stage `ffn_up_exps`, or add a profile-driven preloading path. Any follow-up must first calculate expected cache slots/hit-rate from trace before running another strict cold experiment.

- `profile_n32_run`: `/root/lfz/runs/vendor-ds4-16gb/20260702T085352Z-20260702_ds_deepseek4_up_gate_fused_profile_n32/france-cpu40-vram0gb`.
- `profile_n32_result`: rejected/diagnostic only. `eval_tok_s=0.6`, `prompt_tok_s=0.9`, `TTFT=48876.805293ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=12816809984`, `ram_ok=true`, `correctness_ok=true` for a short 32-token output. It produced `batch_profile.csv` with `4746` entries and `15996` route events.
- `profile_capacity_bound`: On the generated up/gate route trace, a perfect static hot set gives only `16.6%` hit rate with the available `120` slots (`512MiB / 4.25MiB`), `27.3%` at `240` slots, `43.2%` at `512` slots, `61.2%` at `1024` slots, `79.5%` at `2048` slots, and `89.5%` at `3072` slots. Since the accepted SOTA already needs about `3072-3192` gate slots to hold token rate near `2.7 tok/s`, there is no credible VRAM split where fused up/gate batch cache also gets enough slots under the current one-expert-per-slot layout.
- `profile_preload_decision`: Do not run a full strict cold `GGML_MOE_VRAM_PROFILE=batch_profile.csv` promotion attempt with only `512MiB`; the theoretical top-120 hit bound is too low and would mostly move cold loads from runtime misses to preload/TTFT. A small preload smoke may be useful only after changing the cache representation or reusing the existing one-stream gate cache.
- `next_design`: The next viable implementation is not plain batch profile preload. It must reduce slot demand or avoid duplicating gate storage: reuse the one-stream gate cache for `ffn_gate_exps` while computing only `ffn_up_exps` in the batch/fused path, pack paired up+gate experts into one slot when tensor types/strides allow it, or add an up-only GPU op that preserves the existing high-hit gate one-stream path and fuses only the final elementwise `up * silu(gate)`.
- `checkpoint_policy`: Commit/push the current fused GPU-only diagnostic as rejected, not accepted SOTA, so the exact source and plan record can be reproduced later. The accepted SOTA remains the earlier pushed `2.7 tok/s` commit until a later run beats it under RAM/TTFT/correctness gates.


### 当前执行 attempt：ds-gate-up-one-stream-gpu-only-split

- `attempt_id`: `20260702-ds-gate-up-one-stream-gpu-only-split`
- `attempt_kind`: `diagnostic/gate-cache-preserving-up-GPU`
- `status`: rejected
- `bottleneck_basis`: The prior true fused up/gate batch path was rejected because batch cache had `hits=0 misses=98556` and the available `120` slots have only `16.6%` theoretical hit rate on the generated route trace. The accepted SOTA relies on the one-stream gate cache with thousands of slots; therefore a viable direction must avoid the small batch cache and reuse the one-stream machinery.
- `implementation`: Add comma/colon separated matching to `GGML_MOE_STREAM_ONE_NAME_FILTER` so the one-stream path can target both `ffn_gate_exps` and `ffn_up_exps` without enabling `ffn_down_exps`. Add `GGML_MOE_STREAM_ONE_GPU_ONLY_FILTER` in CPU `mul_mat_id`; matching tensors abort if `ggml_cuda_moe_stream_one` declines, proving there is no silent CPU fallback for gate/up. This keeps DeepSeek4's original `gate`, `up`, clamp, and `ggml_swiglu_split` graph instead of the rejected batch fused op.
- `theoretical_bound`: This is not a final fused kernel, but it is the minimum diagnostic for a gate-cache-preserving design. It can only improve if the one-stream cache can serve both gate and up with acceptable hit rate under `13568MiB`; otherwise it will regress due to doubled route volume and cache pressure. It avoids the batch 120-slot upper bound entirely.
- `test_config`: strict cold 16GB cgroup, France prompt, batch build binary, accepted top-k/profile envs, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps,ffn_up_exps`, `GGML_MOE_STREAM_ONE_GPU_ONLY_FILTER=ffn_gate_exps,ffn_up_exps`, no `GGML_MOE_STREAM_FUSED_UP_GATE`, no batch up/gate cache, same CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: promote only if `eval_tok_s > 2.7`, TTFT `<=45449.496149ms`, RAM including page cache `<=16000000000`, France output correct/coherent/complete, and stderr/trace prove both gate and up matched one-stream GPU with no CPU fallback. If it ties/regresses, keep as diagnostic and do not promote.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T090530Z-20260702_ds_gate_up_one_stream_gpu_only_split/france-cpu40-vram0gb`.
- `result`: rejected. `eval_tok_s=1.9`, `prompt_tok_s=0.8`, `TTFT=40727.520952ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15011897344`, `pgmajfault=170261`, `workingset_refault_file=7291876`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`. France output is semantically correct and coherent but truncated by the 192-token cap.
- `gpu_only_evidence`: The run completed with `GGML_MOE_STREAM_ONE_GPU_ONLY_FILTER=ffn_gate_exps,ffn_up_exps`, so any gate/up one-stream decline would have asserted. This proves both matching gate/up tensors were accepted by CUDA one-stream rather than silently falling back to CPU.
- `trace_result`: `rows=73873`, `gate_rows=47871`, `up_rows=26002`, `cache_hits=40947`, `cache_misses=32926`, `cache_inserts=3225`, `hit_rate=55.4%`, `src0_ms=73208.266`, `dontneed_ms=6662.003`, `total_ms=82275.427`. Accepted SOTA gate-only trace had `rows=35151`, `hits=30528`, `misses=4623`, `src0_ms≈23041`; adding up to the same one-stream cache more than doubled route volume and destroyed the high gate hit rate.
- `decision`: do not promote. This is much better than the rejected batch fused path (`0.5 tok/s`) but still far below accepted SOTA (`2.7 tok/s`). The experiment proves that avoiding the 120-slot batch cache is necessary but not sufficient; simply sharing the one-stream cache between gate and up creates too much cache pressure.
- `next_design`: A viable fused/up-GPU design must avoid storing full up experts in the same LRU as gate. Options left: compute up on GPU from a much smaller selected hot set only, add a paired up+gate cache representation that preserves the same number of route keys per expert pair, or keep gate-only SOTA and optimize CPU up math/threading rather than forcing all up experts to GPU.

### 当前执行 attempt：ds-deepseek4-up-gate-fused-batchcache13056

- `attempt_id`: `20260702-ds-deepseek4-up-gate-fused-batchcache13056`
- `attempt_kind`: `diagnostic/fused-up-gate-large-batch-cache`
- `status`: rejected
- `bottleneck_basis`: The prior DeepSeek4 fused GPU-only run used `GGML_MOE_VRAM_CACHE_MIB=512`, giving only about `120` up/gate slots and `0.0%` batch cache hit rate. The profile bound showed that around `3072` slots could reach about `89.5%` perfect static hit rate for the up/gate route trace, so the next experiment must test whether a large batch cache can remove the zero-hit bottleneck without one-stream gate cache.
- `test_config`: strict cold 16GB cgroup, France prompt, batch build binary, `cpu_moe=40`, `--vram-cache-gb 0`, `GGML_MOE_VRAM_CACHE_MIB=13056`, `GGML_MOE_STREAM_ONE_CACHE_MIB=0`, `GGML_MOE_STREAM_ONE_NAME_FILTER=__disabled__`, `GGML_MOE_STREAM_FUSED_UP_GATE=1`, `GGML_MOE_STREAM_FUSED_UP_GATE_GPU_ONLY=1`, `GGML_MOE_STREAM_FUSED_UP_GATE_MIXED_TYPES=1`, accepted top-k envs, same CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T091648Z-20260702_ds_deepseek4_up_gate_fused_gpu_only_batchcache13056/france-cpu40-vram0gb`.
- `result`: rejected. `eval_tok_s=0.8`, `prompt_tok_s=1.1`, `TTFT=45519.453728ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=12675284992`, `pgmajfault=396399`, `workingset_refault_file=22149441`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`. France output is semantically correct and coherent but truncated by the 192-token cap.
- `gpu_only_evidence`: stderr contains `[deepseek4] fused up/gate decode path enabled limit=10` and `[moe_stream] batched up/gate decode path active`, and the GPU-only guard did not abort.
- `cache_result`: batch cache allocated `12.8 GiB`, `3072 slots`, `4.25 MiB` each. Final counters: `hits=75110`, `misses=23446`, `preloads=0`, `hit_rate=76.2%`. This fixes the earlier `0.0%` hit-rate failure but still does not approach accepted SOTA.
- `decision`: do not promote. RAM and correctness pass, but token rate is far below accepted SOTA `2.7 tok/s`; TTFT is about `25.3%` above the accepted pushed SOTA `36333.418687ms`, exceeding the `20%` acceptance gate. Current accepted cold SOTA remains `7aa4e0f709ff020783c58c6f9a2d45dfec6790f6` at `2.7 tok/s`.
- `next_design`: Since large cache improved hit rate but not throughput, the current fused bottleneck is likely inside batch fused staging/compute/D2H rather than only cache capacity. Run a profiled diagnostic with the same cache size and `GGML_MOE_BATCH_PROFILE=1` to split `stage`, `quant`, `up`, `gate`, `fuse`, `d2h`, and `scatter` before changing kernels.

### 当前设计 attempt：ds-deepseek4-up-gate-fused-parallel-mxfp4

- `attempt_id`: `20260702-ds-deepseek4-up-gate-fused-parallel-mxfp4`
- `attempt_kind`: `implementation/fused-up-gate-parallel-compute`
- `status`: planned
- `bottleneck_basis`: The profiled large batch-cache fused run reached only `0.8 tok/s` even with `3072` batch cache slots and `75.5%` hit rate. `GGML_MOE_BATCH_PROFILE=1` split the up/gate fused helper as `stage=0.020ms`, `quant=0.001ms`, `up=7.376ms`, `gate=8.281ms`, `fuse=0.005ms`, `d2h=0.006ms`, `scatter=0.009ms`, `wall=15.714ms/call`. Thus the bottleneck inside fused up/gate is not swiglu, D2H, or cache lookup; it is serial up then gate expert matmul.
- `theoretical_bound`: The current serial up/gate compute costs about `7.376 + 8.281 = 15.657ms/call`. True parallel execution on separate CUDA streams can at best reduce this to `max(7.376, 8.281) + synchronization`, roughly `8.3ms/call`, a `1.88x` improvement for the up/gate portion. With about `4085 / 96 = 42.55` up/gate calls/token in the profile run, the up/gate-only throughput ceiling would improve from about `1 / (42.55 * 0.0157) = 1.5 tok/s` to about `1 / (42.55 * 0.0083) = 2.8 tok/s`. End-to-end token rate will be lower because down/attention/scheduler work still remains.
- `implementation_plan`: The existing `GGML_MOE_STREAM_UP_GATE_PARALLEL` branch is restricted to `GGML_TYPE_IQ2_S`. For DeepSeek4, allow the same parallel branch for non-mixed `GGML_TYPE_MXFP4` / `GGML_TYPE_F8_E4M3_B128` when batch MMVQ supports the type, keeping the fused GPU-only guard enabled. Run strict cold with `GGML_MOE_STREAM_UP_GATE_PARALLEL=1`, large batch cache `13056MiB`, one-stream disabled, and the same correctness/RAM/TTFT gates.
- `acceptance_gate`: promote only if `eval_tok_s > 2.7`, `ram_ok=true`, France output semantically correct/coherent, and TTFT within the current accepted SOTA gate unless explicitly recorded as rejected. Any GPU helper decline under `GGML_MOE_STREAM_FUSED_UP_GATE_GPU_ONLY=1` must abort rather than silently falling back to CPU.

### 方案重置：rollback-to-accepted-sota-and-cold-start-expert-pack

- `reset_id`: `20260702-rollback-to-accepted-sota-and-cold-start-expert-pack`
- `status`: planned
- `rollback_decision`: Roll back unaccepted fused up/gate source changes to accepted SOTA source state `7aa4e0f709ff020783c58c6f9a2d45dfec6790f6`. Keep the rejected fused diagnostics in this plan for audit, but do not carry their source changes forward. The current accepted cold SOTA remains `2.7 tok/s` with gate-only one-stream cache, `ram_ok=true`, `correctness_ok=true`, and TTFT within gate.
- `why_reset`: Large up/gate fused batch cache (`13056MiB`) improved batch hit rate to `75.5-76.2%` but still produced only `0.8 tok/s`. The parallel-stream diagnostic entered the parallel branch but profile stayed serial-shaped: `up≈7.34ms`, `gate≈8.23ms`, `kernel≈15.55ms`, so stream-level overlap did not materialize. This path is not the next highest-return cold-start optimization.
- `new_bottleneck_basis`: In accepted/rejected cold runs, cgroup RAM is dominated by file page cache, not anon heap. Latest fused diagnostic ended with `file≈12.72GB`, `anon≈1.1MB`, `kernel/slab≈165MB`, `pgmajfault≈309k`, `workingset_refault_file≈8.26M`, and many direct page-cache scans/steals. The current vendor runs did not enable `GGML_MOE_EXPERT_PACK`, so cold misses read scattered pages from the original GGUF. The next bottleneck to attack is cold-start random expert file I/O/page-cache churn under strict 16GB, not more VRAM cache capacity or stream-level up/gate fusion.
- `next_primary_direction`: Build or port a vendor-compatible expert pack for DeepSeek4 hot MoE tensors, starting with the accepted SOTA path's tensors and route profile. Use pack layout to make cold expert loads sequential/aligned and test `GGML_MOE_EXPERT_PACK` with direct/io_uring where supported. This must preserve the accepted SOTA gate-only one-stream GPU path first; do not re-enable rejected up/gate fused code unless a later design proves a true fused kernel.
- `design_step_1_pack_inventory`: Identify exact tensor names and expert sizes used by accepted SOTA (`ffn_gate_exps` admission profile and any down/up runtime loads), then verify vendor expert pack reader expects entries keyed by tensor name, expert id, and nbytes. Compare against `/root/lfz/ik_llama/scripts/create-moe-expert-pack.py` and either copy/port the pack creator or create a minimal vendor-side pack builder. Record pack file path, size, entry count, tensor coverage, alignment, and generation command in the plan before testing.
- `design_step_2_baseline_rerun`: Re-run accepted SOTA from source-restored state before enabling pack: strict cold drop_caches, 16GB cgroup, `cpu_moe=40`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, gate-only `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, accepted top-k envs, France prompt. This revalidates rollback and provides current baseline for pgmajfault/refault/page-cache metrics.
- `design_step_3_pack_experiments`: Run pack variants under strict cold 16GB: buffered pack, direct I/O if alignment succeeds, and io_uring if available. Keep `GGML_MOE_STREAM_ONE_CACHE_MIB=13568` unless pack requires small VRAM budget adjustment. For each variant record `eval_tok_s`, `prompt_tok_s`, TTFT, `memory_peak_bytes`, `memory_file_bytes`, `pgmajfault`, `workingset_refault_file`, pack hits/misses/direct/iouring counters, answer text, and exact env.
- `theoretical_bound`: Pack cannot reduce the GPU expert matmul cost on cache hits; it can only reduce cold miss source-load latency and page-cache churn. Bound should be estimated from miss count and expert size: accepted SOTA gate-only trace had about `4623` one-stream misses for the France run, each gate expert about `4.25MiB`. Random GGUF page faults/refaults can dominate cold source load; a sequential/aligned pack can at best approach device read bandwidth plus H2D copy bandwidth. Before promotion, compare observed `src0_ms`, `pgmajfault`, and refault reduction against this bound.
- `acceptance_gate`: A new accepted SOTA requires `eval_tok_s > 2.7`, strict `memory_peak_bytes <= 16000000000` including page cache, `ram_limit_killed=false`, France output semantically correct/coherent, and TTFT not more than 20% above the current accepted baseline unless committed only as an explicitly rejected checkpoint. If accepted, immediately record full reproduction info, commit source/plan/pack metadata, push to `ssd/vendor/deepseek-token-rate-16gb`, and rerun from pushed commit.
- `branch_policy`: All source/plan commits for this vendor work continue to be pushed to GitHub remote `ssd` branch `vendor/deepseek-token-rate-16gb` using the existing `L-Ark` identity. Rejected diagnostics may be committed only when clearly marked rejected; accepted SOTA must be committed and pushed immediately with enough information to reproduce future rollbacks.

### 当前执行：repro-sota-after-restore-before-kimi-merge

- `attempt_id`: `20260702-repro-sota-after-restore-before-kimi-merge`
- `status`: completed_reproduced_current_sota
- `purpose`: Revalidate DeepSeek accepted SOTA after rolling back rejected fused/up-gate source changes and before merging newer Kimi changes.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T103730Z-20260702_repro_sota_after_restore_before_kimi_merge/france-cpu40-vram0gb`.
- `test_config`: strict cold `drop_caches`, 16GB cgroup, `cpu_moe=40`, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `GGML_MOE_STREAM_CACHE_ADMIT_PROFILE=.Agent/profiles/vendor-ds4/current_sota_gate_freq_ge2.tsv`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `result`: reproduced accepted SOTA. `eval_tok_s=2.7`, `prompt_tok_s=0.9`, `TTFT=38591.376792ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15043362816`, `pgmajfault=282230`, `workingset_refault_file=2816754`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `decision`: This is the baseline to preserve while merging newer Kimi changes. Any post-merge checkpoint must keep DeepSeek at `>=2.7 tok/s` with RAM/correctness/TTFT gates passing before push as a guarded merge.

### 当前执行：merge-kimi-new-changes-guarded

- `attempt_id`: `20260702-merge-kimi-new-changes-guarded`
- `status`: accepted_as_guarded_merge_checkpoint
- `merged_commits`: cherry-picked Kimi code commits `1d456eafa` (`cuda: overlap current down staging with upgate`), `342ea9481` (`llama: drop expert mmap cache after prompt`), and `046d8e2de` (`llama: drop dense mmap cache after prompt`) on top of the restored DeepSeek SOTA branch. This preserves the Kimi functionality while avoiding a raw merge that would delete DeepSeek `.Agent` reproduction artifacts.
- `build_result`: `cmake --build build-ds4-moe-stream-batch --target llama-cli -j 8` passed after the cherry-picks.
- `guard_run`: `/root/lfz/runs/vendor-ds4-16gb/20260702T104255Z-20260702_post_kimi_new_changes_ds_sota_guard/france-cpu40-vram0gb`.
- `guard_config`: same accepted DeepSeek SOTA env as pre-merge: strict cold `drop_caches`, 16GB cgroup, `cpu_moe=40`, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, repo cache-admission profile `.Agent/profiles/vendor-ds4/current_sota_gate_freq_ge2.tsv`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `guard_result`: DeepSeek SOTA preserved. `eval_tok_s=2.7`, `prompt_tok_s=0.9`, `TTFT=38843.150588ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15041540096`, `pgmajfault=291550`, `workingset_refault_file=2817341`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `decision`: commit and push this guarded Kimi merge checkpoint to `ssd/vendor/deepseek-token-rate-16gb`, then run a clean pushed-commit reproduction before claiming the branch is fully reproducible. This is not a new DeepSeek token-rate SOTA beyond `2.7 tok/s`; it is a compatibility checkpoint that keeps Kimi changes while maintaining current DeepSeek SOTA.
- `next_design`: continue from the plan reset direction: focus on cold-start expert pack/io_uring or source-load/page-cache churn reduction rather than re-enabling rejected up/gate fused paths.

### 当前执行：pushed-kimi-new-changes-clean-ds-sota

- `attempt_id`: `20260702-pushed-kimi-new-changes-clean-ds-sota`
- `status`: completed_reproduced_guarded_merge_from_pushed_commit
- `pushed_commit`: `0cb2332f4` (`vendor-ds4: record kimi new changes guard run`), pushed to `ssd/vendor/deepseek-token-rate-16gb`.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T104550Z-20260702_pushed_kimi_new_changes_clean_ds_sota/france-cpu40-vram0gb`.
- `result`: DeepSeek SOTA preserved after Kimi new changes. `eval_tok_s=2.7`, `prompt_tok_s=1.0`, `TTFT=37346.733086ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15025664000`, `pgmajfault=297401`, `workingset_refault_file=2899805`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `decision`: This confirms the branch can be rolled back to the pushed guarded Kimi-new-changes checkpoint while preserving current DeepSeek SOTA. Continue optimization from this checkpoint, with current accepted token rate still `2.7 tok/s`.

### 当前设计：dense-mmap-drop-cold-start-probe

- `attempt_id`: `20260702-dense-mmap-drop-cold-start-probe`
- `status`: planned
- `bottleneck_basis`: After the Kimi new changes merge, the current DeepSeek SOTA still runs at `2.7 tok/s` with about `15.0GB` file page cache under a strict 16GB cgroup. Direct `GGML_MOE_EXPERT_PACK` is not yet connected to the SOTA one-stream gate path (`moe_stream.cu`); the expert-pack reader is in `moe_stream_batch.cu`. The newly merged Kimi mmap controls are therefore the first default-off cold-start mechanism that can be tested without changing the SOTA compute path.
- `hypothesis`: Enabling `LLAMA_DROP_DENSE_MMAP_CACHE=1` should call `drop_mmap_dense_pages()` after model tensor loading and release dense mmap file pages while keeping expert mmap pages available. Because dense tensors are already loaded/offloaded, this may reduce cgroup page-cache pressure and refaults during generation. It should not change model numerics. The upper bound is limited to reducing page-cache reclaim/refault overhead; it cannot reduce GPU/CPU expert matmul cost.
- `test_config`: strict cold 16GB cgroup, accepted SOTA env, plus `LLAMA_DROP_DENSE_MMAP_CACHE=1`, France prompt, same CLI args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `acceptance_gate`: promote only if `eval_tok_s > 2.7`, `ram_ok=true`, France output semantically correct/coherent, and TTFT within the accepted gate. If token rate ties but pgmajfault/refault improves, record as diagnostic only.

### 当前执行：dense-mmap-drop-cache-probe-result

- `attempt_id`: `20260702-dense-mmap-drop-cache-probe`
- `status`: rejected_tie_diagnostic
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T104903Z-20260702_dense_mmap_drop_cache_ds_sota_probe/france-cpu40-vram0gb`.
- `result`: `eval_tok_s=2.7`, `prompt_tok_s=0.9`, `TTFT=37772.882214ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15017316352`, `pgmajfault=294380`, `workingset_refault_file=2895919`, `ram_ok=true`, `correctness_ok=true`.
- `decision`: do not promote. It ties current SOTA and does not materially improve page-cache/refault metrics versus the pushed clean baseline. No clear dense mmap drop log was observed, so this env path may not release meaningful pages in the current loader mode.

### 当前设计：dense-mmap-drop-after-prompt-probe

- `attempt_id`: `20260702-dense-mmap-drop-after-prompt-probe`
- `status`: planned
- `hypothesis`: `LLAMA_DROP_DENSE_MMAP_AFTER_PROMPT=1` is triggered explicitly after the prompt decode (`batch_inp.n_tokens > 1`). If dense file pages remain resident after prompt evaluation, dropping them before generation may reduce cgroup reclaim pressure without dropping expert pages needed by one-stream misses. It should preserve numerics and Kimi functionality because it is env-gated.
- `test_config`: accepted SOTA env plus `LLAMA_DROP_DENSE_MMAP_AFTER_PROMPT=1`, strict cold 16GB cgroup, France prompt, same CLI args.
- `acceptance_gate`: same as current SOTA promotion gate; ties are diagnostic only.

### 当前执行：dense-mmap-drop-after-prompt-results

- `attempt_id`: `20260702-dense-mmap-drop-after-prompt-probe`
- `status`: rejected_tie_diagnostic
- `run_1`: `/root/lfz/runs/vendor-ds4-16gb/20260702T105127Z-20260702_dense_mmap_drop_after_prompt_ds_sota_probe/france-cpu40-vram0gb`, `eval_tok_s=2.7`, `prompt_tok_s=1.0`, `TTFT=35781.844462ms`, `memory_file_bytes=12457873408`, `pgmajfault=289189`, `workingset_refault_file=3115207`, `ram_ok=true`, `correctness_ok=true`.
- `run_2`: `/root/lfz/runs/vendor-ds4-16gb/20260702T105336Z-20260702_dense_mmap_drop_after_prompt_ds_sota_probe_rerun/france-cpu40-vram0gb`, `eval_tok_s=2.7`, `prompt_tok_s=1.0`, `TTFT=37181.348802ms`, `memory_file_bytes=15031271424`, `pgmajfault=291882`, `workingset_refault_file=2912850`, `ram_ok=true`, `correctness_ok=true`.
- `decision`: do not promote. The first run lowered final file cache and TTFT, but token rate tied and the second run returned to normal file-cache footprint. This env may remain useful as a TTFT/page-cache diagnostic, but it is not a stable token-rate improvement.
- `next_design`: Direct expert pack is not yet connected to the current one-stream gate SOTA path. The next implementation-level optimization should add default-off one-stream pack reads for `ffn_gate_exps` cache misses, then test buffered/direct/io_uring pack reads under strict 16GB.


### 方案重置：rollback-unverified-pack-patch-and-redesign

- `reset_id`: `20260702-rollback-unverified-pack-patch-and-redesign`
- `status`: active_plan
- `time`: `2026-07-02T11:12Z`
- `rollback_action`: Reverted the uncommitted `ggml/src/ggml-cuda/moe_stream.cu` one-stream expert-pack patch before build/test. The source tree is back on the pushed accepted SOTA code path at `25037165d` plus this plan-only update.
- `current_accepted_sota`: `2.7 tok/s`, strict cold `drop_caches`, 16GB cgroup including page cache, `cpu_moe=40`, gate-only one-stream DS4 cache, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `--vram-cache-gb 0`, accepted admission profile `.Agent/profiles/vendor-ds4/current_sota_gate_freq_ge2.tsv`, France correctness OK, TTFT `37346.733086ms` in the pushed clean rerun.
- `important_clarification`: The current accepted SOTA does **not** use an expert pack. Expert data on cache miss still comes from the original GGUF mmap/page-cache path. The SOTA VRAM expert cache mainly stores admitted hot `ffn_gate_exps` experts after they are loaded; it is not backed by a compact pack file today.
- `why_replan`: Adding a pack reader directly into the one-stream path is an implementation change before enough measurement. It could improve cold random source loads, but it also risks serializing miss reads behind one `FILE*` lock, duplicating host buffers, changing page-cache behavior, or creating a new correctness/TTFT risk. The next step must return to the required design/execute loop: measure the per-token cold bottleneck first, compute the bound, then implement only the highest-return path.
- `branch_policy`: All accepted or rejected checkpoints for this vendor work must be committed and pushed to GitHub remote `ssd`, branch `vendor/deepseek-token-rate-16gb`, using `L-Ark <fliangae@connect.ust.hk>`. If a new accepted SOTA appears, immediately record exact source commit, run directory, env, CLI, cgroup limit, page-cache metrics, answer text, pack/profile hashes if any, commit the source/plan/metadata, push, then rerun from the pushed commit so future rollback can reproduce the metric.

#### 新设计阶段 1：重新定位当前 cold-start bottleneck

- `goal`: Produce a fresh per-token profile for the accepted SOTA, not just aggregate tok/s.
- `experiment`: Re-run accepted SOTA with existing trace/profile envs and strict 16GB cgroup. Capture per-token wall time split into source load/page fault, H2D copy, VRAM cache lookup/insert, GPU gate matmul, CPU fallback/up/down work if any, sampler/output, plus `memory.current`, `memory.stat`, `pgmajfault`, `workingset_refault_file`, and page-cache scan/steal counters.
- `expected_bottleneck`: The prior data points to cold source-load/page-cache churn and cache-miss expert movement as the likely limiter, because RAM is almost entirely file cache (`~15GB`) and major faults/refaults are high. Verify rather than assume.
- `acceptance`: No source change in this stage. It only updates the plan with measured bottleneck and a prioritized optimization list.

#### 新设计阶段 2：expert-pack feasibility without touching compute path

- `goal`: Decide whether an expert pack can help the current one-stream SOTA before wiring it into `moe_stream.cu`.
- `inventory`: List all `ffn_gate_exps` `(tensor, expert_id)` pairs admitted by `.Agent/profiles/vendor-ds4/current_sota_gate_freq_ge2.tsv`, estimate entry count, expert bytes, total pack size, and compare to cold miss count from the SOTA trace. Record the profile hash and exact model GGUF hash.
- `theoretical_bound`: Bound the possible gain from pack I/O as `miss_bytes / achievable_read_bandwidth` plus H2D copy. If the observed per-token non-compute time is already below this bound, deprioritize pack work. If random GGUF page faults dominate and sequential pack read can remove many major faults/refaults, promote pack implementation.
- `no-source probe`: If possible, build a minimal offline pack and run a separate reader benchmark inside the same 16GB cgroup to compare random GGUF expert reads vs sequential/aligned pack reads for the exact SOTA miss sequence. This avoids changing inference code before the I/O bound is proven.

#### 新执行 candidate A：default-off one-stream expert pack, only if stage 2 proves value

- `implementation_rule`: Keep default behavior unchanged. Add pack support behind `GGML_MOE_STREAM_ONE_EXPERT_PACK`, limited initially to `ffn_gate_exps` cache misses. Do not change Kimi code paths or batch fused paths.
- `correctness_rule`: For every promoted run, France output must remain semantic/coherent. For large token-rate movement, also run the small prompt set used earlier to verify prompt robustness.
- `performance_rule`: Promote only if strict cold `eval_tok_s > 2.7`, host RAM including page cache remains `<=16000000000`, and TTFT stays within the 20% gate. If TTFT exceeds the gate but token rate improves, commit only as `rejected_ttft` and immediately plan TTFT recovery.
- `gap_analysis`: If measured speed differs from the theoretical bound, analyze pack hit/miss count, read serialization, direct I/O alignment, page-cache residency, H2D overlap, and CPU fallback before guessing.

#### 新执行 candidate B：cache-admission/profile refinement

- `goal`: Improve hit rate without increasing RAM or reducing correctness.
- `method`: Use the fresh trace to identify high-frequency misses not admitted by `current_sota_gate_freq_ge2.tsv`, then test small profile changes under the same 13568MiB one-stream cache. This is lower risk than compute-path fusion because it only changes which gate experts are cached.
- `bound`: Estimate gain from eliminated miss load time and available VRAM cache capacity. Reject profile changes that reduce hit quality, raise TTFT over gate, or push RAM above 16GB.

#### 新执行 candidate C：CPU fallback reduction after I/O work

- `goal`: Only revisit up/down or fused paths after I/O/cache bottleneck is quantified.
- `rule`: Any up/gate/down fusion must first show true parallel work in traces, not just branch entry. Prior rejected fused runs stayed around `0.8 tok/s`, so this path is lower priority until a trace proves CPU fallback or GPU matmul is the dominant remaining time.

### 当前执行：sota-bottleneck-trace-after-replan

- `attempt_id`: `20260702-sota-bottleneck-trace-after-replan`
- `status`: completed_diagnostic
- `time`: `2026-07-02T11:02Z-11:18Z`
- `source_commit`: `cb89fccb9` (`vendor-ds4: replan after pack rollback`).
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T110237Z-20260702_sota_bottleneck_trace_after_replan/france-cpu40-vram0gb`.
- `config`: accepted SOTA config plus `GGML_MOE_STREAM_ONE_TRACE_OUT={case_dir}/one_trace.csv`; strict cold `drop_caches`, 16GB cgroup, France prompt.
- `result`: `eval_tok_s=2.7`, `prompt_tok_s=0.9`, `TTFT=37444.46683ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15026978816`, `pgmajfault=292790`, `workingset_refault_file=2912935`, `ram_ok=true`, `correctness_ok=true`.
- `trace_summary`: `rows=35151`, all `ffn_gate_exps.weight`, `cache_hits=30528`, `cache_misses=4623`, `cache_inserts=3337`, `unique_miss_pairs=4599`, `src0_ms=22933.823`, `miss_src0_ms=22910.543`, `total_ms=25124.006`, `span_ms=67669.820`, `gap_sum_ms=42545.854`.
- `interpretation`: The one-stream gate path itself spends about `25.1s`, dominated by cold miss source load (`~22.9s`). The larger `~42.5s` trace gap is outside one-stream gate and must be CPU fallback/up/down or graph scheduling. Therefore pack/source-load can help but is not the only bottleneck.

### 当前执行：sota-cpu-gap-profile-after-replan

- `attempt_id`: `20260702-sota-cpu-gap-profile-after-replan`
- `status`: completed_diagnostic
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T110537Z-20260702_sota_cpu_gap_profile_after_replan/france-cpu40-vram0gb`.
- `config`: same traced SOTA plus `GGML_KIMI_CPU_MOE_PROFILE=1`, `GGML_KIMI_CPU_MOE_NAME_PROFILE=1`, and `GGML_KIMI_CPU_MOE_FALLBACK_PROFILE_OUT={case_dir}/fallback-profile.csv`.
- `result`: `eval_tok_s=2.7`, `prompt_tok_s=0.9`, `TTFT=38428.228915ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15034163200`, `pgmajfault=273332`, `workingset_refault_file=2881507`, `ram_ok=true`, `correctness_ok=true`.
- `cpu_profile`: profile printed `down calls=16920 total=3.351 ms/call cuda_single=1.533 fallback_t0=1.804`, with `single_accept=35151` and `single_decline=38222`. Since one-stream trace has `35151` gate calls, the accepted single path is gate; the declined/fallback side is up/down.
- `fallback_summary`: decode fallback `up=13142.825ms`, `down=9202.939ms`; prompt fallback `up=3896.147ms`, `down=4261.930ms`. Combined fallback measured in CSV is about `30.5s`, with decode up/down alone about `22.35s`.
- `interpretation`: Current cold SOTA has two comparable bottlenecks: gate miss source load (`~23s`) and up/down CPU fallback (`~30.5s` total, `~22.35s` decode). Previous fused up/down attempts regressed to `0.8 tok/s`, so the next implementation should prefer low-risk gate miss I/O reduction first, while keeping CPU fallback reduction as the next major direction after I/O is quantified.

### 当前执行：expert-pack-feasibility-no-source-probe

- `attempt_id`: `20260702-expert-pack-feasibility-no-source-probe`
- `status`: completed_diagnostic_bound_supports_implementation
- `profile_inventory`: `.Agent/profiles/vendor-ds4/current_sota_gate_freq_ge2.tsv`, `3313` pairs, `sha256=8134c320730e0ba236d103ba4a0b53505b3bab16e69d8bdc2a08607ecfcc274b`. In the traced SOTA, all `3313` profile pairs were seen and inserted; additional `1286` miss pairs were single-use and not admitted. This means cache-admission refinement is unlikely to help without more VRAM, because the non-profile misses have only one event each.
- `model_inventory`: `/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf`, size `156148189760`, mtime `2026-06-22 14:12:05.810122536 +0000`. Full sha256 was intentionally skipped during this diagnostic because hashing 156GB took too long; compute it before promoting a new SOTA if required.
- `minimal_pack`: `/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-gate-miss-firstorder-20260702.pack`, first-miss order, `4599` unique gate miss pairs, payload `20495204352` bytes (`19.088GiB`), file size about `20G`.
- `read_sequences`: `/root/lfz/runs/vendor-ds4-16gb/20260702T110237Z-20260702_sota_bottleneck_trace_after_replan/read-sequences/{orig_miss_sequence.tsv,pack_miss_sequence.tsv}`, `4623` miss events, total read bytes `20602159104` (`19.187GiB`).
- `readbench_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T111430Z-readbench-ds4-gate-pack-feasibility`, each case cold `drop_caches` and 16GB `MemoryMax`.
- `readbench_result`: original GGUF miss sequence `22.505063s`, `0.852575GiB/s`; first-order pack miss sequence `13.656855s`, `1.404954GiB/s`; both read `20602159104` bytes and produced the same checksum.
- `theoretical_bound`: A perfect one-stream pack implementation for this exact France trace can at most save roughly `8.85s` from the measured gate miss source-load path before accounting for integration overhead, read serialization, H2D overlap, and page-cache differences inside inference. Against the traced `span_ms≈67.7s`, this is an upper-bound speedup around `15%` for generation-side wall time, so an accepted improvement from `2.7 tok/s` toward about `3.0-3.1 tok/s` is plausible but not guaranteed.
- `next_design`: Implement default-off one-stream pack reads only after this bound. Use `GGML_MOE_STREAM_ONE_EXPERT_PACK=<pack>` so Kimi and existing batch pack paths are unchanged. Start with buffered reads into existing pinned staging; then test direct/io_uring only if buffered pack improves or shows read serialization as the gap. Do not promote unless strict 16GB RAM, correctness, and TTFT gates pass. If the measured gain is far below the `~8.85s` I/O bound, debug pack hit count, read order, page-cache residency, and H2D overlap before changing compute.

### 当前执行：one-stream-pack-buffered-firstorder-sota

- `attempt_id`: `20260702-one-stream-pack-buffered-firstorder-sota`
- `status`: accepted_new_cold_sota_pending_pushed_rerun
- `time`: `2026-07-02T11:21Z-11:31Z`
- `source_base`: `341ac191c` plus default-off `GGML_MOE_STREAM_ONE_EXPERT_PACK` implementation in `ggml/src/ggml-cuda/moe_stream.cu`.
- `source_change`: Added one-stream expert-pack lookup/read on VRAM cache miss only when `GGML_MOE_STREAM_ONE_EXPERT_PACK` is set. It uses `pread` against the existing v1 pack format and per-slot pinned host staging, then feeds the existing `vram_cache_insert` or transient `cudaMemcpyAsync`. Default behavior is unchanged when the env is unset.
- `pack_builder`: committed as `.Agent/run-tools/create_ds4_gate_trace_pack.py`; generated first-miss-order pack from the traced SOTA `one_trace.csv`.
- `pack_path`: `/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-gate-miss-firstorder-20260702.pack`.
- `pack_sha256`: `7ad26d8b14c20dccd4106a8abbffc9f846eb2fedff4fd00a5af7060941204076`.
- `pack_contents`: `4599` unique `ffn_gate_exps.weight` miss pairs, payload `20495204352` bytes (`19.088GiB`), first-miss order for the France SOTA trace.
- `default_off_guard`: `/root/lfz/runs/vendor-ds4-16gb/20260702T112109Z-20260702_one_pack_patch_default_off_guard/france-cpu40-vram0gb`, no pack env, strict cold 16GB. Result `eval_tok_s=2.7`, `prompt_tok_s=0.9`, `TTFT=38159.157809ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15026552832`, `pgmajfault=294655`, `workingset_refault_file=3122529`, `ram_ok=true`, `correctness_ok=true`. This proves default-off source does not regress current SOTA.
- `accepted_run`: `/root/lfz/runs/vendor-ds4-16gb/20260702T112311Z-20260702_one_pack_buffered_firstorder_probe/france-cpu40-vram0gb`.
- `accepted_config`: accepted SOTA env plus `GGML_MOE_STREAM_ONE_EXPERT_PACK=/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-gate-miss-firstorder-20260702.pack`, strict cold `drop_caches`, 16GB cgroup, France prompt, CLI extra args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `accepted_result`: `eval_tok_s=3.4`, `prompt_tok_s=1.2`, `TTFT=32260.354472ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15040282624`, `pgmajfault=255092`, `workingset_refault_file=1945368`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `answer`: France answer was semantic and coherent: it identifies France as the French Republic in Western Europe, mentions history/culture/global influence, landmarks including the Eiffel Tower/Louvre/Versailles, cuisine/wine/fashion/art/science, EU membership, economy, and historical/modern character.
- `pack_counters`: `[moe_stream] one expert pack: hits=4623 misses=0 reads=4623 bytes=20602159104 failures=0 entries=4599`; VRAM cache `hits=30528 misses=4623 hit_rate=86.8%`.
- `gap_vs_bound`: no-source readbench predicted at most `~8.85s` gate miss I/O savings. Accepted run reduced elapsed from about `85.81s` clean baseline to `72.19s`, and TTFT from `37346.733086ms` to `32260.354472ms`; this is directionally consistent with the I/O bound plus lower refault pressure (`workingset_refault_file 2.90M -> 1.95M`).
- `decision`: This is a new accepted vendor DeepSeek cold-start SOTA if the pushed rerun reproduces. Commit and push immediately to `ssd/vendor/deepseek-token-rate-16gb`, then rerun from the pushed commit with the same pack path and sha to finalize reproducibility.

### 当前执行：pushed-one-stream-pack-buffered-firstorder-sota-rerun

- `attempt_id`: `20260702-pushed-one-stream-pack-buffered-firstorder-sota-rerun`
- `status`: accepted_new_cold_sota_reproduced_from_pushed_commit
- `pushed_commit`: `0312377cec2b617c64b97f30a14e5d474a3b2893` (`vendor-ds4: add one-stream expert pack sota`), pushed to `ssd/vendor/deepseek-token-rate-16gb`.
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T112702Z-20260702_pushed_one_pack_buffered_firstorder_sota_rerun/france-cpu40-vram0gb`.
- `config`: same accepted pack config, strict cold `drop_caches`, 16GB cgroup, France prompt, same pack path and `pack_sha256=7ad26d8b14c20dccd4106a8abbffc9f846eb2fedff4fd00a5af7060941204076`.
- `result`: `eval_tok_s=3.4`, `prompt_tok_s=1.2`, `TTFT=33395.660286ms`, `elapsed_seconds=73.46`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15040389120`, `pgmajfault=270675`, `workingset_refault_file=2207042`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `answer`: Same semantic/coherent France paragraph as the accepted probe; mentions France/French Republic, Western Europe, history/culture/global influence, Eiffel Tower/Louvre/Versailles, cuisine/wine/fashion/art/science, EU membership, economy, and modern vitality.
- `counters`: one expert pack `hits=4623 misses=0 reads=4623 bytes=20602159104 failures=0 entries=4599`; VRAM cache `hits=30528 misses=4623 hit_rate=86.8%`.
- `decision`: This finalizes the current accepted vendor DeepSeek cold-start SOTA at `3.4 tok/s`, reproducible from pushed commit `0312377cec2b617c64b97f30a14e5d474a3b2893` with the recorded pack file/path/hash. Continue next with either small prompt-set validation for this pack strategy or a more general prompt-independent pack/profile construction before claiming robustness beyond the France SOTA prompt.

### 方案重置：rollback-to-odirect-sota-and-redesign

- `reset_id`: `20260702-rollback-to-odirect-sota-and-redesign`
- `status`: active_plan
- `time`: `2026-07-02T15:20Z`
- `rollback_action`: Reverted the uncommitted MXFP4/down-batch diagnostic changes in `ggml/src/ggml-cpu/ggml-cpu.c` and `ggml/src/ggml-cuda/moe_stream_batch.cu`. The working source is back on pushed commit `c37df4434ecdacdf910506f24209b6ccabed341c` plus this plan-only update.
- `current_accepted_sota`: vendor DeepSeek strict cold-start O_DIRECT expert-pack SOTA, `eval_tok_s=4.2`, `prompt_tok_s=1.5`, `TTFT=29984.770823ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15080456192`, `ram_ok=true`, `correctness_ok=true`.
- `current_sota_run`: `/root/lfz/runs/vendor-ds4-16gb/20260702T133103Z-20260702_pushed_onepack_odirect_sota_rerun/france-cpu40-vram0gb`.
- `current_sota_config`: `cpu_moe=40`, `--vram-cache-gb 0`, extra args `-c 256 -b 16 -ub 16 -t 20 -tb 20`, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `GGML_MOE_STREAM_CACHE_ADMIT_PROFILE=.Agent/profiles/vendor-ds4/current_sota_gate_freq_ge2.tsv`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, `GGML_MOE_STREAM_ONE_EXPERT_PACK=/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-gate-miss-firstorder-20260702.pack`, `GGML_MOE_STREAM_ONE_EXPERT_PACK_IO=direct`.
- `repro_files`: expert pack sha256 `7ad26d8b14c20dccd4106a8abbffc9f846eb2fedff4fd00a5af7060941204076`; admission profile sha256 `8134c320730e0ba236d103ba4a0b53505b3bab16e69d8bdc2a08607ecfcc274b`.
- `branch_policy`: All future accepted or rejected checkpoints must be committed and pushed to GitHub remote `ssd`, branch `vendor/deepseek-token-rate-16gb`, using git identity `L-Ark <fliangae@connect.ust.hk>`. Any new accepted SOTA must include exact source commit, run directory, full env/CLI, cgroup limit, page-cache metrics, answer text, pack/profile hashes, and must be rerun from the pushed commit so future rollback can reproduce the metric.
- `reason_for_reset`: The MXFP4/down-batch direction has not met correctness or speed gates. The latest batch-enabled attempt produced incorrect France output and lower token rate, while single-stream MXFP4 compare showed small numerical error. Therefore it is not a valid optimization path until the batch mapping/scatter bug is proven and fixed. Do not leave rejected batch code enabled on the SOTA branch.

#### 新设计阶段 1：复现并锁定回退后的 SOTA

- `goal`: Rebuild with `GGML_CUDA_MOE_STREAM_BATCH=OFF` and rerun the current O_DIRECT SOTA under strict cold `drop_caches` and 16GB cgroup.
- `acceptance`: `eval_tok_s` should reproduce near `4.2 tok/s` within normal run variance, France output must be semantic/coherent, `memory_peak_bytes <= 16000000000`, page cache must be inside the same cgroup, and TTFT must remain within the 20% gate.
- `recording`: Record the fresh run path and metrics in this plan before any new implementation. If reproduction fails, stop optimization and debug reproducibility first.

#### 新设计阶段 2：重新测量 4.2 SOTA 的 token-level bottleneck

- `goal`: Locate the current bottleneck after O_DIRECT, not the old buffered-pack bottleneck.
- `experiment`: Run the accepted O_DIRECT SOTA with one-stream trace, CPU fallback profile, batch decline debug disabled, and strict cgroup metrics. Measure per-token time split into gate pack read/direct I/O, H2D copy/cache insert, gate GPU compute/sync, up/down CPU fallback, sampler/output, and graph scheduling gaps.
- `expected_bottleneck`: Prior post-O_DIRECT trace showed gate source load around `5.3s`, one-stream total around `7.1s`, and CPU fallback around `26.1s`, with decode up/down each around `9.3s`. Verify this on the reset source before choosing implementation.
- `priority_rule`: Rank candidates by measured removable time and theoretical bound. Do not start with a code path unless it can plausibly move wall time more than run noise and has a correctness test plan.

#### 新执行 candidate A：low-risk O_DIRECT/gate-pack I/O tuning

- `hypothesis`: The current pack path still performs many per-expert direct reads (`4623` reads, about `20.6GB`). If direct I/O issue size, alignment, or ordering is suboptimal, batching adjacent reads or using a split reader pool may reduce syscall/device wait without touching math.
- `bound`: Upper bound is the measured gate pack read time from stage 2. If gate I/O is only about `5s`, this path cannot yield more than about `5s` total wall-time reduction and should not be over-prioritized.
- `implementation_rule`: Default-off env only. Keep Kimi paths unchanged. Keep France correctness and pack hit counters identical: `hits=4623`, `misses=0`, `failures=0`.
- `reject_rule`: Reject immediately if RAM exceeds 16GB including page cache, direct read fallback appears, TTFT exceeds the accepted gate for an accepted SOTA, or output correctness changes.

#### 新执行 candidate B：CPU fallback reduction without MXFP4 batch promotion

- `hypothesis`: After O_DIRECT, up/down CPU fallback is likely the largest remaining cost. The next valid compute optimization must first prove numerical equivalence on the exact fallback route before measuring speed.
- `first_step`: Build a compare-only diagnostic for the actual candidate path, limited to a small number of rows, recording CPU vs GPU output error by tensor/expert/row/col. Do not enable it in SOTA runs.
- `allowed_candidates`: Investigate up/down cache admission, down-only single-stream for proven-correct MXFP4 paths, or a corrected batch path only after the mapping/scatter issue is isolated. The previously rejected MXFP4 batch switch must stay disabled until it passes compare and full France correctness.
- `bound`: Use fallback profile totals as the hard upper bound. If decode up/down fallback is about `18-22s`, a perfect elimination sets the maximum possible improvement; expected practical gain must account for H2D, GPU kernel time, and cache pressure.

#### 新执行 candidate C：prompt-robust pack/profile validation

- `goal`: Ensure improvements are not France-only artifacts before claiming general steady behavior.
- `test_set`: warmup France prompt, then test France again, quantum computing, short Python Fibonacci, Japan introduction, and climate-change summary.
- `acceptance`: Under strict 16GB cgroup, outputs must be coherent, RAM must remain within limit, and token rate should stay close to the SOTA envelope. If the first-order France pack misses badly on other prompts, design a general pack/profile from multi-prompt traces before optimizing further.

#### Commit and rollback rules

- `accepted_sota`: If a new run improves accepted cold-start token rate while meeting RAM, correctness, and TTFT gates, immediately commit source/plan/metadata and push to `ssd/vendor/deepseek-token-rate-16gb`, then rerun from the pushed commit.
- `rejected_result`: If speed regresses, correctness fails, RAM exceeds limit, or TTFT is too high, record the run as rejected. Commit only plan/diagnostic records when useful, and revert any default-on or risky source change before continuing.
- `reproducibility`: A SOTA is not considered final until the exact pushed commit can reproduce it with recorded pack/profile hashes and cgroup metrics.

### 当前执行：post-replan-sota-repro-and-kimi-latest-merge-guard

- `attempt_id`: `20260702-post-replan-sota-repro-and-kimi-latest-merge-guard`
- `status`: accepted_guard_pending_push
- `time`: `2026-07-02T15:02Z-15:05Z`
- `pre_merge_source`: `e0df0c26da2f0448726d6343a6cdf94e2eae0a28` (`vendor-ds4: replan from odirect sota`).
- `kimi_latest`: fetched `ssd/vendor/kimi-moe-stream-on-vendor` at `f6175d8e5`; delta from prior Kimi head `58885e449` is only `.Agent/plans/kimi-token-rate-16gb-optimization-plan.md` (`934` inserted lines). No source files changed, so this merge does not change Kimi runtime functionality or DeepSeek runtime code.
- `pre_merge_repro_run`: `/root/lfz/runs/vendor-ds4-16gb/20260702T150201Z-20260702_odirect_sota_repro_after_replan/france-cpu40-vram0gb`.
- `pre_merge_repro_result`: `eval_tok_s=4.1`, `prompt_tok_s=1.5`, `TTFT=30429.314418ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15105679360`, `pgmajfault=271997`, `workingset_refault_file=1657406`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `merge_action`: merged `ssd/vendor/kimi-moe-stream-on-vendor` into `feat/ds4-moe-stream-on-vendor` with merge commit message `vendor-ds4: merge kimi latest plan update`.
- `post_merge_guard_run`: `/root/lfz/runs/vendor-ds4-16gb/20260702T150335Z-20260702_post_kimi_latest_plan_merge_odirect_sota_guard/france-cpu40-vram0gb`.
- `post_merge_guard_config`: current O_DIRECT SOTA config: `cpu_moe=40`, `--vram-cache-gb 0`, extra args `-c 256 -b 16 -ub 16 -t 20 -tb 20`, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `GGML_MOE_STREAM_CACHE_ADMIT_PROFILE=.Agent/profiles/vendor-ds4/current_sota_gate_freq_ge2.tsv`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, `GGML_MOE_STREAM_ONE_EXPERT_PACK=/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-gate-miss-firstorder-20260702.pack`, `GGML_MOE_STREAM_ONE_EXPERT_PACK_IO=direct`, strict cold `drop_caches`, 16GB cgroup.
- `post_merge_guard_result`: `eval_tok_s=4.1`, `prompt_tok_s=1.6`, `TTFT=29574.256369ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15101304832`, `pgmajfault=271842`, `workingset_refault_file=1666518`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `post_merge_answer`: France output is semantic and coherent: it identifies France/French Republic in Western Europe, mentions history/culture/global influence, Eiffel Tower/Louvre/Versailles, cuisine/wine/fashion/art/science, EU membership, economy, and historical/modern vitality.
- `post_merge_counters`: one expert pack `hits=4623 misses=0 reads=4623 bytes=20602159104 failures=0 entries=4599 direct_enabled=1 direct_reads=4623 direct_failures=0 direct_fallbacks=0`; VRAM cache `hits=30528 misses=4623 hit_rate=86.8%`.
- `decision`: Merge is safe for the DeepSeek SOTA path. Push this merge plus plan record to `ssd/vendor/deepseek-token-rate-16gb`, then continue with the next design stage: post-O_DIRECT token-level bottleneck trace.

### 当前设计：post-odirect-bottleneck-trace-and-thread-sweep

- `attempt_id`: `20260702-post-odirect-bottleneck-trace-and-thread-sweep`
- `status`: active_design_before_practice
- `time`: `2026-07-02T15:06Z-15:12Z`
- `source_commit`: `9bafd4377c060e08022dc69f6c56905e4cb1d901` (`vendor-ds4: record kimi latest merge guard`), already pushed to `ssd/vendor/deepseek-token-rate-16gb`.
- `trace_run`: `/root/lfz/runs/vendor-ds4-16gb/20260702T150610Z-20260702_post_kimi_odirect_bottleneck_trace/france-cpu40-vram0gb`.
- `diag_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260702T151000Z-post-odirect-bottleneck-trace`.
- `trace_result`: `eval_tok_s=4.0`, `prompt_tok_s=1.6`, `TTFT=30026.483027ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15078334464`, `ram_ok=true`, `correctness_ok=true`.
- `one_stream_summary`: `rows=35151`, `hits=30528`, `misses=4623`, `inserts=3337`, `hit_rate=86.85%`, `src0_ms=5405.510`, `miss_src0_ms=5383.069`, `src1_ms=164.101`, `kernel_ms=339.665`, `sync_ms=1018.218`, `total_ms=7160.844`, `span_ms=44433.202`, `gap_sum_ms=37272.358`.
- `fallback_summary`: total CPU fallback `26851.367ms`; decode fallback `up=9727.651ms`, `down=9574.938ms`; prompt fallback `up=3208.326ms`, `down=4340.452ms`.
- `bottleneck`: After O_DIRECT, gate I/O is no longer the dominant bottleneck. The hard upper bound for further gate read tuning is roughly the measured `5.38s` miss source-load time, while CPU up/down fallback is about `26.85s` total and decode up/down alone is about `19.30s`.
- `next_practice`: Run a low-risk thread sweep before source changes: keep all SOTA env/pack/profile fixed and test `-t/-tb` values above the current `20` to see whether CPU fallback is under-threaded. This does not change model math, Kimi functionality, pack contents, or admission profile.
- `thread_sweep_bound`: If CPU fallback scaled perfectly from 20 to 28-32 threads, the absolute ceiling is removal of part of the `26.85s` fallback time. In practice memory bandwidth and scheduling overhead will limit gains. A valid accepted improvement must still pass RAM `<=16000000000`, France correctness, and TTFT within the 20% gate.
- `thread_sweep_candidates`: test `-t 24 -tb 24`, then `-t 28 -tb 28`, then `-t 32 -tb 32` only if prior candidates do not regress sharply. If token rate improves, rerun the best candidate once for reproducibility before promoting.

### 当前执行：post-odirect-thread-sweep-results

- `attempt_id`: `20260702-post-odirect-thread-sweep-results`
- `status`: rejected_no_token_rate_sota
- `time`: `2026-07-02T15:08Z-15:13Z`
- `common_config`: current O_DIRECT SOTA env/pack/profile, strict cold `drop_caches`, 16GB cgroup, France prompt, only `-t/-tb` changed.
- `t24_run`: `/root/lfz/runs/vendor-ds4-16gb/20260702T150844Z-20260702_odirect_thread24_sweep/france-cpu40-vram0gb`, `eval_tok_s=4.0`, `prompt_tok_s=1.7`, `TTFT=27911.745904ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15077531648`, `ram_ok=true`, `correctness_ok=true`.
- `t28_run`: `/root/lfz/runs/vendor-ds4-16gb/20260702T151004Z-20260702_odirect_thread28_sweep/france-cpu40-vram0gb`, `eval_tok_s=4.1`, `prompt_tok_s=1.7`, `TTFT=28009.794829ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15098593280`, `ram_ok=true`, `correctness_ok=true`.
- `t32_run`: `/root/lfz/runs/vendor-ds4-16gb/20260702T151127Z-20260702_odirect_thread32_sweep/france-cpu40-vram0gb`, `eval_tok_s=2.3`, `prompt_tok_s=1.6`, `TTFT=30367.436186ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15071858688`, `ram_ok=true`, `correctness_ok=true`.
- `interpretation`: Increasing CPU threads does not improve generation token rate. `t24/t28` improve TTFT but stay below or equal normal SOTA variance, while `t32` likely oversubscribes memory bandwidth or scheduling and strongly regresses decode speed.
- `decision`: Do not promote any thread-sweep config as token-rate SOTA. Keep accepted config at `-t 20 -tb 20` unless a later TTFT-specific pass needs `t28`.
- `next_design`: Since simple CPU thread scaling did not remove the `~26.85s` fallback bottleneck, test up/down fallback reduction via cache/admission/top-k configuration before source changes. First low-risk probe: vary `GGML_MOE_KEEP_TOPK_UPDOWN` while keeping gate pack and cache unchanged. Theoretical upper bound is the measured decode up/down fallback (`~19.30s`) minus added GPU/cache overhead. Reject if VRAM OOM, RAM exceeds 16GB, correctness fails, or token rate/TTFT regress.

### 当前执行：post-odirect-updown-topk-probe-results

- `attempt_id`: `20260702-post-odirect-updown-topk-probe-results`
- `status`: rejected_no_token_rate_sota
- `time`: `2026-07-02T15:14Z-15:18Z`
- `common_config`: current O_DIRECT SOTA env/pack/profile, strict cold `drop_caches`, 16GB cgroup, France prompt, only `GGML_MOE_KEEP_TOPK_UPDOWN` changed.
- `topk5_run`: `/root/lfz/runs/vendor-ds4-16gb/20260702T151413Z-20260702_odirect_keep_topk_updown5_probe/france-cpu40-vram0gb`, `eval_tok_s=3.1`, `prompt_tok_s=1.4`, `TTFT=31613.87263ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15049924608`, `workingset_refault_file=5135535`, `ram_ok=true`, `correctness_ok=true`.
- `topk3_run`: `/root/lfz/runs/vendor-ds4-16gb/20260702T151605Z-20260702_odirect_keep_topk_updown3_probe/france-cpu40-vram0gb`, `eval_tok_s=3.6`, `prompt_tok_s=1.6`, `TTFT=29104.280615ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15061024768`, `workingset_refault_file=3822457`, `ram_ok=true`, `correctness_ok=true`.
- `interpretation`: Simple global up/down top-k changes are worse than the accepted `4` plus layer `10-39 -> 3` strategy. Increasing to `5` raises page/refault pressure and decode slows; decreasing to `3` also loses generation speed, likely because the fallback/GPU balance worsens rather than just reducing work.
- `decision`: Keep accepted top-k config unchanged.
- `next_design`: Probe one-stream gate VRAM cache capacity near the current boundary. The trace has `3337` inserts but the accepted cache reports `3192` slots (`13.2GiB`, `4.25MiB` each), so there are avoidable evictions. Increasing `GGML_MOE_STREAM_ONE_CACHE_MIB` from `13568` toward `14080` may reduce repeat direct reads. The theoretical upper bound is limited to the repeat-miss portion of `miss_src0_ms=5383.069`, not the full CPU fallback. Reject on CUDA OOM, cache allocation failure, RAM >16GB, correctness failure, TTFT regression beyond gate, or token-rate regression.

### 当前执行：post-odirect-onecache-capacity-probe-results

- `attempt_id`: `20260702-post-odirect-onecache-capacity-probe-results`
- `status`: rejected_no_token_rate_sota
- `time`: `2026-07-02T15:18Z-15:24Z`
- `common_config`: current O_DIRECT SOTA env/pack/profile, strict cold `drop_caches`, 16GB cgroup, France prompt, only `GGML_MOE_STREAM_ONE_CACHE_MIB` changed.
- `onecache14080_run`: `/root/lfz/runs/vendor-ds4-16gb/20260702T151826Z-20260702_odirect_onecache14080_probe/france-cpu40-vram0gb`, `eval_tok_s=1.9`, `prompt_tok_s=1.4`, `TTFT=30430.474958ms`, `memory_peak_bytes=16000000000`, `ram_ok=true`, `correctness_ok=true`. Rejected because `cudaMalloc 13.7 GiB FAILED`; VRAM cache disabled and pack reads increased to `35151` (`156648603648` bytes).
- `onecache13760_run`: `/root/lfz/runs/vendor-ds4-16gb/20260702T152039Z-20260702_odirect_onecache13760_probe/france-cpu40-vram0gb`, `eval_tok_s=4.0`, `prompt_tok_s=1.6`, `TTFT=28898.178863ms`, `memory_peak_bytes=16000000000`, `ram_ok=true`, `correctness_ok=true`. It allocated `13.4GiB`, `3237` slots, and reduced pack reads from `4623` to `4612`, but did not improve generation token rate.
- `onecache13888_run`: `/root/lfz/runs/vendor-ds4-16gb/20260702T152209Z-20260702_odirect_onecache13888_probe/france-cpu40-vram0gb`, `eval_tok_s=1.9`, `prompt_tok_s=1.4`, `TTFT=31156.852055ms`, `memory_peak_bytes=16000000000`, `ram_ok=true`, `correctness_ok=true`. Rejected because `cudaMalloc 13.6 GiB FAILED`; VRAM cache disabled and pack reads increased to `35151`.
- `interpretation`: The accepted `13568MiB` cache is near the practical allocation boundary. A small successful increase to `13760MiB` only eliminates 11 direct reads, far too little to move token rate. Larger requests fail allocation and catastrophically disable cache.
- `decision`: Keep accepted `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`. Add a future robustness item to avoid all-or-nothing cache disable on overlarge cache requests, but this is not a token-rate SOTA path.
- `next_design`: Remaining high-value work is CPU fallback/up-down compute. Since thread count and top-k config did not help, the next implementation must be a correctness-first diagnostic for a compute-path change, not another blind parameter sweep. Start by isolating one fallback route with CPU-vs-GPU compare on the exact accepted SOTA trace, then only benchmark after numerical equivalence is proven.

### 当前设计：mxfp4-down-batch-compare-redesign

- `attempt_id`: `20260702-mxfp4-down-batch-compare-redesign`
- `status`: active_design_before_practice
- `time`: `2026-07-02T15:30Z`
- `source_base`: `631846363779940a85490c47b4da829dcb8fcddc`, already pushed to `ssd/vendor/deepseek-token-rate-16gb`.
- `reason`: Post-O_DIRECT bottleneck trace shows CPU fallback dominates (`26851.367ms` total; decode up/down about `19302.589ms`). Thread count, global up/down top-k, and one-stream cache size probes did not improve token rate. The next high-value path is reducing up/down fallback, but previous MXFP4 down batch attempts produced incorrect output and must not be promoted blindly.
- `current_code_fact`: The accepted SOTA uses one-stream gate only via `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`; up/down remain CPU fallback. `ggml-cpu.c` says down batch is eligible for MXFP4 through `ggml_cuda_moe_stream_supports_down_batch()`, but `moe_stream_batch.cu` currently excludes `GGML_TYPE_MXFP4` from the batch implementation type table. Any MXFP4 batch test therefore requires a diagnostic-only source change and batch-on rebuild.
- `diagnostic_plan`: Add a default-off CPU-vs-GPU compare hook after a successful `ggml_cuda_moe_stream_batch()` result and before `matrix_row_counts` is cleared. Temporarily allow MXFP4 in the batch implementation so `GGML_MOE_STREAM_DOWN_BATCH=1` can exercise the rejected path. Run a short `-n 32` strict 16GB probe with `GGML_MOE_STREAM_COMPARE_CPU_OUT` and `GGML_MOE_STREAM_COMPARE_CPU_LIMIT=32`.
- `theoretical_bound`: If down batch became correct and fast, the hard upper bound is approximately the down fallback time from the post-O_DIRECT trace: decode down `9574.938ms` plus prompt down `4340.452ms`, or `13915.390ms` total. Actual gain is lower due to GPU staging, quantization, D2H/scatter, cache pressure, and synchronization.
- `acceptance_for_diagnostic`: Compare file must show small numerical error comparable to previous single-stream MXFP4 compare (`max_abs` around `1e-5` or lower) before any performance claim. Full France correctness must pass before any SOTA candidate is considered.
- `rollback_rule`: If the batch compare shows large error, incorrect France output, token-rate regression, RAM failure, or unclear mapping, record the exact failure and revert the temporary MXFP4 batch enablement before continuing. Do not push a branch where rejected MXFP4 batch support can be accidentally enabled, except as an explicitly documented default-off diagnostic commit after a clean guard.

### 当前执行：mxfp4-down-batch-dstrows-diagnostic-results

- `attempt_id`: `20260702-mxfp4-down-batch-dstrows-diagnostic-results`
- `status`: rejected_no_token_rate_sota_source_reverted`
- `time`: `2026-07-02T15:30Z-15:45Z`
- `diagnostic_source_change`: temporarily enabled MXFP4 down batch, added default-off batch CPU compare plus `GGML_MOE_STREAM_COMPARE_CPU_NAME_FILTER`, protected current batch slots during cache insert, and fixed compact down temporary dst allocation from `dst_cols` rows to `max(dst_cols,n_active)` rows.
- `pre_fix_compare`: `/root/lfz/runs/vendor-ds4-16gb/20260702T153340Z-20260702_mxfp4_down_batch_compare_namefilter/france-cpu40-vram0gb`, diag `/root/lfz/runs/vendor-ds4-16gb/20260702T154500Z-mxfp4-down-batch-compare-namefilter`. Result `eval_tok_s=1.6`, `correctness_ok=false`. Down compare showed huge error: `rows=32`, `max_abs=3.19428262`, `mean_of_mean_abs=0.341392792`, `max_rel=110.616382`.
- `root_cause`: compact down batch wrote `n_active` contiguous temporary rows (`tmp + j*ne01`) but allocated/copy-sized dst buffer by `dst_cols=max_dst_id+1`. With top-k pruning, `dst_cols` can be `4` while `n_active` can be `16`, so row `j>=dst_cols` overflowed and corrupted/reused output. This exactly matched the observed pattern: first four down experts correct, later experts wrong/repeated.
- `post_fix_compare`: `/root/lfz/runs/vendor-ds4-16gb/20260702T153943Z-20260702_mxfp4_down_batch_compare_dstrows/france-cpu40-vram0gb`, diag `/root/lfz/runs/vendor-ds4-16gb/20260702T155600Z-mxfp4-down-batch-compare-dstrows`. Result `eval_tok_s=2.3`, `prompt_tok_s=1.5`, `TTFT=30562.978413ms`, `memory_peak_bytes=16000000000`, `ram_ok=true`, `correctness_ok=true`. Down compare aligned: `rows=32`, `max_abs=1.1920929e-07`, `mean_of_mean_abs=5.27907673e-09`, `max_rel=8.37862445e-07`.
- `full_probe`: `/root/lfz/runs/vendor-ds4-16gb/20260702T154115Z-20260702_mxfp4_down_batch_dstrows_full_probe/france-cpu40-vram0gb`, no compare, full France generation. Result `eval_tok_s=2.6`, `prompt_tok_s=1.6`, `TTFT=29268.52773ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15054217216`, `ram_ok=true`, `correctness_ok=true`.
- `full_probe_answer`: France output is semantically about France and coherent enough for correctness, though repetitive and truncated near the end due the fixed token budget.
- `performance_profile`: fixed down batch accepted `7644` calls and declined `116`; down batch VRAM cache only had `34` slots after `cudaMalloc 0.5GiB FAILED` retry, with `hits=0`, `misses=24890`. `kimi_cpu_moe_profile` showed down total `3.057ms/call`, split into `cuda_batch=1.400ms`, `cuda_single=0.595ms`, `fallback_t0=1.043ms`.
- `interpretation`: The correctness bug in MXFP4 compact down batch is fixed by allocating enough temporary dst rows, but the path is still far slower than current O_DIRECT SOTA because down expert staging/cache misses dominate. This is not an accepted SOTA and must not remain enabled on the SOTA branch.
- `rollback_action`: Revert temporary source changes and rebuild the accepted SOTA path with `GGML_CUDA_MOE_STREAM_BATCH=OFF`. Keep only this plan record. Future work on down batch should first solve staging/cache locality, likely with a down expert pack or a larger/split VRAM allocation strategy that does not starve the existing gate cache.

### 当前执行：post-kimi-upgate-profile-merge-guard

- `attempt_id`: `20260702-post-kimi-upgate-profile-merge-guard`
- `status`: accepted_guard_pending_push
- `time`: `2026-07-02T15:48Z`
- `pre_merge_source`: `15e559953c409dadcfdb5724b92bd2d18adaf341` (`vendor-ds4: record mxfp4 down batch diagnosis`).
- `kimi_latest`: fetched and merged `ssd/vendor/kimi-moe-stream-on-vendor` at `f0d44910233b0bcf52050025a522313a78b93232`. Delta from previous Kimi head `f6175d8e5` includes Kimi plan updates and `ggml/src/ggml-cuda/moe_stream_batch.cu` profile-only up/gate type breakdown (`dd44205e7 cuda: profile kimi upgate type pairs`). This adds diagnostic profile aggregation and does not change the batch-off DeepSeek SOTA path.
- `merge_commit`: `87ddee861` (`vendor-ds4: merge kimi upgate profile update`).
- `build`: rebuilt `build-ds4-moe-stream` with `GGML_CUDA_MOE_STREAM_BATCH=OFF`.
- `guard_run`: `/root/lfz/runs/vendor-ds4-16gb/20260702T154804Z-20260702_post_kimi_upgate_profile_merge_odirect_sota_guard/france-cpu40-vram0gb`.
- `guard_config`: current O_DIRECT SOTA config: `cpu_moe=40`, `--vram-cache-gb 0`, extra args `-c 256 -b 16 -ub 16 -t 20 -tb 20`, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `GGML_MOE_STREAM_CACHE_ADMIT_PROFILE=.Agent/profiles/vendor-ds4/current_sota_gate_freq_ge2.tsv`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, `GGML_MOE_STREAM_ONE_EXPERT_PACK=/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-gate-miss-firstorder-20260702.pack`, `GGML_MOE_STREAM_ONE_EXPERT_PACK_IO=direct`, strict cold `drop_caches`, 16GB cgroup.
- `guard_result`: `eval_tok_s=4.1`, `prompt_tok_s=1.6`, `TTFT=29056.139513ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15087562752`, `pgmajfault=266989`, `workingset_refault_file=1651369`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `guard_answer`: France output is semantic and coherent: it identifies France/French Republic in Western Europe, mentions history/culture/global influence, Eiffel Tower/Louvre/Versailles, cuisine/wine/fashion/art/science, EU membership, economy, and historical/modern vitality.
- `guard_counters`: one expert pack `hits=4623 misses=0 reads=4623 bytes=20602159104 failures=0 entries=4599 direct_enabled=1 direct_reads=4623 direct_failures=0 direct_fallbacks=0`; VRAM cache `hits=30528 misses=4623 hit_rate=86.8%`.
- `decision`: Kimi latest merge is safe for DeepSeek SOTA. Push the merge and this record to `ssd/vendor/deepseek-token-rate-16gb`, then continue with down expert staging/cache feasibility.

### 当前设计：down-top2048-pack-feasibility-and-probe

- `attempt_id`: `20260702-down-top2048-pack-feasibility-and-probe`
- `status`: active_design_before_practice
- `time`: `2026-07-02T16:00Z`
- `source_base`: `1e9b327d57e4f75be101f1b88cf064df7d39a658`, already pushed to `ssd/vendor/deepseek-token-rate-16gb`.
- `reason`: MXFP4 down batch is now numerically understood: compact dst rows must be allocated with `max(dst_cols,n_active)`. The remaining rejection reason is performance: down batch staging/cache had only `34` slots and `0%` down cache hit rate, so expert movement dominates. Before implementing a persistent source change, measure whether a compact down expert pack can reduce staging I/O enough to matter.
- `profile_source`: `/root/lfz/runs/vendor-ds4-16gb/20260702T151000Z-post-odirect-bottleneck-trace/fallback-profile.csv`, the full post-O_DIRECT SOTA CPU fallback profile.
- `fallback_inventory`: down fallback has `2974` unique `(tensor,expert)` pairs, `19111` calls, `13915.390ms` fallback time, and `12.343GiB` unique payload. Up fallback is similarly `2974` pairs, `19111` calls, `12935.977ms`, `12.343GiB`. A full up+down resident set is about `24.7GiB`, so it cannot simply fit in available VRAM/RAM alongside the accepted gate cache.
- `top2048_pack`: generated from the hottest down fallback rows only: `/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-down-top2048-20260702.pack`.
- `top2048_pack_input`: `/root/lfz/runs/vendor-ds4-16gb/20260702T160000Z-down-pack-feasibility/down_top2048_fake_trace.csv`.
- `top2048_pack_contents`: selected top `2048` fallback rows, deduped by pack builder to `1927` unique down experts, payload `8587575296` bytes (`7.998GiB`). The selected rows cover `17644` down calls and `11214.413ms` aggregate down fallback time in the profile.
- `read_sequence`: `/root/lfz/runs/vendor-ds4-16gb/20260702T160000Z-down-pack-feasibility/down_top2048_pack_reads.tsv`, `17644` reads, `78629568512` bytes (`73.229GiB`) when repeated according to fallback call counts.
- `readbench`: isolated 16GB `MemoryMax` readbench with cold `drop_caches`: direct `25.283121s`, `2.896GiB/s`; buffered `7.684598s`, `9.529GiB/s`. Buffered is faster because the compact `~8GiB` pack working set can fit in the 16GB cgroup page cache, while O_DIRECT rereads every repeated expert from storage.
- `theoretical_bound`: Replacing random GGUF fallback/staging with a buffered compact down pack could at most remove part of the selected `11214ms` down fallback time. The readbench lower bound for repeated pack reads is `~7.7s`, so a large net speedup is not guaranteed; the realistic benefit depends on overlap with GPU work, page-cache competition with model mappings, and whether batch cache/staging avoids repeated H2D.
- `next_practice`: Temporarily re-enable the correctness-fixed MXFP4 down batch path (`GGML_TYPE_MXFP4` support and `dst_tmp_rows=max(dst_cols,n_active)`) and run a full strict 16GB France probe with `GGML_MOE_STREAM_DOWN_BATCH=1`, `GGML_MOE_EXPERT_PACK=<down_top2048_pack>`, and buffered pack I/O. Keep the accepted gate O_DIRECT pack config unchanged.
- `acceptance`: Promote only if `eval_tok_s > 4.2`, RAM including page cache remains `<=16000000000`, France output is semantic/coherent, TTFT stays within the accepted gate, pack counters show meaningful down-pack hits, and a rerun from pushed commit reproduces. If it is slower or correctness/RAM fails, record rejected and revert source.

### 当前执行：down-top2048-pack-buffered-full-probe-results

- `attempt_id`: `20260702-down-top2048-pack-buffered-full-probe-results`
- `status`: rejected_no_token_rate_sota_source_reverted_pack_deleted`
- `time`: `2026-07-02T16:01Z-16:06Z`
- `temporary_source_change`: MXFP4 enabled in `moe_stream_batch.cu` down batch allowlist and `dst_tmp_rows=max(dst_cols,n_active)` compact dst fix. Source was reverted after the rejected run and build restored to `GGML_CUDA_MOE_STREAM_BATCH=OFF`.
- `pack_sha256`: deleted rejected pack after recording sha to restore disk: `f64682efb263de46d6eea7389494552305012b403932cf7da77767fa1164d539  ds4-france-down-top2048-20260702.pack`.
- `repro_inputs`: fake trace sha256 `1467bf050ee9d1cba99067b72dfb42054785c8a42d221565324a9b22459076d7`; read sequence sha256 `e5bb9b17279e5d49a6ba224111c13c96152960c74feb5cc00aac292df300b63c`.
- `full_probe_run`: `/root/lfz/runs/vendor-ds4-16gb/20260702T160118Z-20260702_down_top2048_pack_buffered_full_probe/france-cpu40-vram0gb`.
- `full_probe_config`: accepted gate O_DIRECT SOTA env plus temporary `GGML_MOE_STREAM_DOWN_BATCH=1`, `GGML_MOE_VRAM_CACHE_MIB=512`, `GGML_MOE_EXPERT_PACK=/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-down-top2048-20260702.pack`, `GGML_MOE_IO_BACKEND=buffered`, strict cold 16GB cgroup.
- `full_probe_result`: `eval_tok_s=2.3`, `prompt_tok_s=1.5`, `TTFT=31313.021254ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15010816000`, `pgmajfault=202310`, `workingset_refault_file=8747095`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `answer`: France output stayed semantic/coherent but was repetitive and truncated at the token budget.
- `pack_counters`: down expert pack loaded `1927` entries and reported `hits=22911 misses=1979 read_failures=0`; down batch VRAM cache still had only `34` slots with `hits=0 misses=24890`; accepted gate one-pack counters changed to `hits=5062 misses=1235` because the batch-on path changed graph/cache behavior during this rejected probe.
- `profile`: down batch accepted `7644` calls, declined `116`; batch stage `6.015ms/call`, kernel `0.031ms/call`, d2h `0.005ms/call`; `kimi_cpu_moe_profile` down total `3.526ms/call`, `cuda_batch=1.994ms`, `cuda_single=0.577ms`, `fallback_t0=0.931ms`.
- `gap_analysis`: The isolated buffered readbench looked favorable because the `~8GiB` pack fits in a 16GB page cache. In full inference, the same buffered pack competes with model mmap/page cache and gate O_DIRECT staging under the strict 16GB cgroup, increasing `workingset_refault_file` to `8.7M` and making staging slower than the no-pack down-batch probe. This path is rejected.
- `next_design`: A useful down compute path needs either real VRAM residency/split allocation for hot down experts, or a smaller/layer-targeted pack that does not consume enough page cache to destabilize the model. Do not pursue large buffered down packs under the current 16GB cold-start constraint.

### 当前执行：20260703-current-sota-strict-reproduction

- `attempt_id`: `20260703-current-sota-strict-reproduction`
- `status`: reproduced_requirements_but_not_4p2
- `purpose`: User requested another reproduction of the current vendor DeepSeek SOTA and validation against all hard gates: strict 16GB host RAM including page cache, cold `drop_caches`, France correctness, TTFT gate, and current O_DIRECT expert-pack config.
- `source_head`: `9900d191bfbf31d97e21c816f3099246abecb537`, pushed to `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`.
- `binary_build`: stdout reports `build : b14557-1e9b327d5`. The diff from `1e9b327d57e4f75be101f1b88cf064df7d39a658` to `9900d191bfbf31d97e21c816f3099246abecb537` is plan-only (`.Agent/plans/20260701-vendor-ds4-16gb-token-rate-plan.md`), so the runtime source path is unchanged.
- `binary_sha256`: `c70c4f28f972fb7d1b443076961a653d7d05e9d472effb253dcd23311c843f62`.
- `pack_sha256`: `7ad26d8b14c20dccd4106a8abbffc9f846eb2fedff4fd00a5af7060941204076` for `/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-gate-miss-firstorder-20260702.pack`.
- `profile_sha256`: `8134c320730e0ba236d103ba4a0b53505b3bab16e69d8bdc2a08607ecfcc274b` for `.Agent/profiles/vendor-ds4/current_sota_gate_freq_ge2.tsv`.
- `config`: `cpu_moe=40`, `--vram-cache-gb 0`, `GGML_MOE_KEEP_TOPK_UPDOWN=4`, `GGML_MOE_KEEP_TOPK_LAYER_RANGE=10-39`, `GGML_MOE_KEEP_TOPK_LAYER_VALUE=3`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, `GGML_MOE_STREAM_ONE_EXPERT_PACK_IO=direct`, strict cold `drop_caches`, 16GB `systemd-run` cgroup with `MemorySwapMax=0`, CLI extra args `-c 256 -b 16 -ub 16 -t 20 -tb 20`.
- `run_1`: `/root/lfz/runs/vendor-ds4-16gb/20260703T031923Z-20260703T031710Z-current-sota-strict-repro/france-cpu40-vram0gb`.
- `run_1_result`: `eval_tok_s=4.1`, `prompt_tok_s=1.5`, `TTFT=30470.663075ms`, `elapsed_seconds=63.56`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15102509056`, `pgmajfault=275132`, `workingset_refault_file=1656831`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `run_2`: `/root/lfz/runs/vendor-ds4-16gb/20260703T032154Z-20260703T032138Z-current-sota-strict-repro-retry/france-cpu40-vram0gb`.
- `run_2_result`: `eval_tok_s=4.1`, `prompt_tok_s=1.6`, `TTFT=28857.563383ms`, `elapsed_seconds=62.29`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15102316544`, `pgmajfault=264978`, `workingset_refault_file=1688932`, `ram_ok=true`, `ram_limit_killed=false`, `correctness_ok=true`.
- `cgroup_evidence`: both runs reached `memory.peak=16000000000`; `memory.stat file` was about `15.10GB`, so page cache was inside the same 16GB cgroup. `memory.events` had `oom=0`, `oom_kill=0`, and `oom_group_kill=0`; only `max` events occurred from touching the hard cgroup boundary.
- `counters`: both runs reported one expert pack `hits=4623 misses=0 reads=4623 bytes=20602159104 failures=0 entries=4599 direct_enabled=1 direct_reads=4623 direct_failures=0 direct_fallbacks=0`; VRAM cache `hits=30528 misses=4623 hit_rate=86.8%` and `3192` slots (`13.2GiB`).
- `answer`: both runs produced the same semantic and coherent France paragraph: France/French Republic in Western Europe, history/culture/global influence, landmarks including Eiffel Tower/Louvre/Versailles, cuisine/wine/fashion/art/science, EU membership, economy, and modern vitality.
- `decision`: These two fresh strict cold reproductions pass RAM, page-cache accounting, TTFT, O_DIRECT pack, VRAM cache, and correctness requirements, but they reproduce `4.1 tok/s`, not the historical single-run `4.2 tok/s`. Treat `4.2 tok/s` as historical highest observed/current accepted record only when citing its original run; for immediately repeatable evidence from this check, use `4.1 tok/s` as the reproduced strict-cold line.

### 当前计划更新：20260705-sota44-4expert-q4k-coldstart

- `status`: active_plan_disk_blocked_for_full_4expert_validation
- `time`: `2026-07-05T14:05Z`
- `repo`: `/root/lfz/vendor/llama.cpp-deepseek-v4`
- `working_branch`: `feat/ds4-moe-stream-on-vendor`
- `push_target`: `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`
- `required_git_identity`: `L-Ark <fliangae@connect.ust.hk>`
- `source_head_before_this_plan_update`: `e3779ed5a` (`vendor-ds4: reject sparse 4expert validation`), already pushed to `ssd/vendor/deepseek-token-rate-16gb`.

#### 当前有效 SOTA 和硬性门槛

- `accepted_sota`: `eval_tok_s=4.4`, `prompt_tok_s=1.8`, strict cold, vendor framework, 16GB cgroup including page cache, France correctness pass.
- `accepted_sota_guard_run`: `/root/lfz/runs/vendor-ds4-16gb/20260705T070310Z-20260705_current_head_sota44_no_trace_after_sparse_close/france-current-head-sota44-no-trace-cpu40-vram0gb`.
- `accepted_sota_guard_metrics`: `TTFT=32087.738292ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15099523072`, `ram_ok=true`, `oom_seen=false`, `correctness_ok=true`.
- `pushed_source_repro_run`: `/root/lfz/runs/vendor-ds4-16gb/20260703T220820Z-20260704_gate_prefill_top3000_pushed_repro/france-cpu40-vram0gb`.
- `accepted_model`: `/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf`, `156148189760 bytes` (`145.42 GiB`).
- `accepted_gate_pack`: `/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-france-gate-miss-firstorder-20260702.pack`, sha-sensitive SOTA asset, `20495904768 bytes`.
- `accepted_config`: `cpu_moe=40`, `--vram-cache-gb 0`, `-c 256 -b 16 -ub 16 -t 20 -tb 20`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`, `GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`, `GGML_MOE_STREAM_ONE_EXPERT_PACK_IO=direct`, `GGML_MOE_STREAM_ONE_PREFILL_LIMIT=3000`, `GGML_CUDA_DISABLE_GRAPHS=1`, strict `drop_caches`, `MemoryMax=16000000000`, `MemorySwapMax=0`.
- `ttft_promotion_gate`: `<=33617.688744ms`. Runs above this can be committed only as `not_accepted` diagnostics and must not replace the accepted SOTA until TTFT is brought back under the gate.
- `new_sota_rule`: A new accepted SOTA must beat `4.4 tok/s`, pass France semantic/coherent correctness, pass strict 16GB host RAM including page cache, have no OOM/swap, keep TTFT within the gate, record all metrics and exact reproduction inputs, then immediately commit and push source plus records to `ssd/vendor/deepseek-token-rate-16gb`.
- `repro_rule`: After any SOTA promotion, rerun from the pushed commit and record the pushed-source reproduction path, binary/model/pack/profile hashes, cgroup memory evidence, exact env/CLI, answer text, and counters. Future rollback must be able to reproduce the metric from the committed record.

#### 2026-07-05 已完成但未提升 SOTA 的工作

- `q4k_microprobe`: Fixed the standalone Q4_K dot harness by calling `ggml_cpu_init()`. Valid microprobe reaches `116.115 GiB/s` up/gate-like and `115.044 GiB/s` down-like at 32 threads, above the `91.978 GiB/s` hard reopen gate. This is only a microprobe, not a model SOTA. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/q4k-top4-combo-microprobe-result.json`.
- `q4k_stream_one_admission`: Added default-off env `GGML_MOE_STREAM_ONE_Q4K=1` so Q4_K is eligible only for stream-one admission and only when the env/name filter allow it. Generic stream support, down-batch, and up-gate batch behavior were not broadened. Default-off guard passed on repeat and did not change SOTA. Commit: `666713548`; artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/4expert-q4k-defaultoff-admission-validation.json`.
- `deepseek4_tid2eid_alias`: Added default-off env `LLAMA_DEEPSEEK4_TID2EID_WEIGHT_ALIAS=1` to try `blk.%d.ffn_gate_tid2eid.weight` only if the original DeepSeek4 `tid2eid` name is absent. Default-off guard passed and did not change SOTA. Commit: `be6cc73ea`; artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/4expert-tid2eid-alias-defaultoff-validation.json`.
- `4expert_header_probe`: Downloaded only the first `64MiB` of cloudyu 4Expert GGUF for metadata. Header shows `architecture=deepseek4`, `block_count=43`, `expert_count=256`, `expert_used_count=4`, type counts include `Q4_K=129`, and `tid2eid` tensors use `.weight` names. Full model was not downloaded. Artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/4expert-header-alias-probe-and-plan.json`.
- `4expert_manifest_tool`: Added `.Agent/run-tools/create_gguf_header_expert_manifest.py` and produced a header-derived direct manifest with `33024` expert rows, `129` Q4_K expert tensors, per expert slice size `4718592 bytes`, and payload `155826782208 bytes`. Commit: `ccd385c95`; artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/4expert-q4k-header-manifest-summary.json`.
- `sparse_validation_rejection`: Rejected sparse/header-only 4Expert validation. A sparse logical `164.5GB` file made from the header would map missing tensors as zeros, so correctness, TTFT, token rate, and page-cache behavior would all be invalid. Commit: `e3779ed5a`; artifact: `.Agent/runs/20260705-vendor-ds4-coldstart/4expert-sparse-alias-validation-rejection.json`.

#### 当前瓶颈判断

- Current native SOTA still spends most of cold-start time on model page faults, gate expert movement/cache, and CPU fallback for non-cached expert work. Prior down-pack and down-batch attempts either regressed token rate or destabilized page-cache behavior under the 16GB cgroup.
- CUDA graph is disabled in the accepted path (`GGML_CUDA_DISABLE_GRAPHS=1`). A graph experiment has not been promoted; it must be tested only behind the same correctness/RAM/TTFT gates and must not overwrite the SOTA unless it beats `4.4 tok/s`.
- The 4Expert/Q4_K route is the current highest-priority route because the header and Q4_K microprobe show a plausible way to reduce CPU/expert movement cost, but no correctness or token-rate claim is valid until the complete real GGUF is available.

#### 当前阻塞：完整 4Expert GGUF 需要磁盘空间

- `root_fs`: `/dev/root`, ext4, `993G` total, `991G` used, `1.8G` available, `100%`.
- `candidate_4expert_url`: `https://huggingface.co/cloudyu/DeepSeek-V4-Flash-4Expert-GGUF/resolve/main/ds4flash-4expert.gguf`.
- `candidate_4expert_expected_size_bytes`: `164465760544`.
- `minimum_practical_free_space`: `>=180G` before attempting full download or real mmap/load validation.
- `protected_assets_do_not_delete`: accepted native GGUF, accepted SOTA gate pack, accepted SOTA run directories, pushed-source reproduction run directories.
- `space_candidates_requiring_explicit_approval`: `/root/lfz/models/GLM-5.2-UD-IQ3_XXS` (`263G`, unrelated to current vendor DS4 SOTA), `/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.expert-pack` (`147174760448 bytes`, not used by accepted env but adjacent to DeepSeek assets and high risk), `/root/lfz/runs/vendor-ds4-16gb/expert-packs/ds4-promptset-gate-union-firstorder-20260702.pack` (`38420348928 bytes`, prompt-set diagnostic pack), `/root/yibins/ssd16-cache` (`128G`, owner/use unknown), `/opt/models` (`60G`, owner/use unknown), `/tmp` (`15G`, mostly build/test leftovers).
- `deletion_rule`: No deletion or destructive cleanup has been performed for this plan. Free-space actions require explicit user approval for exact paths, followed by a fresh accepted-SOTA guard if any adjacent DeepSeek asset is touched.

#### 下一步执行计划

- `P0_record_and_push_plan`: Commit this plan update and any disk-blocker artifact, then push to `ssd/vendor/deepseek-token-rate-16gb` with identity `L-Ark`.
- `P1_release_or_attach_storage`: Obtain at least `180G` real free space. Preferred low-risk route is explicit approval to remove unrelated large artifacts or attach/use another filesystem. Do not use sparse files as a substitute for the real GGUF.
- `P2_full_4expert_acquisition`: Download the complete 4Expert GGUF, record path, size, sha256, source URL, and free-space state before/after. Abort if file size/hash is incomplete.
- `P3_load_validation_before_benchmark`: Run metadata/load validation with `LLAMA_DEEPSEEK4_TID2EID_WEIGHT_ALIAS=1` and `GGML_MOE_STREAM_ONE_Q4K=1`. Confirm the alias actually resolves `ffn_gate_tid2eid.weight`, Q4_K expert tensors load, no zero/sparse tensors are used, and RAM accounting remains meaningful.
- `P4_correctness_smoke`: Before claiming performance, run short strict-cgroup correctness checks including the France prompt. The France answer must be semantic and coherent. If output is wrong, stop and debug routing/quantization before timing work.
- `P5_strict_cold_benchmark`: Run the exact cold-start benchmark under `MemoryMax=16000000000`, `MemorySwapMax=0`, `drop_caches`, page-cache accounting, no OOM/swap, and the same TTFT gate. Record token rate, prompt rate, TTFT, elapsed time, memory.peak, memory.stat file, faults/refaults, counters, answer, env, CLI, commit, binary hash, model hash, and pack/profile hashes.
- `P6_promote_or_reject`: If `eval_tok_s > 4.4` and all gates pass, immediately commit and push source/records, then rerun from the pushed commit and record pushed-source reproduction. If token rate regresses, correctness fails, RAM exceeds 16GB, TTFT exceeds the accepted gate, or evidence is incomplete, mark the run rejected and keep `4.4 tok/s` as accepted SOTA.
- `P7_fallback_if_4expert_underperforms_or_stays_blocked`: Return to native SOTA bottleneck work only after recording the 4Expert blocker. The next native work should focus on measured CPU fallback/page-refault cost and must avoid previously rejected large buffered down-pack or accidental batch enablement paths unless a new hard-bound analysis shows a clear ceiling above `4.4 tok/s`.


## 2026-07-06 执行记录：down MXFP4 perf probe rejected

- `attempt_id`: `20260706-down-mxfp4-perf-france-probe`
- `status`: `rejected_not_sota`
- `artifact`: `.Agent/runs/20260705-vendor-ds4-coldstart/down-mxfp4-perf-france-probe-rejection-20260706.json`
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260706T120136Z-down-mxfp4-perf-france-probe-20260706/france-cpu40-vram0gb`
- `prompt_scope`: calibration/dev France only. `held_out_test_set_v1_locked` was not used.
- `config`: no prompt-specific pack/profile; strict cold `drop_caches`; `MemoryMax=16000000000`; `MemorySwapMax=0`; `GGML_MOE_STREAM_DOWN_MXFP4_PROBE=perf`; `GGML_MOE_STREAM_DOWN_BATCH=1`; conservative VRAM split with gate one-stream cache `8192MiB` and batch/down cache `512MiB`.
- `metrics`: `eval_tok_s=1.7`, `prompt_tok_s=0.9`, `TTFT=38553.15988ms`, `elapsed_seconds=127.07`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15004889088`, `pgmajfault=156746`, `workingset_refault_file=8071289`, `ram_ok=true`, `oom_seen=false`, `correctness_ok=true`.
- `local_success`: down batch was accepted and down CPU fallback disappeared from `fallback_reason_profile`; remaining fallback rows were only `up`. Down profile printed `batch_accept=6320`, `batch_decline=0`.
- `remaining_bottleneck`: up fallback still dominated (`up decode=19954.796ms`, `up prompt=4304.180ms`). Down batch also added staging/cache cost and required reducing gate cache headroom, so local down fallback removal did not translate into end-to-end gain.
- `decision`: reject. This is slower than the no-prompt-specific calibration France baseline/profile (`~2.6-2.7 tok/s`) and cannot be promoted. Do not run held-out or claim generalized SOTA from this path.
- `next_design`: prioritize prompt-general `ffn_up_exps` fallback reduction, or redesign down/offload only if it avoids sacrificing gate cache and has a hard-bound above the generalized baseline. Any future candidate must first improve the calibration/dev aggregate, then freeze before held-out testing.


## 2026-07-06 执行记录：up one-stream comma filter probe rejected

- `attempt_id`: `20260706-up-one-stream-comma-filter-n64`
- `status`: `rejected_not_sota`
- `artifact`: `.Agent/runs/20260705-vendor-ds4-coldstart/up-one-stream-comma-filter-n64-rejection-20260706.json`
- `source_change`: added default-compatible comma/colon matching for `GGML_MOE_STREAM_ONE_NAME_FILTER` in CPU/CUDA one-stream checks, plus default-off `GGML_MOE_STREAM_ONE_GPU_ONLY_FILTER` in CPU `mul_mat_id` to abort if a matched tensor silently falls back to CPU.
- `default_safety`: when `GGML_MOE_STREAM_ONE_GPU_ONLY_FILTER` is unset and the name filter is a single substring, behavior remains equivalent to the previous path. This is a diagnostic/config capability, not an accepted optimization.
- `candidate_run`: `/root/lfz/runs/vendor-ds4-16gb/20260706T121537Z-20260706T-up-one-stream-comma-filter-n64-dev-smoke/france-up-one-stream-comma-cpu40-vram0gb`.
- `candidate_config`: calibration France only, no held-out, no prompt-specific pack/profile, `n=64`, strict 16GB cgroup, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps,ffn_up_exps`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`.
- `candidate_metrics`: `eval_tok_s=1.9`, `prompt_tok_s=0.8`, `TTFT=38614.208237ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15063433216`, `ram_ok=true`, `oom_seen=false`, heuristic France correctness `true` despite short output.
- `candidate_fallback`: up fallback was removed from fallback profile; remaining fallback was down only (`down decode=8205.193ms`, `down prompt=4904.790ms`). VRAM cache reported `hits=19629`, `misses=6884`, `hit_rate=74.0%`.
- `control_run`: `/root/lfz/runs/vendor-ds4-16gb/20260706T121752Z-20260706T-gate-only-n64-dev-control-after-filter-patch/france-gate-only-n64-control-cpu40-vram0gb` with gate-only filter under the same `n=64` constraints.
- `control_metrics`: `eval_tok_s=2.2`, `prompt_tok_s=0.9`, `TTFT=37992.446775ms`, `memory_peak_bytes=16000000000`, `ram_ok=true`; correctness false only because `n=64` truncates the answer.
- `control_fallback`: gate-only retained `up decode=7469.804ms`, `up prompt=3772.828ms`, `down decode=5037.867ms`, `down prompt=4480.035ms`; VRAM cache `hits=13435`, `misses=3716`, `hit_rate=78.3%`.
- `decision`: reject naive `gate+up` one-stream shared-cache path. It proves `ffn_up_exps` can be routed through one-stream, but end-to-end speed regresses from `2.2` to `1.9 tok/s` on the matched `n=64` control. The local up fallback win is outweighed by cache/staging overhead and remaining down fallback.
- `next_design`: a valid up/down fallback fix must avoid stealing enough gate cache residency to create more misses. Prioritize either a separate hard-bounded up/offload path with better residency policy, or a combined design that reduces total expert movement instead of simply adding up experts to the existing gate cache. Continue using only calibration/dev prompts until a candidate is frozen.


## 2026-07-06 执行记录：up stream with gate-only cache rejected

- `attempt_id`: `20260706-up-stream-gate-cache-only-n64`
- `status`: `rejected_not_sota`
- `artifact`: `.Agent/runs/20260705-vendor-ds4-coldstart/up-stream-gate-cache-only-n64-rejection-20260706.json`
- `source_change`: added default-off `GGML_MOE_STREAM_CACHE_ADMIT_NAME_FILTER` in `moe_stream.cu`. When unset, cache admission behavior is unchanged. When set, one-stream GPU execution can still run for allowed tensors, but VRAM cache insertion is limited to matching tensor names.
- `theoretical_test`: If the previous `gate+up` regression was mainly from up evicting gate experts, then allowing `ffn_up_exps` to run through GPU while only admitting `ffn_gate_exps` into cache should have recovered some of the `~11.24s` n64 up fallback from the gate-only control without destroying gate residency.
- `candidate_run`: `/root/lfz/runs/vendor-ds4-16gb/20260706T122248Z-20260706T-up-stream-gate-cache-only-n64-dev-probe/france-up-stream-gate-cache-only-cpu40-vram0gb`.
- `candidate_config`: calibration France only, no held-out, no prompt-specific pack/profile, strict 16GB cgroup, `n=64`, `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps,ffn_up_exps`, `GGML_MOE_STREAM_CACHE_ADMIT_NAME_FILTER=ffn_gate_exps`, `GGML_MOE_STREAM_ONE_CACHE_MIB=13568`.
- `candidate_metrics`: `eval_tok_s=1.7`, `prompt_tok_s=0.8`, `TTFT=39585.19629ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15086202880`, `ram_ok=true`, `oom_seen=false`, heuristic France correctness `true` despite short output.
- `candidate_fallback`: up fallback was again removed; remaining fallback was down only (`down decode=8142.154ms`, `down prompt=5160.372ms`). VRAM cache reported `hits=13392`, `misses=13121`, `hit_rate=50.5%` because uncached up routes still count as misses and pay repeated transfer/staging.
- `matched_controls`: gate-only `n64` control was `2.2 tok/s`; `gate+up` shared-cache probe was `1.9 tok/s`; gate-cache-only up stream was worse at `1.7 tok/s`.
- `decision`: reject. The bottleneck is not just up evicting gate cache; naive up GPU streaming has too much repeated expert movement/synchronization cost and leaves down fallback untouched.
- `next_design`: stop simple up one-stream/admission sweeps. The next credible path needs a hard-bound design that reduces total expert movement, e.g. prompt-general persistent residency, combined up/down scheduling with bounded VRAM footprint, or a kernel path that reuses already-staged expert data across projections. Continue using calibration/dev prompts only until a candidate is frozen.


## 2026-07-06 下一阶段计划：up/down 总搬运 hard-bound

- `plan_id`: `20260706-updown-total-movement-hard-bound`
- `status`: `next_active_plan`
- `push_target`: `https://github.com/wici-ai/ssd-llama.git` branch `vendor/deepseek-token-rate-16gb`
- `why_new_plan`: 最近三组实验证明，单点把 `ffn_up_exps` 或 `ffn_down_exps` 搬到现有 GPU stream 路径并不能提升泛化 token rate。down MXFP4 batch 能消掉 down fallback 但端到端 `1.7 tok/s`；gate+up shared cache 消掉 up fallback 但 `n64` 从 gate-only `2.2` 掉到 `1.9`；gate-only cache + uncached up stream 进一步掉到 `1.7`。问题不是单个 eligibility guard，而是总 expert movement/cache/staging 成本超过 CPU fallback savings。
- `closed_paths_now`:
  - naive `ffn_up_exps` one-stream with shared gate cache: rejected;
  - naive `ffn_up_exps` one-stream with `ffn_gate_exps`-only cache admission: rejected;
  - down MXFP4 batch with small separate cache: rejected;
  - promptset union gate pack: historical best non-France only about `2.8 tok/s`, France regresses to `3.1-3.2`, not enough for generalized `>5`;
  - extra full MoE GPU layers via lower `cpu_moe`: rejected by existing hard-bound and historical `cpu_moe=39/38` regressions.
- `next_measurement_required`: Build a prompt-general up/down route inventory from `calibration_dev_set_v1` only, not held-out. For each prompt and aggregate, record unique `(tensor,expert)` for up/down, call counts, rows, fallback ms, expert bytes, repeated bytes, and overlap between gate/up/down. This must estimate how much data must move if we try to stream, cache, pack, or persist up/down experts.
- `hard_bound_questions`:
  1. What is the minimum unique up/down payload needed to cover 50/70/90 percent of fallback time across the calibration set?
  2. How much of that payload can fit in available VRAM after reserving enough gate cache to avoid the observed miss cliff?
  3. If not resident, what is the repeated H2D/direct-read lower bound per output token, and is it mathematically compatible with `>5 tok/s`?
  4. Is there meaningful overlap between gate and up/down experts that permits a combined pack/read to reuse one disk read or one pinned staging buffer?
  5. Can a calibration-derived artifact improve dev set min/mean without using held-out prompts?
- `implementation_candidates_after_bound`:
  - `candidate_A`: prompt-general small persistent up/down hotset, capped by VRAM after gate reservation. Only attempt if the bound shows hotset payload is small enough and covers enough fallback time.
  - `candidate_B`: combined gate/up/down pack/read for overlapping experts, but only if overlap is high and pack size/page-cache behavior remains valid under 16GB cgroup.
  - `candidate_C`: grouped up/down compute with a staging reuse window that batches multiple active rows per tensor without inserting every up/down expert into the long-lived gate cache.
  - `candidate_D`: alternate GGUF / smaller expert model route remains blocked by disk unless explicit cleanup/storage approval is available.
- `acceptance_rule`: Before held-out testing, a candidate must improve `calibration_dev_set_v1` min/mean token rate versus no-prompt-specific baseline (`mean=2.18`, `min=1.8`) without correctness regression and with all runs inside 16GB including page cache. After candidate freeze, run `held_out_test_set_v1_locked`; only held-out metrics can be claimed as generalized SOTA.
- `next_action`: produce the calibration up/down movement bound artifact from existing `dev-fallback-profile-no-prompt-specific-20260706` data if sufficient; otherwise run only the missing calibration profiles. Do not run held-out or build another large pack before the bound proves a plausible route above `5 tok/s`.


## 2026-07-06 执行记录：up/down 总搬运 hard-bound

- `attempt_id`: `20260706-updown-total-movement-hard-bound`
- `status`: `completed_measurement_not_sota`
- `artifact`: `.Agent/runs/20260705-vendor-ds4-coldstart/updown-total-movement-hard-bound-20260706.json`
- `source`: existing `calibration_dev_set_v1` fallback profiles from `.Agent/runs/20260705-vendor-ds4-coldstart/dev-fallback-profile-no-prompt-specific-20260706.json`; no held-out prompt was used.
- `aggregate_updown`: `13366` unique `(role,tensor,expert)` pairs, `55.474GiB` unique payload, `946.372GiB` repeated logical expert bytes, `216535.305ms` fallback across the five calibration prompts.
- `by_role`: up has `6683` unique pairs, `27.737GiB` unique payload, `473.186GiB` repeated bytes, `123474.033ms` fallback; down has the same unique/repeated byte footprint and `93061.272ms` fallback.
- `coverage_50`: top fallback rows covering `50.01%` of fallback still require `1857` unique pairs, `7.707GiB` unique payload, and account for `482.022GiB` repeated logical bytes.
- `coverage_70`: `3587` unique pairs, `14.887GiB` unique payload, `657.937GiB` repeated logical bytes.
- `coverage_90`: `6881` unique pairs, `28.559GiB` unique payload, `836.885GiB` repeated logical bytes.
- `transfer_lower_bound`: if all repeated up/down bytes were moved over H2D every time, the pure transfer lower bound is `59.15s @16GiB/s`, `29.57s @32GiB/s`, `19.72s @48GiB/s`, or `14.79s @64GiB/s`, before kernel, synchronization, D2H/scatter, page faults, and cache bookkeeping.
- `overlap_up_down`: all observed up layer/expert pairs also appear in down (`6683/6683`). This supports investigating paired up/down scheduling, but does not by itself solve payload size or transfer cost.
- `missing_gate_overlap`: the source dev fallback runs did not include gate per-expert trace, so this artifact cannot claim gate/up/down overlap. A calibration gate trace is justified only if the next design needs combined gate/up/down pack/read evidence.
- `decision`: simple persistent up/down hotset is not a credible immediate path unless capped to a very small payload and proven to preserve gate cache. Even 50% fallback coverage needs `7.7GiB`, which would steal too much of the `13.2GiB` observed gate cache or exceed available 32GB VRAM headroom; prior reduced-gate-cache probes already regressed.
- `closed_by_bound`: do not run another naive up/down one-stream, small down-batch cache, or broad up/down resident-hotset sweep without a new mechanism that reduces total movement. The data explains why the previous probes regressed: they removed CPU fallback locally but replaced it with repeated expert movement and cache pressure.
- `next_design_choice`: focus on `candidate_C grouped staging / paired up-down scheduling` before source implementation. The design must show how it reduces repeated movement versus `946GiB` logical bytes, how much temporary workspace it needs, and why it will not evict gate cache. If combined gate/up/down pack is considered, first run a calibration-only gate trace to measure overlap; do not use held-out.


## 2026-07-06 设计：hotset-gated up/down stream

- `attempt_id`: `20260706-hotset-gated-updown-stream-design`
- `status`: `ready_for_default_off_implementation`
- `artifact`: `.Agent/runs/20260705-vendor-ds4-coldstart/hotset-gated-stream-design-20260706.json`
- `reason`: full up/down movement is too large (`55.474GiB` unique, `946.372GiB` repeated logical bytes). But a bounded calibration hotset might recover a small part of fallback without the huge regression seen when every up/down expert streams through GPU.
- `cap_bound`: top `0.5GiB` covers only `8.8%` fallback; top `1.0GiB` covers `14.0%`; top `2.0GiB` covers `22.0%`; top `3.0GiB` covers `28.4%`; top `8.0GiB` covers `51.1%`. Therefore this is an incremental probe, not a complete `>5 tok/s` solution.
- `implementation`: add two default-off controls in `moe_stream.cu`:
  - `GGML_MOE_STREAM_CACHE_ADMIT_PROFILE_APPLIES_FILTER=<names>`: cache admission profile applies only to matching tensor names; nonmatching tensors keep default admission. This preserves gate cache while using a hotset profile for up/down.
  - `GGML_MOE_STREAM_ONE_REQUIRE_CACHE_ADMIT_FILTER=<names>`: for matching tensors, if a `(tensor,expert)` is not admitted by the profile and not already cached, one-stream returns `false`, leaving that route on CPU fallback instead of uncached GPU streaming.
- `first_probe`: generate calibration-only top hotset TSV from existing dev fallback CSVs, start with `0.5-1.0GiB`, and test only calibration/dev prompts. Do not use held-out and do not promote unless dev-set min/mean improves over no-prompt-specific baseline (`mean=2.18`, `min=1.8`) with correctness/RAM/TTFT gates.
- `rollback`: if default-off guard changes existing gate-only behavior, if hotset profile causes gate cache miss cliff, or if end-to-end token rate regresses, reject and keep only diagnostic records if default-off behavior is proven safe.


## 2026-07-06 执行记录：up hot1g gated stream rejected/tie

- `attempt_id`: `20260706-up-hot1g-gated-stream-n64`
- `status`: `rejected_tie_not_sota`
- `artifact`: `.Agent/runs/20260705-vendor-ds4-coldstart/up-hot1g-gated-stream-n64-rejection-20260706.json`
- `source_change`: implemented default-off `GGML_MOE_STREAM_CACHE_ADMIT_PROFILE_APPLIES_FILTER` and `GGML_MOE_STREAM_ONE_REQUIRE_CACHE_ADMIT_FILTER` in `moe_stream.cu`. Unset behavior is unchanged; default-off gate-only control matched prior `n64` behavior.
- `profile`: `.Agent/profiles/vendor-ds4/calib-dev-up-hot1g-20260706.tsv`, derived only from `calibration_dev_set_v1`, no held-out. It selects `240` up `(tensor,expert)` pairs, `0.996GiB` payload, covering `26574.059ms` or `21.5%` of aggregate up fallback.
- `candidate_run`: `/root/lfz/runs/vendor-ds4-16gb/20260706T123541Z-20260706T-up-hot1g-gated-stream-n64-smoke/france-up-hot1g-gated-cpu40-vram0gb`.
- `candidate_metrics`: `eval_tok_s=2.2`, `prompt_tok_s=1.0`, `TTFT=36769.252077ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15074291712`, `ram_ok=true`, `correctness_ok=true` for short n64 smoke.
- `candidate_effect`: up fallback decreased versus default-off control (`up decode 7643.444ms -> 7201.154ms`, `up prompt 3953.053ms -> 3271.456ms`), but VRAM cache hit rate dropped from `78.3%` to `58.5%`, so end-to-end token rate only tied.
- `defaultoff_control`: `/root/lfz/runs/vendor-ds4-16gb/20260706T123731Z-20260706T-hotset-gated-defaultoff-gate-only-n64-control/france-defaultoff-gate-only-n64-cpu40-vram0gb`, `eval_tok_s=2.2`, `prompt_tok_s=1.0`, `memory_peak_bytes=16000000000`, `ram_ok=true`; correctness false only due n64 truncation.
- `decision`: reject 1GiB up hotset as performance candidate and do not expand to full dev set. The mechanism is useful as default-off diagnostic/control, but a small resident up hotset still steals enough cache/movement budget to erase fallback savings.
- `next_design`: do not try larger up hotsets unless a gate-cache partition or separate pool prevents hit-rate collapse. The next credible route is a cache-partitioned experiment or a non-cache grouped staging design; both need a hard bound before another full cold benchmark.


## 2026-07-06 设计：one-stream cache tail partition

- `attempt_id`: `20260706-one-stream-cache-tail-partition`
- `status`: `ready_for_default_off_implementation`
- `artifact`: `.Agent/runs/20260705-vendor-ds4-coldstart/one-stream-cache-tail-partition-design-20260706.json`
- `reason`: `up hot1g gated stream` only tied because it reduced up fallback but dropped shared VRAM cache hit rate from `78.3%` to `58.5%`. The cache is a single LRU pool, so up hotset inserts can evict gate entries.
- `implementation`: add default-off tail partition controls in `moe_stream.cu`: `GGML_MOE_STREAM_CACHE_PARTITION_TAIL_FILTER=<names>` and `GGML_MOE_STREAM_CACHE_PARTITION_TAIL_SLOTS=<N>`. Matching tensors evict only in the tail slot range; nonmatching tensors evict only in the main range. Lookup still scans all slots. Unset env keeps existing behavior.
- `first_probe`: use existing calibration-only `up hot1g` profile with `tail_slots=240` under strict `n64` France smoke. Total slots are about `3192`; this leaves about `2952` gate slots and gives up hotset about `1.0GiB`. Do not use held-out.
- `acceptance`: proceed to full calibration/dev only if the n64 smoke beats the gate-only `2.2 tok/s` control and does not break RAM/TTFT/correctness. If it ties or regresses, record rejected and do not expand.


## 2026-07-06 执行记录：up hot1g tail partition rejected/tie

- `attempt_id`: `20260706-up-hot1g-tailpart240-n64`
- `status`: `rejected_tie_not_sota`
- `artifact`: `.Agent/runs/20260705-vendor-ds4-coldstart/up-hot1g-tailpart240-n64-rejection-20260706.json`
- `source_change`: implemented default-off `GGML_MOE_STREAM_CACHE_PARTITION_TAIL_FILTER` and `GGML_MOE_STREAM_CACHE_PARTITION_TAIL_SLOTS` in `moe_stream.cu`. Matching tensors insert only into the tail slot range; nonmatching tensors insert only into the main range; lookup still scans all slots.
- `candidate_run`: `/root/lfz/runs/vendor-ds4-16gb/20260706T124712Z-20260706T-up-hot1g-tailpart240-n64-smoke/france-up-hot1g-tailpart240-cpu40-vram0gb`.
- `candidate_config`: calibration France n64 only, no held-out, existing `up hot1g` profile, `tail_filter=ffn_up_exps`, `tail_slots=240`, strict 16GB cgroup.
- `candidate_metrics`: `eval_tok_s=2.2`, `prompt_tok_s=0.9`, `TTFT=37196.590433ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15083732992`, `ram_ok=true`, `correctness_ok=true` for short n64 smoke.
- `candidate_effect`: fallback moved in the expected direction (`up decode=6781.241ms`, `up prompt=3493.415ms`) but total VRAM cache counter still reported `hits=15512`, `misses=10997`, `hit_rate=58.5%`, and token rate only tied.
- `defaultoff_control`: `/root/lfz/runs/vendor-ds4-16gb/20260706T124907Z-20260706T-tailpart-defaultoff-gate-only-n64-control/france-tailpart-defaultoff-gate-only-n64-cpu40-vram0gb`, `eval_tok_s=2.2`, `prompt_tok_s=0.9`, `memory_peak_bytes=16000000000`, `ram_ok=true`, cache `hits=13435`, `misses=3716`, `hit_rate=78.3%`; correctness false only due n64 truncation.
- `decision`: reject/tie. Tail partition is safe as a default-off diagnostic but does not create a measurable speed improvement. Do not expand to full dev set.
- `next_design`: stop cache-based up hotset experiments unless a new metric separates gate hits from denied up misses and shows a strong path. The remaining credible route is non-cache grouped staging or a model/representation change; both require hard-bound proof before another cold benchmark.


## 2026-07-06 设计：up hot3g tail partition smoke

- `attempt_id`: `20260706-up-hot3g-tailpart-n64`
- `status`: `planned_before_smoke`
- `profile`: `.Agent/profiles/vendor-ds4/calib-dev-up-hot3g-20260706.tsv`, calibration/dev only, no held-out.
- `profile_summary`: selects `722` up pairs, payload `2.997GiB`, covers `51854.272ms` or `42.0%` of aggregate up fallback.
- `theory`: 1GiB up hotset tied because fallback savings were too small. With tail partition active, 3GiB may recover more up fallback while bounding up entries to the tail partition. It sacrifices about `722` gate slots, so promote only if n64 smoke clearly beats the `2.2 tok/s` gate-only control.
- `scope`: run only calibration France `n64` smoke first; no held-out; do not expand to full dev set unless it beats control and remains RAM/TTFT/correctness safe.


## 2026-07-06 执行记录：up hot3g tail partition rejected/tie

- `attempt_id`: `20260706-up-hot3g-tailpart722-n64`
- `status`: `rejected_tie_not_sota`
- `artifact`: `.Agent/runs/20260705-vendor-ds4-coldstart/up-hot3g-tailpart722-n64-rejection-20260706.json`
- `profile`: `.Agent/profiles/vendor-ds4/calib-dev-up-hot3g-20260706.tsv`, calibration/dev only, no held-out; selects `722` up pairs, payload `2.997GiB`, covering `42.0%` of aggregate up fallback.
- `candidate_run`: `/root/lfz/runs/vendor-ds4-16gb/20260706T125410Z-20260706T-up-hot3g-tailpart722-n64-smoke/france-up-hot3g-tailpart722-cpu40-vram0gb`.
- `candidate_metrics`: `eval_tok_s=2.2`, `prompt_tok_s=0.9`, `TTFT=39035.413654ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=15063531520`, `ram_ok=true`; correctness false only because n64 truncates the answer.
- `candidate_effect`: up fallback dropped materially (`up decode=5879.531ms`, `up prompt=2520.818ms`) and cache hit rate was `65.9%`, but end-to-end token rate still tied the `2.2 tok/s` gate-only control while TTFT worsened.
- `decision`: reject and do not expand to full dev set. Larger up hotsets still fail to convert fallback savings into token-rate improvement; stream/cache overhead and remaining down fallback cancel the gain.
- `next_design`: close cache/hotset up streaming as a near-term path. Continue only with a non-cache grouped staging design if it can reduce launch/staging overhead by construction, or switch to model/representation/disk route.


## 2026-07-06 执行记录：dev gate/up/down overlap trace

- `attempt_id`: `20260706-dev-gate-updown-overlap-trace`
- `status`: `completed_measurement_not_sota`
- `artifact`: `.Agent/runs/20260705-vendor-ds4-coldstart/dev-gate-updown-overlap-trace-20260706.json`
- `prompt_scope`: only `calibration_dev_set_v1`; `held_out_test_set_v1_locked` was not used.
- `purpose`: Fill the missing gate overlap evidence called out by `updown-total-movement-hard-bound`. This measurement checks whether combined gate/up/down pack/read or staging can reuse route locality instead of repeating unrelated expert movement.
- `runs`: strict cold 16GB cgroup with `GGML_MOE_STREAM_ONE_TRACE_OUT={case_dir}/one_trace.csv`, gate-only one-stream filter `ffn_gate_exps`, no prompt-specific pack/profile. France `2.5 tok/s`, quantum `1.9 tok/s`, Fibonacci `1.8 tok/s`, Japan `2.4 tok/s`, climate `2.2 tok/s`; all five had `memory_peak_bytes=16000000000`, `ram_ok=true`, `ram_limit_killed=false`, and heuristic correctness pass. These token rates include trace overhead and are not SOTA claims.
- `aggregate_gate_trace`: gate unique `(layer,expert)` pairs `8621`, gate unique payload `35.781GiB`, calls `204420`, rows `210480`, gate cache hit rate `83.5%`, gate source-load time `167285.982ms` across the traced calibration runs.
- `aggregate_updown_reference`: existing up/down fallback profile has `6683` unique `(layer,expert)` pairs but `55.474GiB` tensor payload because up and down are distinct tensors; calls `222040`, rows `228020`, fallback `216535.305ms`.
- `overlap_result`: all observed up/down `(layer,expert)` pairs also appear in the gate trace: `6683/6683` up/down pairs, covering `100.0%` of up/down fallback time. These pairs are `77.52%` of gate pairs. The overlapping full gate+up+down tensor payload is `83.211GiB` (`27.737GiB` gate + `55.474GiB` up/down), far above VRAM and 16GB host RAM budgets.
- `decision`: structural overlap is high enough to consider combined sequential pack/read or staging-window designs, but it is not evidence for another resident-cache attempt. A resident combined cache is impossible under the current payload size and would destroy gate cache. Any next implementation must preserve current gate cache residency and reduce scattered read/page-fault/staging overhead without storing full up/down tensors long-term.
- `next_design`: draft a combined sequential pack/read hard-bound: for each routed `(layer,expert)`, estimate whether reading gate/up/down together from a compact pack can reduce page-cache refaults and source-load time while still streaming only the needed tensors. If the bound cannot beat the calibration baseline without gate-cache loss, reject before source changes. Continue to keep held-out prompts unused until a candidate is frozen.


## 2026-07-06 hard-bound：combined gate/up/down pack-read rejected

- `attempt_id`: `20260706-combined-gate-updown-pack-read-hard-bound`
- `status`: `rejected_before_source_change`
- `artifact`: `.Agent/runs/20260705-vendor-ds4-coldstart/combined-gate-updown-pack-read-hard-bound-20260706.json`
- `prompt_scope`: uses only calibration/dev artifacts and the new calibration gate traces; held-out test set was not used.
- `basis`: gate/up/down overlap is structurally high (`100%` of up/down `(layer,expert)` pairs appear in gate trace), but full overlapping gate+up+down tensor payload is `83.211GiB`, so it cannot be resident in VRAM or host RAM under the target machine.
- `streaming_lower_bound`: estimated gate miss source movement from the trace is `139.989GiB`; up/down repeated logical payload is `946.372GiB`; combined gate-miss + up/down stream amount is `1086.361GiB`. Pure transfer lower bound is `67.898s @16GiB/s`, `33.949s @32GiB/s`, `22.633s @48GiB/s`, `16.974s @64GiB/s`, or `11.316s @96GiB/s`, before GPU kernel, synchronization, D2H/scatter, page faults, and cache bookkeeping.
- `decision`: reject combined sequential pack/read as a standalone generalized `>5 tok/s` route. It may improve locality for a narrower gate-miss problem, but it does not reduce up/down bytes or compute enough; if implemented naively it repeats the same failure pattern as up/down streaming and hotset cache probes.
- `next_allowed_work`: do not write a runtime source patch for combined gate/up/down pack-read from current evidence. The next credible work must either reduce representation/payload, change dataflow so repeated up/down movement is avoided, or empirically validate a smaller model/representation route without using held-out prompts for tuning.

## 2026-07-06 执行记录：4Expert 磁盘释放与真实下载启动

- `attempt_id`: `20260706-4expert-disk-release-download-start`
- `status`: `download_in_progress_not_sota`
- `artifact`: `.Agent/runs/20260705-vendor-ds4-coldstart/4expert-disk-release-download-start-20260706.json`
- `prompt_scope`: 未运行任何 prompt，未使用 held-out test set。
- `action`: 按此前已授权的 GLM 实验清理，只删除 `/root/lfz/models/GLM-5.2-UD-IQ3_XXS`，释放完整 4Expert GGUF 所需磁盘空间；未触碰 DeepSeek SOTA native GGUF、gate pack、SOTA run 目录或 pushed-source repro run。
- `disk_after_cleanup`: 根分区可用空间从约 `1.9GB` 提升到约 `265GB`；aria2 目标文件预分配后当前根分区仍保留约 `112GB` 可用空间。
- `candidate_4expert_url`: `https://huggingface.co/cloudyu/DeepSeek-V4-Flash-4Expert-GGUF/resolve/main/ds4flash-4expert.gguf`
- `expected_size_bytes`: `164465760544`
- `head_etag`: `96bcd717ee6a3715d4ba1fd7946d9ee8a1eabb1247737afd009931f0318f000b`
- `download_path`: `/root/lfz/models/DeepSeek-V4-Flash-4Expert-GGUF/ds4flash-4expert.gguf`
- `download_service`: `ds4-4expert-download.service`，使用单个 systemd transient service 运行 `aria2c -c -x8 -s8 -k16M --file-allocation=none`，避免 SSH 断开导致下载中断。
- `completion_gate`: `.aria2` sidecar 消失，且 `stat size == 164465760544`；随后必须计算并记录 `sha256`，再做 load/correctness/perf。由于 aria2 会预分配文件，不能用 `ls -lh` 判断下载完成。
- `decision`: 当前没有任何正确性或性能结论，不能作为 SOTA。下载完成后继续执行 plan 中 `P3_load_validation_before_benchmark`、`P4_correctness_smoke`、`P5_strict_cold_benchmark`。

## 2026-07-06 执行记录：4Expert load 前置源码检查

- `attempt_id`: `20260706-4expert-load-preflight-source-check`
- `status`: `completed_preflight_not_sota`
- `artifact`: `.Agent/runs/20260705-vendor-ds4-coldstart/4expert-load-preflight-source-check-20260706.json`
- `prompt_scope`: 未运行任何 prompt，未使用 held-out test set。
- `tid2eid_alias`: 当前分支在 `src/llama-model.cpp` 中已有 default-off `LLAMA_DEEPSEEK4_TID2EID_WEIGHT_ALIAS=1`；当原始 `blk.%d.ffn_gate_tid2eid` metadata 不存在而 `.weight` alias 存在时，会选择 `.weight` tensor 创建 `layer.ffn_gate_tid2eid`。
- `q4k_stream_one`: 当前分支在 `ggml/src/ggml-cpu/ggml-cpu.c` 和 `ggml/src/ggml-cuda/moe_stream.cu` 中已有 default-off `GGML_MOE_STREAM_ONE_Q4K=1` admission；未设置时不改变原 MXFP4/F8_E4M3_B128 SOTA 路径。
- `decision`: 前置源码条件已满足，但这不是正确性或性能结果。完整真实 GGUF 下载完成后，必须先记录 size/sha256，再用 `LLAMA_DEEPSEEK4_TID2EID_WEIGHT_ALIAS=1` 和 `GGML_MOE_STREAM_ONE_Q4K=1` 做 P3 load validation；通过后才允许进入严格 16GB correctness/perf benchmark。

## 2026-07-06 执行记录：4Expert ready validator 工具与未完成门禁

- `attempt_id`: `20260706-4expert-ready-validator-tool`
- `status`: `completed_tooling_download_incomplete_not_sota`
- `tool`: `.Agent/run-tools/validate_4expert_ready.py`
- `artifact`: `.Agent/runs/20260705-vendor-ds4-coldstart/4expert-ready-validation-incomplete-20260706.json`
- `prompt_scope`: 未运行任何 prompt，未使用 held-out test set。
- `purpose`: 为 P3 load validation 前增加硬门禁，避免把 aria2 预分配但未完成的 GGUF 当成完整模型跑 benchmark。
- `completion_gate_enforced`: 默认要求 `.aria2` sidecar 不存在、`stat size == 164465760544`；下载完成后可加 `--sha256` 计算完整文件 hash。若 `.aria2` 存在，工具会输出 `complete_and_ready_for_load_validation=false` 并返回非通过状态，除非显式 `--allow-incomplete` 只做诊断记录。
- `current_incomplete_result`: 当前文件 header 可解析，但 `.aria2` 仍存在，所以 `complete_and_ready_for_load_validation=false`，`failures=[aria2_sidecar_present]`。
- `header_observation`: `general.architecture=deepseek4`，`deepseek4.block_count=43`，`deepseek4.expert_count=256`，`deepseek4.expert_used_count=4`；type counts 为 `Q4_K=129`、`Q8_0=366`、`F16=338`、`F32=492`、`I32=3`。
- `alias_observation`: `tid2eid_plain=0`，`tid2eid_weight=3`，这正是后续必须启用 `LLAMA_DEEPSEEK4_TID2EID_WEIGHT_ALIAS=1` 的原因。
- `expert_tensor_observation`: Q4_K expert tensors 共 `129`，其中 gate/up/down 各 `43`，符合 43 层每层 gate/up/down 的预期。
- `decision`: 这只是下载未完成状态下的门禁和 metadata 预检，不是 correctness/perf 结果。下载完成后先运行该工具的非 `--allow-incomplete` + `--sha256` 模式，并把通过结果作为 P3 load validation 的前置证据。

## 2026-07-06 执行记录：4Expert 下载完成后的固定验证入口

- `attempt_id`: `20260706-4expert-after-download-wrapper`
- `status`: `tooling_ready_not_run_download_incomplete`
- `tool`: `.Agent/run-tools/run_4expert_validation_after_download.sh`
- `prompt_scope`: 工具尚未执行；未运行任何 prompt，未使用 held-out test set。
- `reason`: 当前可用 binary 是 `build-ds4-moe-stream-batch/bin/llama-cli`，而 `strict_ds4_runner.py` 的默认 binary 路径不存在；为了避免下载完成后参数错配，新增固定 wrapper 显式传入 binary/model/env。
- `pre_gate`: wrapper 首先运行 `.Agent/run-tools/validate_4expert_ready.py --sha256`；只有 `.aria2` 不存在、size 等于 `164465760544`、metadata 符合 4Expert 预期并且 sha256 记录完成，才允许进入后续 load/correctness smoke。
- `first_smoke_config`: `cpu_moe=40`，`vram_cache_gb=0`，`ONE_CACHE_MIB=13568`，strict `drop_caches`，`MemoryMax=16000000000`，`MemorySwapMax=0`，France prompt，`LLAMA_DEEPSEEK4_TID2EID_WEIGHT_ALIAS=1`，`GGML_MOE_STREAM_ONE_Q4K=1`，`GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1`，`GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps`。
- `claim_rule`: 该 first smoke 只证明完整 4Expert GGUF 能否在严格 16GB 下加载并输出正确 France 回答；不能作为 generalized SOTA。若 smoke 通过，再按 calibration/dev set 评估候选；candidate freeze 后才允许使用 held-out test set。

## 2026-07-06 设计/执行准备：4Expert 下载完成自动验证 watcher

- `attempt_id`: `20260706-4expert-download-watch-tool`
- `status`: `tooling_ready_not_yet_launched`
- `tool`: `.Agent/run-tools/watch_4expert_download_then_validate.sh`
- `prompt_scope`: 工具准备阶段未运行任何 prompt，未使用 held-out test set。
- `purpose`: 当前完整 4Expert GGUF 下载仍在进行，手动等待容易错过完成窗口；新增 watcher 在 `.aria2` sidecar 消失后自动执行 readiness gate、sha256 和 strict France smoke，并把日志固定落盘。
- `safety_gate`: watcher 在 `.aria2` 存在时只轮询，不会运行 load/correctness/perf；若下载 service 非 active 但 `.aria2` 仍存在，会直接失败并记录状态，避免对损坏/未完成文件做 benchmark。
- `post_download_steps`: 先运行 `.Agent/run-tools/validate_4expert_ready.py --sha256`，通过后才调用 `.Agent/run-tools/run_4expert_validation_after_download.sh`。该 wrapper 内部还会再次执行 readiness gate，形成双重门禁。
- `log_location`: 默认 `/root/lfz/runs/vendor-ds4-16gb/<stamp>-4expert-download-watch/watch.log`，同时记录 `ready-validation.json`、`ready_exit_status.txt`、`validation_exit_status.txt`。
- `claim_rule`: watcher 的 first smoke 只用于确认完整 4Expert GGUF 是否能在严格 16GB 下加载并输出正确 France 回答；不是 generalized SOTA。若通过，后续仍需 calibration/dev，再 freeze 后跑 held-out test set。


## 2026-07-06 执行记录：4Expert 下载完成自动验证 watcher 已启动

- `attempt_id`: `20260706-4expert-download-watch-launch`
- `status`: `watcher_running_download_incomplete_not_sota`
- `artifact`: `.Agent/runs/20260705-vendor-ds4-coldstart/4expert-download-watch-launch-20260706.json`
- `unit`: `ds4-4expert-watch-20260706T131914Z.service`
- `log_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260706T131914Z-4expert-download-watch`
- `watch_log`: `/root/lfz/runs/vendor-ds4-16gb/20260706T131914Z-4expert-download-watch/watch.log`
- `prompt_scope`: 启动 watcher 本身未运行任何 prompt，未使用 held-out test set；只有下载完成且 readiness gate 通过后才会自动运行 France strict smoke。
- `source_commit`: `fabb6036d1271b464c6056ab1c51d436928ec6a3`
- `current_download_status`: 下载 service 仍为 `active`，`.aria2` sidecar 仍存在，watcher 仅轮询等待。
- `automatic_next_steps`: `.aria2` 消失后执行 `validate_4expert_ready.py --sha256`；通过后执行 `run_4expert_validation_after_download.sh`，产生 strict 16GB France correctness smoke 记录。
- `claim_rule`: 当前没有任何 correctness/perf/SOTA 结论。只有 watcher 产出完整 `ready-validation.json` 和 strict runner `summary.json`，并确认 RAM/page cache/TTFT/correctness 后，才允许进入后续 decision 和提交记录。

## 2026-07-06 执行记录：4Expert load compatibility 诊断

- `attempt_id`: `20260706-4expert-load-compat-diagnostic`
- `status`: `diagnostic_not_accepted`
- `artifact`: `.Agent/runs/20260705-vendor-ds4-coldstart/4expert-load-compat-diagnostic-20260706.json`
- `prompt_scope`: 只使用 France smoke；未使用 `calibration_dev_set_v1` 做调参，未使用 `held_out_test_set_v1_locked`。该结果不能作为 generalized SOTA。
- `source_scope`: 本轮实现为 4Expert load compatibility 和 CUDA type coverage 诊断；不改变当前 accepted native DeepSeek SOTA。
- `implemented`:
  - default-off GGUF metadata compatibility：`LLAMA_GGUF_ALLOW_U64_TO_U32=1` 和 `LLAMA_GGUF_ALLOW_F64_TO_F32=1`；
  - `tokenizer.ggml.model=bpe` 进入现有 BPE tokenizer 路径；
  - default-off 4Expert tensor aliases：`LLAMA_DEEPSEEK4_4EXPERT_TENSOR_ALIAS=1`，覆盖 `output_hc_* -> hc_head_*`、layer `.weight` 后缀、`attn_kv_latent -> attn_kv`、`compress -> compressor`、`exp_probs_b.bias`；
  - default-off diagnostics：`LLAMA_DUMP_UNCREATED_TENSORS=1`、`GGML_CUDA_BINBCAST_DEBUG=1`、`GGML_CUDA_CONCAT_DEBUG=1`；
  - CUDA `binbcast` 修复 `f32 + f16 -> f32` 分支，避免把 f16 src1 当作 float；
  - CUDA `concat` 增加 pure f16 concat 支持。
- `debug_findings`:
  - readiness gate 已证明完整 4Expert 文件存在，sha256 为 `e9e7e22ba585f83330d08235de39e8dcd8cbb513fd9fad6103764da74a4e64bc`；
  - 初始 blocker 依次为 metadata `u64/f64` type mismatch、`tokenizer.ggml.model=bpe` unknown、`output_hc_*`/layer tensor 命名差异、`attn_kv`/compressor 命名差异、以及 `exp_probs_b.bias` 未消费；
  - `LLAMA_DUMP_UNCREATED_TENSORS=1` 定位最后 40 个未消费 tensor 全部为 `blk.3..42.exp_probs_b.bias`；
  - `GGML_CUDA_BINBCAST_DEBUG=1` 定位 `blk.2.attn_compressor_ape.weight` f16 view 参与 f32 ADD 时触发 stride assert；
  - `GGML_CUDA_CONCAT_DEBUG=1` 定位同一 tensor 的两个 f16 view 做 dim=1 concat，原 CUDA concat 只支持 f32。
- `best_4expert_run_after_fixes`:
  - `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260706T152334Z-4expert-concat-f16-fix-streamoff-france-strict-smoke/france-4expert-concat-f16-fix-streamoff-cpu40-vram0gb-cpu40-vram0gb`
  - `env`: `LLAMA_GGUF_ALLOW_U64_TO_U32=1`, `LLAMA_GGUF_ALLOW_F64_TO_F32=1`, `LLAMA_DEEPSEEK4_4EXPERT_TENSOR_ALIAS=1`, `LLAMA_DEEPSEEK4_TID2EID_WEIGHT_ALIAS=1`, `GGML_MOE_STREAM=0`
  - `metrics`: `eval_tok_s=2.0`, `prompt_tok_s=1.4`, `elapsed_seconds=136.6`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14895955968`, `ram_ok=true`, `ram_limit_killed=false`, `exit_status=0`
  - `correctness`: `false`; answer was empty, missing France/Europe/semantic content.
  - `decision`: rejected / not accepted. It is slower than the native path target and fails correctness, so it cannot replace any SOTA.
- `native_regression_after_changes`:
  - `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260706T152707Z-native-regression-after-4expert-compat-france-strict-smoke/france-native-regression-cpu40-vram0gb-cpu40-vram0gb`
  - `metrics`: `eval_tok_s=1.6`, `prompt_tok_s=0.7`, `TTFT=46905.717439ms`, `memory_peak_bytes=16000000000`, `memory_file_bytes=14978998272`, `ram_ok=true`, `correctness_ok=true`
  - `note`: 该回归 smoke 未使用 accepted SOTA pack/profile，因此不能与 `4.4 tok/s` France SOTA 比较；只证明当前源码没有让 native DeepSeek 路径崩溃。
- `next_decision`:
  - 4Expert 当前不是短期 SOTA 路径，除非先解决空输出/正确率问题；
  - 当前 accepted native DeepSeek SOTA 不变；
  - 后续 token-rate 主线仍回到 plan 中的 generalized prompt CPU fallback / up-down GPU path，而不是继续把 4Expert 空输出结果作为性能优化对象。

## 2026-07-07 执行记录：4Expert token_type 空输出诊断与 reject

- `attempt_id`: `20260707-4expert-token-type-zero-normal-diagnostic`
- `status`: `diagnostic_rejected_not_sota`
- `artifact`: `.Agent/runs/20260705-vendor-ds4-coldstart/4expert-token-type-zero-normal-diagnostic-20260707.json`
- `prompt_scope`: 只使用 France smoke 和 native default-off 回归；未使用 `calibration_dev_set_v1` 调参，未使用 `held_out_test_set_v1_locked`；该结果不能作为 generalized SOTA。
- `source_change`:
  - 新增 default-off `LLAMA_DEBUG_SAMPLE_TOKENS=1`，只在诊断时把 sampled token id 与 `token_to_piece(..., special=true)` 打到 stderr；默认关闭，不改变采样行为。
  - 新增 default-off `LLAMA_GGUF_TOKEN_TYPE_UNDEFINED_AS_NORMAL=1`，只在显式设置时把 GGUF `tokenizer.ggml.token_type=0` 当作 normal token；默认关闭，不影响 native DeepSeek 路径。
- `root_cause`: 4Expert 之前不是没有生成 token，而是 sampled token id 正常但 piece 全为空。原因是 4Expert GGUF 中大量正常 token 的 `token_type` 为 `0`，当前 loader 将其解释为 `LLAMA_TOKEN_TYPE_UNDEFINED` 并覆盖默认 normal attr，导致 BPE `token_to_piece` 抑制输出。
- `before_fix_evidence`: `/root/lfz/runs/vendor-ds4-16gb/20260706T162828Z-20260707-4expert-sample-token-debug-completion-n16/france-4expert-sample-debug-n16-cpu40-vram0gb`，sample debug 显示 `id=20/16/2619/...` 但 `piece=""`；stdout 只有换行；`token to piece cache size` 约 `0.0001 MB`。
- `after_token_type_fix_evidence`: `/root/lfz/runs/vendor-ds4-16gb/20260706T163214Z-20260707-4expert-token-type-zero-normal-completion-n16/france-4expert-ttype0normal-n16-cpu40-vram0gb`，sample debug 显示 `id=20 piece="2"`, `id=2619 piece=" **"`, `id=71343 piece="Identify"`；`token to piece cache size` 约 `1.0267 MB`，证明空输出问题被 tokenizer attr 层修复。
- `correctness_full_smoke`: `/root/lfz/runs/vendor-ds4-16gb/20260706T163339Z-20260707-4expert-token-type-zero-normal-cli-france-n192/france-4expert-ttype0normal-cli-cpu40-vram0gb`，`eval_tok_s=2.0`、`prompt_tok_s=1.4`、`TTFT=42943.682616 ms`、`memory_peak_bytes=16000000000`、`memory_file_bytes=14923116544`、`ram_ok=true`，但 `correctness_ok=false`，原因 `degenerate_text,truncated_or_corrupt_landmark`；输出前半段有 France 语义，但随后 markdown/code-fence 重复并出现 `Eiff Tower` 截断。
- `template_probe`: `/root/lfz/runs/vendor-ds4-16gb/20260706T163647Z-20260707-4expert-ttype0normal-deepseek3-template-france-n128/france-4expert-deepseek3-template-n128-cpu40-vram0gb`，显式 `--chat-template deepseek3 --reasoning off` 后 `eval_tok_s=1.6`、`ram_ok=true`、`correctness_ok=false`，输出时间戳/Chat 噪声；模板不能修复正确性。
- `native_defaultoff_regression`: `/root/lfz/runs/vendor-ds4-16gb/20260706T163913Z-20260707-native-defaultoff-regression-after-token-debug-n64/france-native-defaultoff-regression-n64-cpu40-vram0gb`，新 env 默认关闭时 native DeepSeek gate-only n64 smoke `eval_tok_s=1.5`、`TTFT=47080.892513 ms`、`memory_peak_bytes=16000000000`、`ram_ok=true`、`correctness_ok=true`。
- `decision`: 4Expert 从空输出推进到可见文本，但仍未通过正确性，且速度低于 native generalized baseline，不是 accepted SOTA。短期 token-rate 主线不要基于 4Expert 做性能 benchmark；如果之后继续 4Expert，必须先定位剩余 tensor alias / 数值路径 / 模板不匹配导致的退化，再进入性能实验。
- `next_allowed_work`: 回到 native generalized prompt 的主线。下一步优先选择能够减少 up/down fallback payload 或改变数据流的方案；不要把 4Expert 作为 token-rate 候选，除非 correctness 先过。

## 2026-07-07 hard-bound：generalized grouped up/down staging rejected

- `attempt_id`: `20260707-generalized-grouped-staging-hard-bound`
- `status`: `rejected_before_source_change`
- `artifact`: `.Agent/runs/20260705-vendor-ds4-coldstart/generalized-grouped-staging-hard-bound-20260707.json`
- `prompt_scope`: 只使用 `calibration_dev_set_v1` 的 no-prompt-specific fallback profile；未使用 `held_out_test_set_v1_locked`。
- `purpose`: 重新按当前任务背景评估 native generalized prompt 路线：16GB host RAM + 32GB 5090、随机 prompt 稳定 `>5 tok/s`。该 bound 不沿用 France-only 4.4/10 tok/s 结论，而是基于 generalized calibration baseline (`mean=2.18`, `min=1.7/1.8`) 估算 paired up/down grouped staging 的上限。
- `method`: 从每个 calibration prompt 的 decode up/down fallback profile 聚合 `(layer, expert)`，把 up/down 同一 expert 作为 paired staging 单元；用 `eval_tok_s` 和 decode calls 反推 decode window，并在 zero-overhead 模型中扣除被选 hot set 覆盖的 decode fallback ms。该结果是乐观上限，不包含 H2D/D2H、GPU kernel、launch/sync、scatter、cache bookkeeping 或 gate-cache 退化。
- `gate_preserving_pool_bound`: 沿用之前 top768/Q8 hard-bound 中可保留 gate cache 的约 `3.2GiB` extra pool。`3.2GiB` paired layer-expert hot set 选择 `385` 个 layer-expert，payload `3.196GiB`；zero-overhead 上限约为 `mean=2.64-2.68 tok/s`、`min=1.94-1.96 tok/s`。
- `large_payload_sensitivity`: `8GiB` 上限约 `mean=3.07`、`min=2.22`；`16GiB` 上限约 `mean=3.69`、`min=2.60`；`25.5GiB` 上限约 `mean=4.24`、`min=3.03`。这些 payload 已经会挤压 gate cache/VRAM，且仍不到 generalized `5 tok/s`。
- `decision`: reject grouped up/down staging as a direct route before source change. 它只改变 staging/scheduling，不减少 expert payload，也不能在 gate-cache-safe VRAM budget 内覆盖足够 generalized fallback；任何真实实现开销都会低于 zero-overhead 上限。
- `next_allowed_work`: 不写 grouped staging runtime patch。后续必须转向能减少 payload/改变数据流的路线：例如有 top1 proof 的表示压缩/非 native representation、能完全消除 up/down bytes 的 graph/dataflow 证明，或者先解决 4Expert correctness 后再重新评估 smaller representation。继续保持 held-out set 未使用，直到 candidate freeze。

## 2026-07-07 执行记录：4Expert correctness follow-up rejected

- `attempt_id`: `20260707-4expert-correctness-followup-reject`
- `status`: `rejected_not_sota_correctness_still_fails`
- `artifact`: `.Agent/runs/20260705-vendor-ds4-coldstart/4expert-correctness-followup-reject-20260707.json`
- `prompt_scope`: 只使用 France smoke；未使用 calibration prompt 调参，未使用 `held_out_test_set_v1_locked`。
- `purpose`: 在 `LLAMA_GGUF_TOKEN_TYPE_UNDEFINED_AS_NORMAL=1` 修复空输出后，做低成本 follow-up，判断 4Expert 是否只是模板/漏 tensor 问题。
- `uncreated_check`: `/root/lfz/runs/vendor-ds4-16gb/20260706T164826Z-20260707-4expert-token-type-uncreated-dump-n1/france-4expert-ttype-uncreated-n1-cpu40-vram0gb`，启用 `LLAMA_DUMP_UNCREATED_TENSORS=1` 后没有出现 uncreated/unused tensor dump；`memory_peak_bytes=16000000000`、`ram_ok=true`。
- `template_check_deepseek`: `/root/lfz/runs/vendor-ds4-16gb/20260706T165001Z-20260707-4expert-ttype0normal-deepseek-template-france-n128/france-4expert-deepseek-template-n128-cpu40-vram0gb`，`eval_tok_s=1.7`、`TTFT=32685.844057 ms`、`ram_ok=true`、`correctness_ok=false`，输出 malformed `deep2` version-map 风格文本，缺少 France/Europe/context。
- `template_check_deepseek3`: 之前的 `/root/lfz/runs/vendor-ds4-16gb/20260706T163647Z-20260707-4expert-ttype0normal-deepseek3-template-france-n128/france-4expert-deepseek3-template-n128-cpu40-vram0gb`，同样 `correctness_ok=false`，输出 timestamp/Chat 噪声。
- `decision`: 4Expert 不是短期 token-rate 候选。tokenizer attr 层已修复空输出，但 default/deepseek/deepseek3 模板都无法通过 France correctness，且没有明显未消费 tensor；剩余问题更可能是 tensor alias / numerical / route compatibility，需要单独的 layer-output parity 计划。除非 France correctness 先过，否则不要对 4Expert 做性能 benchmark 或 generalized SOTA claim。
- `next_allowed_work`: 回到 native generalized path；若未来重开 4Expert，先写 dedicated numerical parity plan，而不是继续试模板或 token-rate 参数。

## 2026-07-07 hard-bound：generalized full up/down removal still insufficient alone

- `attempt_id`: `20260707-generalized-full-updown-removal-bound`
- `status`: `completed_bound_no_source_change`
- `artifact`: `.Agent/runs/20260705-vendor-ds4-coldstart/generalized-full-updown-removal-bound-20260707.json`
- `prompt_scope`: 只使用 `calibration_dev_set_v1`；未使用 `held_out_test_set_v1_locked`。
- `purpose`: 在关闭 grouped staging 后，重新回答一个核心问题：如果理想地移除 native decode up/down CPU fallback，本任务要求的 randomized/generalized `>5 tok/s` 是否自然成立。
- `method`: 从 no-prompt-specific fallback profile 中用 decode up calls 反推 decode token 数，按 `40` 和 `43` 层做 sensitivity；用当前 `eval_tok_s` 反推 decode window，然后 zero-overhead 扣除 measured decode up/down fallback ms。
- `result_40_layer`: mean upper-bound `4.916 tok/s`，min `3.592 tok/s`；`quantum` 和 `fibonacci` 即使理想移除 decode up/down fallback 仍低于 `5 tok/s`。
- `result_43_layer`: mean upper-bound `5.433 tok/s`，min `3.919 tok/s`；`fibonacci` 仍低于 `5 tok/s`。
- `decision`: up/down fallback 是大瓶颈，但“只做 up/down exact path”不足以保证 generalized `>5 tok/s`。后续候选必须同时处理 gate/source/residual decode cost，或者在表示/模型层减少整体 payload。

## 2026-07-07 hard-bound：generalized residual bottleneck after up/down removal

- `attempt_id`: `20260707-generalized-residual-bottleneck-after-updown`
- `status`: `completed_measurement_not_sota`
- `artifact`: `.Agent/runs/20260705-vendor-ds4-coldstart/generalized-residual-bottleneck-after-updown-bound-20260707.json`
- `prompt_scope`: 只使用 calibration/dev artifacts；未使用 held-out。
- `inputs`: `.Agent/runs/20260705-vendor-ds4-coldstart/generalized-full-updown-removal-bound-20260707.json` 和 `.Agent/runs/20260705-vendor-ds4-coldstart/dev-gate-updown-overlap-trace-20260706.json`。
- `observation`: gate/source trace 与 up/down fallback 同量级。示例：`fibonacci` 的 gate `src0_ms` trace total 为 `48157.035 ms`，decode up/down fallback 为约 `48085.4 ms`；`quantum` gate `src0_ms=38041.296 ms`，decode up/down fallback 约 `49631.8 ms`。注意 gate trace 未按 phase 分离且有 instrumentation，只能作为 bottleneck 量级，不直接从 eval decode window 扣除。
- `decision`: 不允许写只替换 up/down fallback 的 runtime patch 来 claim generalized `>5 tok/s`。下一步必须先做 joint hard-bound：要么 exact graph/dataflow 证明可以同时减少/隐藏 gate source + up/down movement，要么 representation/top1 proof 证明能整体减少 payload 并保持 correctness。
- `next_allowed_work`: 评估 current graph 是否能保留 gate results 并避免 up/down repeated source movement；若当前 graph 不具备该数据流，则记录为 generalized 5 的 blocker。另一条可重开路线是 representation，但必须先有 fixed-text top1/correctness proof；4Expert 已因 correctness 失败暂时关闭。

## 2026-07-07 hard-bound：generalized exact graph/dataflow recheck rejected

- `attempt_id`: `20260707-generalized-exact-graph-dataflow-recheck`
- `status`: `rejected_before_source_change`
- `artifact`: `.Agent/runs/20260705-vendor-ds4-coldstart/generalized-exact-graph-dataflow-recheck-20260707.json`
- `prompt_scope`: 未运行新 prompt；使用 calibration-only generalized bounds 和 source audit；未使用 held-out。
- `why_recheck`: 当前目标是随机/generalized prompt 在 16GB RAM + 32GB 5090 上稳定 `>5 tok/s`，不是 France-only 10 tok/s。新的 generalized bound 证明 up/down-only removal 不足，因此 graph/dataflow 若要重开，必须同时减少/隐藏 gate source 和 up/down movement。
- `source_audit`: `src/models/deepseek4.cpp` 仍然 hardcode `graph_gate_output_input_available=false`，`build_expert_mix` 接收的是 `selected_experts` 和 `weights`，不是可复用 retained gate output；当前 `DS4_HOT_DISPATCH` 会重新计算 gate/up/down，并通过 hot tensor subset 增加 payload/VRAM 压力。
- `decision`: 不允许从当前证据写 exact graph/dataflow runtime patch。因为 generalized 5 需要 joint gate/source + up/down reduction，而当前图缺少 retained-gate/source-reuse 接口；这不是调参问题，而是数据流 blocker。
- `reopen_condition`: 需要先有 source/dataflow probe 证明 accepted graph 中存在 retained CUDA gate/topk/weights path，或者新增接口后有 hard-bound 证明所有 calibration prompt 在计入 overhead、16GB page cache、gate cache、workspace 后仍 `>5 tok/s`，并且 fixed-text top1/correctness 先过。
- `next_allowed_work`: 剩余可行类别转为 external verifier/speculative decoding 高接受率路线，或 representation/model 路线的 correctness/top1 proof。若这些也没有证据，则应记录 generalized `>5 tok/s` 需要新的 dataflow interface，而不是继续 stream/cache 局部调优。

## 2026-07-07 route triage：转向 alternate low-bit DeepSeek GGUF 候选

- `attempt_id`: `20260707-generalized-route-triage-alt-gguf-plan`
- `status`: `plan_recorded_before_download_no_sota`
- `artifact`: `.Agent/runs/20260705-vendor-ds4-coldstart/generalized-route-triage-alt-gguf-plan-20260707.json`
- `prompt_scope`: 未运行新 prompt；未使用 `held_out_test_set_v1_locked`。
- `why_now`: 当前 native 路线的局部优化已被 generalized hard-bound 限制住：up/down grouped staging、up/down-only fallback removal、当前 exact graph/dataflow、DFlash/MTP verifier、native exact compact representation、4Expert 都没有足够证据能在 `16GB RAM + 32GB 5090` 下让随机/泛化 prompt 稳定 `>5 tok/s`。
- `closed_route_summary`:
  - up/down grouped staging：gate-preserving `3.2GiB` zero-overhead 上限只有 `mean≈2.64-2.68 tok/s`, `min≈1.94-1.96 tok/s`；更大 payload 会挤压 gate cache 且仍不到 generalized `5 tok/s`。
  - up/down-only removal：理想 zero-overhead 40-layer 上限 `mean≈4.916`, `min≈3.592 tok/s`；43-layer sensitivity 仍有 Fibonacci 低于 `5 tok/s`。
  - exact graph/dataflow：当前 source 没有 retained gate output/source-reuse 接口，不能从现有证据直接写 runtime patch。
  - DFlash/MTP/speculative：oracle target verifier 的 W=2/4/8 speedup 只有 `1.010x/1.013x/1.002x`，低于 reopen gate；vendor 也没有 source-ready DFlash/MTP runtime。
  - native exact compact representation：MXFP4 hotset 实测/entropy reduction 只有约 `1.04x-1.085x`，远低于 `4x-5.33x` 需求。
  - 4Expert：空输出 token_type 问题已定位并 default-off 修复，但 France correctness 仍失败，约 `2.0 tok/s`，不是短期 token-rate 候选。
- `reopened_candidate`: alternate low-bit DeepSeek GGUF，优先 `0xSero/DeepSeek-V4-Flash-162B-GGUF` 的 `DeepSeek-V4-Flash-Spark-Mini-Q2-REAP-ds4.gguf`。该路线必须标注为 alternate model/quantization candidate，不能混同 native GGUF SOTA；只有 correctness 和泛化 prompt-set 都过关后才可作为任务候选。
- `disk_and_head_audit`: 当前 `/root` 可用约 `111G`，不删除 native SOTA evidence 也足够下载一个 `52,593,532,000 bytes` 的 0xSero candidate；HEAD 当前返回 `x-linked-size=52593532000`，`x-linked-etag=e917278028d7a9e25dfc9d04bf5848375dad7573c5aeab1720d6a83714352406`，final `ETag=7f4deb0dc07cdbc01ff88ae11e616fd8d2d1d8263efec15b034c5d731fe83070`。
- `next_action`: 先记录并 push 本计划；之后只下载这一条 candidate，记录 size/sha256 和 metadata/header validation。France strict 16GB correctness 通过前禁止 token-rate SOTA claim；calibration/dev 通过并 freeze 前禁止使用 held-out test。
- `promotion_rule`: 如果 alternate candidate 在 `calibration_dev_set_v1` 上形成候选，必须冻结 model sha256、source commit、env/CLI 和参数后，再运行 `held_out_test_set_v1_locked`。accepted generalized SOTA 必须报告 held-out per-prompt `eval_tok_s`、TTFT、RAM/page-cache、correctness 和完整输出，并立即 commit/push 到 `ssd/vendor/deepseek-token-rate-16gb` 后从 pushed commit 复现。

## 2026-07-07 执行记录：0xSero alternate GGUF 下载启动

- `attempt_id`: `20260707-0xsero-alt-gguf-download-start`
- `status`: `download_started_not_sota`
- `artifact`: `.Agent/runs/20260705-vendor-ds4-coldstart/alt-gguf-0xsero-download-start-20260707.json`
- `prompt_scope`: 未运行任何 prompt；未使用 `held_out_test_set_v1_locked`。
- `candidate`: `0xSero/DeepSeek-V4-Flash-162B-GGUF` / `DeepSeek-V4-Flash-Spark-Mini-Q2-REAP-ds4.gguf`。
- `download_path`: `/root/lfz/models/DeepSeek-V4-Flash-162B-GGUF/DeepSeek-V4-Flash-Spark-Mini-Q2-REAP-ds4.gguf`。
- `expected_size_bytes`: `52593532000`。
- `download_unit`: `ds4-0xsero-alt-gguf-download-20260706T170631Z.service`。
- `download_log`: `/root/lfz/runs/vendor-ds4-16gb/20260706T170631Z-0xsero-alt-gguf-download/download.log`。
- `completion_gate`: `.aria2` sidecar 消失、`stat size == 52593532000`、记录 sha256、GGUF header/metadata validation 通过后，才允许进入 France strict 16GB correctness smoke。
- `claim_rule`: 当前只是下载启动记录，没有 correctness、token-rate 或 SOTA 结论。France correctness 通过前禁止 benchmark/SOTA claim；calibration/dev 候选 freeze 前禁止使用 held-out test。

## 2026-07-07 执行记录：0xSero alternate GGUF ready validation 通过

- `attempt_id`: `20260707-0xsero-alt-gguf-ready-validation`
- `status`: `ready_for_france_strict_smoke`
- `artifact`: `.Agent/runs/20260705-vendor-ds4-coldstart/alt-gguf-0xsero-ready-validation-20260707.json`
- `prompt_scope`: 未运行任何 prompt；未使用 `held_out_test_set_v1_locked`。
- `model_path`: `/root/lfz/models/DeepSeek-V4-Flash-162B-GGUF/DeepSeek-V4-Flash-Spark-Mini-Q2-REAP-ds4.gguf`。
- `size_and_sha256`: size `52593532000` bytes；sha256 `e917278028d7a9e25dfc9d04bf5848375dad7573c5aeab1720d6a83714352406`。
- `download_correction`: aria2 起初因 `systemd --same-dir` 写入 repo root；下载完成后已把完整文件移动到 canonical `model_path`，repo root 的 partial file 和 `.aria2` sidecar 均已清理，未加入 git。
- `header_validation`: GGUF version `3`，`n_kv=58`，`n_tensors=1328`，`general.architecture=deepseek4`，`general.file_type=19`，`deepseek4.block_count=43`，`deepseek4.expert_count=144`，`deepseek4.expert_used_count=6`，`deepseek4.nextn_predict_layers=1`。
- `tensor_validation`: type counts 为 `F32=492`, `F16=359`, `Q8_0=345`, `IQ2_XXS=86`, `Q2_K=43`, `I32=3`；gate/up expert tensors 各 `43` 个，type `IQ2_XXS`，down expert tensors `43` 个，type `Q2_K`；`ffn_gate_tid2eid.weight` 为 `3` 个。
- `next_action`: 运行 France strict 16GB correctness smoke。若 load 或 France correctness 失败，立即 reject，不做 calibration/dev token-rate benchmark；若通过，才进入 `calibration_dev_set_v1` strict cold no-prompt-specific baseline。
- `claim_rule`: 当前只有下载和 metadata ready 结论，没有 correctness、token-rate 或 SOTA 结论。

## 2026-07-07 执行记录：0xSero alternate GGUF France load smoke rejected

- `attempt_id`: `20260707-0xsero-alt-gguf-france-load-smoke`
- `status`: `rejected_load_incompatible_not_sota`
- `artifact`: `.Agent/runs/20260705-vendor-ds4-coldstart/alt-gguf-0xsero-france-load-smoke-reject-20260707.json`
- `prompt_scope`: 只运行 France smoke；未使用 `calibration_dev_set_v1` 调参，未使用 `held_out_test_set_v1_locked`。
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260706T172636Z-20260707-0xsero-alt-gguf-france-strict-smoke/france-0xsero-alt-cpu40-vram0gb-cpu40-vram0gb`。
- `config`: strict 16GB cgroup、drop_caches、`cpu_moe=40`、`vram_cache=0`、`LLAMA_DEEPSEEK4_TID2EID_WEIGHT_ALIAS=1`、`GGML_MOE_STREAM=0`、无 prompt-specific pack/profile。
- `result`: exit status `1`，`eval_tok_s=None`，`prompt_tok_s=None`，`TTFT=None`，`memory_peak_bytes=387440640`，`memory_file_bytes=196919296`，`ram_ok=true`；没有进入真实推理。
- `loader_error`: stderr 明确报 `missing tensor 'hc_head_base'`。0xSero header 中有 per-layer `blk.N.hc_attn_*` / `blk.N.hc_ffn_*`，但没有 vendor 当前期望的 global `hc_head_base/hc_head_fn/hc_head_scale`，也没有现有 alias 支持的 `output_hc_*`。
- `decision`: reject，不进入 token-rate benchmark，不作为 SOTA。该问题是模型/loader/dataflow compatibility blocker，不是 RAM、TTFT 或 cache 参数问题。
- `next_allowed_work`: 不使用 held-out。若继续 alternate GGUF 路线，必须先用 header-only probe 筛掉缺少 global `hc_head_*`/`output_hc_*` 或其他必需 tensor 的 candidate；或者单独写 loader/dataflow correctness 计划，证明 per-layer hc tensor 如何等价替代当前 global hc_head 后再改 source。

## 2026-07-07 header triage：选择 sleepy K128 alternate GGUF 作为下一候选

- `attempt_id`: `20260707-alt-gguf-header-compat-triage`
- `status`: `completed_header_only_no_sota`
- `artifact`: `.Agent/runs/20260705-vendor-ds4-coldstart/alt-gguf-header-compat-triage-20260707.json`
- `prompt_scope`: 未运行 prompt；未使用 `calibration_dev_set_v1` 或 `held_out_test_set_v1_locked`。
- `method`: 对 sleepy、teamblobfish、antirez、tarruda 候选只做 `curl --range 0-16777215 --max-filesize 20000000` header probe；没有下载完整模型。
- `key_finding`: 0xSero 失败的根因是缺 global `output_hc_*`/`hc_head_*`；sleepy 和 antirez 单文件候选都带 `output_hc_*`、`.weight` tensor aliases、`attn_kv.weight` 和 `compressor` 命名，理论上可用现有 default-off `LLAMA_DEEPSEEK4_4EXPERT_TENSOR_ALIAS=1` + `LLAMA_DEEPSEEK4_TID2EID_WEIGHT_ALIAS=1` 先做 load smoke。
- `candidate_comparison`:
  - sleepy `DeepSeek-V4-Flash-REAP-K128-uniform.gguf`: size `50439361920`，single file，`expert_count=256`，`expert_used_count=6`，gate/up `IQ2_XXS`，down `Q2_K`，`output_hc=3`，作为下一候选。
  - antirez IQ2XXS chat-v2: size `86720111200`，header-compatible，但更大，磁盘/下载成本更高，作为 fallback。
  - teamblobfish IQ1_M split: part1 size `49882725408`，split_count `2`，header-compatible，但 split 加载和 IQ1_M path 风险更高，暂不优先。
  - tarruda Q2_K split: first shard是 metadata-only，`n_tensors=0`，需要更多 split 处理，不作为下一步。
- `next_action`: 删除已 rejected 的本地 0xSero 大文件释放磁盘（记录和 sha256 已 push），下载 sleepy K128 uniform，完成 size/sha256/header validation 后才允许 France strict 16GB smoke。
- `claim_rule`: header triage 不是 correctness/token-rate/SOTA。sleepy 若 load 或 France correctness 失败，立即 reject，不进入 calibration/dev；若通过，才跑 calibration/dev，freeze 后才可使用 held-out。

## 2026-07-07 执行记录：sleepy K128 alternate GGUF 下载启动

- `attempt_id`: `20260707-sleepy-k128-alt-gguf-download-start`
- `status`: `download_started_not_sota`
- `artifact`: `.Agent/runs/20260705-vendor-ds4-coldstart/alt-gguf-sleepy-k128-download-start-20260707.json`
- `prompt_scope`: 未运行 prompt；未使用 `held_out_test_set_v1_locked`。
- `cleanup`: 已删除本地 rejected 0xSero GGUF 大文件 `/root/lfz/models/DeepSeek-V4-Flash-162B-GGUF/DeepSeek-V4-Flash-Spark-Mini-Q2-REAP-ds4.gguf` 释放磁盘；该 candidate 的 sha256、ready validation、reject 记录已 push。
- `candidate`: `sleepyeldrazi/deepseek-v4-flash-reap-k128-Q2-GGUF` / `DeepSeek-V4-Flash-REAP-K128-uniform.gguf`。
- `download_path`: `/root/lfz/models/DeepSeek-V4-Flash-REAP-K128-uniform-GGUF/DeepSeek-V4-Flash-REAP-K128-uniform.gguf`。
- `expected_size_bytes`: `50439361920`。
- `download_unit`: `ds4-sleepy-k128-alt-gguf-download-20260706T173353Z.service`。
- `download_log`: `/root/lfz/runs/vendor-ds4-16gb/20260706T173353Z-sleepy-k128-alt-gguf-download/download.log`。
- `working_directory`: `/root/lfz/models/DeepSeek-V4-Flash-REAP-K128-uniform-GGUF`，本次不使用 `systemd-run --same-dir`，避免写入 repo root。
- `completion_gate`: `.aria2` sidecar 消失、`stat size == 50439361920`、sha256/header validation 通过后，才允许 France strict 16GB smoke。
- `claim_rule`: 当前只有下载启动记录，没有 correctness/token-rate/SOTA 结论。

## 2026-07-07 执行记录：sleepy K128 alternate GGUF ready validation 通过

- `attempt_id`: `20260707-sleepy-k128-alt-gguf-ready-validation`
- `status`: `ready_for_france_strict_smoke`
- `artifact`: `.Agent/runs/20260705-vendor-ds4-coldstart/alt-gguf-sleepy-k128-ready-validation-20260707.json`
- `prompt_scope`: 未运行 prompt；未使用 `held_out_test_set_v1_locked`。
- `model_path`: `/root/lfz/models/DeepSeek-V4-Flash-REAP-K128-uniform-GGUF/DeepSeek-V4-Flash-REAP-K128-uniform.gguf`。
- `size_and_sha256`: size `50439361920` bytes；sha256 `54927e791ae4e0fbc848e5f2574681b0d8c32721fddd1da82918ece04c0d928b`。
- `header_validation`: GGUF version `3`，`n_kv=64`，`n_tensors=1328`，`general.architecture=deepseek4`，`general.file_type=19`，`expert_count=256`，`expert_used_count=6`，`nextn_predict_layers=1`。
- `tensor_validation`: type counts 为 `F32=492`, `F16=359`, `Q8_0=345`, `IQ2_XXS=86`, `Q2_K=43`, `I32=3`；`output_hc=3`、`tid2eid_weight=3`、gate/up/down expert tensors 各 `43` 个。
- `first_smoke_env`: `LLAMA_DEEPSEEK4_TID2EID_WEIGHT_ALIAS=1`、`LLAMA_DEEPSEEK4_4EXPERT_TENSOR_ALIAS=1`、`GGML_MOE_STREAM=0`。
- `next_action`: 运行 France strict 16GB load/correctness smoke；如果 load 或 correctness 失败，立即 reject，不进入 token-rate benchmark。
- `claim_rule`: 当前只有下载和 metadata ready 结论，没有 correctness/token-rate/SOTA 结论。

## 2026-07-07 执行记录：sleepy K128 France load smoke rejected

- `attempt_id`: `20260707-sleepy-k128-alt-gguf-france-load-smoke`
- `status`: `rejected_load_incompatible_not_sota`
- `artifact`: `.Agent/runs/20260705-vendor-ds4-coldstart/alt-gguf-sleepy-k128-france-load-smoke-reject-20260707.json`
- `prompt_scope`: 只运行 France smoke；未使用 `calibration_dev_set_v1` 调参，未使用 `held_out_test_set_v1_locked`。
- `run_dir`: `/root/lfz/runs/vendor-ds4-16gb/20260706T174958Z-20260707-sleepy-k128-alt-gguf-france-strict-smoke/france-sleepy-k128-alt-cpu40-vram0gb-cpu40-vram0gb`。
- `config`: strict 16GB cgroup、drop_caches、`cpu_moe=40`、`vram_cache=0`、`LLAMA_DEEPSEEK4_TID2EID_WEIGHT_ALIAS=1`、`LLAMA_DEEPSEEK4_4EXPERT_TENSOR_ALIAS=1`、`LLAMA_GGUF_TOKEN_TYPE_UNDEFINED_AS_NORMAL=1`、`GGML_MOE_STREAM=0`。
- `result`: exit status `1`，`eval_tok_s=None`，`prompt_tok_s=None`，`TTFT=None`，`memory_peak_bytes=393523200`，`ram_ok=true`；未进入真实推理。
- `loader_error`: `blk.3.ffn_gate_inp.weight` shape mismatch，vendor 期望 `[4096,256]`，sleepy REAP K128 从 layer 3 开始是 `[4096,128]`。
- `decision`: reject，不进入 token-rate benchmark。该问题是 REAP K128 gate/dataflow 结构差异，不是简单 alias、RAM 或 cache 参数问题。
- `next_action`: 记录并 push 后删除本地 sleepy 大文件释放磁盘；下一候选改为 antirez IQ2XXS chat-v2，因为 header probe 显示其 `output_hc` alias 存在且 `ffn_gate_inp` 为 `[4096,256]`。

## 2026-07-07 执行记录：antirez IQ2XXS alternate GGUF 下载启动

- `attempt_id`: `20260707-antirez-iq2xxs-alt-gguf-download-start`
- `status`: `download_started_not_sota`
- `artifact`: `.Agent/runs/20260705-vendor-ds4-coldstart/alt-gguf-antirez-iq2xxs-download-start-20260707.json`
- `prompt_scope`: 未运行 prompt；未使用 `held_out_test_set_v1_locked`。
- `cleanup`: 已删除本地 rejected sleepy K128 GGUF 大文件释放磁盘；该 candidate 的 sha256、ready validation、reject 记录已 push。
- `candidate`: `antirez/deepseek-v4-gguf` / `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf`。
- `download_path`: `/root/lfz/models/DeepSeek-V4-Flash-antirez-IQ2XXS-chat-v2-GGUF/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf`。
- `expected_size_bytes`: `86720111200`。
- `download_unit`: `ds4-antirez-iq2xxs-alt-gguf-download-20260706T175230Z.service`。
- `download_log`: `/root/lfz/runs/vendor-ds4-16gb/20260706T175230Z-antirez-iq2xxs-alt-gguf-download/download.log`。
- `why_this_candidate`: header probe 显示 `output_hc=3`、`tid2eid_weight=3`、`ffn_gate_inp=[4096,256]`，避免了 0xSero 的 missing `hc_head_base` 和 sleepy 的 K128 gate shape mismatch。
- `completion_gate`: `.aria2` sidecar 消失、`stat size == 86720111200`、sha256/header validation 通过后，才允许 France strict 16GB smoke。
- `claim_rule`: 当前只有下载启动记录，没有 correctness/token-rate/SOTA 结论。

## 2026-07-07 执行记录：antirez IQ2XXS alternate GGUF ready validation 通过

- `attempt_id`: `20260707-antirez-iq2xxs-alt-gguf-ready-validation`
- `status`: `ready_for_france_strict_smoke`
- `artifact`: `.Agent/runs/20260705-vendor-ds4-coldstart/alt-gguf-antirez-iq2xxs-ready-validation-20260707.json`
- `prompt_scope`: 未运行 prompt；未使用 `held_out_test_set_v1_locked`。
- `model_path`: `/root/lfz/models/DeepSeek-V4-Flash-antirez-IQ2XXS-chat-v2-GGUF/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf`。
- `size_and_sha256`: size `86720111200` bytes；sha256 `31598c67c8b8744d3bcebcd19aa62253c6dc43cef3b8adf9f593656c9e86fd8c`。
- `header_validation`: GGUF version `3`，`n_kv=58`，`n_tensors=1328`，`general.architecture=deepseek4`，`general.file_type=19`，`expert_count=256`，`expert_used_count=6`，`nextn_predict_layers=1`。
- `tensor_validation`: type counts 为 `F32=492`, `F16=359`, `Q8_0=345`, `IQ2_XXS=86`, `Q2_K=43`, `I32=3`；`output_hc=3`、`tid2eid_weight=3`、gate/up/down expert tensors 各 `43` 个，全部 `ffn_gate_inp=[4096,256]`。
- `first_smoke_env`: `LLAMA_DEEPSEEK4_TID2EID_WEIGHT_ALIAS=1`、`LLAMA_DEEPSEEK4_4EXPERT_TENSOR_ALIAS=1`、`LLAMA_GGUF_TOKEN_TYPE_UNDEFINED_AS_NORMAL=1`、`GGML_MOE_STREAM=0`。
- `next_action`: 运行 France strict 16GB load/correctness smoke；如果 load 或 correctness 失败，立即 reject，不进入 token-rate benchmark。
- `claim_rule`: 当前只有下载和 metadata ready 结论，没有 correctness/token-rate/SOTA 结论。

## 2026-07-07 执行记录：antirez IQ2XXS France correctness rejected

- `attempt_id`: `20260707-antirez-iq2xxs-alt-gguf-france-correctness`
- `status`: `rejected_correctness_not_sota`
- `artifact`: `.Agent/runs/20260705-vendor-ds4-coldstart/alt-gguf-antirez-iq2xxs-france-correctness-reject-20260707.json`
- `prompt_scope`: 只运行 France smoke；未使用 `calibration_dev_set_v1` 调参，未使用 `held_out_test_set_v1_locked`。
- `model_sha256`: `31598c67c8b8744d3bcebcd19aa62253c6dc43cef3b8adf9f593656c9e86fd8c`。
- `default_smoke`: `/root/lfz/runs/vendor-ds4-16gb/20260706T181656Z-20260707-antirez-iq2xxs-alt-gguf-france-strict-smoke/france-antirez-iq2xxs-alt-cpu40-vram0gb-cpu40-vram0gb`；`eval_tok_s=2.7`，`prompt_tok_s=1.2`，`TTFT=25684.853561ms`，`memory_peak_bytes=16000000000`，`ram_ok=true`，但输出混合中文 thinking、重复 token 和不完整英文，`correctness_ok=false`。
- `template_smoke`: `/root/lfz/runs/vendor-ds4-16gb/20260706T181941Z-20260707-antirez-iq2xxs-alt-gguf-france-deepseek3-reasonoff-smoke/france-antirez-iq2xxs-deepseek3-reasonoff-cpu40-vram0gb-cpu40-vram0gb`；`--chat-template deepseek3 --reasoning off --single-turn` 后 `eval_tok_s=3.4`，`prompt_tok_s=0.9`，`TTFT=25495.83811ms`，`memory_peak_bytes=16000000000`，`ram_ok=true`，但输出模型名/数字噪声，缺少 France/Europe/context，`correctness_ok=false`。
- `decision`: reject，不进入 calibration/dev 或 held-out。虽然 token-rate 数字看起来接近/超过当前 dev baseline，但 France 正确率失败，不能作为 SOTA 或下一阶段泛化候选。
- `cleanup`: 已删除本地 antirez 86.7GB GGUF 释放磁盘；sha256、ready validation、run dir、完整输出和 reject 记录已保留。
- `next_direction`: alternate low-bit route 已连续暴露 load/correctness blocker（0xSero 缺 global hc、sleepy K128 gate shape、antirez correctness 失败）。下一步不应继续盲下大模型；若继续 alternate，需要先做 header+小样本 correctness proof 或回到 native source/dataflow 设计。

## 2026-07-07 hard-bound：generalized sparse pair hotset route rejected

- `attempt_id`: `20260707-generalized-sparse-pair-hard-bound`
- `status`: `rejected_by_generalized_zero_overhead_bound_not_sota`
- `artifact`: `.Agent/runs/20260705-vendor-ds4-coldstart/generalized-sparse-pair-profile-hard-bound-20260707.json`
- `prompt_scope`: 只使用 `calibration_dev_set_v1` 五个 prompt 的 fallback CSV；没有使用 `held_out_test_set_v1_locked`，没有使用 prompt-specific France profile。
- `generated_profiles`: 已生成 generalized top48/top64/top128/top256/top512 sparse pair profile、TSV 和 offset manifest，路径位于 `.Agent/profiles/vendor-ds4/calib-dev-sparse-pair-top{N}-updown-20260707.*`。这些 profile 只作为可复现 hard-bound 输入，不是 SOTA。
- `method`: 对五个 calibration/dev prompt 的 decode fallback CSV 按完整 `(layer, expert)` up/down pair 聚合，按总 decode fallback time 排序；对每个 topN 计算“选中 pair 的 fallback time 可零开销完全消失”时的 token-rate 上界。该上界忽略 staging、H2D/D2H、kernel launch、indexing、同步和 prompt/TTFT 开销，因此是乐观上界。
- `topN_bound_43_layer_sensitivity`:
  - top512：payload `4352.0 MiB`，mean `2.792 tok/s`，min `2.039 tok/s`，五个 dev prompt 全部低于 5。
  - top1024：payload `8704.0 MiB`，mean `3.221 tok/s`，min `2.299 tok/s`，五个 dev prompt 全部低于 5。
  - top2048：payload `17408.0 MiB`，mean `3.973 tok/s`，min `2.764 tok/s`，五个 dev prompt 全部低于 5。
  - top3072：payload `26112.0 MiB`，mean `4.569 tok/s`，min `3.216 tok/s`，只有 France 超过 5；quantum/fibonacci/Japan/climate 仍低于 5。
  - top4096：payload `34816.0 MiB`，mean `4.997 tok/s`，min `3.559 tok/s`，payload 本身已超过 32GB 5090 可用 VRAM 预算，quantum/fibonacci 仍低于 5。
  - all 6485 observed complete pairs：payload `55122.5 MiB`，mean `5.433 tok/s`，min `3.919 tok/s`，Fibonacci 仍低于 5；这与 full up/down removal bound 中 Fibonacci 仍低于 5 的结论一致。
- `decision`: generalized sparse pair/hotset staging 不能作为主路线。原因不是实现细节，而是 expert 命中在泛化 prompt 上过于分散：在可承受 VRAM payload 内覆盖不够；超过可承受 VRAM 后即使零开销也无法让最差 prompt 达到 5 tok/s。
- `source_edit_allowed_by_this_artifact`: `false`。不能因为 France-only 或小 topN profile 看起来可行就写 runtime 行为改动；这会违反“不能 prompt-specific、必须泛化”的任务背景。
- `next_plan`: 回到精确通用 dataflow，目标是消除/大幅压缩所有 prompt 的 up/down CPU fallback，而不是缓存少量历史热 expert。
  1. 定位当前 fallback 的精确触发点：`batch_env_missing + one_name_filter`，确认 `selected_experts`/`weights` 已在 graph 中作为 tensor 传入 `build_expert_mix`，但当前 fused/hot route 没有使用这些 graph tensor 直接完成精确 gather/compute。
  2. 设计 default-off probe：记录每层 decode 时 selected_experts/weights tensor 的 backend、shape、lifetime、是否可在 CUDA kernel 内读取；同时记录 up/down source tensor backend 和 fallback source path。probe 不能改变默认行为、输出或 token-rate。
  3. 若 probe 证明 selected_experts/weights 可被 CUDA path 使用，则实现 compact exact-gather 原型：每 token 每层只为 topK selected experts 建立 compact id list，在 GPU 上直接对 up/down selected experts 做 MMVQ/compute，避免 CPU fallback 和全 expert/hotset staging。
  4. 若 graph tensor 不能直接供 CUDA kernel 使用，则先实现最小 D2H selected id copy hard-bound：每层 topK id/weight 的数据量极小，计算 id copy 和调度开销上限，再决定是否把 selected ids 显式传给 vendor CUDA path。
  5. 每个 source 改动前必须先写 plan；每个实验只用 calibration/dev；held-out locked test set 只在最终 candidate freeze 后运行。出现新的合规 generalized SOTA 时，必须详细记录复现信息并立刻 commit/push 到 `ssd/vendor/deepseek-token-rate-16gb`，随后从 pushed commit 复现。

## 2026-07-07 implementation plan：default-off MUL_MAT_ID exact dataflow probe

- `attempt_id`: `20260707-ds4-exact-mmid-dataflow-probe`
- `status`: `planned_before_source_edit`
- `purpose`: 验证通用 exact up/down CPU fallback 修正是否可行，不做 prompt-specific hotset，不使用 held-out。
- `source_scope`: 只在 `ggml/src/ggml-cuda/ggml-cuda.cu::ggml_cuda_mul_mat_id` 增加 default-off CSV probe；不改变默认路径、kernel 选择、logits、fallback 行为或 token-rate。
- `env`: `DS4_EXACT_MMID_DATAFLOW_PROBE_OUT=<csv>`。未设置或为 `0` 时完全不写文件。
- `csv_fields`: 记录每个 `GGML_OP_MUL_MAT_ID` 的 selected ids/source tensors 信息，包括 tensor name、role(gate/up/down)、chosen CUDA path(`mmvq/mmq/mmf/slow_sync`)、src0/src1/ids/dst backend buffer、type、shape、nbytes、ids bytes、ids 是否在 CUDA buffer、dst tokens/picks、是否 quantized、是否满足 CUDA graph fast path。
- `success_gate`: 用 France calibration/dev smoke 打开 probe，输出必须正确、默认关闭时 build 不变；probe CSV 应能回答 selected experts ids 是否已在 CUDA 可读 buffer 内，以及 up/down fallback 是因为 source tensor placement/type/path 还是 ids/dataflow 不可用。
- `claim_rule`: 该 probe 是诊断提交，不是 SOTA；若输出/性能异常则回退 probe 或标为 rejected。后续任何行为改变必须基于该 probe 的证据重新写 plan。

## 2026-07-07 执行记录：exact MMID + CPU fallback source probe

- `attempt_id`: `20260707-ds4-exact-mmid-cpu-source-probe-france-smoke`
- `status`: `completed_probe_smoke_not_sota`
- `artifact`: `.Agent/runs/20260705-vendor-ds4-coldstart/exact-mmid-cpu-source-probe-france-smoke-20260707.json`
- `source_change`: 增加两个 default-off probe，不改变默认 logits/path：
  - `DS4_EXACT_MMID_DATAFLOW_PROBE_OUT=<csv>`：记录 CUDA backend `GGML_OP_MUL_MAT_ID` 的 path、src0/src1/ids/dst backend/type/shape。
  - `GGML_MOE_FALLBACK_SOURCE_PROBE_OUT=<csv>`：记录 CPU fallback 残留 rows 的 src0/src1/ids/dst backend/type/shape/reason，不改变既有 `fallback_reason_profile.csv` 格式。
- `validation_run`: `/root/lfz/runs/vendor-ds4-16gb/20260706T184846Z-20260707-exact-mmid-cpu-source-probe-france/france-exact-mmid-cpu-source-probe-cpu40-vram0gb`
- `config`: vendor DeepSeek native GGUF、strict cold 16GB cgroup、`cpu_moe=40`、`vram_cache=0`、gate one-stream cache only、`-n 96` France smoke；这是 calibration diagnostic，不是 SOTA，不使用 held-out。
- `result`: exit `0`，`eval_tok_s=2.5`，`prompt_tok_s=0.9`，`TTFT=36704.18 ms`，`memory_peak_bytes=16000000000`，`memory_file_bytes=15062511616`，`ram_ok=true`，`correctness_ok=true`。France 输出语义正确但因 `-n 96` 被截断在一句中部；该 run 只验证 probe，不作为 SOTA。
- `cuda_mmid_probe`: `882` rows，全部 `path=mmvq`，角色为 gate/up/down 各 `294`；所有观测到的 `src0_buft=CUDA0`、`ids_buft=CUDA0`。这说明 CUDA backend 只看到了已经被调度到 CUDA 的 MMID，不能代表 CPU fallback 残留。
- `fallback_source_probe`: `6176` rows，残留耗时为 `up/decode=9880.61 ms`、`down/decode=6639.33 ms`、`up/prompt=3782.40 ms`、`down/prompt=4247.47 ms`；所有残留 rows 的 `src0_buft=CPU_Mapped`，`src1_buft=CUDA_Host`，`ids_buft=CUDA_Host`，`dst_buft=CUDA_Host`。
- `bottleneck_update`: 当前合规路径的 up/down CPU fallback 不是因为 selected ids 在 CUDA graph 内完全不可得，而是因为这些 up/down tensor 没有被允许进入 one-stream CUDA path：`single_reason=one_name_filter`，`batch_reason=batch_env_missing`。gate 已通过 name_filter/cache 路线覆盖，up/down 被 name_filter 排除后落在 CPU_Mapped + CUDA_Host staging 的 CPU fallback。
- `next_plan`: 先做 default-off up/down one-stream enablement hard-bound，而不是 prompt-specific hotset。
  1. 在 calibration/dev 上用 `GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps,ffn_up_exps,ffn_down_exps` 或等价 role filter 运行小 smoke，观察 `fallback_source_probe` 是否显著下降，以及 correctness/TTFT/RAM 是否受影响。
  2. 若 up/down 进入 one-stream 后 correctness 正常但 token-rate 不升或下降，继续拆分 one-stream 内部时间：CPU_Mapped expert read、H2D、src1 staging、kernel、D2H/scatter，确认是不是每 expert 单独搬运导致吞吐差。
  3. 如果单 expert one-stream 对 up/down 太慢，再设计 compact exact gather/batched route：同一 layer 同一 token 的 topK up/down 统一处理，减少 per-expert launch/copy/scatter；理论上限按 fallback_source_probe 的 up/down decode ms 计算。
  4. 所有实验仍只用 calibration/dev；held-out locked test set 只在 candidate freeze 后测试。若产生合规 generalized SOTA，必须完整记录复现信息并立刻 push 到 `ssd/vendor/deepseek-token-rate-16gb`。
