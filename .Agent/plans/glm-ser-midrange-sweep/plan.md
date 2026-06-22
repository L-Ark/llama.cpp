# GLM SER Midrange Sweep Plan

## Goal

Find whether GLM-5.1 on RTX 5090 under strict 2 GB host RAM can trade a small
accuracy drop for the largest possible speedup using the existing training-free
`--smart-expert-reduction` mechanism.

## Required Constraints

- Every target model run must use `MemoryMax=2G` and `MemorySwapMax=0`.
- RAM tier remains disabled: `GGML_MOE_RAM_TIER_MIB=0`.
- Use the current 5090 32 GB VRAM strict line: top8 runtime profile,
  `GGML_MOE_VRAM_CACHE_MIB=12288`, direct expert-pack reads.
- Do not implement runtime routing code until the sweep identifies a stable
  candidate and a clear mechanism.
- Record all commands, outputs, summaries, failures, and final conclusions in
  this file.

## Baseline From Previous Task

Full 4-item smoke under the same 2 GB line:

| config | accuracy | direct_reads | VRAM hit | prompt tok/s | eval tok/s | total_ms | wall_s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SER off | 3/4 | 22578 | 14.7% | 0.904 | 0.451 | 642134.90 | 656.51 |
| SER 1,0.95 | 3/4 | 20580 | 21.5% | 0.978 | 0.590 | 607250.91 | 619.24 |

Known rejected points from the previous task:

- `SER=1,1.0` and `SER=1,1.1` produced large speedups but visibly corrupted
  free-form output.
- `SER<=1,0.8` had little or no traffic effect in the earlier free-form test.

## Sweep Design

Phase 1, fast single-item screen:

- Dataset: `/root/lfz/data/glm_resource_eval/processed/smoke_one.jsonl`.
- SER points: `1,0.96`, `1,0.97`, `1,0.975`, `1,0.98`, `1,0.99`.
- Required metrics: accuracy, output letter, direct reads, VRAM hit, RAM hit,
  prompt/eval tok/s, total_ms, wall_s, read_failures.
- Accept for phase 2 if the answer is still correct and eval tok/s improves
  materially over `SER=1,0.95` on the same single-item task, or if it is the
  fastest non-corrupt point worth verifying under a small accuracy-loss policy.

Phase 2, 4-item smoke verification:

- Dataset: `/root/lfz/data/glm_resource_eval/processed/smoke.jsonl`.
- Run the best 1-2 phase-1 candidates.
- Primary objective: maximize eval tok/s and wall-time reduction while allowing
  at most a small accuracy drop from the current smoke baseline.
- Since smoke has only 4 items, any drop from `3/4` to `2/4` is a large 25-point
  drop. Treat `2/4` as a warning candidate, not a default recommendation. It
  may justify daily-set testing only if speedup is very large.

## Progress

- 2026-06-12: Read `/root/lfz/ik_llama/.Agent/Agent.md`.
- 2026-06-12: Created this task folder. Worktree already contains uncommitted
  `.Agent` artifacts from the prior dynamic-k/cache-aware task; preserve them.
- 2026-06-12: GPU idle before new runs (`223 MiB` used, `31887 MiB` free,
  `0%` util). No model process running.
- 2026-06-12: Completed phase-1 `smoke_one` screen for SER `1,0.96`,
  `1,0.97`, `1,0.975`, `1,0.98`, and `1,0.99` under
  `MemoryMax=2G`, `MemorySwapMax=0`.
- 2026-06-12: Selected `SER=1,0.975` for phase-2 4-item smoke validation.
  It is the highest tested threshold that still answered the single smoke item
  correctly. `SER=1,0.98` and `SER=1,0.99` were faster but already wrong on
  the single item, so they are not good default candidates.
- 2026-06-12: Completed phase-2 `SER=1,0.975` 4-item smoke run. It improved
  throughput and read count but dropped accuracy from the prior `3/4` line to
  `2/4`, so it is not acceptable as a small-accuracy-loss default.
- 2026-06-12: Started a more conservative phase-2 `SER=1,0.96` 4-item smoke
  validation. This checks whether the useful region is closer to the previous
  `SER=1,0.95` point.
- 2026-06-12: Completed phase-2 `SER=1,0.96` 4-item smoke run. It preserved
  the prior `3/4` smoke accuracy and improved speed modestly over `SER=1,0.95`.
  This is the best current default candidate from this sweep.
- 2026-06-12: User ended the task before additional fine sweep points such as
  `SER=1,0.97` on 4-item smoke. Final recommendation remains `SER=1,0.96`
  as the best measured quality-preserving point. Reproduction instructions are
  in `REPRODUCE.md`.

## Results

### Phase 1: `smoke_one`

Command template:

```bash
systemd-run --same-dir --wait --collect --quiet \
  -p MemoryMax=2G -p MemorySwapMax=0 \
  python3 /root/lfz/ik_llama/.Agent/plans/glm-dynamic-k-cache-aware-routing/runs/ser-local-2gb/run_smoke_accuracy.py \
  --inner-no-systemd \
  --dataset /root/lfz/data/glm_resource_eval/processed/smoke_one.jsonl \
  --output-dir /root/lfz/ik_llama/.Agent/plans/glm-ser-midrange-sweep/runs/accuracy \
  --label smoke-one-ser-<point> \
  --ser <point>
```

| SER | accuracy | output | direct_reads | VRAM hit | prompt tok/s | eval tok/s | total_ms | wall_s | read_failures |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1,0.96` | 1/1 | B | 4421 | 30.3% | 0.686 | 0.861 | 119645.46 | 122.81 | 0 |
| `1,0.97` | 1/1 | B | 4354 | 25.9% | 0.710 | 0.664 | 111323.09 | 114.61 | 0 |
| `1,0.975` | 1/1 | B | 4092 | 28.3% | 0.725 | 0.810 | 111552.06 | 114.00 | 0 |
| `1,0.98` | 0/1 | C | 3864 | 25.1% | 0.811 | 1.027 | 100273.15 | 102.93 | 0 |
| `1,0.99` | 0/1 | `,` | 3829 | 8.2% | 0.901 | 0.893 | 95433.68 | 97.58 | 0 |

Interpretation:

- Yes, speed improves as the threshold rises, mainly through fewer expert-pack
  direct reads and faster prompt processing.
- The usable region appears narrow. `1,0.98` already changes the answer from
  `B` to `C` on the first smoke item, so the current no-fine-tune SER knob has
  a sharp quality cliff.
- `1,0.975` is the best phase-2 candidate because it keeps the single answer
  correct while reducing direct reads from the prior single-item `SER=1,0.95`
  result (`5172`) to `4092`.

### Phase 2: 4-item `smoke`

Command template:

```bash
systemd-run --same-dir --wait --collect --quiet \
  -p MemoryMax=2G -p MemorySwapMax=0 \
  python3 /root/lfz/ik_llama/.Agent/plans/glm-dynamic-k-cache-aware-routing/runs/ser-local-2gb/run_smoke_accuracy.py \
  --inner-no-systemd \
  --dataset /root/lfz/data/glm_resource_eval/processed/smoke.jsonl \
  --output-dir /root/lfz/ik_llama/.Agent/plans/glm-ser-midrange-sweep/runs/accuracy \
  --label smoke4-ser-<point> \
  --ser <point>
```

| SER | accuracy | direct_reads | VRAM hit | prompt tok/s | eval tok/s | total_ms | wall_s | read_failures | interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `1,0.96` | 3/4 | 18938 | 25.8% | 1.021 | 0.641 | 577421.79 | 589.08 | 0 | Best current quality-preserving candidate. |
| `1,0.975` | 2/4 | 16409 | 27.0% | 1.094 | 0.846 | 546904.13 | 559.95 | 0 | Faster, but accuracy loss is too large for default use. |

`SER=1,0.96` scoring details:

| sample | expected | output | correct |
| --- | --- | --- | --- |
| `ceval_computer_network_0009` | D | D | yes |
| `cmmlu_chinese_civil_service_exam_0098` | D | D | yes |
| `ceval_computer_network_0010` | B | B | yes |
| `cmmlu_chinese_civil_service_exam_0150` | C | D | no |

`SER=1,0.975` scoring details:

| sample | expected | output | correct |
| --- | --- | --- | --- |
| `ceval_computer_network_0009` | D | D | yes |
| `cmmlu_chinese_civil_service_exam_0098` | D | C | no |
| `ceval_computer_network_0010` | B | B | yes |
| `cmmlu_chinese_civil_service_exam_0150` | C | D | no |

Compared with the prior `SER=1,0.95` 4-item smoke (`3/4`, `20580`
direct reads, `21.5%` VRAM hit, `0.978` prompt tok/s, `0.590` eval tok/s,
`607250.91` total_ms, `619.24` wall_s), `SER=1,0.975` reduces direct reads by
20.3%, improves eval tok/s by 43.5%, and reduces wall time by 9.6%, but the
accuracy drop from `3/4` to `2/4` is not a small loss on this test.

Compared with the prior `SER=1,0.95` 4-item smoke, `SER=1,0.96` keeps the same
`3/4` accuracy while reducing direct reads by 8.0%, improving eval tok/s by
8.7%, and reducing wall time by 4.9%. Compared with SER off, it keeps the same
`3/4` accuracy while reducing direct reads by 16.1%, improving eval tok/s by
42.2%, and reducing wall time by 10.3%.

## Current Conclusion

The answer is partly yes, but not yet at the "small accuracy loss for large
speedup" level with the current training-free SER knob:

- `SER=1,0.96` is usable as a conservative candidate. It preserves smoke
  accuracy and gives a modest but real improvement.
- `SER=1,0.975` demonstrates the possible speed direction, but the quality
  cliff appears quickly: `3/4` to `2/4` is too large a drop for a default.
- The next mechanism should be cache/residency-aware routing rather than a
  global threshold only. A global SER threshold does not know whether a skipped
  expert is semantically important for a specific token/layer.

## Final Task Status

Ended by user request on 2026-06-12. No further SER fine sweep points were run
after `SER=1,0.96` and `SER=1,0.975` on the 4-item smoke set. The task output
is an experimental recommendation, not a source-code change:

- default candidate: `SER=1,0.96`;
- rejected faster candidate: `SER=1,0.975`;
- required strict runtime constraint: host RAM capped at 2 GB, swap disabled,
  RAM tier disabled.
