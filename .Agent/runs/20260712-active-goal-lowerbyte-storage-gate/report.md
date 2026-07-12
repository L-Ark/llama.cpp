# Kimi lower-byte target and storage gate

Timestamp: 2026-07-12 CST

Branch: `vendor/kimi-deepseek-41d205-additive`

Input profile root:
`/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-copyio-n32-005717`

This is a planning and feasibility report only. It does not change runtime
behavior and does not claim SOTA.

## Artifacts

- optimistic exact-byte ceiling for `2 tok/s`:
  `.Agent/runs/20260712-active-goal-lowerbyte-storage-gate/exact-byte-scheduler-ceiling-2tps.md`
- optimistic exact-byte ceiling for `5 tok/s`:
  `.Agent/runs/20260712-active-goal-lowerbyte-storage-gate/exact-byte-scheduler-ceiling-5tps.md`
- profiled-floor byte target for `2 tok/s`:
  `.Agent/runs/20260712-active-goal-lowerbyte-storage-gate/byte-target-2tps.md`
- profiled-floor byte target for `5 tok/s`:
  `.Agent/runs/20260712-active-goal-lowerbyte-storage-gate/byte-target-5tps.md`
- Hugging Face current candidate size probe:
  `.Agent/runs/20260712-active-goal-lowerbyte-storage-gate/hf-candidate-size-probe.json`
- local storage inventory:
  `.Agent/runs/20260712-active-goal-lowerbyte-storage-gate/storage-inventory.txt`

## Byte Target

Two bounds are now recorded because they answer different questions.

The optimistic exact-byte ceiling uses the prior fixed all-hit floor of
`40.1 ms/token`. It is useful as a best-case IO scheduling upper bound:

| target | France required ratio | Intelligence required ratio | mean required ratio |
|---|---:|---:|---:|
| `2 tok/s` | `0.708x` | `0.741x` | `0.725x` |
| `5 tok/s` | `0.246x` | `0.258x` | `0.252x` |

The profiled-floor bound estimates the all-hit MoE floor from observed all-hit
profile rows. It is stricter and should be treated as the admission warning:

| target | France required ratio | Intelligence required ratio | median required ratio |
|---|---:|---:|---:|
| `2 tok/s` | `0.41x` | `0.59x` | `0.50x` |
| `5 tok/s` | `0.00x` | `0.11x` | `0.05x` |

Interpretation:

- Pure scheduler/co-submit remains rejected as a primary route. With current
  exact bytes, even perfect movement at `10.3 GiB/s` is below `2 tok/s`.
- A complete lower-byte model near `0.50x` is a plausible `2 tok/s` smoke
  candidate, but not a guaranteed SOTA. It passes the optimistic `2 tok/s`
  byte gate and is borderline under the profiled-floor gate.
- No current complete GGUF candidate is a `5 tok/s` solution by itself. `5 tok/s`
  still needs a much smaller effective representation, stronger prediction, or
  additional VRAM-resident reuse.

## Storage State

Current local storage from `df -B1`:

- available on `/root/lfz`: `113131085824 bytes` (`105.35 GiB`);
- current IQ3_S model directory:
  `/root/lfz/models/Kimi-K2.7-Code-GGUF-IQ3_S`,
  `405392165559 bytes`;
- old France-only expert pack:
  `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-france-l12-upgate-v2.expert-pack`,
  `175133036544 bytes`;
- old trace-first pack:
  `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-tracefirst-n64-20260630.expert-pack`,
  `79544299520 bytes`;
- current general-dev budget overlay:
  `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-general-dev-budget16-overlay.expert-pack`,
  `17179111424 bytes`.

Current free space cannot download any complete lower-byte candidate while
keeping a `50 GiB` reserve.

If the old France-only pack is removed, available space becomes approximately
`288264122368 bytes` (`268.43 GiB`). That is enough for `i1-IQ1_S` with a
`50 GiB` reserve.

## Candidate Gate

Candidate sizes were probed by metadata only; no model payload was downloaded.

| candidate | size bytes | ratio vs current IQ3_S | current storage gate | gate after deleting old France pack | notes |
|---|---:|---:|---|---|---|
| `i1-IQ1_S` | `204430872480` | `0.504x` | blocked, short `144986877856` bytes with `50 GiB` reserve | pass, leaves `83833249888` bytes | best complete-model `2 tok/s` smoke candidate |
| `i1-IQ1_M` | `227937931680` | `0.562x` | blocked, short `168493937056` bytes | pass, leaves `60326190688` bytes | weaker byte reduction, tight reserve |
| `i1-IQ2_XXS` | `267116363680` | `0.659x` | blocked | still blocked with `50 GiB` reserve | too large for the target |
| `deep55 pruned` | `202657355840` | `0.500x` | blocked, short `143213361216` bytes | pass, leaves `85606766528` bytes | pruned model, quality/identity risk |
| `unsloth UD-IQ1_M` | `303909170464` | `0.750x` | blocked | blocked | too large |
| `huihui UD-IQ1_M-MXFP4` | `303627382432` | `0.749x` | blocked | blocked | different abliterated model, too large |

## Decision

For the next runtime experiment, only `i1-IQ1_S` is worth a complete-model
smoke under the current task constraints:

- It is close to the `0.50x` profiled-floor `2 tok/s` median gate.
- It is well below the optimistic `0.725x` `2 tok/s` byte gate.
- It is still far above the `0.252x` optimistic `5 tok/s` byte gate, so it
  should not be framed as a `5 tok/s` solution.

Before the smoke can run, storage must be freed. The old France-only pack is the
only single cleanup candidate large enough to unblock `i1-IQ1_S`, but it cannot
be deleted blindly.

Reference audit after this report found that some repro/historical scripts still
reference the France pack path, including:

- `.Agent/run-tools/kimi-general-prompt-repro.sh`;
- `.Agent/run-tools/kimi_phase0_io_trace_remote.sh`;
- historical run scripts under `/root/lfz/runs/vendor-kimi-token-rate`;
- old side worktrees under `/root/lfz`.

Therefore the cleanup gate is:

1. first migrate the current accepted repro path away from the France-only pack
   or record an equivalent prompt-general replacement;
2. confirm the current SOTA rollback/repro command no longer requires that file;
3. then delete or archive the old France-only pack to unlock the `i1-IQ1_S`
   smoke.

Until that migration is done, the storage gate remains blocked even though the
byte math says deleting the pack would be sufficient.

## Reproduce

```bash
ROOT=/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-copyio-n32-005717
RUN=.Agent/runs/20260712-active-goal-lowerbyte-storage-gate

python3 .Agent/run-tools/kimi_exact_byte_scheduler_ceiling.py \
  --profile-root "$ROOT" \
  --out-json "$RUN/exact-byte-scheduler-ceiling-2tps.json" \
  --out-md "$RUN/exact-byte-scheduler-ceiling-2tps.md" \
  --bandwidth-gib-s 10.3 \
  --all-hit-floor-ms-per-token 40.1 \
  --target-tok-s 2.0

python3 .Agent/run-tools/kimi_exact_byte_scheduler_ceiling.py \
  --profile-root "$ROOT" \
  --out-json "$RUN/exact-byte-scheduler-ceiling-5tps.json" \
  --out-md "$RUN/exact-byte-scheduler-ceiling-5tps.md" \
  --bandwidth-gib-s 10.3 \
  --all-hit-floor-ms-per-token 40.1 \
  --target-tok-s 5.0

python3 .Agent/run-tools/kimi_byte_reduction_target_bound.py \
  --runs-root "$ROOT" \
  --out "$RUN/byte-target-2tps.md" \
  --target-tps 2.0 \
  --peak-gib-s 10.3 \
  --evidence-scope current-copyio-dev-generalized

python3 .Agent/run-tools/kimi_byte_reduction_target_bound.py \
  --runs-root "$ROOT" \
  --out "$RUN/byte-target-5tps.md" \
  --target-tps 5.0 \
  --peak-gib-s 10.3 \
  --evidence-scope current-copyio-dev-generalized

python3 .Agent/run-tools/kimi_hf_candidate_size_probe.py \
  --timeout-s 45 \
  --out-json "$RUN/hf-candidate-size-probe.json"

df -B1 /root/lfz /tmp
du -sb /root/lfz/models/* 2>/dev/null | sort -n
du -sb /root/lfz/runs/ik_llama/kimi-iq3s-assets/* 2>/dev/null | sort -n
```
