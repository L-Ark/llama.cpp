# GP104 Remote Cleanup Inventory

Date: `2026-07-08`

Branch: `vendor/kimi-speculative-general-token-rate-16gb`

Status: completed non-destructive inventory only. No remote files were deleted.

## Purpose

GP103 found that the complete `i1-IQ1_S` GGUF candidate is the most direct
full-model lower-byte smoke to try next, but the remote disk did not satisfy
the safety gate while preserving the current IQ3_S SOTA assets.

This run inventories remote disk usage and identifies cleanup candidates that
could be deleted only after explicit user approval.

## Disk Gate

- Current remote free space:
  - `243991724032 bytes` (`227.235 GiB`)
- `i1-IQ1_S` candidate size:
  - `204430872480 bytes` (`190.391 GiB`)
- Required reserve after download:
  - `50 GiB`
- Required free space before download:
  - `258117963680 bytes` (`240.391 GiB`)
- Additional space needed:
  - `14126239648 bytes` (`13.156 GiB`)

## Preserve List

These paths are part of the current Kimi reproduction surface or are too
important/ambiguous to clean without a separate explicit decision:

| path | bytes | reason |
|---|---:|---|
| `/root/lfz/models/Kimi-K2.7-Code-GGUF-IQ3_S` | `405392165559` | current IQ3_S model |
| `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-france-l12-upgate-v2.expert-pack` | `175133036544` | current main expert pack used by scripts |
| `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-l1l2down-overlay.expert-pack` | `4844539904` | current overlay expert pack used by scripts |
| `/root/lfz/tmp/kimi-stage2m-align` | `1254787626` | current remote worktree |
| `/root/lfz/runs/vendor-kimi-token-rate/20260707-020500Z-gp4-postcommit-test-n96-profile` | `117005201` | accepted held-out GP4 run evidence |

Remote script grep confirms current Kimi repro scripts reference only:

- `GGML_MOE_EXPERT_PACK=/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-france-l12-upgate-v2.expert-pack`
- `GGML_MOE_EXPERT_PACK_OVERLAY=/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-l1l2down-overlay.expert-pack`

## Cleanup Candidates

These are candidates only. They were not deleted.

| recommendation | path | bytes | GiB | reason |
|---|---|---:|---:|---|
| minimal candidate | `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-general-dev-budget16-overlay.expert-pack` | `17179111424` | `15.999` | not referenced by current scripts; enough by itself to pass the IQ1_S + 50 GiB gate |
| larger candidate | `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-tracefirst-n64-20260630.expert-pack` | `79544299520` | `74.087` | historical trace-first pack; not referenced by current scripts |
| optional old overlay | `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-l1l2down-l4l60missing-overlay.expert-pack` | `7689551872` | `7.161` | rejected/old overlay; not enough by itself |
| optional old overlay | `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-phase7gz-combined-overlay.expert-pack` | `5042724864` | `4.696` | old combined overlay; not enough by itself |
| optional temp pack | `/root/lfz/runs/ik_llama/kimi-iq3s-assets/tmp-hot-upgate-pair-smoke.expert-pack` | `343846912` | `0.320` | temp smoke pack |
| optional old overlay | `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-phase7gz-missing-down-overlay.expert-pack` | `198184960` | `0.185` | old missing-down overlay |
| optional run artifact | `/root/lfz/runs/vendor-kimi-token-rate/20260705-7ir-overlay-firstuse` | `4844588365` | `4.512` | large old run directory; not enough by itself |

## Minimal Cleanup Math

Deleting only the minimal candidate would give:

- free before IQ1_S:
  - `261170835456 bytes` (`243.234 GiB`)
- free after IQ1_S:
  - `56739962976 bytes` (`52.843 GiB`)
- safety gate:
  - pass, because `52.843 GiB >= 50 GiB`

Deleting the old trace-first pack would give:

- free before IQ1_S:
  - `323536023552 bytes` (`301.316 GiB`)
- free after IQ1_S:
  - `119105151072 bytes` (`110.925 GiB`)
- safety gate:
  - pass with much larger margin, but it is a larger cleanup decision.

## Commands Used

```bash
ssh -p 51056 root@92.180.27.82 'df -B1 /root /root/lfz 2>/dev/null; find /root/lfz/models -maxdepth 2 -mindepth 1 -type d -exec du -sb {} + 2>/dev/null | sort -nr | head -80'
ssh -p 51056 root@92.180.27.82 'find /root/lfz/runs/ik_llama/kimi-iq3s-assets -maxdepth 1 -type f -printf "%s\t%p\n" 2>/dev/null | sort -nr | head -80'
ssh -p 51056 root@92.180.27.82 'find /root/lfz/runs/vendor-kimi-token-rate -maxdepth 2 -mindepth 1 -type d -exec du -sb {} + 2>/dev/null | sort -nr | head -120'
ssh -p 51056 root@92.180.27.82 'find /root/lfz/tmp -maxdepth 2 -mindepth 1 -type d -exec du -sb {} + 2>/dev/null | sort -nr | head -120'
ssh -p 51056 root@92.180.27.82 'cd /root/lfz/tmp/kimi-stage2m-align 2>/dev/null && grep -R "general-dev-budget16\|tracefirst-n64\|france-l12-upgate-v2\|l1l2down-overlay\|l1l2down-l4l60missing\|phase7gz-combined\|EXPERT_PACK\|expert-pack" -n scripts .Agent/run-tools .Agent/plans 2>/dev/null | head -200'
```

## Decision

Lower-quant smoke is no longer blocked on finding a cleanup candidate. It is
blocked on explicit user approval to delete or move at least one candidate,
preferably the minimal `kimi-iq3s-general-dev-budget16-overlay.expert-pack`.

After approval and cleanup, the next step should be the already planned
`i1-IQ1_S` full-model download/smoke with strict cold-start, 16 GB host RAM,
France quality, TTFT, and prompt-general validation gates.
