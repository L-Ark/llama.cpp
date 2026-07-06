# GP51 Remaining Fallback Eligibility Audit

Timestamp: `2026-07-07T08:18:00+08:00`.

Branch: `vendor/kimi-speculative-general-token-rate-16gb`.

Input runs:

- Baseline:
  `/root/lfz/tmp/runs/20260707-gp50-partial-iouring/baseline_dev_france_n32`
- GP50:
  `/root/lfz/tmp/runs/20260707-gp50-partial-iouring/dev_france_n32_batch_on`

This is an analysis-only step. No runtime behavior changed.

## Commands

Decode fallback by tensor and type:

```bash
run=/root/lfz/tmp/runs/20260707-gp50-partial-iouring/dev_france_n32_batch_on
awk -F, 'NR>1 && $8=="decode" {
  key=$12 ",type=" $7
  us[key]+=$4
  cnt[key]+=$2
  calls[key]+=$3
} END {
  for (k in us) printf "%.3f\t%d\t%d\t%s\n", us[k]/1000, cnt[k], calls[k], k
}' "$run/fallback-profile.csv" | sort -nr | head -40
```

Name-profile decode fallback where CUDA batch was not eligible:

```bash
grep "kimi_cpu_moe_name_profile" "$run/stderr.txt" |
perl -ne 'if (/name=(\S+).*batch_eligible=(\d+) batch_accept=(\d+) batch_decline=(\d+) decode_calls=(\d+) decode_total=([0-9.]+) ms\/call decode_fallback=([0-9.]+) ms\/call/) {
  $total=$5*$7;
  printf "%.3f\t%s\tbe=%s\tacc=%s\tdec=%s\tdec_fb_ms=%s\n", $total,$1,$2,$3,$5,$7 if $2==0 && $5>0
}' | sort -nr | head -40
```

Decode wall by up/gate and down batch type:

```bash
awk -F, 'NR>1 && $2=="decode" {
  key="up=" $5 "/gate=" $6
  wall[key]+=$26
  stage[key]+=$14
  up[key]+=$16
  gate[key]+=$17
  misses[key]+=$9+$11
  jobs[key]+=$12+$13
} END {
  for (k in wall) printf "%.3f\t%.3f\t%.3f\t%.3f\t%d\t%d\t%s\n",
    wall[k], stage[k], up[k], gate[k], misses[k], jobs[k], k
}' "$run/up-gate-profile.csv" | sort -nr

awk -F, 'NR>1 {
  key="type=" $3
  wall[key]+=$13
  stage[key]+=$8
  kernel[key]+=$10
  misses[key]+=$6
  jobs[key]+=$7
} END {
  for (k in wall) printf "%.3f\t%.3f\t%.3f\t%d\t%d\t%s\n",
    wall[k], stage[k], kernel[k], misses[k], jobs[k], k
}' "$run/down-batch-profile.csv" | sort -nr
```

## Findings

The remaining decode paths with `batch_eligible=0` are limited to seven
`type=2` down tensors:

| Decode fallback ms | CSV rows | Calls | Tensor |
| ---: | ---: | ---: | --- |
| 439.360 | 248 | 248 | `blk.6.ffn_down_exps.weight,type=2` |
| 357.944 | 248 | 248 | `blk.8.ffn_down_exps.weight,type=2` |
| 326.200 | 248 | 248 | `blk.7.ffn_down_exps.weight,type=2` |
| 325.600 | 248 | 248 | `blk.18.ffn_down_exps.weight,type=2` |
| 324.768 | 248 | 248 | `blk.10.ffn_down_exps.weight,type=2` |
| 305.496 | 248 | 248 | `blk.9.ffn_down_exps.weight,type=2` |
| 246.424 | 248 | 248 | `blk.15.ffn_down_exps.weight,type=2` |

Name-profile independently reports the same seven tensors:

| Estimated total ms | Tensor | Decode calls | Decode fallback ms/call |
| ---: | --- | ---: | ---: |
| 439.456 | `blk.6.ffn_down_exps.weight` | 31 | 14.176 |
| 358.050 | `blk.8.ffn_down_exps.weight` | 31 | 11.550 |
| 326.306 | `blk.7.ffn_down_exps.weight` | 31 | 10.526 |
| 325.717 | `blk.18.ffn_down_exps.weight` | 31 | 10.507 |
| 324.880 | `blk.10.ffn_down_exps.weight` | 31 | 10.480 |
| 305.598 | `blk.9.ffn_down_exps.weight` | 31 | 9.858 |
| 246.512 | `blk.15.ffn_down_exps.weight` | 31 | 7.952 |

Total remaining decode fallback from `batch_eligible=0` down tensors is about
`2.326 s / 31 tokens`, or `75 ms/token`.

Other decode wall sums from CSV:

| Area | Wall ms | Stage ms | Compute-ish ms | Misses/jobs |
| --- | ---: | ---: | ---: | --- |
| up/gate `type=22/22` | 3508.889 | 17.509 | up `3375.139`, gate `106.456` | 5105 / 5105 |
| up/gate `type=18/18` | 2877.195 | 10.297 | up `1559.851`, gate `1268.603` | 3004 / 3004 |
| down batch `type=11` | 2260.062 | 2031.841 | kernel `163.863` | 1814 / 1814 |
| down batch `type=23` | 2206.576 | 2131.144 | kernel `36.490` | 1707 / 1707 |

## Interpretation

Supporting the seven `type=2` down tensors on the CUDA batch path can at most
recover about `75 ms/token` before accounting for replacement GPU/H2D cost. From
the GP50 rate of `1.25 tok/s` (`800 ms/token`), a perfect removal would only
raise the rate to roughly:

```text
1 / (0.800 - 0.075) = 1.38 tok/s
```

The realistic gain is lower because those tensors would still need staging and
GPU compute. Therefore Q4_0 down batch eligibility is useful cleanup but is not
the next highest-leverage path toward `5 tok/s`.

The larger remaining wall components are:

- up/gate compute/wait path: about `6.386 s / 31 tokens`;
- down staged batch path: about `4.467 s / 31 tokens`;
- expert-pack iouring wait remains large in aggregate and queue depth is still
  usually below the pure IO peak.

## Decision

Do not implement Q4_0 down CUDA batch support as the immediate next step unless
it is bundled with a broader down-path cleanup. The next implementation plan
should target one of:

- reducing up/gate wall time for `type=22` and `type=18`;
- improving sustained iouring queue depth / earlier known-job submission;
- reducing down stage time for `type=11` and `type=23`.

No SOTA or token-rate improvement is claimed for GP51.
