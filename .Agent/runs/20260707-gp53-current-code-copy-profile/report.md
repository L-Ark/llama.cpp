# GP53 Current-Code Slow Dev Copy Profile

Timestamp: `2026-07-07T09:18:00+08:00`.

Branch: `vendor/kimi-speculative-general-token-rate-16gb`.

Remote repo: `/root/lfz/tmp/vendor-kimi-speculative-gp33`.

HEAD: `5144c977c`.

## Goal

GP52 showed that slow dev prompts have high direct-read ratio and a large
residual decode component. GP53 runs current GP50 code on a slow dev prompt with
copy profiling to identify the concrete copy paths behind that residual.

No runtime code changed in this step.

## Commands

The first `N=32` run was useful for attribution but failed quality because the
answer was truncated before the final value. The accepted attribution run used
`N=48`.

```bash
repo=/root/lfz/tmp/vendor-kimi-speculative-gp33
cd "$repo"
ln -sfn build-gp50-runtime build-cuda-batch
run=/root/lfz/tmp/runs/20260707-gp53-current-code-copy-profile/dev_linear_equation_n48
systemd-run --wait --collect --same-dir \
  -p MemoryMax=15900000000 -p MemorySwapMax=0 \
  env REPO="$repo" RUN="$run" N=48 PROFILE=1 COPY_PROFILE=1 \
      PROMPT_ID=dev_linear_equation \
      PROMPT_USER_TEXT="Solve: if x + 3 = 10, what is x?" \
      QUALITY_KEYWORDS="7|seven" \
      .Agent/run-tools/kimi-general-prompt-repro.sh
```

Aggregation command:

```bash
run=/root/lfz/tmp/runs/20260707-gp53-current-code-copy-profile/dev_linear_equation_n48
awk -F, 'NR>1 && $8==0 {
  key=$2 ",pack=" $6 "," $3
  n[key]++
  bytes[key]+=$5
  host[key]+=$10
  iow[key]+=$11
  h2d[key]+=$13
  wall[key]+=$14
} END {
  for (k in n) printf "%.3f\t%.3f\t%.3f\t%.3f\t%d\t%.3f GiB\t%s\n",
    wall[k],host[k],iow[k],h2d[k],n[k],bytes[k]/1024/1024/1024,k
}' "$run/copy-profile.csv" | sort -nr | head -40

awk -F, 'NR>1 {
  key="op=" $2 ",pack=" $6 ",iouring=" $8
  n[key]++
  bytes[key]+=$5
  host[key]+=$10
  iow[key]+=$11
  h2d[key]+=$13
  wall[key]+=$14
} END {
  for (k in n) printf "%.3f\t%.3f\t%.3f\t%.3f\t%d\t%.3f GiB\t%s\n",
    wall[k],host[k],iow[k],h2d[k],n[k],bytes[k]/1024/1024/1024,k
}' "$run/copy-profile.csv" | sort -nr
```

## Runtime Result

Accepted run:

- run dir:
  `/root/lfz/tmp/runs/20260707-gp53-current-code-copy-profile/dev_linear_equation_n48`;
- quality: pass;
- output:
  `To solve for x, subtract 3 from both sides: x + 3 = 10 x = 10 − 3 **x = 7**`;
- TTFT: `95795.13 ms`;
- decode: `208177.33 ms / 34 runs`;
- token rate: `0.16 tok/s`;
- RAM peak: `15899996160` bytes;
- direct reads: `2768`;
- iouring reads: `16547`;
- iouring bytes: `90322534400`;
- iouring wait: `21569.993 ms`.

The rejected `N=32` run:

- run dir:
  `/root/lfz/tmp/runs/20260707-gp53-current-code-copy-profile/dev_linear_equation_n32`;
- quality: fail because output stopped at `**x =`;
- token rate: `0.16 tok/s`;
- used only as a discarded attribution sanity check.

## Copy Attribution

By backend:

| Wall ms | Host ms | IO wait ms | H2D ms | Jobs | GiB | Path |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 183050.117 | 179895.411 | 0.000 | 2718.867 | 12856 | 64.888 | `op=runtime_load,pack=0,iouring=0` |
| 85246.742 | 0.000 | 85246.742 | 3447.104 | 16251 | 82.381 | `op=runtime_load,pack=1,iouring=1` |
| 41181.273 | 40423.557 | 0.000 | 659.996 | 2691 | 15.809 | `op=current_down_overlap,pack=0,iouring=0` |
| 7143.719 | 6404.296 | 0.000 | 663.982 | 2768 | 16.261 | `op=current_down_overlap,pack=1,iouring=0` |
| 1194.879 | 0.000 | 1194.879 | 69.935 | 296 | 1.739 | `op=current_down_overlap,pack=1,iouring=1` |

Top non-iouring tensor paths:

| Wall ms | Host ms | H2D ms | Jobs | GiB | Path |
| ---: | ---: | ---: | ---: | ---: | --- |
| 2285.238 | 2249.468 | 32.532 | 110 | 0.799 | `runtime_load,pack=0,blk.25.ffn_down_exps.weight` |
| 2194.251 | 2157.655 | 33.185 | 111 | 0.806 | `runtime_load,pack=0,blk.58.ffn_down_exps.weight` |
| 1922.781 | 1887.635 | 32.001 | 107 | 0.777 | `runtime_load,pack=0,blk.21.ffn_down_exps.weight` |
| 1874.275 | 1840.399 | 30.168 | 124 | 0.728 | `runtime_load,pack=0,blk.60.ffn_down_exps.weight` |
| 1830.799 | 1799.592 | 27.034 | 111 | 0.652 | `current_down_overlap,pack=0,blk.48.ffn_down_exps.weight` |
| 1807.187 | 1779.170 | 24.363 | 99 | 0.582 | `current_down_overlap,pack=0,blk.41.ffn_down_exps.weight` |
| 1800.161 | 1769.958 | 27.291 | 92 | 0.668 | `runtime_load,pack=0,blk.4.ffn_down_exps.weight` |
| 1794.040 | 1765.947 | 25.522 | 86 | 0.625 | `runtime_load,pack=0,blk.24.ffn_down_exps.weight` |
| 1747.666 | 1719.024 | 25.913 | 87 | 0.632 | `runtime_load,pack=0,blk.57.ffn_down_exps.weight` |
| 1743.153 | 1714.769 | 24.422 | 111 | 0.581 | `runtime_load,pack=0,blk.48.ffn_gate_exps.weight` |

## Interpretation

The dominant slow path is not a small number of pack-hit jobs bypassing iouring.
It is broad `pack=0` coverage failure:

- `runtime_load,pack=0,iouring=0`: `183.050 s` wall, `64.888 GiB`;
- `current_down_overlap,pack=0,iouring=0`: `41.181 s` wall, `15.809 GiB`.

Together these account for about `224.231 s` of copy wall in a run whose decode
wall is `208.177 s`; the overlap and profiling scopes are not directly additive,
but the direction is unambiguous. The current France-oriented expert pack does
not cover the slow general prompt well enough.

The pack-hit direct path is smaller:

- `current_down_overlap,pack=1,iouring=0`: `7.144 s`, `16.261 GiB`.

Moving that pack-hit path to iouring is useful cleanup, but it is not the main
reason the prompt is slow.

## Decision

Next implementation should target prompt-agnostic expert-pack coverage for slow
general prompts:

- build a dev-trained general hotset pack from committed dev route profiles;
- avoid held-out test prompts;
- use the same 16GB cold-start gate;
- compare current France pack vs general hotset pack on dev prompts first;
- accept only if quality passes, TTFT does not exceed +20%, and token rate
  improves on non-France dev prompts.

Standalone iouring scheduler work for pack-hit direct jobs is lower priority
than fixing pack misses, because the largest measured path is `pack=0`.
