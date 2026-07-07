# GP63 layer-wise IO/cache model

Timestamp: `2026-07-07T12:45:00+08:00`.

Scope: dev-only Phase 1 analysis for GP63. Held-out test prompts were not used.

## Inputs

- Command: `.Agent/run-tools/kimi_layerwise_io_cache_model.py --runs-root .Agent/runs/20260707-gp4-aligned-alias-dev-n96-profile-correct --out-dir .Agent/runs/20260707-gp63-layerwise-io-cache-model`
- Runs root: `.Agent/runs/20260707-gp4-aligned-alias-dev-n96-profile-correct`
- Prompts: `7`
- Average token rate: `1.331 tok/s`
- Token-rate range: `1.100-1.430 tok/s`
- Max TTFT: `98983.24 ms`
- Max memory peak: `15899996160` bytes
- Aggregate expert-pack iouring bytes: `2098.71 GiB`
- Aggregate expert-pack iouring wait: `359.15 s`
- Observed bytes/wait throughput: `5.84 GiB/s`
- Aggregate down VRAM hit rate: `71.45%`
- Aggregate up/gate VRAM hit rate: `40.16%`

## Interpretation

- This is a demand-side and trace-side model, not a runtime SOTA claim.
- The current profiles expose aggregate VRAM cache hits by group, but not exact per-layer hit/miss events.
- Per-layer cache rows therefore use a fixed prompt-agnostic LFU proxy with the current slot counts.
- Overfetch rows use leave-one-prompt-out static priors and are gated by false-byte ratio.

## Cache-Increase Candidates

| layer | kind | traffic GiB | proxy hit % | next64 GiB | MiB/slot | avg prompts |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 25 | down | 28.355 | 23.668 | 11.309 | 180.940 | 4.828 |
| 16 | down | 28.355 | 30.507 | 10.582 | 169.319 | 4.188 |
| 22 | upgate | 34.193 | 25.666 | 10.160 | 162.559 | 5.234 |
| 23 | down | 28.355 | 36.655 | 10.045 | 160.720 | 4.469 |
| 19 | down | 28.355 | 34.734 | 10.016 | 160.255 | 4.422 |
| 20 | down | 28.355 | 33.991 | 10.001 | 160.022 | 4.703 |
| 4 | down | 28.355 | 23.181 | 9.849 | 157.582 | 5.062 |
| 26 | down | 28.355 | 34.657 | 9.725 | 155.606 | 4.203 |
| 56 | upgate | 40.865 | 24.744 | 9.578 | 153.245 | 4.562 |
| 10 | upgate | 34.193 | 16.906 | 9.564 | 153.029 | 4.797 |

## Cache-Decrease Candidates

| layer | kind | traffic GiB | proxy hit % | remove64 GiB | MiB/slot | avg prompts |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 60 | down | 23.264 | 16.995 | 3.954 | 449.835 | 4.556 |
| 53 | down | 22.935 | 18.058 | 4.142 | 424.102 | 4.800 |
| 51 | down | 22.935 | 19.262 | 4.418 | 452.375 | 4.800 |
| 1 | down | 22.935 | 19.851 | 4.553 | 466.211 | 4.500 |
| 29 | down | 22.935 | 19.903 | 4.565 | 467.414 | 5.300 |
| 55 | down | 22.935 | 20.210 | 4.635 | 395.527 | 4.417 |
| 59 | down | 22.935 | 20.338 | 4.664 | 477.641 | 4.500 |
| 56 | down | 22.935 | 20.389 | 4.676 | 598.555 | 5.125 |
| 3 | down | 22.935 | 20.748 | 4.758 | 487.266 | 4.800 |
| 28 | down | 22.935 | 21.440 | 4.917 | 387.314 | 5.077 |

## Bounded Overfetch Candidates

No leave-one-prompt-out static-prior overfetch setting passed the GP63 false-byte gate.

## Same-Budget Cache Swap Bound

| free MiB | removed | added | net GiB | saved s | decode % |
| --- | --- | --- | ---: | --- | ---: |
| 256 | 39 | 56 | 1.663 | 0.285 | 0.078 |
| 512 | 82 | 106 | 2.746 | 0.470 | 0.129 |
| 1024 | 171 | 207 | 3.365 | 0.576 | 0.158 |
| 2048 | 350 | 405 | -1.462 | -0.250 | -0.069 |
| 4096 | 734 | 795 | -26.135 | -4.472 | -1.229 |

## Do-Not-Touch Layers

| layer | kind | traffic GiB | proxy hit % | next64 GiB | remove64 GiB |
| ---: | --- | ---: | ---: | ---: | ---: |

## Phase 2 Gate

- Best cache-increase next64 gain: `11.309 GiB`.
- Best accepted overfetch candidate count: `0`.
- Best same-budget cache-swap decode improvement estimate: `0.158%`.
- Phase 2 recommendation: `do not implement runtime change from this model; Phase 2 gate not met`.

Required next action:

Record GP63 Phase 1 as insufficient for runtime implementation. The next plan should target a stronger predictor or a lower-byte expert representation rather than simple layer-wise cache/overfetch.
