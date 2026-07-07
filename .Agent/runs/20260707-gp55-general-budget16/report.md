# GP55 general budget16 overlay runtime result

Status: accepted dev runtime improvement.

This run used only dev prompt data for candidate selection. It did not use any
held-out test prompt.

## Overlay Build

- overlay:
  `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-general-dev-budget16-overlay.expert-pack`
- source entries:
  `.Agent/runs/20260707-gp54-general-hotset-pack-feasibility/budget16gib-entries.txt`
- entries: `2978`
- size: `17179111424` bytes (`16 GiB` shown by `ls -lh`)
- build wall time: `0:40.28`
- max RSS: `378664 KB`

Build command:

```bash
cd /root/lfz/tmp/vendor-kimi-speculative-gp33
python3 scripts/kimi-build-missing-down-overlay.py \
  --model-glob '/root/lfz/models/Kimi-K2.7-Code-GGUF-IQ3_S/IQ3_S/Kimi-K2.7-Code-IQ3_S-*-of-00010.gguf' \
  --reject-pack /root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-france-l12-upgate-v2.expert-pack \
  --reject-pack /root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-l1l2down-overlay.expert-pack \
  --entry-file .Agent/runs/20260707-gp54-general-hotset-pack-feasibility/budget16gib-entries.txt \
  --out /root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-general-dev-budget16-overlay.expert-pack
```

## Runtime Command

```bash
repo=/root/lfz/tmp/vendor-kimi-speculative-gp33
run=/root/lfz/tmp/runs/20260707-gp55-general-budget16/dev_linear_equation_n48
cd "$repo"
ln -sfn build-gp50-runtime build-cuda-batch
systemd-run --wait --collect --same-dir \
  -p MemoryMax=15900000000 -p MemorySwapMax=0 \
  env REPO="$repo" RUN="$run" N=48 PROFILE=1 COPY_PROFILE=1 \
      PROMPT_ID=dev_linear_equation \
      PROMPT_USER_TEXT="Solve: if x + 3 = 10, what is x?" \
      QUALITY_KEYWORDS="7|seven" \
      EXTRA_RUNTIME_ENV="GGML_MOE_EXPERT_PACK_LIST=/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-general-dev-budget16-overlay.expert-pack GGML_MOE_EXPERT_PACK_REPLACE_DUPLICATES=1" \
      .Agent/run-tools/kimi-general-prompt-repro.sh
```

## Result

Baseline is GP53 current-code `dev_linear_equation N=48`.

| metric | GP53 baseline | GP55 budget16 overlay |
|---|---:|---:|
| quality | pass | pass |
| token rate | `0.16 tok/s` | `0.22 tok/s` |
| decode wall | `208177.33 ms` | `156463.54 ms` |
| decode runs | `34` | `34` |
| TTFT | `95795.13 ms` | `95778.96 ms` |
| RAM peak | `15899996160` | `15899996160` |
| direct reads | `2768` | `3164` |
| iouring reads | `16547` | `19894` |
| iouring bytes | `90322534400` | `109455474688` |
| iouring wait | `21569993 us` | `21943125 us` |

Answer:

```text
To solve for x, subtract 3 from both sides: x + 3 = 10 x = 10 − 3 **x = 7**
```

Acceptance gates:

- Quality: pass.
- TTFT: pass, `95778.96 ms` is below the `114954.16 ms` GP55 limit.
- RAM: pass under the configured `15900000000` byte cgroup limit, but with no
  margin (`memory.peak=15899996160`).
- Token rate: pass, `0.16 -> 0.22 tok/s`.
- Cold start: pass, run used `systemd-run --wait --collect` with the 16GB
  memory cgroup.

## Copy Attribution

| op | pack hit | iouring | jobs | GiB | host ms | io wait ms | h2d ms | wall ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| runtime_load | 0 | 0 | 9863 | 49.149 | 125695.146 | 0.000 | 2043.028 | 128042.814 |
| runtime_load | 1 | 1 | 19244 | 98.120 | 0.000 | 98074.676 | 4089.149 | 98074.676 |
| current_down_overlap | 0 | 0 | 1941 | 11.403 | 26951.775 | 0.000 | 471.850 | 27491.438 |
| current_down_overlap | 1 | 0 | 3164 | 18.587 | 7001.056 | 0.000 | 756.888 | 7839.119 |
| current_down_overlap | 1 | 1 | 650 | 3.819 | 0.000 | 2600.181 | 154.650 | 2600.181 |

Compared with GP53:

- `runtime_load,pack=0` dropped from `64.888 GiB / 183050.117 ms` to
  `49.149 GiB / 125695.146 ms`.
- `current_down_overlap,pack=0` dropped from `15.809 GiB / 41181.273 ms` to
  `11.403 GiB / 26951.775 ms`.
- Direct GGUF copy traffic dropped by about `20.145 GiB` on this prompt.
- The replacement iouring traffic increased, but decode still improved by
  `51713.79 ms`.

## Next

- Validate on additional dev prompts before considering this a general SOTA.
- If the improvement holds, run the held-out test set once as the SOTA gate.
- The RAM peak has no margin, so later candidates should also inspect page cache
  and memory distribution before increasing overlay size.
