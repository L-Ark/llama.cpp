# Kimi multi-prompt activation sample

This is a dev-only activation sample for representation screening. It is not a SOTA run.

- root: `.Agent/runs/20260708-gp85-strided-activation-smoke-v2`
- prompts: `1`
- total activation records: `216`
- all quality passed: `True`
- all direct_reads zero: `True`
- max memory peak: `15899996160`

## Prompts

| prompt | quality | token rate | decode | TTFT | memory peak | direct reads | activation records | layers | tensors | role counts |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `dev_france_regression` | `pass` | `1.80` | `17187.99 ms / 31` | `71294.95 ms` | `15899996160` | `0` | `216` | `60` | `173` | `{'down,decode': 65, 'gate,decode': 76, 'up,decode': 75}` |

## Decision

Accept this dev-only activation sample for prompt-general representation screening.

## Reproduce

```bash
.Agent/run-tools/kimi_activation_sample_summary.py --root .Agent/runs/20260708-gp85-strided-activation-smoke-v2 --out-json .Agent/runs/20260708-gp85-strided-activation-smoke-v2/summary.json --out-md .Agent/runs/20260708-gp85-strided-activation-smoke-v2/summary.md --expected-records-per-prompt 216 --min-layers-per-prompt 60
```
