# Kimi multi-prompt activation sample

This is a dev-only activation sample for representation screening. It is not a SOTA run.

- root: `.Agent/runs/20260708-gp85-strided-activation-corpus`
- prompts: `3`
- total activation records: `1536`
- all quality passed: `True`
- all direct_reads zero: `True`
- max memory peak: `15899996160`

## Prompts

| prompt | quality | token rate | decode | TTFT | memory peak | direct reads | activation records | layers | tensors | role counts |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `dev_japan_factual` | `pass` | `1.85` | `33962.63 ms / 63` | `86401.75 ms` | `15899996160` | `0` | `512` | `60` | `173` | `{'down,decode': 154, 'gate,decode': 181, 'up,decode': 177}` |
| `dev_mixed_summary` | `pass` | `1.71` | `31661.19 ms / 54` | `115322.57 ms` | `15899996160` | `0` | `512` | `60` | `173` | `{'down,decode': 154, 'gate,decode': 181, 'up,decode': 177}` |
| `dev_python_reverse` | `pass` | `1.69` | `37230.60 ms / 63` | `97770.03 ms` | `15899996160` | `0` | `512` | `60` | `173` | `{'down,decode': 154, 'gate,decode': 181, 'up,decode': 177}` |

## Decision

Accept this dev-only activation sample for prompt-general representation screening.

## Reproduce

```bash
.Agent/run-tools/kimi_activation_sample_summary.py --root .Agent/runs/20260708-gp85-strided-activation-corpus --out-json .Agent/runs/20260708-gp85-strided-activation-corpus/summary.json --out-md .Agent/runs/20260708-gp85-strided-activation-corpus/summary.md --expected-records-per-prompt 512 --min-layers-per-prompt 60
```
