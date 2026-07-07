# Kimi multi-prompt activation sample

This is a dev-only activation sample for representation screening. It is not a SOTA run.

- root: `.Agent/runs/20260708-gp88-callstride-activation-corpus`
- prompts: `3`
- total activation records: `6144`
- all quality passed: `True`
- all direct_reads zero: `True`
- max memory peak: `15899996160`

## Prompts

| prompt | quality | token rate | decode | TTFT | memory peak | direct reads | activation records | layers | tensors | role counts |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `dev_japan_factual` | `pass` | `1.91` | `33032.75 ms / 63` | `82411.84 ms` | `15899996160` | `0` | `2048` | `60` | `173` | `{'down,decode': 624, 'gate,decode': 712, 'up,decode': 712}` |
| `dev_mixed_summary` | `pass` | `1.66` | `32611.60 ms / 54` | `115262.19 ms` | `15899996160` | `0` | `2048` | `60` | `173` | `{'down,decode': 624, 'gate,decode': 712, 'up,decode': 712}` |
| `dev_python_reverse` | `pass` | `1.67` | `37706.24 ms / 63` | `83502.24 ms` | `15899996160` | `0` | `2048` | `60` | `173` | `{'down,decode': 624, 'gate,decode': 712, 'up,decode': 712}` |

## Decision

Accept this dev-only activation sample for prompt-general representation screening.

## Reproduce

```bash
.Agent/run-tools/kimi_activation_sample_summary.py --root .Agent/runs/20260708-gp88-callstride-activation-corpus --out-json .Agent/runs/20260708-gp88-callstride-activation-corpus/summary.json --out-md .Agent/runs/20260708-gp88-callstride-activation-corpus/summary.md --expected-records-per-prompt 2048 --min-layers-per-prompt 30
```
