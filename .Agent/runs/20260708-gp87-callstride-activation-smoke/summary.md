# Kimi multi-prompt activation sample

This is a dev-only activation sample for representation screening. It is not a SOTA run.

- root: `.Agent/runs/20260708-gp87-callstride-activation-smoke`
- prompts: `1`
- total activation records: `512`
- all quality passed: `True`
- all direct_reads zero: `True`
- max memory peak: `15899996160`

## Prompts

| prompt | quality | token rate | decode | TTFT | memory peak | direct reads | activation records | layers | tensors | role counts |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `dev_france_regression` | `pass` | `1.83` | `16941.86 ms / 31` | `82667.35 ms` | `15899996160` | `0` | `512` | `26` | `65` | `{'down,decode': 152, 'gate,decode': 180, 'up,decode': 180}` |

## Decision

Accept this dev-only activation sample for prompt-general representation screening.

## Reproduce

```bash
.Agent/run-tools/kimi_activation_sample_summary.py --root .Agent/runs/20260708-gp87-callstride-activation-smoke --out-json .Agent/runs/20260708-gp87-callstride-activation-smoke/summary.json --out-md .Agent/runs/20260708-gp87-callstride-activation-smoke/summary.md --expected-records-per-prompt 512 --min-layers-per-prompt 20
```
