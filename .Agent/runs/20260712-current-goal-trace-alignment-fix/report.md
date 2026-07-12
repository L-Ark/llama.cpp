# Kimi trace alignment audit

This is a default-off instrumentation audit. It does not claim SOTA.

## Runs

### before

- run: `/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-trace-alignment-smoke/dev_france_n4`
- exit: `0`
- token_rate: `1.37`
- TTFT ms: `8451.92`
- memory peak: `12740333568`
- quality: `fail`

Route score:

- rows: `181`
- calls: `4`
- positions: `{'17': 60, '18': 60, '19': 60, '0': 1}`

Route detail:

- rows: `28416`
- mode: `{'prompt': 24072, 'decode': 4344}`
- kind: `{'gate': 9472, 'up': 9472, 'down': 9472}`
- token_id top: `{'-1': 25520, '0': 2896}`
- prompt token_id top: `{'-1': 24072}`
- decode token_id top: `{'0': 2896, '-1': 1448}`

Activations:

- rows: `512`
- role: `{'down': 512}`
- mode: `{'decode': 512}`
- tensor_kind: `{'gate': 240, 'up': 136, 'down': 136}`
- token_id top: `{'0': 31, '1': 31, '5': 31, '6': 31, '7': 31, '3': 31, '13': 30, '14': 30, '8': 30, '9': 30, '10': 30, '2': 30}`
- role/tensor mismatches: `376`

### after

- run: `/root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-trace-alignment-smoke-after-fix/dev_france_n4`
- exit: `0`
- token_rate: `1.36`
- TTFT ms: `8110.83`
- memory peak: `12740292608`
- quality: `fail`

Route score:

- rows: `181`
- calls: `4`
- positions: `{'17': 60, '18': 60, '19': 60, '0': 1}`

Route detail:

- rows: `28416`
- mode: `{'prompt': 24072, 'decode': 4344}`
- kind: `{'gate': 9472, 'up': 9472, 'down': 9472}`
- token_id top: `{'0': 5760, '13': 1416, '1': 1416, '5': 1416, '6': 1416, '14': 1416, '7': 1416, '8': 1416, '9': 1416, '10': 1416, '11': 1416, '15': 1416}`
- prompt token_id top: `{'0': 1416, '13': 1416, '1': 1416, '5': 1416, '6': 1416, '14': 1416, '7': 1416, '8': 1416, '9': 1416, '10': 1416, '11': 1416, '15': 1416}`
- decode token_id top: `{'0': 4344}`

Activations:

- rows: `512`
- role: `{'up': 172, 'gate': 172, 'down': 168}`
- mode: `{'decode': 512}`
- tensor_kind: `{'up': 172, 'gate': 172, 'down': 168}`
- token_id top: `{'0': 512}`
- role/tensor mismatches: `0`

## Decision

Pass: after the instrumentation fix, activation dump rows are decode-only for this smoke, and role names match tensor kinds.
Before fix, role/tensor mismatches were `376`; after fix they are `0`.

Remaining limitation:

- activation `token_id` is still the per-ubatch tensor-column index. In decode it is expected to be `0`; global decode position should be recovered from `route-score-trace.csv` (`pos`) and layer/order alignment.
- This fix makes a future full N96 hidden/router admission corpus feasible, but it is not itself a token-rate optimization.

## Reproduce

```bash
.Agent/run-tools/kimi_trace_alignment_audit.py --before /root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-trace-alignment-smoke/dev_france_n4 --after /root/lfz/runs/vendor-kimi-token-rate/20260712-current-goal-trace-alignment-smoke-after-fix/dev_france_n4 --out-json .Agent/runs/20260712-current-goal-trace-alignment-fix/report.json --out-md .Agent/runs/20260712-current-goal-trace-alignment-fix/report.md
```
