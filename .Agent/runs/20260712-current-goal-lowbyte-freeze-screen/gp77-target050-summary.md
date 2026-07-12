# Kimi multi-prompt representation screen

This is a dev-only offline screen. It does not change runtime behavior or claim SOTA.

- screens: `3`
- aggregate matvec candidates: `12`
- aggregate fused candidates: `4`
- passing matvec candidates: `0`
- passing fused candidates: `0`
- passing mixed-role candidates: `0`

## Best Mixed-Role Under Budget

| down | fused up/gate | global ratio | down rel L2 | fused rel L2 | worst rel L2 | decision |
|---|---|---:|---:|---:|---:|---|
| `down:aw_mse:bits1:block256` | `fused_up_gate:aw_mse_keep_input0p1:bits1:block256` | `0.3861` | `0.499277` | `0.598174` | `0.598174` | reject |
| `down:aw_mse_keep_input0p02:bits1:block256` | `fused_up_gate:aw_mse_keep_input0p1:bits1:block256` | `0.3913` | `0.358095` | `0.598174` | `0.598174` | reject |
| `down:aw_mse_keep_input0p05:bits1:block256` | `fused_up_gate:aw_mse_keep_input0p1:bits1:block256` | `0.3991` | `0.297230` | `0.598174` | `0.598174` | reject |
| `down:aw_mse_keep_input0p1:bits1:block256` | `fused_up_gate:aw_mse_keep_input0p1:bits1:block256` | `0.4121` | `0.234819` | `0.598174` | `0.598174` | reject |
| `down:aw_mse:bits1:block256` | `fused_up_gate:aw_mse_keep_input0p05:bits1:block256` | `0.3665` | `0.499277` | `0.670288` | `0.670288` | reject |
| `down:aw_mse_keep_input0p02:bits1:block256` | `fused_up_gate:aw_mse_keep_input0p05:bits1:block256` | `0.3717` | `0.358095` | `0.670288` | `0.670288` | reject |
| `down:aw_mse_keep_input0p05:bits1:block256` | `fused_up_gate:aw_mse_keep_input0p05:bits1:block256` | `0.3794` | `0.297230` | `0.670288` | `0.670288` | reject |
| `down:aw_mse_keep_input0p1:bits1:block256` | `fused_up_gate:aw_mse_keep_input0p05:bits1:block256` | `0.3924` | `0.234819` | `0.670288` | `0.670288` | reject |
| `down:aw_mse:bits1:block256` | `fused_up_gate:aw_mse_keep_input0p02:bits1:block256` | `0.3547` | `0.499277` | `0.721826` | `0.721826` | reject |
| `down:aw_mse_keep_input0p02:bits1:block256` | `fused_up_gate:aw_mse_keep_input0p02:bits1:block256` | `0.3599` | `0.358095` | `0.721826` | `0.721826` | reject |
| `down:aw_mse_keep_input0p05:bits1:block256` | `fused_up_gate:aw_mse_keep_input0p02:bits1:block256` | `0.3676` | `0.297230` | `0.721826` | `0.721826` | reject |
| `down:aw_mse_keep_input0p1:bits1:block256` | `fused_up_gate:aw_mse_keep_input0p02:bits1:block256` | `0.3806` | `0.234819` | `0.721826` | `0.721826` | reject |
| `down:aw_mse:bits1:block256` | `fused_up_gate:aw_mse:bits1:block256` | `0.3468` | `0.499277` | `0.771296` | `0.771296` | reject |
| `down:aw_mse_keep_input0p02:bits1:block256` | `fused_up_gate:aw_mse:bits1:block256` | `0.3520` | `0.358095` | `0.771296` | `0.771296` | reject |
| `down:aw_mse_keep_input0p05:bits1:block256` | `fused_up_gate:aw_mse:bits1:block256` | `0.3598` | `0.297230` | `0.771296` | `0.771296` | reject |
| `down:aw_mse_keep_input0p1:bits1:block256` | `fused_up_gate:aw_mse:bits1:block256` | `0.3728` | `0.234819` | `0.771296` | `0.771296` | reject |

## Best Fused Up/Gate

| candidate | rows | byte ratio | mean rel L2 | max rel L2 |
|---|---:|---:|---:|---:|
| `fused_up_gate:aw_mse_keep_input0p1:bits1:block256` | `72` | `0.4326` | `0.598174` | `0.659500` |
| `fused_up_gate:aw_mse_keep_input0p05:bits1:block256` | `72` | `0.4010` | `0.670288` | `0.733526` |
| `fused_up_gate:aw_mse_keep_input0p02:bits1:block256` | `72` | `0.3821` | `0.721826` | `0.793158` |
| `fused_up_gate:aw_mse:bits1:block256` | `72` | `0.3695` | `0.771296` | `0.826939` |

## Decision

No candidate passes the multi-prompt offline gate. Do not implement this blockwise low-bit residual family as a runtime path; move to a qualitatively different representation.

## Reproduce

```bash
.Agent/run-tools/kimi_multi_prompt_screen_summary.py --screen-json .Agent/runs/20260707-gp77-multiprompt-representation-screen/dev_japan_factual/screen.json --screen-json .Agent/runs/20260707-gp77-multiprompt-representation-screen/dev_mixed_summary/screen.json --screen-json .Agent/runs/20260707-gp77-multiprompt-representation-screen/dev_python_reverse/screen.json --out-json .Agent/runs/20260712-current-goal-lowbyte-freeze-screen/gp77-target050-summary.json --out-md .Agent/runs/20260712-current-goal-lowbyte-freeze-screen/gp77-target050-summary.md --target-byte-ratio 0.50 --target-mean-rel-l2 0.10
```
