# Kimi mixed-role byte/error budget

This is an offline budget screen. It does not change runtime behavior or claim SOTA.

- source screen: `.Agent/runs/20260712-current-goal-lowbyte-freeze-screen/gp77-target050-summary.json`
- down weight: `0.376`
- fused up/gate weight: `0.624`
- target global byte ratio: `<= 0.50x`
- target mean rel L2 per component: `<= 0.10`
- combinations under byte target: `16`
- passing combinations: `0`

## Best Under Budget

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

## Conclusion

No mixed-role combination passes. Even the best under-budget pair has worst mean rel L2 0.598, so GP68-GP70 blockwise residual tuning should stop as a primary 5 tok/s path.

## Reproduce

```bash
.Agent/run-tools/kimi_mixed_role_byte_error_budget.py --screen-json .Agent/runs/20260712-current-goal-lowbyte-freeze-screen/gp77-target050-summary.json --out-json .Agent/runs/20260712-current-goal-nondestructive-lowbyte-closure/mixed-role-target050.json --out-md .Agent/runs/20260712-current-goal-nondestructive-lowbyte-closure/mixed-role-target050.md --target-global-ratio 0.50 --target-mean-rel-l2 0.10
```
