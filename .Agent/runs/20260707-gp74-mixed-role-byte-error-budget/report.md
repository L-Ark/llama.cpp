# Kimi mixed-role byte/error budget

This is an offline budget screen. It does not change runtime behavior or claim SOTA.

- source screen: `.Agent/runs/20260707-gp70-activation-input-residual-screen-n16-france/screen.json`
- down weight: `0.376`
- fused up/gate weight: `0.624`
- target global byte ratio: `<= 0.40x`
- target mean rel L2 per component: `<= 0.10`
- combinations under byte target: `15`
- passing combinations: `0`

## Best Under Budget

| down | fused up/gate | global ratio | down rel L2 | fused rel L2 | worst rel L2 | decision |
|---|---|---:|---:|---:|---:|---|
| `down:aw_mse:bits1:block256` | `fused_up_gate:aw_mse_keep_input0p1:bits1:block256` | `0.3861` | `0.498586` | `0.594271` | `0.594271` | reject |
| `down:aw_mse_keep_input0p02:bits1:block256` | `fused_up_gate:aw_mse_keep_input0p1:bits1:block256` | `0.3913` | `0.358662` | `0.594271` | `0.594271` | reject |
| `down:aw_mse_keep_input0p05:bits1:block256` | `fused_up_gate:aw_mse_keep_input0p1:bits1:block256` | `0.3991` | `0.297049` | `0.594271` | `0.594271` | reject |
| `down:aw_mse:bits1:block256` | `fused_up_gate:aw_mse_keep_input0p05:bits1:block256` | `0.3665` | `0.498586` | `0.668969` | `0.668969` | reject |
| `down:aw_mse_keep_input0p02:bits1:block256` | `fused_up_gate:aw_mse_keep_input0p05:bits1:block256` | `0.3717` | `0.358662` | `0.668969` | `0.668969` | reject |
| `down:aw_mse_keep_input0p05:bits1:block256` | `fused_up_gate:aw_mse_keep_input0p05:bits1:block256` | `0.3794` | `0.297049` | `0.668969` | `0.668969` | reject |
| `down:aw_mse_keep_input0p1:bits1:block256` | `fused_up_gate:aw_mse_keep_input0p05:bits1:block256` | `0.3924` | `0.234399` | `0.668969` | `0.668969` | reject |
| `down:aw_mse:bits1:block256` | `fused_up_gate:aw_mse_keep_input0p02:bits1:block256` | `0.3547` | `0.498586` | `0.719166` | `0.719166` | reject |
| `down:aw_mse_keep_input0p02:bits1:block256` | `fused_up_gate:aw_mse_keep_input0p02:bits1:block256` | `0.3599` | `0.358662` | `0.719166` | `0.719166` | reject |
| `down:aw_mse_keep_input0p05:bits1:block256` | `fused_up_gate:aw_mse_keep_input0p02:bits1:block256` | `0.3676` | `0.297049` | `0.719166` | `0.719166` | reject |
| `down:aw_mse_keep_input0p1:bits1:block256` | `fused_up_gate:aw_mse_keep_input0p02:bits1:block256` | `0.3806` | `0.234399` | `0.719166` | `0.719166` | reject |
| `down:aw_mse:bits1:block256` | `fused_up_gate:aw_mse:bits1:block256` | `0.3468` | `0.498586` | `0.769351` | `0.769351` | reject |
| `down:aw_mse_keep_input0p02:bits1:block256` | `fused_up_gate:aw_mse:bits1:block256` | `0.3520` | `0.358662` | `0.769351` | `0.769351` | reject |
| `down:aw_mse_keep_input0p05:bits1:block256` | `fused_up_gate:aw_mse:bits1:block256` | `0.3598` | `0.297049` | `0.769351` | `0.769351` | reject |
| `down:aw_mse_keep_input0p1:bits1:block256` | `fused_up_gate:aw_mse:bits1:block256` | `0.3728` | `0.234399` | `0.769351` | `0.769351` | reject |

## Conclusion

No mixed-role combination passes. Even the best under-budget pair has worst mean rel L2 0.594, so GP68-GP70 blockwise residual tuning should stop as a primary 5 tok/s path.

## Reproduce

```bash
.Agent/run-tools/kimi_mixed_role_byte_error_budget.py --screen-json .Agent/runs/20260707-gp70-activation-input-residual-screen-n16-france/screen.json --out-json .Agent/runs/20260707-gp74-mixed-role-byte-error-budget/report.json --out-md .Agent/runs/20260707-gp74-mixed-role-byte-error-budget/report.md
```
