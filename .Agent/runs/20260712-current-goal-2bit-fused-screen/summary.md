# Kimi 2-bit fused up/gate screen

This is dev-only and non-destructive. It does not change runtime behavior or claim SOTA.

| candidate | prompts | pairs | byte ratio | mean rel L2 | max rel L2 |
|---|---:|---:|---:|---:|---:|
| `fused_up_gate:aw_mse:bits2:block64` | 3 | 48 | `0.7673` | `0.575718` | `0.640693` |
| `fused_up_gate:aw_mse:bits2:block128` | 3 | 48 | `0.7247` | `0.581918` | `0.640225` |
| `fused_up_gate:aw_mse:bits2:block256` | 3 | 48 | `0.7034` | `0.583060` | `0.645486` |
| `fused_up_gate:maxabs:bits2:block64` | 3 | 48 | `0.7673` | `1.032088` | `1.125518` |
| `fused_up_gate:maxabs:bits2:block128` | 3 | 48 | `0.7247` | `1.084651` | `1.187158` |
| `fused_up_gate:maxabs:bits2:block256` | 3 | 48 | `0.7034` | `1.112494` | `1.266916` |

## Per Prompt

| prompt | candidate | pairs | byte ratio | mean rel L2 | max rel L2 |
|---|---|---:|---:|---:|---:|
| `dev_japan_factual` | `fused_up_gate:aw_mse:bits2:block64` | 16 | `0.7673` | `0.575900` | `0.604534` |
| `dev_japan_factual` | `fused_up_gate:aw_mse:bits2:block128` | 16 | `0.7247` | `0.584991` | `0.621643` |
| `dev_japan_factual` | `fused_up_gate:aw_mse:bits2:block256` | 16 | `0.7034` | `0.585079` | `0.620837` |
| `dev_japan_factual` | `fused_up_gate:maxabs:bits2:block64` | 16 | `0.7673` | `1.038414` | `1.125029` |
| `dev_japan_factual` | `fused_up_gate:maxabs:bits2:block128` | 16 | `0.7247` | `1.098310` | `1.187158` |
| `dev_japan_factual` | `fused_up_gate:maxabs:bits2:block256` | 16 | `0.7034` | `1.126448` | `1.213644` |
| `dev_mixed_summary` | `fused_up_gate:aw_mse:bits2:block64` | 16 | `0.7673` | `0.575319` | `0.640693` |
| `dev_mixed_summary` | `fused_up_gate:aw_mse:bits2:block128` | 16 | `0.7247` | `0.580951` | `0.640225` |
| `dev_mixed_summary` | `fused_up_gate:aw_mse:bits2:block256` | 16 | `0.7034` | `0.582113` | `0.645486` |
| `dev_mixed_summary` | `fused_up_gate:maxabs:bits2:block64` | 16 | `0.7673` | `1.029527` | `1.107082` |
| `dev_mixed_summary` | `fused_up_gate:maxabs:bits2:block128` | 16 | `0.7247` | `1.078265` | `1.168349` |
| `dev_mixed_summary` | `fused_up_gate:maxabs:bits2:block256` | 16 | `0.7034` | `1.118518` | `1.266916` |
| `dev_python_reverse` | `fused_up_gate:aw_mse:bits2:block64` | 16 | `0.7673` | `0.575935` | `0.633592` |
| `dev_python_reverse` | `fused_up_gate:aw_mse:bits2:block128` | 16 | `0.7247` | `0.579812` | `0.614970` |
| `dev_python_reverse` | `fused_up_gate:aw_mse:bits2:block256` | 16 | `0.7034` | `0.581987` | `0.633574` |
| `dev_python_reverse` | `fused_up_gate:maxabs:bits2:block64` | 16 | `0.7673` | `1.028322` | `1.125518` |
| `dev_python_reverse` | `fused_up_gate:maxabs:bits2:block128` | 16 | `0.7247` | `1.077378` | `1.184663` |
| `dev_python_reverse` | `fused_up_gate:maxabs:bits2:block256` | 16 | `0.7034` | `1.092515` | `1.184150` |
