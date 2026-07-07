# GP54 prompt-agnostic hotset pack feasibility

This is a dev-only simulation. It uses `route-profile.csv` files and
expert-pack indexes; it does not run inference and does not inspect
held-out test prompts.

## Inputs

- route root: `.Agent/runs/20260706-kimi-general-dev-baseline-n96-profile`
- route traffic model: `count * expert_bytes` per `route-profile.csv` row
- assumed direct-copy cost: `2.70 s/GiB`

## Existing Pack Sources

| path | entries | new unique keys | duplicates | new unique payload GiB |
|---|---:|---:|---:|---:|
| `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-france-l12-upgate-v2.expert-pack` | 30831 | 30831 | 0 | 163.10 |
| `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-l1l2down-overlay.expert-pack` | 768 | 768 | 0 | 4.51 |

## Current Dev Coverage

- routed traffic: `3506.57 GiB`
- existing pack-hit traffic: `2459.33 GiB`
- existing pack-miss traffic: `1047.24 GiB`
- existing pack-hit traffic ratio: `70.1%`
- unique missing keys: `27797`

## Prompt Coverage

| prompt | traffic GiB | hit GiB | hit traffic | miss GiB |
|---|---:|---:|---:|---:|
| `dev_france_regression` | 553.28 | 550.08 | 99.4% | 3.20 |
| `dev_japan_factual` | 610.75 | 503.24 | 82.4% | 107.51 |
| `dev_linear_equation` | 244.38 | 132.15 | 54.1% | 112.22 |
| `dev_mixed_summary` | 388.05 | 226.88 | 58.5% | 161.17 |
| `dev_photosynthesis_factual` | 675.40 | 398.66 | 59.0% | 276.74 |
| `dev_python_reverse` | 682.58 | 368.59 | 54.0% | 313.99 |
| `dev_zh_france` | 352.13 | 279.72 | 79.4% | 72.41 |

## Role Misses

| role | traffic GiB | hit traffic | miss GiB |
|---|---:|---:|---:|
| down | 1280.91 | 70.7% | 374.92 |
| gate | 1156.20 | 69.7% | 350.30 |
| up | 1069.46 | 69.9% | 322.02 |

## Top Missing Layer/Role Buckets

| layer | role | miss GiB | miss events |
|---:|---|---:|---:|
| 57 | down | 9.40 | 1294 |
| 58 | down | 9.20 | 1266 |
| 19 | down | 9.20 | 1266 |
| 16 | down | 9.01 | 1241 |
| 21 | down | 8.87 | 1221 |
| 43 | down | 8.82 | 1501 |
| 45 | down | 8.79 | 1496 |
| 40 | down | 8.55 | 1456 |
| 46 | down | 8.44 | 1436 |
| 39 | down | 8.25 | 1405 |
| 52 | down | 7.95 | 1354 |
| 44 | down | 7.94 | 1351 |
| 43 | gate | 7.86 | 1501 |
| 37 | down | 7.84 | 1334 |
| 45 | gate | 7.83 | 1496 |
| 55 | down | 7.81 | 1329 |
| 60 | down | 7.76 | 1321 |
| 13 | down | 7.75 | 1319 |
| 26 | down | 7.71 | 1061 |
| 35 | down | 7.65 | 1302 |
| 40 | gate | 7.62 | 1456 |
| 23 | down | 7.58 | 1043 |
| 46 | gate | 7.52 | 1436 |
| 48 | down | 7.49 | 1275 |
| 34 | down | 7.43 | 1264 |
| 39 | gate | 7.35 | 1405 |
| 24 | down | 7.34 | 1011 |
| 41 | down | 7.31 | 1244 |
| 25 | down | 7.31 | 1006 |
| 54 | down | 7.28 | 1239 |
| 17 | down | 7.25 | 1234 |
| 30 | down | 7.23 | 1230 |
| 29 | down | 7.23 | 1230 |
| 51 | down | 7.20 | 1225 |
| 56 | down | 7.17 | 1221 |
| 47 | down | 7.15 | 1217 |
| 52 | gate | 7.09 | 1354 |
| 44 | gate | 7.07 | 1351 |
| 20 | down | 7.04 | 969 |
| 49 | down | 7.03 | 1197 |

## Candidate Hotsets

| candidate | entries | pack GiB | added miss traffic GiB | added miss traffic | worst prompt added miss | optimistic direct-copy save s | remaining miss GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `top512` | 512 | 2.77 | 165.77 | 15.8% | 2.8% | 447.6 | 881.46 |
| `top1024` | 1024 | 5.52 | 251.87 | 24.1% | 4.6% | 680.1 | 795.37 |
| `top2048` | 2048 | 11.01 | 368.36 | 35.2% | 10.2% | 994.6 | 678.88 |
| `top4096` | 4096 | 22.14 | 526.03 | 50.2% | 24.0% | 1420.3 | 521.21 |
| `top8192` | 8192 | 43.62 | 728.40 | 69.6% | 48.3% | 1966.7 | 318.84 |
| `top16384` | 16384 | 86.12 | 945.26 | 90.3% | 75.0% | 2552.2 | 101.98 |
| `budget4gib` | 744 | 4.00 | 209.25 | 20.0% | 3.1% | 565.0 | 837.98 |
| `budget8gib` | 1477 | 8.00 | 308.86 | 29.5% | 6.4% | 833.9 | 738.37 |
| `budget16gib` | 2978 | 16.00 | 448.14 | 42.8% | 16.1% | 1210.0 | 599.10 |
| `budget32gib` | 5954 | 32.00 | 629.81 | 60.1% | 35.6% | 1700.5 | 417.42 |
| `budget64gib` | 12152 | 64.00 | 855.60 | 81.7% | 60.4% | 2310.1 | 191.64 |

## Top Missing Keys

| rank | traffic GiB | count | tensor | expert | bytes |
|---:|---:|---:|---|---:|---:|
| 1 | 0.83 | 142 | `blk.45.ffn_down_exps.weight` | 90 | 6307840 |
| 2 | 0.74 | 142 | `blk.45.ffn_gate_exps.weight` | 90 | 5619712 |
| 3 | 0.68 | 115 | `blk.55.ffn_down_exps.weight` | 310 | 6307840 |
| 4 | 0.65 | 111 | `blk.35.ffn_down_exps.weight` | 167 | 6307840 |
| 5 | 0.64 | 88 | `blk.24.ffn_down_exps.weight` | 128 | 7798784 |
| 6 | 0.62 | 106 | `blk.36.ffn_down_exps.weight` | 19 | 6307840 |
| 7 | 0.62 | 142 | `blk.45.ffn_up_exps.weight` | 90 | 4702208 |
| 8 | 0.62 | 85 | `blk.23.ffn_down_exps.weight` | 52 | 7798784 |
| 9 | 0.61 | 103 | `blk.39.ffn_down_exps.weight` | 260 | 6307840 |
| 10 | 0.61 | 103 | `blk.43.ffn_down_exps.weight` | 346 | 6307840 |
| 11 | 0.60 | 83 | `blk.57.ffn_down_exps.weight` | 348 | 7798784 |
| 12 | 0.60 | 115 | `blk.55.ffn_gate_exps.weight` | 310 | 5619712 |
| 13 | 0.59 | 100 | `blk.5.ffn_down_exps.weight` | 187 | 6307840 |
| 14 | 0.58 | 111 | `blk.35.ffn_gate_exps.weight` | 167 | 5619712 |
| 15 | 0.57 | 97 | `blk.38.ffn_down_exps.weight` | 361 | 6307840 |
| 16 | 0.57 | 97 | `blk.43.ffn_down_exps.weight` | 157 | 6307840 |
| 17 | 0.56 | 95 | `blk.51.ffn_down_exps.weight` | 7 | 6307840 |
| 18 | 0.55 | 106 | `blk.36.ffn_gate_exps.weight` | 19 | 5619712 |
| 19 | 0.55 | 94 | `blk.37.ffn_down_exps.weight` | 248 | 6307840 |
| 20 | 0.55 | 93 | `blk.3.ffn_down_exps.weight` | 259 | 6307840 |
| 21 | 0.54 | 92 | `blk.40.ffn_down_exps.weight` | 382 | 6307840 |
| 22 | 0.54 | 103 | `blk.39.ffn_gate_exps.weight` | 260 | 5619712 |
| 23 | 0.54 | 103 | `blk.43.ffn_gate_exps.weight` | 346 | 5619712 |
| 24 | 0.54 | 74 | `blk.22.ffn_down_exps.weight` | 253 | 7798784 |
| 25 | 0.53 | 91 | `blk.56.ffn_down_exps.weight` | 124 | 6307840 |
| 26 | 0.53 | 90 | `blk.37.ffn_down_exps.weight` | 307 | 6307840 |
| 27 | 0.52 | 100 | `blk.5.ffn_gate_exps.weight` | 187 | 5619712 |
| 28 | 0.52 | 100 | `blk.5.ffn_up_exps.weight` | 187 | 5619712 |
| 29 | 0.52 | 72 | `blk.19.ffn_down_exps.weight` | 352 | 7798784 |
| 30 | 0.52 | 89 | `blk.45.ffn_down_exps.weight` | 354 | 6307840 |
| 31 | 0.52 | 88 | `blk.45.ffn_down_exps.weight` | 131 | 6307840 |
| 32 | 0.51 | 97 | `blk.38.ffn_gate_exps.weight` | 361 | 5619712 |
| 33 | 0.51 | 97 | `blk.43.ffn_gate_exps.weight` | 157 | 5619712 |
| 34 | 0.51 | 86 | `blk.49.ffn_down_exps.weight` | 325 | 6307840 |
| 35 | 0.51 | 86 | `blk.41.ffn_down_exps.weight` | 326 | 6307840 |
| 36 | 0.50 | 115 | `blk.55.ffn_up_exps.weight` | 310 | 4702208 |
| 37 | 0.50 | 85 | `blk.43.ffn_down_exps.weight` | 362 | 6307840 |
| 38 | 0.50 | 85 | `blk.32.ffn_down_exps.weight` | 272 | 6307840 |
| 39 | 0.50 | 85 | `blk.46.ffn_down_exps.weight` | 279 | 6307840 |
| 40 | 0.50 | 95 | `blk.51.ffn_gate_exps.weight` | 7 | 5619712 |
| 41 | 0.49 | 68 | `blk.19.ffn_down_exps.weight` | 300 | 7798784 |
| 42 | 0.49 | 84 | `blk.35.ffn_down_exps.weight` | 312 | 6307840 |
| 43 | 0.49 | 94 | `blk.37.ffn_gate_exps.weight` | 248 | 5619712 |
| 44 | 0.49 | 83 | `blk.52.ffn_down_exps.weight` | 373 | 6307840 |
| 45 | 0.49 | 83 | `blk.52.ffn_down_exps.weight` | 106 | 6307840 |
| 46 | 0.49 | 93 | `blk.3.ffn_gate_exps.weight` | 259 | 5619712 |
| 47 | 0.49 | 93 | `blk.3.ffn_up_exps.weight` | 259 | 5619712 |
| 48 | 0.49 | 111 | `blk.35.ffn_up_exps.weight` | 167 | 4702208 |
| 49 | 0.48 | 82 | `blk.54.ffn_down_exps.weight` | 185 | 6307840 |
| 50 | 0.48 | 82 | `blk.37.ffn_down_exps.weight` | 44 | 6307840 |
| 51 | 0.48 | 92 | `blk.40.ffn_gate_exps.weight` | 382 | 5619712 |
| 52 | 0.48 | 91 | `blk.56.ffn_gate_exps.weight` | 124 | 5619712 |
| 53 | 0.48 | 91 | `blk.56.ffn_up_exps.weight` | 124 | 5619712 |
| 54 | 0.47 | 65 | `blk.19.ffn_down_exps.weight` | 212 | 7798784 |
| 55 | 0.47 | 90 | `blk.37.ffn_gate_exps.weight` | 307 | 5619712 |
| 56 | 0.47 | 80 | `blk.14.ffn_down_exps.weight` | 88 | 6307840 |
| 57 | 0.47 | 80 | `blk.13.ffn_down_exps.weight` | 178 | 6307840 |
| 58 | 0.47 | 89 | `blk.45.ffn_gate_exps.weight` | 354 | 5619712 |
| 59 | 0.46 | 106 | `blk.36.ffn_up_exps.weight` | 19 | 4702208 |
| 60 | 0.46 | 88 | `blk.45.ffn_gate_exps.weight` | 131 | 5619712 |
| 61 | 0.46 | 78 | `blk.54.ffn_down_exps.weight` | 35 | 6307840 |
| 62 | 0.46 | 63 | `blk.16.ffn_down_exps.weight` | 349 | 7798784 |
| 63 | 0.46 | 87 | `blk.8.ffn_up_exps.weight` | 271 | 5619712 |
| 64 | 0.45 | 77 | `blk.60.ffn_down_exps.weight` | 310 | 6307840 |
| 65 | 0.45 | 77 | `blk.38.ffn_down_exps.weight` | 275 | 6307840 |
| 66 | 0.45 | 103 | `blk.39.ffn_up_exps.weight` | 260 | 4702208 |
| 67 | 0.45 | 103 | `blk.43.ffn_up_exps.weight` | 346 | 4702208 |
| 68 | 0.45 | 86 | `blk.49.ffn_gate_exps.weight` | 325 | 5619712 |
| 69 | 0.45 | 86 | `blk.41.ffn_gate_exps.weight` | 326 | 5619712 |
| 70 | 0.45 | 86 | `blk.7.ffn_up_exps.weight` | 321 | 5619712 |
| 71 | 0.44 | 85 | `blk.43.ffn_gate_exps.weight` | 362 | 5619712 |
| 72 | 0.44 | 85 | `blk.32.ffn_gate_exps.weight` | 272 | 5619712 |
| 73 | 0.44 | 85 | `blk.46.ffn_gate_exps.weight` | 279 | 5619712 |
| 74 | 0.44 | 61 | `blk.23.ffn_down_exps.weight` | 214 | 7798784 |
| 75 | 0.44 | 61 | `blk.57.ffn_down_exps.weight` | 33 | 7798784 |
| 76 | 0.44 | 75 | `blk.17.ffn_down_exps.weight` | 262 | 6307840 |
| 77 | 0.44 | 84 | `blk.35.ffn_gate_exps.weight` | 312 | 5619712 |
| 78 | 0.44 | 60 | `blk.21.ffn_down_exps.weight` | 105 | 7798784 |
| 79 | 0.43 | 83 | `blk.2.ffn_gate_exps.weight` | 332 | 5619712 |
| 80 | 0.43 | 83 | `blk.2.ffn_up_exps.weight` | 332 | 5619712 |

## Reproduction

Run from repository root after replacing pack paths as needed:

```bash
python3 .Agent/run-tools/kimi_general_hotset_pack_feasibility.py \
  --route-root .Agent/runs/20260706-kimi-general-dev-baseline-n96-profile \
  --pack /root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-france-l12-upgate-v2.expert-pack \
  --pack /root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-l1l2down-overlay.expert-pack \
  --out-dir .Agent/runs/20260707-gp54-general-hotset-pack-feasibility
```

The generated `top-missing-entries.txt` can be converted to repeated
`--entry` arguments for `scripts/kimi-build-missing-down-overlay.py`.

## Interpretation

- A high added miss-traffic percentage means the candidate can remove many
  GGUF direct-copy fallback events if the generated overlay is loaded in the
  runtime pack list.
- The estimate is an upper bound: replacement iouring wait, H2D, scheduling,
  TTFT, disk space, and 16GB host-RAM cold-start gates still require a real
  runtime run before any SOTA claim.
- Candidate selection is based only on dev prompts. Held-out prompts remain
  reserved for final validation.
