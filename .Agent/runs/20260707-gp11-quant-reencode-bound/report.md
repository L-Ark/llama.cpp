# Kimi quant re-encode bound

This is an offline error screen. It does not use held-out test prompts and does not create a runtime pack.

- generated_at: `2026-07-06T18:31:12+0000`
- sample_json: `.Agent/runs/20260707-gp11-quant-reencode-bound/hot-upgate-sample.json`
- inventory: `.Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv`
- libggml_base: `build-cuda-batch/bin/libggml-base.so`

| tensor | expert | type | current MiB | bits | block | ratio | rel L2 | rel max | finite |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| `blk.41.ffn_gate_exps.weight` | 27 | IQ3_XXS | 5.36 | 1 | 64 | 0.408 | 1.8616 | 0.9813 | True |
| `blk.41.ffn_gate_exps.weight` | 27 | IQ3_XXS | 5.36 | 1 | 128 | 0.367 | 2.0741 | 0.9813 | True |
| `blk.41.ffn_gate_exps.weight` | 27 | IQ3_XXS | 5.36 | 1 | 256 | 0.347 | 2.2746 | 0.9854 | True |
| `blk.41.ffn_gate_exps.weight` | 27 | IQ3_XXS | 5.36 | 2 | 64 | 0.735 | 0.6974 | 0.3226 | True |
| `blk.41.ffn_gate_exps.weight` | 27 | IQ3_XXS | 5.36 | 2 | 128 | 0.694 | 0.7488 | 0.3226 | True |
| `blk.41.ffn_gate_exps.weight` | 27 | IQ3_XXS | 5.36 | 2 | 256 | 0.673 | 0.7911 | 0.3226 | True |
| `blk.41.ffn_gate_exps.weight` | 27 | IQ3_XXS | 5.36 | 3 | 64 | 1.061 | 0.2325 | 0.1648 | True |
| `blk.41.ffn_gate_exps.weight` | 27 | IQ3_XXS | 5.36 | 3 | 128 | 1.020 | 0.2570 | 0.1648 | True |
| `blk.41.ffn_gate_exps.weight` | 27 | IQ3_XXS | 5.36 | 3 | 256 | 1.000 | 0.2786 | 0.1648 | True |
| `blk.41.ffn_gate_exps.weight` | 27 | IQ3_XXS | 5.36 | 4 | 64 | 1.388 | 0.1123 | 0.0645 | True |
| `blk.41.ffn_gate_exps.weight` | 27 | IQ3_XXS | 5.36 | 4 | 128 | 1.347 | 0.1207 | 0.0687 | True |
| `blk.41.ffn_gate_exps.weight` | 27 | IQ3_XXS | 5.36 | 4 | 256 | 1.327 | 0.1278 | 0.0700 | True |
| `blk.41.ffn_up_exps.weight` | 27 | IQ2_S | 4.48 | 1 | 64 | 0.488 | 1.7774 | 0.9515 | True |
| `blk.41.ffn_up_exps.weight` | 27 | IQ2_S | 4.48 | 1 | 128 | 0.439 | 1.9797 | 0.9515 | True |
| `blk.41.ffn_up_exps.weight` | 27 | IQ2_S | 4.48 | 1 | 256 | 0.415 | 2.1731 | 0.9580 | True |
| `blk.41.ffn_up_exps.weight` | 27 | IQ2_S | 4.48 | 2 | 64 | 0.878 | 0.6873 | 0.4397 | True |
| `blk.41.ffn_up_exps.weight` | 27 | IQ2_S | 4.48 | 2 | 128 | 0.829 | 0.7390 | 0.4839 | True |
| `blk.41.ffn_up_exps.weight` | 27 | IQ2_S | 4.48 | 2 | 256 | 0.805 | 0.7811 | 0.4839 | True |
| `blk.41.ffn_up_exps.weight` | 27 | IQ2_S | 4.48 | 3 | 64 | 1.268 | 0.3015 | 0.1645 | True |
| `blk.41.ffn_up_exps.weight` | 27 | IQ2_S | 4.48 | 3 | 128 | 1.220 | 0.3160 | 0.1645 | True |
| `blk.41.ffn_up_exps.weight` | 27 | IQ2_S | 4.48 | 3 | 256 | 1.195 | 0.3252 | 0.1645 | True |
| `blk.41.ffn_up_exps.weight` | 27 | IQ2_S | 4.48 | 4 | 64 | 1.659 | 0.0879 | 0.0651 | True |
| `blk.41.ffn_up_exps.weight` | 27 | IQ2_S | 4.48 | 4 | 128 | 1.610 | 0.0976 | 0.0691 | True |
| `blk.41.ffn_up_exps.weight` | 27 | IQ2_S | 4.48 | 4 | 256 | 1.585 | 0.1079 | 0.0691 | True |
| `blk.13.ffn_gate_exps.weight` | 270 | IQ2_S | 4.48 | 1 | 64 | 0.488 | 1.7792 | 0.9220 | True |
| `blk.13.ffn_gate_exps.weight` | 270 | IQ2_S | 4.48 | 1 | 128 | 0.439 | 1.9816 | 0.9460 | True |
| `blk.13.ffn_gate_exps.weight` | 270 | IQ2_S | 4.48 | 1 | 256 | 0.415 | 2.1757 | 0.9460 | True |
| `blk.13.ffn_gate_exps.weight` | 270 | IQ2_S | 4.48 | 2 | 64 | 0.878 | 0.6874 | 0.4497 | True |
| `blk.13.ffn_gate_exps.weight` | 270 | IQ2_S | 4.48 | 2 | 128 | 0.829 | 0.7390 | 0.4516 | True |
| `blk.13.ffn_gate_exps.weight` | 270 | IQ2_S | 4.48 | 2 | 256 | 0.805 | 0.7811 | 0.4839 | True |
| `blk.13.ffn_gate_exps.weight` | 270 | IQ2_S | 4.48 | 3 | 64 | 1.268 | 0.3015 | 0.1473 | True |
| `blk.13.ffn_gate_exps.weight` | 270 | IQ2_S | 4.48 | 3 | 128 | 1.220 | 0.3159 | 0.1645 | True |
| `blk.13.ffn_gate_exps.weight` | 270 | IQ2_S | 4.48 | 3 | 256 | 1.195 | 0.3252 | 0.1645 | True |
| `blk.13.ffn_gate_exps.weight` | 270 | IQ2_S | 4.48 | 4 | 64 | 1.659 | 0.0880 | 0.0648 | True |
| `blk.13.ffn_gate_exps.weight` | 270 | IQ2_S | 4.48 | 4 | 128 | 1.610 | 0.0977 | 0.0691 | True |
| `blk.13.ffn_gate_exps.weight` | 270 | IQ2_S | 4.48 | 4 | 256 | 1.585 | 0.1082 | 0.0706 | True |
| `blk.13.ffn_up_exps.weight` | 270 | IQ2_S | 4.48 | 1 | 64 | 0.488 | 1.7782 | 0.9580 | True |
| `blk.13.ffn_up_exps.weight` | 270 | IQ2_S | 4.48 | 1 | 128 | 0.439 | 1.9812 | 0.9700 | True |
| `blk.13.ffn_up_exps.weight` | 270 | IQ2_S | 4.48 | 1 | 256 | 0.415 | 2.1748 | 0.9700 | True |
| `blk.13.ffn_up_exps.weight` | 270 | IQ2_S | 4.48 | 2 | 64 | 0.878 | 0.6872 | 0.3257 | True |
| `blk.13.ffn_up_exps.weight` | 270 | IQ2_S | 4.48 | 2 | 128 | 0.829 | 0.7391 | 0.3257 | True |
| `blk.13.ffn_up_exps.weight` | 270 | IQ2_S | 4.48 | 2 | 256 | 0.805 | 0.7811 | 0.3257 | True |
| `blk.13.ffn_up_exps.weight` | 270 | IQ2_S | 4.48 | 3 | 64 | 1.268 | 0.3013 | 0.1645 | True |
| `blk.13.ffn_up_exps.weight` | 270 | IQ2_S | 4.48 | 3 | 128 | 1.220 | 0.3158 | 0.1645 | True |
| `blk.13.ffn_up_exps.weight` | 270 | IQ2_S | 4.48 | 3 | 256 | 1.195 | 0.3251 | 0.1645 | True |
| `blk.13.ffn_up_exps.weight` | 270 | IQ2_S | 4.48 | 4 | 64 | 1.659 | 0.0879 | 0.0634 | True |
| `blk.13.ffn_up_exps.weight` | 270 | IQ2_S | 4.48 | 4 | 128 | 1.610 | 0.0977 | 0.0634 | True |
| `blk.13.ffn_up_exps.weight` | 270 | IQ2_S | 4.48 | 4 | 256 | 1.585 | 0.1082 | 0.0634 | True |
| `blk.27.ffn_gate_exps.weight` | 280 | IQ2_S | 4.48 | 1 | 64 | 0.488 | 1.7778 | 0.9460 | True |
| `blk.27.ffn_gate_exps.weight` | 280 | IQ2_S | 4.48 | 1 | 128 | 0.439 | 1.9802 | 0.9460 | True |
| `blk.27.ffn_gate_exps.weight` | 280 | IQ2_S | 4.48 | 1 | 256 | 0.415 | 2.1739 | 0.9460 | True |
| `blk.27.ffn_gate_exps.weight` | 280 | IQ2_S | 4.48 | 2 | 64 | 0.878 | 0.6873 | 0.4835 | True |
| `blk.27.ffn_gate_exps.weight` | 280 | IQ2_S | 4.48 | 2 | 128 | 0.829 | 0.7389 | 0.4835 | True |
| `blk.27.ffn_gate_exps.weight` | 280 | IQ2_S | 4.48 | 2 | 256 | 0.805 | 0.7809 | 0.4835 | True |
| `blk.27.ffn_gate_exps.weight` | 280 | IQ2_S | 4.48 | 3 | 64 | 1.268 | 0.3015 | 0.1645 | True |
| `blk.27.ffn_gate_exps.weight` | 280 | IQ2_S | 4.48 | 3 | 128 | 1.220 | 0.3159 | 0.1645 | True |
| `blk.27.ffn_gate_exps.weight` | 280 | IQ2_S | 4.48 | 3 | 256 | 1.195 | 0.3251 | 0.1645 | True |
| `blk.27.ffn_gate_exps.weight` | 280 | IQ2_S | 4.48 | 4 | 64 | 1.659 | 0.0879 | 0.0675 | True |
| `blk.27.ffn_gate_exps.weight` | 280 | IQ2_S | 4.48 | 4 | 128 | 1.610 | 0.0977 | 0.0691 | True |
| `blk.27.ffn_gate_exps.weight` | 280 | IQ2_S | 4.48 | 4 | 256 | 1.585 | 0.1081 | 0.0691 | True |
| `blk.27.ffn_up_exps.weight` | 280 | IQ2_S | 4.48 | 1 | 64 | 0.488 | 1.7776 | 0.9460 | True |
| `blk.27.ffn_up_exps.weight` | 280 | IQ2_S | 4.48 | 1 | 128 | 0.439 | 1.9801 | 0.9460 | True |
| `blk.27.ffn_up_exps.weight` | 280 | IQ2_S | 4.48 | 1 | 256 | 0.415 | 2.1729 | 0.9460 | True |
| `blk.27.ffn_up_exps.weight` | 280 | IQ2_S | 4.48 | 2 | 64 | 0.878 | 0.6872 | 0.4839 | True |
| `blk.27.ffn_up_exps.weight` | 280 | IQ2_S | 4.48 | 2 | 128 | 0.829 | 0.7391 | 0.4839 | True |
| `blk.27.ffn_up_exps.weight` | 280 | IQ2_S | 4.48 | 2 | 256 | 0.805 | 0.7809 | 0.4839 | True |
| `blk.27.ffn_up_exps.weight` | 280 | IQ2_S | 4.48 | 3 | 64 | 1.268 | 0.3013 | 0.1645 | True |
| `blk.27.ffn_up_exps.weight` | 280 | IQ2_S | 4.48 | 3 | 128 | 1.220 | 0.3158 | 0.1645 | True |
| `blk.27.ffn_up_exps.weight` | 280 | IQ2_S | 4.48 | 3 | 256 | 1.195 | 0.3250 | 0.1645 | True |
| `blk.27.ffn_up_exps.weight` | 280 | IQ2_S | 4.48 | 4 | 64 | 1.659 | 0.0879 | 0.0691 | True |
| `blk.27.ffn_up_exps.weight` | 280 | IQ2_S | 4.48 | 4 | 128 | 1.610 | 0.0977 | 0.0691 | True |
| `blk.27.ffn_up_exps.weight` | 280 | IQ2_S | 4.48 | 4 | 256 | 1.585 | 0.1080 | 0.0691 | True |
| `blk.41.ffn_gate_exps.weight` | 179 | IQ3_XXS | 5.36 | 1 | 64 | 0.408 | 1.8593 | 0.9688 | True |
| `blk.41.ffn_gate_exps.weight` | 179 | IQ3_XXS | 5.36 | 1 | 128 | 0.367 | 2.0716 | 0.9688 | True |
| `blk.41.ffn_gate_exps.weight` | 179 | IQ3_XXS | 5.36 | 1 | 256 | 0.347 | 2.2723 | 0.9688 | True |
| `blk.41.ffn_gate_exps.weight` | 179 | IQ3_XXS | 5.36 | 2 | 64 | 0.735 | 0.6969 | 0.4839 | True |
| `blk.41.ffn_gate_exps.weight` | 179 | IQ3_XXS | 5.36 | 2 | 128 | 0.694 | 0.7483 | 0.4839 | True |
| `blk.41.ffn_gate_exps.weight` | 179 | IQ3_XXS | 5.36 | 2 | 256 | 0.673 | 0.7907 | 0.4839 | True |
| `blk.41.ffn_gate_exps.weight` | 179 | IQ3_XXS | 5.36 | 3 | 64 | 1.061 | 0.2322 | 0.1564 | True |
| `blk.41.ffn_gate_exps.weight` | 179 | IQ3_XXS | 5.36 | 3 | 128 | 1.020 | 0.2567 | 0.1564 | True |
| `blk.41.ffn_gate_exps.weight` | 179 | IQ3_XXS | 5.36 | 3 | 256 | 1.000 | 0.2783 | 0.1570 | True |
| `blk.41.ffn_gate_exps.weight` | 179 | IQ3_XXS | 5.36 | 4 | 64 | 1.388 | 0.1122 | 0.0674 | True |
| `blk.41.ffn_gate_exps.weight` | 179 | IQ3_XXS | 5.36 | 4 | 128 | 1.347 | 0.1205 | 0.0685 | True |
| `blk.41.ffn_gate_exps.weight` | 179 | IQ3_XXS | 5.36 | 4 | 256 | 1.327 | 0.1277 | 0.0685 | True |
| `blk.41.ffn_up_exps.weight` | 179 | IQ2_S | 4.48 | 1 | 64 | 0.488 | 1.7775 | 0.9700 | True |
| `blk.41.ffn_up_exps.weight` | 179 | IQ2_S | 4.48 | 1 | 128 | 0.439 | 1.9792 | 0.9700 | True |
| `blk.41.ffn_up_exps.weight` | 179 | IQ2_S | 4.48 | 1 | 256 | 0.415 | 2.1720 | 0.9700 | True |
| `blk.41.ffn_up_exps.weight` | 179 | IQ2_S | 4.48 | 2 | 64 | 0.878 | 0.6874 | 0.4487 | True |
| `blk.41.ffn_up_exps.weight` | 179 | IQ2_S | 4.48 | 2 | 128 | 0.829 | 0.7391 | 0.4839 | True |
| `blk.41.ffn_up_exps.weight` | 179 | IQ2_S | 4.48 | 2 | 256 | 0.805 | 0.7809 | 0.4839 | True |
| `blk.41.ffn_up_exps.weight` | 179 | IQ2_S | 4.48 | 3 | 64 | 1.268 | 0.3016 | 0.1613 | True |
| `blk.41.ffn_up_exps.weight` | 179 | IQ2_S | 4.48 | 3 | 128 | 1.220 | 0.3161 | 0.1613 | True |
| `blk.41.ffn_up_exps.weight` | 179 | IQ2_S | 4.48 | 3 | 256 | 1.195 | 0.3252 | 0.1645 | True |
| `blk.41.ffn_up_exps.weight` | 179 | IQ2_S | 4.48 | 4 | 64 | 1.659 | 0.0878 | 0.0648 | True |
| `blk.41.ffn_up_exps.weight` | 179 | IQ2_S | 4.48 | 4 | 128 | 1.610 | 0.0975 | 0.0691 | True |
| `blk.41.ffn_up_exps.weight` | 179 | IQ2_S | 4.48 | 4 | 256 | 1.585 | 0.1079 | 0.0691 | True |

## Best By Sample Under Ratio

| tensor | expert | max ratio | bits | block | ratio | rel L2 |
|---|---:|---:|---:|---:|---:|---:|
| `blk.41.ffn_gate_exps.weight` | 27 | 0.55 | 1 | 64 | 0.408 | 1.8616 |
| `blk.41.ffn_up_exps.weight` | 27 | 0.55 | 1 | 64 | 0.488 | 1.7774 |
| `blk.13.ffn_gate_exps.weight` | 270 | 0.55 | 1 | 64 | 0.488 | 1.7792 |
| `blk.13.ffn_up_exps.weight` | 270 | 0.55 | 1 | 64 | 0.488 | 1.7782 |
| `blk.27.ffn_gate_exps.weight` | 280 | 0.55 | 1 | 64 | 0.488 | 1.7778 |
| `blk.27.ffn_up_exps.weight` | 280 | 0.55 | 1 | 64 | 0.488 | 1.7776 |
| `blk.41.ffn_gate_exps.weight` | 179 | 0.55 | 1 | 64 | 0.408 | 1.8593 |
| `blk.41.ffn_up_exps.weight` | 179 | 0.55 | 1 | 64 | 0.488 | 1.7775 |
| `blk.41.ffn_gate_exps.weight` | 27 | 0.50 | 1 | 64 | 0.408 | 1.8616 |
| `blk.41.ffn_up_exps.weight` | 27 | 0.50 | 1 | 64 | 0.488 | 1.7774 |
| `blk.13.ffn_gate_exps.weight` | 270 | 0.50 | 1 | 64 | 0.488 | 1.7792 |
| `blk.13.ffn_up_exps.weight` | 270 | 0.50 | 1 | 64 | 0.488 | 1.7782 |
| `blk.27.ffn_gate_exps.weight` | 280 | 0.50 | 1 | 64 | 0.488 | 1.7778 |
| `blk.27.ffn_up_exps.weight` | 280 | 0.50 | 1 | 64 | 0.488 | 1.7776 |
| `blk.41.ffn_gate_exps.weight` | 179 | 0.50 | 1 | 64 | 0.408 | 1.8593 |
| `blk.41.ffn_up_exps.weight` | 179 | 0.50 | 1 | 64 | 0.488 | 1.7775 |
| `blk.41.ffn_gate_exps.weight` | 27 | 0.40 | 1 | 128 | 0.367 | 2.0741 |
| `blk.41.ffn_gate_exps.weight` | 179 | 0.40 | 1 | 128 | 0.367 | 2.0716 |

## Reproduce

```bash
.Agent/run-tools/kimi_quant_reencode_bound.py --inventory .Agent/runs/20260706-kimi-d2moe-phase0/kimi-iq3s-expert-inventory.tsv --sample-json .Agent/runs/20260707-gp11-quant-reencode-bound/hot-upgate-sample.json --libggml-base build-cuda-batch/bin/libggml-base.so --out-json .Agent/runs/20260707-gp11-quant-reencode-bound/report.json --out-md .Agent/runs/20260707-gp11-quant-reencode-bound/report.md --bits 1,2,3,4 --blocks 64,128,256 --torch-threads 8
```
