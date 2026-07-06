# Kimi route predictor bound

This is an offline dev-trace diagnostic. It does not use held-out test prompts.

## Aggregate

| predictor | k | group | steps | byte recall | full steps | pred/actual bytes | hit GiB | wasted GiB |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| dev_global_lfu | 8 | down | 25871 | 12.3% | 0.0% | 1.00x | 158.01 | 1122.90 |
| dev_global_lfu | 16 | down | 25871 | 19.3% | 0.0% | 2.00x | 247.39 | 2314.43 |
| dev_global_lfu | 32 | down | 25871 | 28.6% | 0.1% | 4.00x | 366.14 | 4757.51 |
| history_lfu | 8 | down | 25871 | 29.9% | 0.0% | 0.99x | 382.81 | 879.73 |
| history_lfu | 16 | down | 25871 | 40.2% | 0.2% | 1.95x | 514.30 | 1984.77 |
| history_lfu | 32 | down | 25871 | 51.8% | 1.3% | 3.79x | 663.52 | 4196.15 |
| hybrid_lfu | 8 | down | 25871 | 25.7% | 0.0% | 1.00x | 328.62 | 952.29 |
| hybrid_lfu | 16 | down | 25871 | 35.5% | 0.0% | 2.00x | 454.36 | 2107.46 |
| hybrid_lfu | 32 | down | 25871 | 47.7% | 0.6% | 4.00x | 611.41 | 4512.24 |
| previous_same_layer | 0 | down | 25871 | 33.9% | 0.1% | 0.99x | 433.99 | 828.55 |
| dev_global_lfu | 8 | upgate | 29287 | 12.3% | 0.0% | 1.00x | 274.20 | 1951.46 |
| dev_global_lfu | 16 | upgate | 29287 | 19.2% | 0.0% | 2.00x | 426.82 | 4024.50 |
| dev_global_lfu | 32 | upgate | 29287 | 28.4% | 0.1% | 4.00x | 631.57 | 8271.06 |
| history_lfu | 8 | upgate | 29287 | 29.9% | 0.0% | 0.99x | 665.92 | 1527.82 |
| history_lfu | 16 | upgate | 29287 | 40.1% | 0.1% | 1.95x | 892.19 | 3450.10 |
| history_lfu | 32 | upgate | 29287 | 51.6% | 1.2% | 3.79x | 1149.48 | 7295.59 |
| hybrid_lfu | 8 | upgate | 29287 | 25.7% | 0.0% | 1.00x | 571.86 | 1653.79 |
| hybrid_lfu | 16 | upgate | 29287 | 35.4% | 0.0% | 2.00x | 788.02 | 3663.29 |
| hybrid_lfu | 32 | upgate | 29287 | 47.6% | 0.6% | 4.00x | 1059.70 | 7842.92 |
| previous_same_layer | 0 | upgate | 29287 | 33.6% | 0.1% | 0.99x | 748.60 | 1445.14 |

## Per Prompt

| prompt | predictor | k | group | byte recall | full steps | pred/actual bytes | hit GiB | wasted GiB |
|---|---|---:|---|---:|---:|---:|---:|---:|
| `dev_france_regression` | dev_global_lfu | 8 | down | 22.0% | 0.0% | 1.00x | 44.50 | 157.61 |
| `dev_france_regression` | dev_global_lfu | 16 | down | 33.6% | 0.0% | 2.00x | 67.85 | 336.37 |
| `dev_france_regression` | dev_global_lfu | 32 | down | 46.3% | 0.2% | 4.00x | 93.65 | 714.77 |
| `dev_france_regression` | history_lfu | 8 | down | 33.7% | 0.0% | 0.99x | 68.12 | 131.36 |
| `dev_france_regression` | history_lfu | 16 | down | 44.8% | 0.1% | 1.96x | 90.46 | 304.83 |
| `dev_france_regression` | history_lfu | 32 | down | 57.1% | 1.6% | 3.81x | 115.40 | 653.74 |
| `dev_france_regression` | hybrid_lfu | 8 | down | 31.9% | 0.0% | 1.00x | 64.47 | 137.64 |
| `dev_france_regression` | hybrid_lfu | 16 | down | 42.5% | 0.0% | 2.00x | 85.99 | 318.22 |
| `dev_france_regression` | hybrid_lfu | 32 | down | 55.4% | 1.0% | 4.00x | 111.94 | 696.49 |
| `dev_france_regression` | previous_same_layer | 0 | down | 36.7% | 0.0% | 0.99x | 74.15 | 125.33 |
| `dev_france_regression` | dev_global_lfu | 8 | upgate | 21.9% | 0.0% | 1.00x | 76.90 | 274.27 |
| `dev_france_regression` | dev_global_lfu | 16 | upgate | 33.3% | 0.0% | 2.00x | 116.83 | 585.51 |
| `dev_france_regression` | dev_global_lfu | 32 | upgate | 46.0% | 0.2% | 4.00x | 161.67 | 1243.01 |
| `dev_france_regression` | history_lfu | 8 | upgate | 33.5% | 0.0% | 0.99x | 117.55 | 229.06 |
| `dev_france_regression` | history_lfu | 16 | upgate | 44.4% | 0.1% | 1.96x | 155.79 | 531.03 |
| `dev_france_regression` | history_lfu | 32 | upgate | 56.6% | 1.4% | 3.81x | 198.86 | 1137.87 |
| `dev_france_regression` | hybrid_lfu | 8 | upgate | 31.7% | 0.0% | 1.00x | 111.40 | 239.77 |
| `dev_france_regression` | hybrid_lfu | 16 | upgate | 42.2% | 0.0% | 2.00x | 148.09 | 554.25 |
| `dev_france_regression` | hybrid_lfu | 32 | upgate | 54.9% | 0.9% | 4.00x | 192.81 | 1211.88 |
| `dev_france_regression` | previous_same_layer | 0 | upgate | 36.1% | 0.0% | 0.99x | 126.65 | 219.97 |
| `dev_japan_factual` | dev_global_lfu | 8 | down | 18.7% | 0.0% | 1.00x | 41.64 | 181.46 |
| `dev_japan_factual` | dev_global_lfu | 16 | down | 28.1% | 0.0% | 2.00x | 62.61 | 383.59 |
| `dev_japan_factual` | dev_global_lfu | 32 | down | 39.4% | 0.0% | 4.00x | 87.92 | 804.48 |
| `dev_japan_factual` | history_lfu | 8 | down | 33.5% | 0.0% | 0.99x | 74.76 | 145.72 |
| `dev_japan_factual` | history_lfu | 16 | down | 45.0% | 0.2% | 1.96x | 100.32 | 336.90 |
| `dev_japan_factual` | history_lfu | 32 | down | 57.4% | 1.9% | 3.82x | 128.10 | 724.88 |
| `dev_japan_factual` | hybrid_lfu | 8 | down | 30.5% | 0.0% | 1.00x | 68.14 | 154.96 |
| `dev_japan_factual` | hybrid_lfu | 16 | down | 41.5% | 0.0% | 2.00x | 92.53 | 353.67 |
| `dev_japan_factual` | hybrid_lfu | 32 | down | 54.8% | 0.8% | 4.00x | 122.22 | 770.17 |
| `dev_japan_factual` | previous_same_layer | 0 | down | 38.5% | 0.2% | 0.99x | 85.80 | 134.67 |
| `dev_japan_factual` | dev_global_lfu | 8 | upgate | 18.5% | 0.0% | 1.00x | 71.90 | 315.75 |
| `dev_japan_factual` | dev_global_lfu | 16 | upgate | 27.8% | 0.0% | 2.00x | 107.88 | 667.42 |
| `dev_japan_factual` | dev_global_lfu | 32 | upgate | 38.9% | 0.1% | 4.00x | 150.98 | 1399.61 |
| `dev_japan_factual` | history_lfu | 8 | upgate | 33.3% | 0.0% | 0.99x | 129.18 | 253.91 |
| `dev_japan_factual` | history_lfu | 16 | upgate | 44.6% | 0.2% | 1.96x | 172.74 | 586.91 |
| `dev_japan_factual` | history_lfu | 32 | upgate | 56.8% | 1.8% | 3.82x | 220.33 | 1262.24 |
| `dev_japan_factual` | hybrid_lfu | 8 | upgate | 30.4% | 0.0% | 1.00x | 117.75 | 269.89 |
| `dev_japan_factual` | hybrid_lfu | 16 | upgate | 41.1% | 0.0% | 2.00x | 159.18 | 616.12 |
| `dev_japan_factual` | hybrid_lfu | 32 | upgate | 54.2% | 0.7% | 4.00x | 210.17 | 1340.42 |
| `dev_japan_factual` | previous_same_layer | 0 | upgate | 37.8% | 0.2% | 0.99x | 146.46 | 236.63 |
| `dev_linear_equation` | dev_global_lfu | 8 | down | 5.6% | 0.0% | 1.00x | 5.00 | 84.27 |
| `dev_linear_equation` | dev_global_lfu | 16 | down | 10.4% | 0.0% | 2.00x | 9.29 | 169.24 |
| `dev_linear_equation` | dev_global_lfu | 32 | down | 17.9% | 0.0% | 4.00x | 15.97 | 341.10 |
| `dev_linear_equation` | history_lfu | 8 | down | 24.3% | 0.0% | 0.97x | 21.72 | 64.92 |
| `dev_linear_equation` | history_lfu | 16 | down | 33.8% | 0.0% | 1.90x | 30.21 | 139.50 |
| `dev_linear_equation` | history_lfu | 32 | down | 44.2% | 0.3% | 3.61x | 39.49 | 282.71 |
| `dev_linear_equation` | hybrid_lfu | 8 | down | 19.9% | 0.0% | 1.00x | 17.81 | 71.46 |
| `dev_linear_equation` | hybrid_lfu | 16 | down | 28.3% | 0.0% | 2.00x | 25.27 | 153.26 |
| `dev_linear_equation` | hybrid_lfu | 32 | down | 39.9% | 0.3% | 4.00x | 35.63 | 321.44 |
| `dev_linear_equation` | previous_same_layer | 0 | down | 22.1% | 0.0% | 0.97x | 19.77 | 66.88 |
| `dev_linear_equation` | dev_global_lfu | 8 | upgate | 5.8% | 0.0% | 1.00x | 8.97 | 146.13 |
| `dev_linear_equation` | dev_global_lfu | 16 | upgate | 10.4% | 0.0% | 2.00x | 16.19 | 294.03 |
| `dev_linear_equation` | dev_global_lfu | 32 | upgate | 17.9% | 0.0% | 4.00x | 27.76 | 592.68 |
| `dev_linear_equation` | history_lfu | 8 | upgate | 24.6% | 0.0% | 0.97x | 38.11 | 112.44 |
| `dev_linear_equation` | history_lfu | 16 | upgate | 34.0% | 0.0% | 1.90x | 52.71 | 242.19 |
| `dev_linear_equation` | history_lfu | 32 | upgate | 44.4% | 0.2% | 3.61x | 68.82 | 490.72 |
| `dev_linear_equation` | hybrid_lfu | 8 | upgate | 20.3% | 0.0% | 1.00x | 31.48 | 123.63 |
| `dev_linear_equation` | hybrid_lfu | 16 | upgate | 28.6% | 0.0% | 2.00x | 44.41 | 265.81 |
| `dev_linear_equation` | hybrid_lfu | 32 | upgate | 40.1% | 0.2% | 4.00x | 62.20 | 558.24 |
| `dev_linear_equation` | previous_same_layer | 0 | upgate | 22.3% | 0.0% | 0.97x | 34.61 | 115.94 |
| `dev_mixed_summary` | dev_global_lfu | 8 | down | 8.4% | 0.0% | 1.00x | 11.90 | 129.85 |
| `dev_mixed_summary` | dev_global_lfu | 16 | down | 14.7% | 0.0% | 2.00x | 20.78 | 262.72 |
| `dev_mixed_summary` | dev_global_lfu | 32 | down | 22.7% | 0.0% | 4.00x | 32.23 | 534.77 |
| `dev_mixed_summary` | history_lfu | 8 | down | 28.6% | 0.0% | 0.98x | 40.60 | 98.52 |
| `dev_mixed_summary` | history_lfu | 16 | down | 39.0% | 0.0% | 1.94x | 55.22 | 219.27 |
| `dev_mixed_summary` | history_lfu | 32 | down | 51.1% | 0.7% | 3.74x | 72.47 | 458.05 |
| `dev_mixed_summary` | hybrid_lfu | 8 | down | 23.0% | 0.0% | 1.00x | 32.67 | 109.08 |
| `dev_mixed_summary` | hybrid_lfu | 16 | down | 32.9% | 0.0% | 2.00x | 46.68 | 236.82 |
| `dev_mixed_summary` | hybrid_lfu | 32 | down | 45.7% | 0.1% | 4.00x | 64.77 | 502.24 |
| `dev_mixed_summary` | previous_same_layer | 0 | down | 31.5% | 0.0% | 0.98x | 44.59 | 94.54 |
| `dev_mixed_summary` | dev_global_lfu | 8 | upgate | 8.4% | 0.0% | 1.00x | 20.81 | 225.49 |
| `dev_mixed_summary` | dev_global_lfu | 16 | upgate | 14.5% | 0.0% | 2.00x | 35.75 | 456.85 |
| `dev_mixed_summary` | dev_global_lfu | 32 | upgate | 22.6% | 0.0% | 4.00x | 55.55 | 929.66 |
| `dev_mixed_summary` | history_lfu | 8 | upgate | 28.2% | 0.0% | 0.98x | 69.49 | 172.25 |
| `dev_mixed_summary` | history_lfu | 16 | upgate | 38.3% | 0.0% | 1.94x | 94.36 | 382.62 |
| `dev_mixed_summary` | history_lfu | 32 | upgate | 50.3% | 0.6% | 3.74x | 123.94 | 798.26 |
| `dev_mixed_summary` | hybrid_lfu | 8 | upgate | 22.8% | 0.0% | 1.00x | 56.23 | 190.07 |
| `dev_mixed_summary` | hybrid_lfu | 16 | upgate | 32.5% | 0.0% | 2.00x | 80.16 | 412.44 |
| `dev_mixed_summary` | hybrid_lfu | 32 | upgate | 45.0% | 0.1% | 4.00x | 110.92 | 874.28 |
| `dev_mixed_summary` | previous_same_layer | 0 | upgate | 30.7% | 0.0% | 0.98x | 75.59 | 166.15 |
| `dev_photosynthesis_factual` | dev_global_lfu | 8 | down | 7.0% | 0.0% | 1.00x | 17.31 | 229.41 |
| `dev_photosynthesis_factual` | dev_global_lfu | 16 | down | 12.0% | 0.0% | 2.00x | 29.70 | 463.73 |
| `dev_photosynthesis_factual` | dev_global_lfu | 32 | down | 21.0% | 0.0% | 4.00x | 51.73 | 935.14 |
| `dev_photosynthesis_factual` | history_lfu | 8 | down | 32.3% | 0.0% | 0.99x | 79.69 | 164.40 |
| `dev_photosynthesis_factual` | history_lfu | 16 | down | 42.2% | 0.1% | 1.96x | 104.11 | 380.35 |
| `dev_photosynthesis_factual` | history_lfu | 32 | down | 53.1% | 0.9% | 3.85x | 131.10 | 817.90 |
| `dev_photosynthesis_factual` | hybrid_lfu | 8 | down | 25.7% | 0.0% | 1.00x | 63.52 | 183.20 |
| `dev_photosynthesis_factual` | hybrid_lfu | 16 | down | 35.6% | 0.0% | 2.00x | 87.94 | 405.49 |
| `dev_photosynthesis_factual` | hybrid_lfu | 32 | down | 47.7% | 0.3% | 4.00x | 117.64 | 869.23 |
| `dev_photosynthesis_factual` | previous_same_layer | 0 | down | 33.5% | 0.0% | 0.99x | 82.72 | 161.37 |
| `dev_photosynthesis_factual` | dev_global_lfu | 8 | upgate | 7.3% | 0.0% | 1.00x | 31.26 | 397.42 |
| `dev_photosynthesis_factual` | dev_global_lfu | 16 | upgate | 12.2% | 0.0% | 2.00x | 52.21 | 805.15 |
| `dev_photosynthesis_factual` | dev_global_lfu | 32 | upgate | 21.1% | 0.0% | 4.00x | 90.49 | 1624.25 |
| `dev_photosynthesis_factual` | history_lfu | 8 | upgate | 32.4% | 0.0% | 0.99x | 138.69 | 285.43 |
| `dev_photosynthesis_factual` | history_lfu | 16 | upgate | 42.2% | 0.1% | 1.96x | 180.80 | 661.01 |
| `dev_photosynthesis_factual` | history_lfu | 32 | upgate | 53.1% | 1.0% | 3.85x | 227.67 | 1421.64 |
| `dev_photosynthesis_factual` | hybrid_lfu | 8 | upgate | 25.8% | 0.0% | 1.00x | 110.75 | 317.93 |
| `dev_photosynthesis_factual` | hybrid_lfu | 16 | upgate | 35.6% | 0.0% | 2.00x | 152.49 | 704.87 |
| `dev_photosynthesis_factual` | hybrid_lfu | 32 | upgate | 47.8% | 0.3% | 4.00x | 204.86 | 1509.88 |
| `dev_photosynthesis_factual` | previous_same_layer | 0 | upgate | 33.6% | 0.0% | 0.99x | 143.98 | 280.14 |
| `dev_python_reverse` | dev_global_lfu | 8 | down | 4.0% | 0.0% | 1.00x | 10.07 | 239.27 |
| `dev_python_reverse` | dev_global_lfu | 16 | down | 7.4% | 0.0% | 2.00x | 18.37 | 480.32 |
| `dev_python_reverse` | dev_global_lfu | 32 | down | 13.4% | 0.0% | 4.00x | 33.50 | 963.86 |
| `dev_python_reverse` | history_lfu | 8 | down | 20.5% | 0.0% | 0.99x | 51.11 | 195.60 |
| `dev_python_reverse` | history_lfu | 16 | down | 28.9% | 0.0% | 1.97x | 72.09 | 418.04 |
| `dev_python_reverse` | history_lfu | 32 | down | 40.0% | 0.1% | 3.87x | 99.64 | 864.44 |
| `dev_python_reverse` | hybrid_lfu | 8 | down | 15.3% | 0.0% | 1.00x | 38.10 | 211.24 |
| `dev_python_reverse` | hybrid_lfu | 16 | down | 23.2% | 0.0% | 2.00x | 57.88 | 440.80 |
| `dev_python_reverse` | hybrid_lfu | 32 | down | 34.3% | 0.1% | 4.00x | 85.46 | 911.90 |
| `dev_python_reverse` | previous_same_layer | 0 | down | 30.3% | 0.0% | 0.99x | 75.45 | 171.27 |
| `dev_python_reverse` | dev_global_lfu | 8 | upgate | 4.1% | 0.0% | 1.00x | 17.82 | 415.42 |
| `dev_python_reverse` | dev_global_lfu | 16 | upgate | 7.3% | 0.0% | 2.00x | 31.52 | 834.96 |
| `dev_python_reverse` | dev_global_lfu | 32 | upgate | 13.2% | 0.0% | 4.00x | 57.00 | 1675.97 |
| `dev_python_reverse` | history_lfu | 8 | upgate | 21.3% | 0.0% | 0.99x | 92.36 | 336.33 |
| `dev_python_reverse` | history_lfu | 16 | upgate | 29.9% | 0.0% | 1.97x | 129.40 | 722.21 |
| `dev_python_reverse` | history_lfu | 32 | upgate | 40.9% | 0.2% | 3.87x | 177.04 | 1497.63 |
| `dev_python_reverse` | hybrid_lfu | 8 | upgate | 15.9% | 0.0% | 1.00x | 68.81 | 364.43 |
| `dev_python_reverse` | hybrid_lfu | 16 | upgate | 24.0% | 0.0% | 2.00x | 103.97 | 762.52 |
| `dev_python_reverse` | hybrid_lfu | 32 | upgate | 35.0% | 0.1% | 4.00x | 151.66 | 1581.31 |
| `dev_python_reverse` | previous_same_layer | 0 | upgate | 30.7% | 0.0% | 0.99x | 133.12 | 295.57 |
| `dev_zh_france` | dev_global_lfu | 8 | down | 21.5% | 0.0% | 1.00x | 27.60 | 101.03 |
| `dev_zh_france` | dev_global_lfu | 16 | down | 30.2% | 0.0% | 2.00x | 38.80 | 218.46 |
| `dev_zh_france` | dev_global_lfu | 32 | down | 39.8% | 0.1% | 4.00x | 51.14 | 463.38 |
| `dev_zh_france` | history_lfu | 8 | down | 36.4% | 0.0% | 0.98x | 46.80 | 79.20 |
| `dev_zh_france` | history_lfu | 16 | down | 48.1% | 0.7% | 1.93x | 61.89 | 185.88 |
| `dev_zh_france` | history_lfu | 32 | down | 60.1% | 3.8% | 3.67x | 77.31 | 394.44 |
| `dev_zh_france` | hybrid_lfu | 8 | down | 34.1% | 0.0% | 1.00x | 43.92 | 84.71 |
| `dev_zh_france` | hybrid_lfu | 16 | down | 45.1% | 0.2% | 2.00x | 58.07 | 199.19 |
| `dev_zh_france` | hybrid_lfu | 32 | down | 57.3% | 1.8% | 4.00x | 73.75 | 440.77 |
| `dev_zh_france` | previous_same_layer | 0 | down | 40.0% | 0.2% | 0.98x | 51.51 | 74.50 |
| `dev_zh_france` | dev_global_lfu | 8 | upgate | 20.8% | 0.0% | 1.00x | 46.53 | 176.97 |
| `dev_zh_france` | dev_global_lfu | 16 | upgate | 29.7% | 0.0% | 2.00x | 66.44 | 380.57 |
| `dev_zh_france` | dev_global_lfu | 32 | upgate | 39.4% | 0.1% | 4.00x | 88.12 | 805.89 |
| `dev_zh_france` | history_lfu | 8 | upgate | 36.0% | 0.0% | 0.98x | 80.54 | 138.40 |
| `dev_zh_france` | history_lfu | 16 | upgate | 47.6% | 0.6% | 1.93x | 106.40 | 324.13 |
| `dev_zh_france` | history_lfu | 32 | upgate | 59.4% | 3.6% | 3.67x | 132.83 | 687.23 |
| `dev_zh_france` | hybrid_lfu | 8 | upgate | 33.8% | 0.0% | 1.00x | 75.44 | 148.06 |
| `dev_zh_france` | hybrid_lfu | 16 | upgate | 44.6% | 0.1% | 2.00x | 99.72 | 347.28 |
| `dev_zh_france` | hybrid_lfu | 32 | upgate | 56.9% | 1.7% | 4.00x | 127.08 | 766.93 |
| `dev_zh_france` | previous_same_layer | 0 | upgate | 39.5% | 0.2% | 0.98x | 88.20 | 130.74 |

Interpretation gates:

- Runtime predictor work is justified only if slow dev prompts reach at least 65% byte recall,
  at most 1.35x predicted bytes, and at least 40% full-step coverage for the bottleneck group.
- Otherwise prediction cannot plausibly close the gap from the current ~1.4 tok/s SOTA to 5 tok/s.
