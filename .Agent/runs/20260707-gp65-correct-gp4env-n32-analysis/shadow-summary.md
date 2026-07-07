# Kimi next-gate shadow sweep summary

- Prompt file: `.Agent/evals/kimi-general-dev-prompts.jsonl`
- Mode: dev-only unless explicitly stated by the caller
- N: `32`

| label | prompts | recall | precision | false/actual | quality failures | max record us | avg tok/s | max memory bytes |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| top8_gp4env_n32 | 7/7 | 77.35% | 77.35% | 0.2265 | dev_linear_equation | 17089 | 1.594 | 15899996160 |

## top8_gp4env_n32

- Root: `/root/lfz/runs/vendor-kimi-token-rate/20260707-gp65-correct-gp4env-n32-shadow-top8`
- Missing CSV: `[]`
- Matched pairs: `12803`
- False-prefetch byte units: `23199`

| prompt | quality | tok/s | pairs | recall | precision | false/actual | record avg us | output |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| dev_france_regression | pass | 1.71 | 1829 | 79.34% | 79.34% | 0.2066 | 1787.1 | France is a country in Western Europe known for its rich history, culture, and influence on art, fas |
| dev_japan_factual | pass | 1.67 | 1829 | 79.70% | 79.70% | 0.2030 | 1719.6 | Japan is an island nation in East Asia known for its unique blend of ancient traditions and cutting- |
| dev_linear_equation | fail | 1.42 | 1829 | 71.11% | 71.11% | 0.2889 | 1731.2 | To solve for x, subtract 3 from both sides: x + 3 = 10 x = 10 − 3 **x = |
| dev_mixed_summary | pass | 1.52 | 1829 | 78.03% | 78.03% | 0.2197 | 1745.1 | Solar power offers predictable daytime generation and works well on rooftops, making it easy for sma |
| dev_photosynthesis_factual | pass | 1.56 | 1829 | 79.11% | 79.11% | 0.2089 | 2257.4 | Photosynthesis is the process by which plants, algae, and some bacteria convert sunlight, water, and |
| dev_python_reverse | pass | 1.54 | 1829 | 73.30% | 73.30% | 0.2670 | 1824.9 | Here's a Python function to reverse a string: ```python def reverse_string(s): return s[::-1] ``` ** |
| dev_zh_france | pass | 1.74 | 1829 | 80.86% | 80.86% | 0.1914 | 1764.9 | 法国是西欧国家，首都巴黎。面积约55万平方公里，人口约6700万。法国以浪漫文化、美食、葡萄酒和艺术闻名，拥有埃菲尔 |
