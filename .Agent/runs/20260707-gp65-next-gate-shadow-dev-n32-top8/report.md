# Kimi next-gate shadow sweep summary

- Prompt file: `.Agent/evals/kimi-general-dev-prompts.jsonl`
- Mode: dev-only unless explicitly stated by the caller
- N: `32`

| label | prompts | recall | precision | false/actual | quality failures | max record us | avg tok/s | max memory bytes |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| top8-n32-dev | 7/7 | 77.35% | 77.35% | 0.2265 | dev_linear_equation | 2633 | 1.659 | 15899996160 |

## top8-n32-dev

- Root: `/root/lfz/runs/vendor-kimi-token-rate/20260707-gp65-next-gate-shadow-dev-n32-top8`
- Missing CSV: `[]`
- Matched pairs: `12803`
- False-prefetch byte units: `23199`

| prompt | quality | tok/s | pairs | recall | precision | false/actual | record avg us | output |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| dev_france_regression | pass | 1.76 | 1829 | 79.34% | 79.34% | 0.2066 | 1768.6 | France is a country in Western Europe known for its rich history, culture, and influence on art, fas |
| dev_japan_factual | pass | 1.72 | 1829 | 79.70% | 79.70% | 0.2030 | 1715.3 | Japan is an island nation in East Asia known for its unique blend of ancient traditions and cutting- |
| dev_linear_equation | fail | 1.47 | 1829 | 71.11% | 71.11% | 0.2889 | 1788.5 | To solve for x, subtract 3 from both sides: x + 3 = 10 x = 10 − 3 **x = |
| dev_mixed_summary | pass | 1.58 | 1829 | 78.03% | 78.03% | 0.2197 | 1762.4 | Solar power offers predictable daytime generation and works well on rooftops, making it easy for sma |
| dev_photosynthesis_factual | pass | 1.70 | 1829 | 79.11% | 79.11% | 0.2089 | 1769.3 | Photosynthesis is the process by which plants, algae, and some bacteria convert sunlight, water, and |
| dev_python_reverse | pass | 1.57 | 1829 | 73.30% | 73.30% | 0.2670 | 1817.2 | Here's a Python function to reverse a string: ```python def reverse_string(s): return s[::-1] ``` ** |
| dev_zh_france | pass | 1.81 | 1829 | 80.86% | 80.86% | 0.1914 | 1774.9 | 法国是西欧国家，首都巴黎。面积约55万平方公里，人口约6700万。法国以浪漫文化、美食、葡萄酒和艺术闻名，拥有埃菲尔 |
