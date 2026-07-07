# Kimi next-gate shadow sweep summary

- Prompt file: `.Agent/evals/kimi-general-dev-prompts.jsonl`
- Mode: dev-only unless explicitly stated by the caller
- N: `96`

| label | prompts | recall | precision | false/actual | quality failures | max record us | avg tok/s | max memory bytes |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| top8_gp4env_n96 | 7/7 | 77.25% | 77.25% | 0.2275 | none | 2718 | 1.677 | 15899996160 |

## top8_gp4env_n96

- Root: `/root/lfz/runs/vendor-kimi-token-rate/20260707-gp65-correct-gp4env-n96-shadow-top8`
- Missing CSV: `[]`
- Matched pairs: `28792`
- False-prefetch byte units: `52400`

| prompt | quality | tok/s | pairs | recall | precision | false/actual | record avg us | output |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| dev_france_regression | pass | 1.78 | 4543 | 78.91% | 78.91% | 0.2109 | 1757.6 | France is a country in Western Europe known for its rich history, culture, and influence on art, fas |
| dev_japan_factual | pass | 1.80 | 5015 | 79.53% | 79.53% | 0.2047 | 1710.1 | Japan is an island nation in East Asia known for its unique blend of ancient traditions and cutting- |
| dev_linear_equation | pass | 1.44 | 2006 | 70.84% | 70.84% | 0.2916 | 1792.1 | To solve for x, subtract 3 from both sides: x + 3 = 10 x = 10 − 3 **x = 7** |
| dev_mixed_summary | pass | 1.59 | 3186 | 77.92% | 77.92% | 0.2208 | 1759.2 | Solar power offers predictable daytime generation and works well on rooftops, making it easy for sma |
| dev_photosynthesis_factual | pass | 1.75 | 5546 | 77.80% | 77.80% | 0.2220 | 1752.0 | Photosynthesis is the process by which plants, algae, and some bacteria convert sunlight, water, and |
| dev_python_reverse | pass | 1.62 | 5605 | 73.51% | 73.51% | 0.2649 | 1799.2 | Here's a Python function to reverse a string: ```python def reverse_string(s): return s[::-1] ``` ** |
| dev_zh_france | pass | 1.76 | 2891 | 80.59% | 80.59% | 0.1941 | 1804.0 | 法国是西欧国家，首都巴黎。面积约55万平方公里，人口约6700万。法国以浪漫文化、美食、葡萄酒和艺术闻名，拥有埃菲尔铁塔、卢浮宫等著名景点。经济发达，是欧盟重要成员国。 |
