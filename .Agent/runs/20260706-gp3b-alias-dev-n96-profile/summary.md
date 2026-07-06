# Kimi general prompt dev baseline

| prompt | quality | tok/s | TTFT ms | decode ms/runs | memory peak GiB | output |
|---|---:|---:|---:|---:|---:|---|
| dev_france_regression | pass | 1.37 | 77277.87 | 56050.91/77 | 14.81 | France is a country in Western Europe known for its rich history, culture, and influence on art, fashion, and cuisine. Its capital, Paris, is famous for landmarks like the Eiffel T |
| dev_japan_factual | pass | 1.0 | 76059.99 | 84599.85/85 | 14.81 | Japan is an island nation in East Asia known for its unique blend of ancient traditions and cutting-edge modernity. With a population of about 125 million, it features bustling cit |
| dev_photosynthesis_factual | pass | 0.8 | 66326.23 | 118019.29/94 | 14.81 | Photosynthesis is the process by which plants, algae, and some bacteria convert sunlight, water, and carbon dioxide into glucose (a sugar) and oxygen. It takes place mainly in the  |
| dev_linear_equation | pass | 0.52 | 97004.71 | 65967.93/34 | 14.81 | To solve for x, subtract 3 from both sides: x + 3 = 10 x = 10 − 3 **x = 7** |
| dev_python_reverse | pass | 0.7 | 79811.88 | 135110.05/95 | 14.81 | Here's a Python function to reverse a string: ```python def reverse_string(s): return s[::-1] ``` **Example usage:** ```python text = "Hello, World!" reversed_text = reverse_string |
| dev_zh_france | pass | 0.84 | 70104.82 | 58142.21/49 | 14.81 | 法国是西欧国家，首都巴黎。面积约55万平方公里，人口约6700万。法国以浪漫文化、美食、葡萄酒和艺术闻名，拥有埃菲尔铁塔、卢浮宫等著名景点。经济发达，是欧盟重要成员国。 |
| dev_mixed_summary | pass | 0.74 | 97707.15 | 73139.15/54 | 14.81 | Solar power offers predictable daytime generation and works well on rooftops, making it easy for small towns to scale gradually. Wind power can produce energy day and night but dep |
