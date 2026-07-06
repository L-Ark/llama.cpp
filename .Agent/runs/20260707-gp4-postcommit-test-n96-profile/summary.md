# Kimi general prompt dev baseline

| prompt | quality | tok/s | TTFT ms | decode ms/runs | memory peak GiB | output |
|---|---:|---:|---:|---:|---:|---|
| test_english_factual_01 | pass | 1.48 | 53196.0 | 64332.16/95 | 14.81 | Brazil is the largest country in South America and the fifth-largest in the world by area. It has a population of about 215 million people and its capital is Brasília, while São Pa |
| test_english_factual_02 | pass | 1.49 | 69820.25 | 63006.07/94 | 14.81 | The Moon has phases because it orbits Earth and we see varying amounts of its sunlit half from our perspective. When the Moon is between Earth and the Sun, its lit side faces away  |
| test_reasoning_math_01 | pass | 1.32 | 231632.04 | 41772.79/55 | 14.81 | I need to find the arrival time by adding 2 hours and 15 minutes to 3 PM. 3 PM + 2 hours = 5 PM 5 PM + 15 minutes = 5:15 PM The train arrives at **5:15 PM**. |
| test_coding_01 | pass | 1.14 | 84072.9 | 83682.56/95 | 14.81 | Here's a JavaScript function that returns the largest number in an array: ```javascript function findLargestNumber(arr) { if (arr.length === 0) { return undefined; // or throw an e |
| test_chinese_01 | pass | 1.44 | 70343.09 | 66012.01/95 | 14.81 | 节约用水的原因主要有以下几点： 1. **水资源有限**：地球上淡水资源仅占约2.5%，且大部分难以直接利用，可利用的淡水资源十分稀缺。 2. **人口增长与需求增加**：随着人口增加和经济发展，用水量不断上升，供需矛盾加剧。 3. **保护生态环境**：过度用水会导致河流干涸、湿地减少、地下水位下降，破坏生态系统平衡。 4. **减少能源消耗**：水的抽取 |
| test_mixed_instruction_01 | pass | 1.33 | 79872.16 | 71383.52/95 | 14.81 | Here are three practical tips for preparing a home office for remote work: 1. **Choose a dedicated, quiet space** — Set up in an area away from high-traffic parts of your home to m |
