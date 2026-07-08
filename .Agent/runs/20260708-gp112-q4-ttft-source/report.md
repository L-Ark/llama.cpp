# GP112 Q4 TTFT Source Differential

## gp110_full_q4

- prompts compared: `6`
- mean token rate: `1.367 -> 1.802 tok/s`
- mean TTFT: `98156.1 -> 119898.4 ms`
- max TTFT ratio: `1.611`
- mean decode: `65031.5 -> 47587.5 ms`
- mean iouring wait: `65394.7 -> 38532.2 ms`
- mean iouring bytes: `385.2 -> 409.6 GiB`
- mean prompt fallback: `57101.9 -> 76693.6 ms`
- mean decode fallback: `4522.9 -> 0.0 ms`

| prompt | tok/s | TTFT ratio | TTFT delta ms | decode delta ms | IO wait delta ms | IO GiB delta | prompt fallback delta ms | decode fallback delta ms | upgate hit | down hit | down slot MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_chinese_01` | 1.44->1.88 | 1.254 | 17879.8 | -15442.7 | -25679.8 | 44.5 | 17219.4 | -4562.7 | 44.4->44.4% | 73.5->65.8% | 7.44->7.88 |
| `test_coding_01` | 1.14->1.59 | 1.380 | 31910.0 | -23787.9 | -36836.4 | 38.4 | 22387.6 | -6003.6 | 26.3->26.4% | 66.7->60.6% | 7.44->7.88 |
| `test_english_factual_01` | 1.48->1.91 | 1.611 | 32482.5 | -14479.6 | -23964.1 | 39.2 | 30248.9 | -4449.3 | 45.3->46.2% | 73.4->66.1% | 7.44->7.88 |
| `test_english_factual_02` | 1.49->1.83 | 1.334 | 23308.1 | -24277.1 | -31412.4 | -64.2 | 20644.8 | -4002.0 | 42.0->41.6% | 71.1->64.4% | 7.44->7.88 |
| `test_mixed_instruction_01` | 1.33->1.83 | 1.359 | 28678.6 | -19405.1 | -29467.6 | 32.8 | 27049.8 | -4903.8 | 40.7->42.1% | 71.6->65.1% | 7.44->7.88 |
| `test_reasoning_math_01` | 1.32->1.77 | 0.984 | -3805.0 | -7271.5 | -13814.4 | 55.7 | 0.0 | -3216.0 | 35.8->35.7% | 69.1->62.5% | 7.44->7.88 |

### Interpretation

- TTFT gate fails on `5/6` comparable prompts.
- Mean io_uring wait is lower while TTFT is higher, so the TTFT regression is not explained by decode IO wait.
- Candidate moves more expert bytes, mainly from Q4 down slot/cache changes, but the measured IO wait does not increase in the same direction.

### Prompt fallback role deltas

| key | mean delta ms |
|---|---:|
| `prompt,down` | 8184.8 |
| `prompt,gate` | 5925.4 |
| `prompt,up` | 5481.6 |

### Prompt fallback type deltas

| key | mean delta ms |
|---|---:|
| `prompt,type=22` | 6407.8 |
| `prompt,type=11` | 5099.3 |
| `prompt,type=18` | 4999.2 |
| `prompt,type=23` | 1941.4 |
| `prompt,type=2` | 1144.1 |

## gp111_decode_only_q4

- prompts compared: `2`
- mean token rate: `1.310 -> 1.770 tok/s`
- mean TTFT: `68634.4 -> 95308.6 ms`
- max TTFT ratio: `1.586`
- mean decode: `74007.4 -> 54177.9 ms`
- mean iouring wait: `74790.0 -> 44179.4 ms`
- mean iouring bytes: `439.3 -> 478.1 GiB`
- mean prompt fallback: `66098.4 -> 88166.6 ms`
- mean decode fallback: `5226.4 -> 0.0 ms`

| prompt | tok/s | TTFT ratio | TTFT delta ms | decode delta ms | IO wait delta ms | IO GiB delta | prompt fallback delta ms | decode fallback delta ms | upgate hit | down hit | down slot MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_coding_01` | 1.14->1.60 | 1.264 | 22198.6 | -24347.0 | -37351.4 | 38.4 | 13280.0 | -6003.6 | 26.3->26.4% | 66.7->60.6% | 7.44->7.88 |
| `test_english_factual_01` | 1.48->1.94 | 1.586 | 31149.7 | -15311.9 | -23869.7 | 39.2 | 30856.3 | -4449.3 | 45.3->46.2% | 73.4->66.1% | 7.44->7.88 |

### Interpretation

- TTFT gate fails on `2/2` comparable prompts.
- Mean io_uring wait is lower while TTFT is higher, so the TTFT regression is not explained by decode IO wait.
- Candidate moves more expert bytes, mainly from Q4 down slot/cache changes, but the measured IO wait does not increase in the same direction.

### Prompt fallback role deltas

| key | mean delta ms |
|---|---:|
| `prompt,down` | 9287.0 |
| `prompt,gate` | 6631.8 |
| `prompt,up` | 6149.3 |

### Prompt fallback type deltas

| key | mean delta ms |
|---|---:|
| `prompt,type=22` | 7016.3 |
| `prompt,type=11` | 5977.5 |
| `prompt,type=18` | 5764.9 |
| `prompt,type=23` | 1919.0 |
| `prompt,type=2` | 1390.5 |
