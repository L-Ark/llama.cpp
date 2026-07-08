# GP112 Q4 TTFT Source Differential

## heldout_q4_on

- prompts compared: `6`
- mean token rate: `1.713 -> 1.820 tok/s`
- mean TTFT: `108041.2 -> 105251.7 ms`
- max TTFT ratio: `1.131`
- mean decode: `51468.0 -> 47041.3 ms`
- mean iouring wait: `37485.4 -> 38834.5 ms`
- mean iouring bytes: `385.2 -> 409.6 GiB`
- mean prompt fallback: `65451.1 -> 68475.5 ms`
- mean decode fallback: `4991.6 -> 0.0 ms`

| prompt | tok/s | TTFT ratio | TTFT delta ms | decode delta ms | IO wait delta ms | IO GiB delta | prompt fallback delta ms | decode fallback delta ms | upgate hit | down hit | down slot MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_chinese_01` | 1.81->1.88 | 1.008 | 617.5 | -2055.2 | 2877.5 | 44.5 | 3924.4 | -5235.5 | 44.4->44.4% | 73.5->65.8% | 7.44->7.88 |
| `test_coding_01` | 1.52->1.60 | 0.997 | -287.0 | -3168.8 | 3023.4 | 38.4 | 1934.1 | -5770.9 | 26.3->26.4% | 66.7->60.6% | 7.44->7.88 |
| `test_english_factual_01` | 1.84->1.96 | 1.131 | 8784.6 | -3215.3 | 2433.6 | 39.2 | 13940.0 | -4941.5 | 45.3->46.2% | 73.4->66.1% | 7.44->7.88 |
| `test_english_factual_02` | 1.79->1.85 | 1.057 | 4920.8 | -14059.4 | -6670.3 | -64.2 | 3025.4 | -4729.8 | 42.0->41.6% | 71.1->64.4% | 7.44->7.88 |
| `test_mixed_instruction_01` | 1.72->1.86 | 0.927 | -7378.6 | -4276.4 | 2236.7 | 32.8 | -4677.9 | -5722.5 | 40.7->42.1% | 71.6->65.1% | 7.44->7.88 |
| `test_reasoning_math_01` | 1.60->1.77 | 0.894 | -23393.6 | 214.8 | 4193.5 | 55.7 | 0.0 | -3549.5 | 35.8->35.7% | 69.1->62.5% | 7.44->7.88 |

### Interpretation

- TTFT gate passes on comparable prompts.
- Candidate moves more expert bytes, mainly from Q4 down slot/cache changes, but the measured IO wait does not increase in the same direction.

### Prompt fallback role deltas

| key | mean delta ms |
|---|---:|
| `prompt,down` | 1456.5 |
| `prompt,up` | 836.6 |
| `prompt,gate` | 731.3 |

### Prompt fallback type deltas

| key | mean delta ms |
|---|---:|
| `prompt,type=18` | 1251.4 |
| `prompt,type=11` | 1180.8 |
| `prompt,type=22` | 316.5 |
| `prompt,type=23` | 173.0 |
| `prompt,type=2` | 102.7 |
