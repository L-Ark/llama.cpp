# GP112 Q4 TTFT Source Differential

## paired_q4_on

- prompts compared: `2`
- mean token rate: `1.680 -> 1.780 tok/s`
- mean TTFT: `81057.6 -> 85306.4 ms`
- max TTFT ratio: `1.131`
- mean decode: `57138.6 -> 53946.5 ms`
- mean iouring wait: `41987.5 -> 44716.0 ms`
- mean iouring bytes: `439.3 -> 478.1 GiB`
- mean prompt fallback: `73735.6 -> 81672.6 ms`
- mean decode fallback: `5356.2 -> 0.0 ms`

| prompt | tok/s | TTFT ratio | TTFT delta ms | decode delta ms | IO wait delta ms | IO GiB delta | prompt fallback delta ms | decode fallback delta ms | upgate hit | down hit | down slot MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `test_coding_01` | 1.52->1.60 | 0.997 | -287.0 | -3168.8 | 3023.4 | 38.4 | 1934.1 | -5770.9 | 26.3->26.4% | 66.7->60.6% | 7.44->7.88 |
| `test_english_factual_01` | 1.84->1.96 | 1.131 | 8784.6 | -3215.3 | 2433.6 | 39.2 | 13940.0 | -4941.5 | 45.3->46.2% | 73.4->66.1% | 7.44->7.88 |

### Interpretation

- TTFT gate passes on comparable prompts.
- Candidate moves more expert bytes, mainly from Q4 down slot/cache changes, but the measured IO wait does not increase in the same direction.

### Prompt fallback role deltas

| key | mean delta ms |
|---|---:|
| `prompt,down` | 3044.1 |
| `prompt,gate` | 2532.3 |
| `prompt,up` | 2360.7 |

### Prompt fallback type deltas

| key | mean delta ms |
|---|---:|
| `prompt,type=18` | 3207.5 |
| `prompt,type=11` | 2307.8 |
| `prompt,type=22` | 1685.5 |
| `prompt,type=2` | 396.3 |
| `prompt,type=23` | 340.0 |
