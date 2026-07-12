# Kimi static expert-prior predictability

- Input traces: `2`
- Folds: `2`

Method: leave-one-prompt-out; train per-tensor top-K experts on the other dev prompts.

| top K | recall | precision | byte recall | predicted/actual bytes |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 0.0908 | 0.1118 | 0.0908 | 0.8117 |
| 16 | 0.1572 | 0.0969 | 0.1573 | 1.6233 |
| 32 | 0.2455 | 0.0756 | 0.2454 | 3.2467 |
| 64 | 0.3789 | 0.0584 | 0.3788 | 6.4933 |
| 128 | 0.5728 | 0.0462 | 0.5729 | 12.3965 |
| 192 | 0.5832 | 0.0452 | 0.5836 | 12.9049 |
| 256 | 0.5832 | 0.0452 | 0.5836 | 12.9049 |

## By Kind At Largest K

| kind | recall | precision | byte recall | predicted/actual bytes |
| --- | ---: | ---: | ---: | ---: |
| down | 0.5832 | 0.0452 | 0.5855 | 12.8706 |
| gate | 0.5832 | 0.0452 | 0.5806 | 12.9265 |
| up | 0.5832 | 0.0452 | 0.5841 | 12.9295 |
