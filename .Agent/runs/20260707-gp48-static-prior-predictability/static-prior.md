# Kimi static expert-prior predictability

- Input traces: `7`
- Folds: `7`

Method: leave-one-prompt-out; train per-tensor top-K experts on the other dev prompts.

| top K | recall | precision | byte recall | predicted/actual bytes |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 0.1236 | 0.1236 | 0.1232 | 1.0000 |
| 16 | 0.1925 | 0.0963 | 0.1922 | 2.0000 |
| 32 | 0.2847 | 0.0712 | 0.2846 | 4.0000 |
| 64 | 0.4170 | 0.0521 | 0.4169 | 8.0000 |
| 128 | 0.6010 | 0.0376 | 0.6009 | 16.0000 |

## By Kind At Largest K

| kind | recall | precision | byte recall | predicted/actual bytes |
| --- | ---: | ---: | ---: | ---: |
| down | 0.6009 | 0.0376 | 0.6029 | 16.0000 |
| gate | 0.6011 | 0.0376 | 0.5992 | 16.0000 |
| up | 0.6011 | 0.0376 | 0.6004 | 16.0000 |
