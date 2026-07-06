# Kimi v2 shadow pack from dev routes

- Output pack: `.Agent/runs/20260707-gp45-v2-shadow-pack-from-dev-routes/dev-hotset-iq1s-r055.shadow-v2.expert-pack`
- Packed type id: `24`
- Packed ratio: `0.55`
- Selected entries: `4096`
- Input profiles: `7`

This is a metadata-only pack for `GGML_MOE_EXPERT_PACK_V2` shadow profiling.
It does not contain real expert payload bytes and must not be used for runtime H2D.

## Aggregate

| events | event coverage | byte coverage | hybrid byte ratio | logical GiB | hybrid GiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 675560 | 0.3597 | 0.3666 | 0.8350 | 3506.568 | 2928.115 |

## By Kind

| kind | events | event coverage | byte coverage | hybrid byte ratio |
| --- | ---: | ---: | ---: | ---: |
| down | 206968 | 0.4153 | 0.4195 | 0.8112 |
| gate | 234296 | 0.3466 | 0.3475 | 0.8436 |
| up | 234296 | 0.3237 | 0.3238 | 0.8543 |
