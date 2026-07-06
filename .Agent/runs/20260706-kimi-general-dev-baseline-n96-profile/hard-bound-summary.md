# Kimi general dev baseline hard-bound analysis

This report is derived from the formal n96 dev baseline metrics. It does
not use held-out test prompts.

- prompts: `7`
- min token rate: `0.16 tok/s`
- median token rate: `0.24 tok/s`
- mean token rate: `0.404 tok/s`

| prompt | tok/s | decode ms/runs | target ms @5 tok/s | need speedup | iouring GiB | iouring wait ms | wait share | visible H2D ms | visible host-stage ms | tok/s if wait=0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `dev_france_regression` | 1.26 | 61015/77 | 15400 | 4.0x | 293.72 | 52397 | 85.9% | 12536 | 8358 | 8.94 |
| `dev_japan_factual` | 0.42 | 202575/85 | 17000 | 11.9x | 136.73 | 26679 | 13.2% | 14059 | 204332 | 0.48 |
| `dev_linear_equation` | 0.16 | 218566/34 | 6800 | 32.1x | 12.86 | 2392 | 1.1% | 7592 | 269610 | 0.16 |
| `dev_mixed_summary` | 0.20 | 266891/54 | 10800 | 24.7x | 32.06 | 6673 | 2.5% | 10382 | 318865 | 0.21 |
| `dev_photosynthesis_factual` | 0.24 | 393774/94 | 18800 | 20.9x | 62.77 | 13517 | 3.4% | 16788 | 463721 | 0.25 |
| `dev_python_reverse` | 0.17 | 566553/95 | 19000 | 29.8x | 33.28 | 6939 | 1.2% | 18523 | 693121 | 0.17 |
| `dev_zh_france` | 0.38 | 128862/49 | 9800 | 13.1x | 58.30 | 11866 | 9.2% | 7781 | 137578 | 0.42 |

Interpretation:

- France remains the only prompt where removing visible iouring wait would
  theoretically exceed `5 tok/s`.
- For non-France/general prompts, visible iouring wait is too small a share
  of decode time to explain the gap. Eliminating it would still leave
  `0.16-0.48 tok/s` for the slow prompts.
- The next optimization branch must therefore reduce host staging / CPU-side
  fallback / expert materialization cost and bytes moved per routed expert,
  not only increase iouring queue depth or fixed hotset hit rate.
