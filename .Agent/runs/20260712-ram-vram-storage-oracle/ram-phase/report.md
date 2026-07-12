# Kimi Phase 5C RAM/page-cache baseline

This report is diagnostic only. It does not claim a SOTA result or change runtime behavior.

## Run Summary

| prompt | phase | tok/s | TTFT ms | RAM peak GiB | final file GiB | final file_mapped MiB | clean unmapped file GiB | decode iouring GiB | decode file delta MiB | decode refault | direct reads | fallback GGUF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `dev_france_regression` | no | 1.51 | 13538.0 | 11.90 | 11.24 | 0.031 | 11.22 | 0.0 | 0.0 | 0 | 0 | 0 |
| `dev_intelligence_general` | no | 1.54 | 10050.8 | 11.74 | 11.23 | 0.031 | 11.21 | 0.0 | 0.0 | 0 | 0 | 0 |

## Interpretation

- No normal low-refault run was found in the selected inputs.

## Next Gate

- Build RAM-slab candidates from dev-only foreground IO/wait profiles.
- Start with pageable, batchable layer/role or pack-contiguous slabs.
- Accept a RAM tier only if it reduces demand wait without increasing TTFT, refaults, direct reclaim, or mixed SSD/RAM batch fragmentation.
