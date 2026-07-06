# Kimi External Low-Byte Asset Survey

Generated at: `2026-07-07T02:46:48+0800`

This survey checks whether an externally available lower-byte Kimi asset can
unblock the 16GB Host RAM + 32GB RTX 5090 target after GP10-GP14 ruled out the
current IQ3_S runtime-only paths.

## Sources

- AesSedai Kimi-K2.7-Code GGUF:
  `https://huggingface.co/AesSedai/Kimi-K2.7-Code-GGUF`
- unsloth Kimi-K2.7-Code GGUF:
  `https://huggingface.co/unsloth/Kimi-K2.7-Code-GGUF`
- decart-ai Kimi-K2.7-Code NVFP4:
  `https://huggingface.co/decart-ai/Kimi-K2.7-Code-NVFP4`
- AMD Kimi-K2.7-Code MXFP4:
  `https://huggingface.co/amd/Kimi-K2.7-Code-MXFP4`

Sizes were measured with HTTP HEAD requests against each file's
`/resolve/main/...` URL and summing `Content-Length`. No model weights were
downloaded.

## Candidate Sizes

| repo | format | files | total GiB | total GB | direct vendor path? |
|---|---|---:|---:|---:|---|
| `AesSedai/Kimi-K2.7-Code-GGUF` | `IQ2_XXS` GGUF | 7 | 262.79 | 282.17 | yes, GGUF after download |
| `unsloth/Kimi-K2.7-Code-GGUF` | `UD-IQ1_M` GGUF | 8 | 283.04 | 303.91 | maybe, depends on type support |
| `unsloth/Kimi-K2.7-Code-GGUF` | `UD-IQ2_M` GGUF | 8 | 296.14 | 317.97 | maybe, depends on type support |
| `unsloth/Kimi-K2.7-Code-GGUF` | `UD-IQ2_XXS` GGUF | 8 | 296.00 | 317.83 | maybe, depends on type support |
| `AesSedai/Kimi-K2.7-Code-GGUF` | `IQ2_S` GGUF | 8 | 311.80 | 334.80 | yes, GGUF after download |
| `amd/Kimi-K2.7-Code-MXFP4` | MXFP4 safetensors | 64 | 514.87 | 552.84 | no, requires conversion/runtime work |
| `decart-ai/Kimi-K2.7-Code-NVFP4` | NVFP4 safetensors | 60 | 554.31 | 595.19 | no, requires conversion/runtime work |

## Current Server Constraint

- Current free disk from GP14:
  - `/dev/root`: `86G` free.
- Therefore none of the externally available GGUF candidates can be downloaded
  without freeing at least hundreds of GiB.
- Downloading NVFP4/MXFP4 safetensors would need even more disk and would not
  directly plug into the current GGUF/expert-pack runtime path.

## Decision

The most plausible immediate candidate is
`AesSedai/Kimi-K2.7-Code-GGUF` `IQ2_XXS`, because it is the smallest GGUF
variant found at about `262.79 GiB`.

It still cannot be tested now because the server has only `86G` free. Testing it
would require an explicit disk-space operation, such as moving or deleting old
artifacts. That should be a separate user-approved step because it may require
removing current IQ3_S model shards or old run artifacts.

NVFP4 is available externally as safetensors, but it is not a direct solution
for this vendor runtime:

- it is larger than the GGUF low-bit candidates;
- it is not currently in GGUF/expert-pack form;
- it would require conversion and kernel/runtime support before the 16GB
  cold-start gates can even be evaluated.
