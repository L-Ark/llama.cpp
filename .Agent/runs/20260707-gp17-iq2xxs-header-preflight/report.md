# AesSedai IQ2_XXS Header Preflight

Generated at: `2026-07-07T02:53:41+0800`

This is a metadata-only compatibility preflight for the smallest external GGUF
candidate found in GP15. It downloads only the first split header shard
(`6.6 MiB`), not the full model.

## Remote State

- Repo: `/root/lfz/llama.cpp-vendor-kimi-gp2-6b5c`
- Branch: `vendor/kimi-general-prompt-token-rate-16gb`
- Commit: `4f3a2518f`

## Downloaded Header Shard

```text
URL: https://huggingface.co/AesSedai/Kimi-K2.7-Code-GGUF/resolve/main/IQ2_XXS/Kimi-K2.7-Code-IQ2_XXS-00001-of-00007.gguf
Local temp path: /root/lfz/tmp/gp17-iq2xxs-header/Kimi-K2.7-Code-IQ2_XXS-00001-of-00007.gguf
Size: 6.6 MiB
SHA256: d89b9a9945205f70dbe5bce6f79ff1047b6295efc98237b0936637dc34052298
```

The binary shard itself is not committed to git.

## GGUF Tool Check

Command:

```bash
build-cuda-batch/bin/llama-gguf \
  /root/lfz/tmp/gp17-iq2xxs-header/Kimi-K2.7-Code-IQ2_XXS-00001-of-00007.gguf \
  r n
```

Result:

- GGUF version: `3`
- alignment: `32`
- key/value entries: `57`
- tensors in first shard: `0`
- the tool can read the header in no-check mode.

## Decoded Metadata

```text
general.architecture='deepseek2'
general.name='Kimi K2.7 Code'
general.size_label='384x14B'
general.file_type=20
deepseek2.block_count=61
deepseek2.expert_count=384
deepseek2.expert_used_count=8
deepseek2.expert_feed_forward_length=2048
deepseek2.leading_dense_block_count=1
split.no=0
split.count=7
split.tensors.count=1096
general.quantization_version=2
n_tensors=0
n_fields=60
```

## Compatibility Notes

- The model metadata is `deepseek2`, matching the Kimi/DeepSeek-MoE loader path.
- The split structure is visible and expects `7` shards.
- Current source includes CPU and CUDA support for `GGML_TYPE_IQ2_XXS` and
  `GGML_TYPE_IQ2_XS`, including CUDA MMQ/MMVQ entries and `moe_stream_batch`
  handling.
- Important mismatch:
  - the Hugging Face path says `IQ2_XXS`;
  - `general.file_type=20` maps to `LLAMA_FTYPE_MOSTLY_IQ2_XS`, not
    `LLAMA_FTYPE_MOSTLY_IQ2_XXS` (`19`) in this checkout.

## Decision

The candidate is not blocked at metadata/header level. It should be treated as a
low-byte GGUF candidate, but the exact quantization type must be verified after
full download from tensor metadata and runtime logs rather than inferred from
the folder name alone.

The next blocker remains disk space: full download needs about `262.79 GiB`,
while the server currently has `86G` free unless the GP16 cleanup is approved.
