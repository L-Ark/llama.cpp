# GP32 mradermacher i1-IQ1_S header preflight

Timestamp: `2026-07-07T08:45:00+0800`

Branch: `vendor/kimi-speculative-general-token-rate-16gb`

Status: completed header/range preflight; no full model downloaded.

## Input

Repository:

<https://huggingface.co/mradermacher/Kimi-K2.7-Code-i1-GGUF>

File prefix checked:

```text
Kimi-K2.7-Code.i1-IQ1_S.gguf.part1of5
```

Downloaded range:

```text
0-16777215
```

Local remote temp file:

```text
/root/lfz/tmp/gp32-iq1s-header-preflight/Kimi-K2.7-Code.i1-IQ1_S.gguf.part1of5.head16m
```

The 16 MiB prefix binary is not committed. Committed evidence includes:

- `url.txt`
- `sha256.txt`
- `gguf-dump-no-tensors.txt`
- `parsed-header.json`
- `parsed-header-full.json`

SHA256 of the downloaded 16 MiB prefix:

```text
47eb7d745164141fd86ed11bf8479053b5f76db58771a0ab6abab2f09f6b48a4
```

## Method

1. Downloaded only a range prefix with `curl -L -r 0-16777215`.
2. Tried repository `gguf_dump.py`; it parsed far enough to reach tensor data
   but failed because the partial file intentionally lacks full tensor data.
3. Used a minimal GGUF metadata parser to read:
   - GGUF header;
   - all key/value metadata;
   - all tensor metadata entries;
   - no tensor data.

## Metadata

Key fields:

| key | value |
| --- | --- |
| `magic` | `GGUF` |
| `version` | `3` |
| `n_tensors` | `1096` |
| `n_kv` | `61` |
| `general.architecture` | `deepseek2` |
| `general.name` | `Kimi K2.7 Code` |
| `general.file_type` | `24` |
| `general.quantization_version` | `2` |
| `general.size_label` | `384x14B` |
| `deepseek2.block_count` | `61` |
| `deepseek2.expert_count` | `384` |
| `deepseek2.expert_used_count` | `8` |
| `deepseek2.expert_feed_forward_length` | `2048` |
| `deepseek2.leading_dense_block_count` | `1` |

`general.file_type=24` maps to `LLAMA_FTYPE_MOSTLY_IQ1_S` in this checkout.

Tensor type counts:

| GGML type id | type name | tensor count |
| ---: | --- | ---: |
| `0` | `F32` | 365 |
| `10` | `Q2_K` | 8 |
| `13` | `Q5_K` | 1 |
| `16` | `IQ2_XXS` | 61 |
| `19` | `IQ1_S` | 600 |
| `20` | `IQ4_NL` | 61 |

Expert tensor type counts:

| kind/type | count |
| --- | ---: |
| `up/IQ1_S` | 60 |
| `gate/IQ1_S` | 60 |
| `down/IQ1_S` | 57 |
| `down/Q2_K` | 3 |

The non-IQ1_S expert tensors are only three `down` tensors with `Q2_K`.
`Q2_K` is already accepted by the current vendor stream path.

## Compatibility Read

Passes:

- Header is readable from a 16 MiB prefix.
- Metadata is Kimi/deepseek2-compatible.
- Model dimensions match current Kimi assumptions:
  - 61 blocks;
  - 384 experts;
  - 8 active experts;
  - hidden size 7168;
  - expert FFN length 2048.
- `file_type=24` confirms IQ1_S, not only folder naming.
- Expert tensor types are stream-plausible today:
  - `IQ1_S` supported in `moe_stream_batch.cu`;
  - `Q2_K` supported in `moe_stream_batch.cu`.

Risks:

- This is still only metadata, not quality or runtime validation.
- The provider labels `IQ1_S` as "for the desperate", so semantic quality risk
  is high.
- The file uses `.gguf.partNofM` multipart layout. A future full download
  should either stream-concatenate the parts into one final GGUF or verify that
  the loader can consume this exact part layout. The current assumption is that
  concatenation is required.
- Prompt path may still have unsupported/fallback behavior for IQ1_S; runtime
  smoke must record stream activation and CPU fallback.

## Decision

GP32 passes the metadata gate.

The next material step is a full `i1-IQ1_S` download and n32 smoke plan, but it
requires explicit disk cleanup approval first. Current free disk is about
`85G`, while the final concatenated IQ1_S GGUF is about `190.39 GiB`.

No deletion or full download was performed in GP32.

## Next Plan

Before downloading:

1. Ask for explicit approval to delete old, non-SOTA packs:
   - old `kimi-iq3s-france.expert-pack` (`160G`);
   - old `kimi-iq3s-tracefirst-n64-20260630.expert-pack` (`75G`);
   - optionally obsolete overlay packs.
2. Preserve:
   - current IQ3_S model;
   - current SOTA `kimi-iq3s-france-l12-upgate-v2.expert-pack`;
   - current `kimi-iq3s-l1l2down-overlay.expert-pack`;
   - committed run records.
3. Stream-concatenate five `i1-IQ1_S` parts into a final GGUF without keeping
   both all parts and the final file.
4. Run n32 cold-start France smoke under 16GB cgroup.
5. Gate on:
   - quality;
   - stream activation;
   - CPU fallback;
   - TTFT <= current baseline +20%;
   - host RAM <=16GB including page cache.
