# GP92 Complete Lower-Quant GGUF Bound

Date: 2026-07-08

This is an offline file-list and bound calculation. It does not change runtime
behavior and does not claim SOTA.

## Question

Can an existing complete lower-quant Kimi-K2.7-Code GGUF replace the current
`IQ3_S` model and, by reducing expert bytes, plausibly move prompt-general Kimi
decode toward `5 tok/s` under the strict deployment target?

Constraints preserved:

- cold start only;
- host RAM below 16 GB including page cache;
- current SOTA reproduction assets must remain available;
- no prompt-specific hotset or pack;
- quality and TTFT gates still required before any future SOTA claim.

## Sources Checked

- AesSedai Kimi-K2.7-Code GGUF:
  `https://huggingface.co/AesSedai/Kimi-K2.7-Code-GGUF`
- AesSedai `IQ3_S` tree:
  `https://huggingface.co/AesSedai/Kimi-K2.7-Code-GGUF/tree/main/IQ3_S`
- AesSedai `IQ2_XXS` tree:
  `https://huggingface.co/AesSedai/Kimi-K2.7-Code-GGUF/tree/main/IQ2_XXS`
- AesSedai `IQ2_S` tree:
  `https://huggingface.co/AesSedai/Kimi-K2.7-Code-GGUF/tree/main/IQ2_S`
- mradermacher imatrix Kimi-K2.7-Code GGUF:
  `https://huggingface.co/mradermacher/Kimi-K2.7-Code-GGUF`

The local CLI could not reach Hugging Face API reliably from this machine
(`curl` returned `HTTP:000`), so the file-list sizes below were taken from the
public Hugging Face file trees rather than by downloading model files.

## Current Local Reference

Remote command:

```bash
ssh -p 51056 root@92.180.27.82 \
  'du -ch /root/lfz/models/Kimi-K2.7-Code-GGUF-IQ3_S/IQ3_S/*.gguf 2>/dev/null | tail -1; df -h /root/lfz/models /root/lfz/tmp 2>/dev/null'
```

Output:

```text
378G    total
Filesystem      Size  Used Avail Use% Mounted on
/dev/root       993G  765G  228G  78% /
/dev/root       993G  765G  228G  78% /
```

Implication: the current IQ3_S SOTA model must be preserved, and only `228G`
is free. A complete candidate above this size cannot be downloaded without
deleting current reproduction assets or other data.

## Candidate Sizes

| source | quant | advertised files | advertised size | ratio vs Aes IQ3_S 405GB |
|---|---:|---:|---:|---:|
| AesSedai | `IQ3_S` | 10 | ~405 GB | 1.000 |
| AesSedai | `IQ2_S` | 8 | ~335 GB | 0.827 |
| AesSedai | `IQ2_XXS` | 7 | ~282 GB | 0.696 |
| mradermacher imatrix | `i1-IQ2_XXS` | 5 | ~267.2 GB | 0.660 |
| mradermacher imatrix | `i1-IQ1_M` | 4 | ~228 GB | 0.563 |
| mradermacher imatrix | `i1-IQ1_S` | 4 | ~204.5 GB | 0.505 |

Only `i1-IQ1_S` clearly fits the current free disk budget without deleting
assets. `i1-IQ1_M` is effectively at the free-space limit and leaves no
practical room for temporary files or derived packs.

## Bound Calculation

Accepted prompt-general SOTA reference:

- profile root: `.Agent/runs/20260707-gp4-postcommit-test-n96-profile`
- mean token rate: `1.367 tok/s`
- current moved expert bytes: mean `4.39 GiB/token`
- active expert footprint: about `8.10 GiB/token`
- all-hit MoE compute floor: about `40 ms/token`
- pure IO peak: about `10.4 GiB/s`
- current real runtime bandwidth: about `6.1 GiB/s`

Approximate lower bound:

```text
token_time >= compute_floor + moved_bytes / bandwidth
```

For an optimistic perfect-overlap view:

```text
token_time >= max(compute_floor, moved_bytes / bandwidth)
```

Required moved bytes for `5 tok/s`:

- non-overlap with `10.4 GiB/s`: `(0.2 - 0.04) * 10.4 = 1.664 GiB/token`;
- perfect-overlap with `10.4 GiB/s`: `0.2 * 10.4 = 2.08 GiB/token`;
- current moved-byte ratio needed:
  - non-overlap: `1.664 / 4.39 = 0.379x`;
  - perfect-overlap: `2.08 / 4.39 = 0.474x`.

Candidate moved-byte bounds if moved bytes scale with advertised size:

| quant | ratio | moved bytes | 10.4 GiB/s non-overlap bound | 10.4 GiB/s perfect-overlap bound |
|---|---:|---:|---:|---:|
| `IQ2_S` | 0.827 | 3.63 GiB/token | ~2.57 tok/s | ~2.86 tok/s |
| `IQ2_XXS` | 0.696 | 3.06 GiB/token | ~2.99 tok/s | ~3.40 tok/s |
| `i1-IQ1_S` | 0.505 | 2.22 GiB/token | ~3.94 tok/s | ~4.68 tok/s |

At the current real runtime bandwidth (`~6.1 GiB/s`), all three candidates are
lower:

| quant | moved bytes | 6.1 GiB/s non-overlap bound |
|---|---:|---:|
| `IQ2_S` | 3.63 GiB/token | ~1.58 tok/s |
| `IQ2_XXS` | 3.06 GiB/token | ~1.85 tok/s |
| `i1-IQ1_S` | 2.22 GiB/token | ~2.47 tok/s |

## Decision

- Do not download a complete lower-quant model yet.
- AesSedai `IQ2_S` and `IQ2_XXS` are rejected as standalone paths to `5 tok/s`:
  their byte ratios are too high and they do not fit the current free disk
  budget anyway.
- mradermacher `i1-IQ1_S` is close enough to the disk budget to be possible,
  but it still does not clear `5 tok/s` by itself under the optimistic bound,
  and it carries much higher quality risk.
- A full low-quant GGUF replacement could still be useful as a component if
  combined with another improvement that reduces moved bytes or exposes enough
  IO parallelism, but it is not the next primary experiment by itself.

## Next Direction

The next candidate should target at least one of:

- a smaller auxiliary expert representation that reaches `0.30x-0.40x` moved
  bytes without replacing the whole model;
- a quality-preserving residual/shared-base form that can be screened on dev
  activations before runtime work;
- a compatible draft/speculative path with proven tokenizer match and high
  acceptance;
- a cache or prefetch change only after byte reduction makes the remaining IO
  target plausible.
