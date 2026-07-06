# GP31 external asset and disk feasibility refresh

Timestamp: `2026-07-07T08:15:00+0800`

Branch: `vendor/kimi-speculative-general-token-rate-16gb`

Status: completed metadata refresh; no large model downloaded.

## Purpose

GP27, GP29, and GP30 rejected the near-term runtime/cache-only paths. GP10
shows the remaining target needs moved expert bytes near `0.39x-0.55x` of the
current IQ3_S path unless a separate speculative mechanism provides a large
accepted-token multiplier.

GP31 refreshes current external assets and disk feasibility before any large
download or implementation.

## Sources

Hugging Face metadata was queried with the official model API using
`?blobs=true`; only metadata and small README/config files were downloaded.

Primary repositories checked:

- <https://huggingface.co/AesSedai/Kimi-K2.7-Code-GGUF>
- <https://huggingface.co/unsloth/Kimi-K2.7-Code-GGUF>
- <https://huggingface.co/mradermacher/Kimi-K2.7-Code-GGUF>
- <https://huggingface.co/mradermacher/Kimi-K2.7-Code-i1-GGUF>
- <https://huggingface.co/freakyskittle/kimi-k2.7-code-GGUF>
- <https://huggingface.co/decart-ai/Kimi-K2.7-Code-NVFP4>
- <https://huggingface.co/amd/Kimi-K2.7-Code-MXFP4>
- <https://huggingface.co/cm00cm/Kimi-K2.7-Code-DFlash>
- <https://huggingface.co/cm00cm/Kimi-K2.7-Code-EAGLE3>
- <https://huggingface.co/novita/kimi-k2.7-code-eagle3-mla>
- <https://huggingface.co/freakyskittle/Kimi-K2.7-Code-Dflash>

Raw records:

- `hf-asset-summary.txt`
- `hf-raw-metadata.json`
- `remote-disk-summary.txt`
- `code-support-summary.txt`

## Current disk state

Remote filesystem:

- `/dev/root`: `993G` total, `909G` used, `85G` available.

Required current reproduction assets:

- `/root/lfz/models/Kimi-K2.7-Code-GGUF-IQ3_S`: `378G`
- current SOTA main pack:
  `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-france-l12-upgate-v2.expert-pack`: `164G`
- current overlay pack:
  `/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-l1l2down-overlay.expert-pack`: `4.6G`

Potential cleanup candidates, not deleted:

- old `kimi-iq3s-france.expert-pack`: `160G`
- old `kimi-iq3s-tracefirst-n64-20260630.expert-pack`: `75G`
- old `kimi-iq3s-l1l2down-l4l60missing-overlay.expert-pack`: `7.2G`
- old `kimi-iq3s-phase7gz-combined-overlay.expert-pack`: `4.7G`
- old `tmp-hot-upgate-pair-smoke.expert-pack`: `328M`

Deleting files requires explicit approval. GP31 only records the safe cleanup
plan.

## Full-model candidates

Size ratio below is relative to AesSedai IQ3_S `377.55 GiB`.

| repo | variant | size GiB | ratio | direct vendor path | notes |
| --- | --- | ---: | ---: | --- | --- |
| `mradermacher/Kimi-K2.7-Code-i1-GGUF` | `i1-IQ1_S` | 190.39 | 0.504x | plausible | Best new candidate. Needs concat/download plan and header/runtime preflight. |
| `freakyskittle/kimi-k2.7-code-GGUF` | `TQ1_0` | 203.37 | 0.539x | no | Size passes, but TQ is not supported by current `moe_stream_batch` path. |
| `mradermacher/Kimi-K2.7-Code-i1-GGUF` | `i1-IQ1_M` | 212.28 | 0.562x | no | Slightly above target and current stream path lacks IQ1_M. |
| `mradermacher/Kimi-K2.7-Code-i1-GGUF` | `i1-IQ2_XXS` | 248.77 | 0.659x | plausible | Too large for GP10 target. |
| `AesSedai/Kimi-K2.7-Code-GGUF` | `IQ2_XXS` | 262.79 | 0.696x | plausible | Previously known; too large for target. |
| `unsloth/Kimi-K2.7-Code-GGUF` | `UD-IQ1_M` | 283.04 | 0.750x | uncertain | Too large for target. |
| `decart-ai/Kimi-K2.7-Code-NVFP4` | `NVFP4 safetensors` | 554.31 | 1.468x | no | Larger than IQ3_S and not GGUF/expert-pack. |
| `amd/Kimi-K2.7-Code-MXFP4` | `MXFP4 safetensors` | 514.87 | 1.364x | no | Larger than IQ3_S and not GGUF/expert-pack. |

Important implementation details:

- `mradermacher` parts use `.gguf.partNofM`; README points users to concatenate
  multipart files. To avoid needing both parts and final file on disk, a future
  download plan should stream each part into the final GGUF or concatenate while
  deleting parts.
- `freakyskittle` TQ1_0 is a single `203.37 GiB` GGUF and size-plausible, but
  current vendor MoE streaming does not support TQ types.

## Draft candidates

| repo | size GiB | format | current direct path | notes |
| --- | ---: | --- | --- | --- |
| `cm00cm/Kimi-K2.7-Code-DFlash` | 6.48 | safetensors + remote code | no | DFlash draft for SGLang/SpecForge-style target-hidden-state flow. |
| `cm00cm/Kimi-K2.7-Code-EAGLE3` | 2.70 | safetensors | no | EAGLE3, reduced draft vocab 32000; work-in-progress snapshot. |
| `novita/kimi-k2.7-code-eagle3-mla` | 3.43 | safetensors | no | Documented for vLLM `Eagle3DeepseekV2ForCausalLM`, accept length around 2.3-3.2 in their benchmarks. |
| `freakyskittle/Kimi-K2.7-Code-Dflash` | 1.04 | Q4_0 GGUF | no | Oxidize `dflash-draft` architecture, not llama.cpp common speculative draft model. |

Current llama.cpp speculative examples require a compatible draft model with
matching tokenizer/vocab behavior. These draft assets are useful research
candidates, but none is a drop-in `--model-draft` for the current vendor Kimi
runtime.

## Code support check

From `ggml/src/ggml-cuda/moe_stream_batch.cu`:

- `moe_stream_type_supported()` includes `GGML_TYPE_IQ1_S`.
- `launch_moe_mmvq_compact_batch()` includes `GGML_TYPE_IQ1_S`.
- up/gate mixed pair support includes `IQ1_S` with `IQ2_XXS`.
- `IQ1_M`, `TQ1_0`, and `TQ2_0` are not accepted by the current vendor stream
  path.

Therefore the only newly found full-model candidate that both passes the rough
byte-ratio gate and has a plausible direct vendor stream path is
`mradermacher/Kimi-K2.7-Code-i1-GGUF` `i1-IQ1_S`.

## Decision

GP31 finds one plausible next branch:

1. `mradermacher/Kimi-K2.7-Code-i1-GGUF` `i1-IQ1_S`
   - size: `190.39 GiB`;
   - ratio vs current IQ3_S: `0.504x`;
   - direct stream support appears plausible because current vendor stream
     supports `IQ1_S`;
   - blocked by current disk (`85G` free) until old packs are cleaned up;
   - quality risk is high because the provider labels it "for the desperate",
     so it must pass France and dev quality gates before any SOTA claim.

Do not pursue these as the next primary path:

- AesSedai/unsloth IQ2-family selected hotsets: too large versus GP10 target.
- NVFP4/MXFP4: larger than IQ3_S and not in current GGUF/vendor path.
- TQ1_0/TQ2_0: size-plausible but stream support missing.
- DFlash/EAGLE3 draft assets: potentially useful, but require a separate
  architecture integration and acceptance experiment; they are not immediate
  llama.cpp `--model-draft` assets.

## Next plan

Write GP32 before any action:

1. Request or record explicit approval for deleting only old, non-SOTA packs
   if a full IQ1_S download is attempted.
2. First do a no-cleanup header/range preflight for `i1-IQ1_S`:
   - download only the first few MiB of `part1of5`;
   - verify GGUF metadata and tensor type declarations;
   - verify `general.architecture=deepseek2` / Kimi-compatible metadata.
3. If metadata passes and cleanup is approved:
   - stream-concatenate the five parts into a final GGUF without retaining all
     parts simultaneously;
   - run n32 cold-start France smoke under 16GB cgroup;
   - check stream activation, CPU fallback, output quality, TTFT, and RAM.
4. Only after n32 passes, run n96 dev. Held-out test remains sealed until a
   candidate is frozen.
