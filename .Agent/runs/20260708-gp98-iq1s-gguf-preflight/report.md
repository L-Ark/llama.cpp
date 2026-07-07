# GP98 Complete IQ1_S GGUF Preflight

Timestamp: `2026-07-08T03:50:00+0800`.

Status: completed non-destructive preflight; no full model download.

## Purpose

Check whether the mradermacher complete `i1-IQ1_S` Kimi GGUF is worth a full
download/runtime smoke after GP92-GP97 showed that:

- available complete `IQ2_*` GGUFs are too large for the `5 tok/s` byte budget;
- a `0.505x` full-model representation is only borderline;
- several custom `~0.4x` oracle representation families failed the offline
  output-error gates.

This run intentionally does not claim a new SOTA. The accepted prompt-general
runtime SOTA remains GP4 until a full cold-start runtime candidate passes all
quality, TTFT, RAM, and held-out gates.

## Source

- HF repo: `mradermacher/Kimi-K2.7-Code-i1-GGUF`.
- File family: `Kimi-K2.7-Code.i1-IQ1_S.gguf.part1of5` through
  `part5of5`.
- HF repo commit from resolver headers:
  `06671d52123acf7ad3dad395d3628c96d8b690a8`.

## Commands

Remote disk check:

```bash
ssh -p 51056 root@92.180.27.82 \
  'df -h /root/lfz/models /root/lfz/tmp; \
   du -sh /root/lfz/models/Kimi-K2.7-Code-GGUF-IQ3_S/IQ3_S 2>/dev/null || true; \
   du -sh /root/lfz/models 2>/dev/null || true'
```

Remote HF resolver size check:

```bash
ssh -p 51056 root@92.180.27.82 '
for i in 1 2 3 4 5; do
  url="https://huggingface.co/mradermacher/Kimi-K2.7-Code-i1-GGUF/resolve/main/Kimi-K2.7-Code.i1-IQ1_S.gguf.part${i}of5"
  echo "== part${i} =="
  curl -sS -D - -o /dev/null --max-time 30 -H "Range: bytes=0-4095" "$url" |
    tr -d "\r" |
    egrep -i "^(HTTP/|content-length:|location:|accept-ranges:|x-linked-size:|x-linked-etag:|x-repo-commit:)"
done'
```

Local source-code support check:

```bash
rg -n "IQ1_S|IQ2_XXS|packed type|LLAMA_FTYPE_MOSTLY_IQ1_S|GGML_TYPE_IQ1_S" \
  include src ggml/src/ggml-cuda | head -80
```

## Current Remote Disk

```text
Filesystem      Size  Used Avail Use% Mounted on
/dev/root       993G  765G  228G  78% /
/dev/root       993G  765G  228G  78% /
378G    /root/lfz/models/Kimi-K2.7-Code-GGUF-IQ3_S/IQ3_S
378G    /root/lfz/models
```

Current free space is only about `228 GB` decimal. The current IQ3_S SOTA model
asset is preserved.

## Resolver Size Results

All five parts responded with `HTTP/2 302`, `accept-ranges: bytes`, and the same
HF repo commit.

```text
part1 x-linked-size: 41875931136
part2 x-linked-size: 41875931136
part3 x-linked-size: 41875931136
part4 x-linked-size: 41875931136
part5 x-linked-size: 36927147936
```

Total:

```text
204430872480 bytes
190.391086489 GiB
204.430872480 GB
```

If downloaded while preserving the current IQ3_S SOTA asset, the final GGUF
alone would leave only about `21.95 GiB` from the current filesystem free-space
budget. That is not enough margin for a safe full download, failed/resumable
partials, logs, pack-building, or runtime scratch space.

## Metadata Compatibility Evidence

This GP98 run did not download a new prefix because prior GP32 already performed
a stricter non-destructive 16 MiB prefix parse for the same model family.

Prior GP32 record:

- `.Agent/plans/kimi-token-rate-16gb-optimization-plan.md`, section
  `GP32: mradermacher i1-IQ1_S header/range preflight`.
- Remote temp:
  `/root/lfz/tmp/gp32-iq1s-header-preflight/`.
- Downloaded range: `0-16777215` only.
- Prefix SHA256:
  `47eb7d745164141fd86ed11bf8479053b5f76db58771a0ab6abab2f09f6b48a4`.

Relevant GP32 metadata:

```text
magic=GGUF
version=3
n_tensors=1096
n_kv=61
general.architecture=deepseek2
general.name=Kimi K2.7 Code
general.file_type=24
deepseek2.block_count=61
deepseek2.expert_count=384
deepseek2.expert_used_count=8
deepseek2.expert_feed_forward_length=2048
deepseek2.leading_dense_block_count=1
```

Tensor type counts from the prefix metadata:

```text
F32: 365
Q2_K: 8
Q5_K: 1
IQ2_XXS: 61
IQ1_S: 600
IQ4_NL: 61
```

Expert tensor type counts:

```text
up/IQ1_S: 60
gate/IQ1_S: 60
down/IQ1_S: 57
down/Q2_K: 3
```

Code inspection in the current branch shows generic CUDA IQ1_S support and Kimi
MoE stream-path support sites:

- `include/llama.h`: `LLAMA_FTYPE_MOSTLY_IQ1_S`.
- `src/llama-model-loader.cpp`: file type display and loader mapping.
- `ggml/src/ggml-cuda/mmq.cu`, `mmvq.cu`, `vecdotq.cuh`: IQ1_S CUDA matmul
  support.
- `ggml/src/ggml-cuda/moe_stream_batch.cu`: IQ1_S appears in packed/down
  support checks and mixed IQ1_S/IQ2_XXS handling.

## Decision

Defer full download and runtime smoke for now.

Reasons:

1. Disk margin is too small. The full `i1-IQ1_S` GGUF is `190.39 GiB`, while the
   remote currently has only `228 GB` free and must preserve the current IQ3_S
   SOTA assets.
2. The candidate is only borderline for the `5 tok/s` target even under
   optimistic byte scaling; GP93 indicated `0.505x` is not a clean margin.
3. Metadata and current code support look plausible, so this is a disk/quality
   risk decision, not a hard compatibility rejection.

## Reproducibility

To reproduce this preflight:

1. Use branch `vendor/kimi-speculative-general-token-rate-16gb`.
2. Run the disk and resolver commands above from this report.
3. Verify that the five `x-linked-size` values sum to
   `204430872480 bytes`.
4. Cross-check the GP32 metadata record in
   `.Agent/plans/kimi-token-rate-16gb-optimization-plan.md`.

Do not download the full model unless either:

- enough non-SOTA disk is explicitly approved for cleanup; or
- a separate disk/storage target is provided.

