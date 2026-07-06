# GP18: IQ2_XXS download preparation

Timestamp: `2026-07-07T03:03:00+0800`.

Status: completed non-destructive preparation; no cleanup or model download was executed.

## Rationale

GP15-GP17 identified AesSedai `IQ2_XXS` as the smallest currently found GGUF
candidate that may reduce expert bytes enough to move the general-prompt Kimi
token rate materially closer to the `>5 tok/s` target. GP16 also showed the
server does not currently have enough free disk for the full candidate unless
old expert packs are removed.

This step prepares a reproducible, default-safe path for that artifact decision
without deleting current files or downloading hundreds of GiB implicitly.

## Artifacts

- Download manifest:
  `.Agent/runs/20260707-gp18-iq2xxs-download-prep/download-manifest.json`
- Helper script:
  `.Agent/run-tools/kimi_iq2xxs_prepare_download.sh`

Manifest summary:

- Repository: `AesSedai/Kimi-K2.7-Code-GGUF`
- Candidate: `IQ2_XXS/Kimi-K2.7-Code-IQ2_XXS-00001-of-00007.gguf` through
  `00007-of-00007.gguf`
- Total size: `282167679808` bytes, `262.789 GiB`
- Shard count: `7`
- First shard SHA256:
  `d89b9a9945205f70dbe5bce6f79ff1047b6295efc98237b0936637dc34052298`

## Safety behavior

The helper script is default-safe:

- Running `.Agent/run-tools/kimi_iq2xxs_prepare_download.sh` only prints the
  intended paths, preserved artifacts, old-pack cleanup candidates, and free
  space.
- It does not delete old expert packs unless
  `KIMI_IQ2_CLEANUP_OLD_PACKS=YES` is set.
- It does not download the candidate unless `KIMI_IQ2_DOWNLOAD=YES` is set.
- Before download, it requires at least `MIN_FREE_GIB=290` GiB free on
  `/root/lfz`.
- After each shard download, it verifies the exact byte size from the manifest.

Syntax check:

```bash
bash -n .Agent/run-tools/kimi_iq2xxs_prepare_download.sh
```

Result: pass.

## Reproduction commands

Dry-run plan:

```bash
.Agent/run-tools/kimi_iq2xxs_prepare_download.sh
```

Cleanup command, only after explicit approval:

```bash
KIMI_IQ2_CLEANUP_OLD_PACKS=YES \
  .Agent/run-tools/kimi_iq2xxs_prepare_download.sh
```

Download command, only after enough disk is available:

```bash
KIMI_IQ2_DOWNLOAD=YES \
  .Agent/run-tools/kimi_iq2xxs_prepare_download.sh
```

Optional overrides:

```bash
TARGET_DIR=/root/lfz/models/Kimi-K2.7-Code-GGUF-AesSedai-IQ2_XXS/IQ2_XXS \
MIN_FREE_GIB=290 \
LOG_DIR=/root/lfz/runs/vendor-kimi-token-rate/iq2xxs-download-prep \
KIMI_IQ2_DOWNLOAD=YES \
  .Agent/run-tools/kimi_iq2xxs_prepare_download.sh
```

## Decision

This step does not claim any token-rate improvement. It creates the reproducible
asset acquisition gate for the next material branch: lower-byte Kimi GGUF
evaluation.

Next required steps after explicit disk/artifact approval:

1. Remove only the old expert-pack cleanup candidates from GP16, preserving the
   current IQ3_S model and SOTA expert packs.
2. Download and byte-verify all seven AesSedai `IQ2_XXS` GGUF shards.
3. Verify full tensor metadata and actual quantization types.
4. Run a cold-start `n32` dev quality/token-rate smoke under the 16 GB cgroup.
5. If quality passes, adapt or rebuild the expert pack for the lower-byte
   candidate and run dev `n96`.
6. Do not run held-out test prompts until the candidate is frozen.
