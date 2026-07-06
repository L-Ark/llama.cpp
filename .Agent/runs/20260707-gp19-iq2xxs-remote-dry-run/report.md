# GP19: IQ2_XXS remote dry-run verification

Timestamp: `2026-07-07T03:10:00+0800`.

Status: completed remote dry-run; no cleanup or model download was executed.

## Rationale

GP18 prepared the default-safe cleanup/download helper locally and pushed it to
`wici/vendor/kimi-general-prompt-token-rate-16gb`. This step verifies the helper
against the actual remote server state without changing the remote working tree
or deleting artifacts.

## Remote state

Host:

```text
ssh -p 51056 root@92.180.27.82
hostname: ubuntu
remote time: 2026-07-06T18:59:06+00:00
```

Disk:

```text
Filesystem      Size  Used Avail Use% Mounted on
/dev/root       993G  907G   86G  92% /
```

The remote checkout at `/root/lfz/llama.cpp-vendor-kimi` is dirty, with many
tracked files deleted or modified. To avoid overwriting user/runtime state, this
step did not run `git pull` or reset the checkout.

## Verification method

Fetch only the pushed branch ref, extract the GP18 script and manifest from
`FETCH_HEAD` into a temporary directory, then run syntax check and dry-run:

```bash
cd /root/lfz/llama.cpp-vendor-kimi
git fetch wici vendor/kimi-general-prompt-token-rate-16gb --quiet
TMP=/root/lfz/tmp/gp18-iq2xxs-dry-run
rm -rf "$TMP"
mkdir -p "$TMP"
git show FETCH_HEAD:.Agent/run-tools/kimi_iq2xxs_prepare_download.sh \
  > "$TMP/kimi_iq2xxs_prepare_download.sh"
git show FETCH_HEAD:.Agent/runs/20260707-gp18-iq2xxs-download-prep/download-manifest.json \
  > "$TMP/download-manifest.json"
chmod +x "$TMP/kimi_iq2xxs_prepare_download.sh"
bash -n "$TMP/kimi_iq2xxs_prepare_download.sh"
REPO_ROOT=/root/lfz/llama.cpp-vendor-kimi \
MANIFEST="$TMP/download-manifest.json" \
  "$TMP/kimi_iq2xxs_prepare_download.sh"
```

## Dry-run result

```text
repo_root=/root/lfz/llama.cpp-vendor-kimi
manifest=/root/lfz/tmp/gp18-iq2xxs-dry-run/download-manifest.json
target_dir=/root/lfz/models/Kimi-K2.7-Code-GGUF-AesSedai-IQ2_XXS/IQ2_XXS
min_free_gib=290
free_gib=85.71
```

Preserved artifacts:

```text
/root/lfz/models/Kimi-K2.7-Code-GGUF-IQ3_S/IQ3_S
/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-france-l12-upgate-v2.expert-pack
/root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-l1l2down-overlay.expert-pack
```

Old pack cleanup candidates observed on the remote server:

```text
160G  /root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-france.expert-pack
75G   /root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-tracefirst-n64-20260630.expert-pack
7.2G  /root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-l1l2down-l4l60missing-overlay.expert-pack
4.7G  /root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-phase7gz-combined-overlay.expert-pack
328M  /root/lfz/runs/ik_llama/kimi-iq3s-assets/tmp-hot-upgate-pair-smoke.expert-pack
190M  /root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-phase7gz-missing-down-overlay.expert-pack
```

Script safety output:

```text
cleanup skipped: set KIMI_IQ2_CLEANUP_OLD_PACKS=YES to delete old packs
download skipped: set KIMI_IQ2_DOWNLOAD=YES to download candidate
```

## Decision

- The remote disk state matches GP16/GP18: the server still has only about
  `86G` free, so the full `262.789 GiB` IQ2_XXS candidate cannot be downloaded
  yet.
- The cleanup candidates required for the download are present.
- No destructive action was taken.
- The next step remains an explicit artifact decision: either approve deletion
  of the listed old packs, or attach/free enough additional storage for the
  lower-byte Kimi GGUF candidate.
