# Kimi q2_k Regeneration Feasibility

Generated at: `2026-07-07T02:43:01+0800`

## Server Checks

```bash
df -h /root/lfz /root/lfz/models
du -sh /root/lfz/models/Kimi-K2.7-Code
find /root/lfz -maxdepth 4 -name convert_hf_to_gguf.py
grep -E "Elapsed|Maximum resident|File system outputs|Model successfully exported" \
  /root/lfz/runs/ik_llama/convert-logs/kimi-k27-full-q2_k-fast-20260623-012612Z.log
```

## Findings

- Current free disk on `/`: `86G`.
- HF source model directory is not present:
  - `/root/lfz/models/Kimi-K2.7-Code` is missing.
- Conversion scripts exist in several checkouts, including:
  - `/root/lfz/llama.cpp-vendor-kimi-gp2-6b5c/convert_hf_to_gguf.py`.
- Historical q2_k conversion:
  - rc `0`;
  - exported to `/root/lfz/models/Kimi-K2.7-Code-GGUF/`;
  - elapsed `5:51:43`;
  - max RSS `57017852 KB`;
  - file system outputs `660167272` blocks, roughly hundreds of GiB.

## Decision

Regenerating q2_k is not currently feasible on this server without restoring
the HF source model and freeing substantial disk space. It is also not an
inference-time optimization and would need a separate artifact-generation plan.

For the current 16GB Host RAM + 32GB VRAM runtime goal, the actionable next
dependency is an externally supplied or newly generated calibrated lower-byte
Kimi GGUF/expert pack.
