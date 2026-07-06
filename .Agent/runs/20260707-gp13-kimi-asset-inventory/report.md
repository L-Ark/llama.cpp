# Kimi Local Asset Inventory

Generated at: `2026-07-07T02:41:32+0800`

This is a server-side asset inventory for the remaining optimization branch:
test a smaller or lower-byte Kimi variant under the same 16GB Host RAM and
32GB VRAM constraints.

## Commands

```bash
find /root/lfz/models -maxdepth 3 -type f \( -iname "*Kimi*gguf" -o -iname "*.gguf" \) -printf "%s %p\n"
find /root/lfz/runs /root/lfz/models -maxdepth 5 -type f \( -iname "*kimi*expert-pack" -o -iname "*.expert-pack" \) -printf "%s %p\n"
find /root/lfz -maxdepth 6 \( -iname "*kimi*iq2*" -o -iname "*kimi*nvfp4*" -o -iname "*kimi*q2*" -o -iname "*kimi*q3*" -o -iname "*kimi*q4*" -o -iname "*hot-upgate-pair*" \) -printf "%y %s %p\n"
```

## Findings

- Runnable Kimi GGUF currently present:
  - `/root/lfz/models/Kimi-K2.7-Code-GGUF-IQ3_S/IQ3_S/Kimi-K2.7-Code-IQ3_S-00001-of-00010.gguf`
  - plus shards `00002-of-00010` through `00010-of-00010`.
- Existing Kimi expert packs are all IQ3_S-derived assets, including:
  - `kimi-iq3s-france-l12-upgate-v2.expert-pack`;
  - `kimi-iq3s-l1l2down-overlay.expert-pack`;
  - older trace/hot overlay packs.
- A historical q2_k conversion log exists:
  - `/root/lfz/runs/ik_llama/convert-logs/kimi-k27-full-q2_k-fast-20260623-012612Z.log`;
  - rc file reports `0`;
  - log says it exported to `/root/lfz/models/Kimi-K2.7-Code-GGUF/`.
- The exported q2_k directory is not present now:
  - `/root/lfz/models/Kimi-K2.7-Code-GGUF` does not exist.
- q4_0 conversion log exists but was terminated:
  - rc `143`.
- No local NVFP4 Kimi GGUF or expert pack was found.

## Decision

There is no currently available lower-byte Kimi model variant on the server that
can be used for an immediate 16GB cold-start dev baseline. The remaining path
requires obtaining or generating a calibrated lower-byte Kimi model/pack,
especially for up/gate experts, before another runtime implementation attempt.
