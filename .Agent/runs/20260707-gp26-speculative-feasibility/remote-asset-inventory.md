# GP26 remote asset inventory

Timestamp: `2026-07-07T05:35:00+0800`.

Command:

```bash
ssh -p 51056 root@92.180.27.82 'set -e; echo HOST=$(hostname); date -Is; echo MODELS; find /root/lfz/models -maxdepth 3 -type f \( -name "*.gguf" -o -name "*.safetensors" -o -name "*.bin" \) -printf "%s %p\n" | sort -nr | head -80; echo DIRS; find /root/lfz/models -maxdepth 2 -type d -printf "%p\n" | sort'
```

Result summary:

- Host: `ubuntu`
- Remote timestamp: `2026-07-06T19:29:22+00:00`
- Present model assets:
  - `/root/lfz/models/Kimi-K2.7-Code-GGUF-IQ3_S/IQ3_S/Kimi-K2.7-Code-IQ3_S-00001-of-00010.gguf`
  - `/root/lfz/models/Kimi-K2.7-Code-GGUF-IQ3_S/IQ3_S/Kimi-K2.7-Code-IQ3_S-00002-of-00010.gguf`
  - `/root/lfz/models/Kimi-K2.7-Code-GGUF-IQ3_S/IQ3_S/Kimi-K2.7-Code-IQ3_S-00003-of-00010.gguf`
  - `/root/lfz/models/Kimi-K2.7-Code-GGUF-IQ3_S/IQ3_S/Kimi-K2.7-Code-IQ3_S-00004-of-00010.gguf`
  - `/root/lfz/models/Kimi-K2.7-Code-GGUF-IQ3_S/IQ3_S/Kimi-K2.7-Code-IQ3_S-00005-of-00010.gguf`
  - `/root/lfz/models/Kimi-K2.7-Code-GGUF-IQ3_S/IQ3_S/Kimi-K2.7-Code-IQ3_S-00006-of-00010.gguf`
  - `/root/lfz/models/Kimi-K2.7-Code-GGUF-IQ3_S/IQ3_S/Kimi-K2.7-Code-IQ3_S-00007-of-00010.gguf`
  - `/root/lfz/models/Kimi-K2.7-Code-GGUF-IQ3_S/IQ3_S/Kimi-K2.7-Code-IQ3_S-00008-of-00010.gguf`
  - `/root/lfz/models/Kimi-K2.7-Code-GGUF-IQ3_S/IQ3_S/Kimi-K2.7-Code-IQ3_S-00009-of-00010.gguf`
  - `/root/lfz/models/Kimi-K2.7-Code-GGUF-IQ3_S/IQ3_S/Kimi-K2.7-Code-IQ3_S-00010-of-00010.gguf`
- No `.gguf`, `.safetensors`, or `.bin` draft model was found under
  `/root/lfz/models` at max depth 3.
- No lower-byte full Kimi GGUF was found under `/root/lfz/models` at max depth
  3.

Conclusion:

- GP26 model-based speculative decoding is asset-blocked for now.
- Downloading a draft model is not approved by this record; it would require a
  separate size/RAM/VRAM/TTFT feasibility plan first.
