# Kimi general prompt SOTA reproduction

Cold-start 16GB cgroup command shape:

systemd-run --wait --collect --same-dir \
  -p MemoryMax=15900000000 -p MemorySwapMax=0 \
  env RUN=<new-run-dir> PROMPT_ID=<id> PROMPT_USER_TEXT='<prompt>' \
      QUALITY_KEYWORDS='<keyword1|keyword2,...>' \
      .Agent/run-tools/kimi-general-prompt-repro.sh
