# Kimi Phase 7FB production/min-profile reproduction

Run from /root/lfz/llama.cpp-vendor-kimi at the git commit recorded in git.txt.

Cold-start/cgroup command shape:

systemd-run --wait --collect --same-dir \
  -p MemoryMax=15900000000 -p MemorySwapMax=0 \
  env RUN=<new-run-dir> N=32 VRAM_MIB=15000 THREADS=32 PINNED_SLOTS=12 \
      UPGATE_PCT=62 IQ2_UPGATE_PARALLEL=1 MIN_PROFILE=1 \
      MOE_IO_DEPTH=16 MOE_IO_REFILL_BATCH=8 \
      MOE_PREFETCH_DOWN_DEPTH=2 \
      EXTRA_RUNTIME_ENV='<optional KEY=VALUE lines>' \
      scripts/kimi-phase7fb-min-profile-repro.sh

This script keeps the accepted Phase 7FB production runtime. With MIN_PROFILE=1
it omits diagnostic CSV/trace/profile envs while keeping standard run metadata,
stdout/stderr, cgroup memory files, and metrics.txt.
