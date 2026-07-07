# Kimi Phase 7OG priority wrapper reproduction

This wrapper executes the accepted Phase 7OG launch recipe around
`scripts/kimi-phase7fb-min-profile-repro.sh`. The model command, prompt, runtime MoE env, cold-start cache
drop, and run metadata are produced by the base script inside the systemd unit.

Reproduce from a clean checkout at the commit recorded in `git.txt`:

```bash
cd /root/lfz/tmp/kimi-stage2m-align
RUN=<new-run-dir> scripts/kimi-phase7og-priority-repro.sh
```

Key systemd properties:

```text
MemoryMax=15900000000
MemorySwapMax=0
IOAccounting=yes
IOWeight=10000
CPUWeight=10000
Nice=-10
IOSchedulingClass=realtime
IOSchedulingPriority=0
```
