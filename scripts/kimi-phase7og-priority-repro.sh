#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO:-/root/lfz/llama.cpp-vendor-kimi}"
cd "$REPO_DIR"

: "${N:=96}"
: "${PINNED_SLOTS:=12}"
: "${VRAM_MIB:=15000}"
: "${THREADS:=32}"
: "${UPGATE_PCT:=62}"
: "${IQ2_UPGATE_PARALLEL:=1}"
: "${MIN_PROFILE:=1}"
: "${MOE_IO_DEPTH:=8}"
: "${MOE_IO_REFILL_BATCH:=4}"
: "${MOE_PREFETCH_DOWN_DEPTH:=2}"
: "${EXTRA_RUNTIME_ENV:=}"

: "${MEMORY_MAX:=15900000000}"
: "${MEMORY_SWAP_MAX:=0}"
: "${IO_ACCOUNTING:=yes}"
: "${IO_WEIGHT:=10000}"
: "${CPU_WEIGHT:=10000}"
: "${NICE_LEVEL:=-10}"
: "${IO_SCHEDULING_CLASS:=realtime}"
: "${IO_SCHEDULING_PRIORITY:=0}"
: "${BASE_SCRIPT:=scripts/kimi-phase7fb-min-profile-repro.sh}"

if [ ! -x "$BASE_SCRIPT" ]; then
  echo "base script is not executable: $BASE_SCRIPT" >&2
  exit 1
fi

RUN_ROOT="${RUN_ROOT:-/root/lfz/runs/vendor-kimi-token-rate}"
if [ -z "${RUN:-}" ]; then
  ts="$(date -u +%Y%m%d-%H%M%SZ)"
  RUN="$RUN_ROOT/${ts}-phase7og-priority-wrapper-n${N}"
fi
mkdir -p "$RUN"

systemd_props=(
  "MemoryMax=$MEMORY_MAX"
  "MemorySwapMax=$MEMORY_SWAP_MAX"
  "IOAccounting=$IO_ACCOUNTING"
  "IOWeight=$IO_WEIGHT"
  "CPUWeight=$CPU_WEIGHT"
  "Nice=$NICE_LEVEL"
  "IOSchedulingClass=$IO_SCHEDULING_CLASS"
  "IOSchedulingPriority=$IO_SCHEDULING_PRIORITY"
)

env_args=(
  "REPO=$REPO_DIR"
  "RUN=$RUN"
  "N=$N"
  "VRAM_MIB=$VRAM_MIB"
  "THREADS=$THREADS"
  "PINNED_SLOTS=$PINNED_SLOTS"
  "UPGATE_PCT=$UPGATE_PCT"
  "IQ2_UPGATE_PARALLEL=$IQ2_UPGATE_PARALLEL"
  "MIN_PROFILE=$MIN_PROFILE"
  "MOE_IO_DEPTH=$MOE_IO_DEPTH"
  "MOE_IO_REFILL_BATCH=$MOE_IO_REFILL_BATCH"
  "MOE_PREFETCH_DOWN_DEPTH=$MOE_PREFETCH_DOWN_DEPTH"
  "EXTRA_RUNTIME_ENV=$EXTRA_RUNTIME_ENV"
)

cmd=(systemd-run --wait --collect --same-dir)
for prop in "${systemd_props[@]}"; do
  cmd+=(-p "$prop")
done
cmd+=(env "${env_args[@]}" "$BASE_SCRIPT")

{
  echo "repo=$REPO_DIR"
  echo "pwd=$PWD"
  echo "branch=$(env -u GIT_DIR -u GIT_WORK_TREE git -c safe.directory="$REPO_DIR" -C "$REPO_DIR" branch --show-current 2>&1)"
  echo "head=$(env -u GIT_DIR -u GIT_WORK_TREE git -c safe.directory="$REPO_DIR" -C "$REPO_DIR" rev-parse HEAD 2>&1)"
  echo "status_short:"
  env -u GIT_DIR -u GIT_WORK_TREE git -c safe.directory="$REPO_DIR" -C "$REPO_DIR" status --short 2>&1
  echo
  echo "wrapper_argv:"
  printf '%q ' "$0" "$@"
  echo
  echo
  echo "systemd_properties:"
  printf '%s\n' "${systemd_props[@]}"
  echo
  echo "env_args:"
  printf '%s\n' "${env_args[@]}"
  echo
  echo "command:"
  printf '%q ' "${cmd[@]}"
  echo
} > "$RUN/wrapper-command.txt"
cp "$0" "$RUN/wrapper.sh"
chmod +x "$RUN/wrapper.sh" 2>/dev/null || true

cat > "$RUN/wrapper-readme.md" <<EOF
# Kimi Phase 7OG priority wrapper reproduction

This wrapper executes the accepted Phase 7OG launch recipe around
\`$BASE_SCRIPT\`. The model command, prompt, runtime MoE env, cold-start cache
drop, and run metadata are produced by the base script inside the systemd unit.

Reproduce from a clean checkout at the commit recorded in \`git.txt\`:

\`\`\`bash
cd $REPO_DIR
RUN=<new-run-dir> scripts/kimi-phase7og-priority-repro.sh
\`\`\`

Key systemd properties:

\`\`\`text
$(printf '%s\n' "${systemd_props[@]}")
\`\`\`
EOF

set +e
"${cmd[@]}" > "$RUN/systemd-run.txt" 2>&1
rc=$?
set -e

{
  echo "exit=$rc"
  echo "finished=$(date -Is)"
} > "$RUN/wrapper-result.txt"

cat "$RUN/systemd-run.txt"
exit "$rc"
