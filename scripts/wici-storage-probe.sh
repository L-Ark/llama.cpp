#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-/home/wici/models/glm-5.1/glm51-iq3xxs.expert-pack}"
OUT_DIR="${OUT_DIR:-$ROOT/bench/wici-storage}"
RUN_FIO="${RUN_FIO:-1}"
RUN_GDSIO="${RUN_GDSIO:-1}"
PROBE_SECONDS="${PROBE_SECONDS:-10}"
PROBE_SIZE="${PROBE_SIZE:-4G}"
IO_SIZE="${IO_SIZE:-4M}"

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/$(date +%Y%m%d-%H%M%S)-storage-probe.log"

section() {
    printf '\n## %s\n' "$1" | tee -a "$LOG"
}

run() {
    printf '$ %s\n' "$*" | tee -a "$LOG"
    "$@" 2>&1 | tee -a "$LOG"
    return 0
}

run_sh() {
    printf '$ %s\n' "$*" | tee -a "$LOG"
    bash -lc "$*" 2>&1 | tee -a "$LOG"
    return 0
}

section "target"
printf 'target=%s\n' "$TARGET" | tee -a "$LOG"
if [[ -e "$TARGET" ]]; then
    run ls -lh "$TARGET"
else
    printf 'missing target\n' | tee -a "$LOG"
fi

section "host"
run uname -a
run_sh 'nvidia-smi --query-gpu=name,driver_version,pci.bus_id --format=csv,noheader || true'
run findmnt -T "$TARGET" -o TARGET,SOURCE,FSTYPE,OPTIONS
run df -hT "$TARGET"
run lsblk -o NAME,MODEL,SIZE,ROTA,TYPE,MOUNTPOINTS

section "cufile"
run_sh 'ldconfig -p 2>/dev/null | grep -i cufile || true'
run_sh 'find /usr/local/cuda* -path "*cufile.h" -o -path "*libcufile.so*" 2>/dev/null | sort | head -80'
run_sh 'find /usr/local/cuda* -path "*/gds/tools/gdscheck.py" -o -path "*/gds/tools/gdsio" 2>/dev/null | sort'
run_sh 'lsmod | grep -E "nvidia_fs|nvidia" || true'
run_sh 'modinfo nvidia-fs 2>/dev/null | head -40 || true'
run_sh 'test -r /etc/cufile.json && sed -n "1,220p" /etc/cufile.json || true'

GDSCHECK="$(find /usr/local/cuda* -path '*/gds/tools/gdscheck.py' 2>/dev/null | sort | head -1)"
GDSIO="$(find /usr/local/cuda* -path '*/gds/tools/gdsio' 2>/dev/null | sort | head -1)"

if [[ -n "$GDSCHECK" ]]; then
    section "gdscheck"
    run "$GDSCHECK" -p
fi

if [[ "$RUN_FIO" != "0" ]] && command -v fio >/dev/null 2>&1 && [[ -e "$TARGET" ]]; then
    section "fio direct read"
    run fio \
        --name=expert-pack-direct-read \
        --filename="$TARGET" \
        --readonly \
        --rw=read \
        --bs="$IO_SIZE" \
        --iodepth=32 \
        --ioengine=libaio \
        --direct=1 \
        --time_based \
        --runtime="$PROBE_SECONDS" \
        --size="$PROBE_SIZE" \
        --group_reporting
fi

if [[ "$RUN_GDSIO" != "0" && -n "$GDSIO" && -e "$TARGET" ]]; then
    section "gdsio gpu direct read"
    run "$GDSIO" \
        -f "$TARGET" \
        -d 0 \
        -n 0 \
        -m 0 \
        -w 1 \
        -s "$PROBE_SIZE" \
        -i "$IO_SIZE" \
        -x 0 \
        -I 0 \
        -T "$PROBE_SECONDS"

    section "gdsio cpu-only read"
    run "$GDSIO" \
        -f "$TARGET" \
        -d 0 \
        -n 0 \
        -m 0 \
        -w 1 \
        -s "$PROBE_SIZE" \
        -i "$IO_SIZE" \
        -x 1 \
        -I 0 \
        -T "$PROBE_SECONDS"
fi

section "summary"
printf 'log=%s\n' "$LOG" | tee -a "$LOG"
