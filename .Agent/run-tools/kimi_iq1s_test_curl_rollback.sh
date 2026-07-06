#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "$TMP_ROOT"' EXIT

MODEL_DIR="$TMP_ROOT/model"
MODEL_PATH="$MODEL_DIR/model.gguf"
FAKE_CURL="$TMP_ROOT/fake-curl.sh"

cat > "$FAKE_CURL" <<'EOF'
#!/usr/bin/env bash
printf 'partial-bytes'
exit 23
EOF
chmod +x "$FAKE_CURL"

set +e
OUTPUT="$(
  EXECUTE=1 \
  DOWNLOAD=1 \
  RUN_SMOKE=0 \
  VALIDATE_PARTS=0 \
  KIMI_IQ1S_SYNTHETIC_DOWNLOAD_TEST=1 \
  KIMI_IQ1S_SYNTHETIC_IQ1S_BYTES=16 \
  MODEL_DIR="$MODEL_DIR" \
  MODEL_PATH="$MODEL_PATH" \
  CURL_BIN="$FAKE_CURL" \
    "$SCRIPT_DIR/kimi_iq1s_prepare_full_smoke.sh" 2>&1
)"
RC=$?
set -e

printf '%s\n' "$OUTPUT"

if [ "$RC" -eq 0 ]; then
  printf 'ERROR expected prepare script to fail when fake curl exits nonzero\n' >&2
  exit 1
fi

if ! printf '%s\n' "$OUTPUT" | grep -q 'ERROR part 1 curl failed rc=23; truncating temp back to 0'; then
  printf 'ERROR expected rollback log line was not found\n' >&2
  exit 1
fi

TMP_MODEL="$MODEL_PATH.tmp"
if [ ! -e "$TMP_MODEL" ]; then
  printf 'ERROR expected temporary model path to exist: %s\n' "$TMP_MODEL" >&2
  exit 1
fi

TMP_SIZE="$(stat -c '%s' "$TMP_MODEL")"
if [ "$TMP_SIZE" != "0" ]; then
  printf 'ERROR temporary model was not truncated: size=%s path=%s\n' "$TMP_SIZE" "$TMP_MODEL" >&2
  exit 1
fi

printf '[kimi_iq1s_test_curl_rollback] pass tmp_size=%s rc=%s\n' "$TMP_SIZE" "$RC"
