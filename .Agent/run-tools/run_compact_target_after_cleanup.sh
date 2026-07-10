#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  .Agent/run-tools/run_compact_target_after_cleanup.sh [options]

Options:
  --candidate ID                       Candidate from the exact manifest.
                                       Default: 0xsero-spark-mini-q2-reap-ds4.
  --execute-download                   Actually delete/download/validate/run.
                                       Default is dry-run only.
  --confirm-delete-rejected-iq2s       Required with --execute-download before
                                       deleting the already-rejected local IQ2_S
                                       cleanup candidate.
  --download-root DIR                  Destination root for compact candidates.
                                       Default: /root/lfz/models/DeepSeek-V4-Flash-compact-targets.
  --skip-run                           Download and validate sizes only.
  -h, --help                           Show this help.

Purpose:
  Guarded next-step runner for the compact-target correctness gate. By default
  it performs no deletion, no download, and no model execution. It prints the
  exact candidate, cleanup file, required flags, disk estimate, and destination.

Execution safety:
  A real run requires both --execute-download and
  --confirm-delete-rejected-iq2s. Without both flags, this script exits before
  touching model files.
USAGE
}

fail() {
  echo "error: $*" >&2
  exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

MANIFEST="${MANIFEST:-$REPO_ROOT/.Agent/runs/20260705-vendor-ds4-coldstart/compact-target-exact-download-manifest-refresh-0xsero-20260708.json}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-/root/lfz/models/DeepSeek-V4-Flash-compact-targets}"
OUT_ROOT="${OUT_ROOT:-/root/lfz/runs/vendor-ds4-16gb}"
BINARY="${BINARY:-$REPO_ROOT/build-ds4-moe-stream/bin/llama-cli}"
CPU_MOE="${CPU_MOE:-40}"
CANDIDATE_ID="0xsero-spark-mini-q2-reap-ds4"
EXECUTE_DOWNLOAD=0
CONFIRM_DELETE=0
SKIP_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --candidate)
      [[ $# -ge 2 ]] || fail "missing value for --candidate"
      CANDIDATE_ID="$2"
      shift 2
      ;;
    --execute-download)
      EXECUTE_DOWNLOAD=1
      shift
      ;;
    --confirm-delete-rejected-iq2s)
      CONFIRM_DELETE=1
      shift
      ;;
    --download-root)
      [[ $# -ge 2 ]] || fail "missing value for --download-root"
      DOWNLOAD_ROOT="$2"
      shift 2
      ;;
    --skip-run)
      SKIP_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      fail "unknown option: $1"
      ;;
  esac
done

[[ -f "$MANIFEST" ]] || fail "missing manifest: $MANIFEST"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ART_DIR="$REPO_ROOT/.Agent/runs/20260705-vendor-ds4-coldstart"
mkdir -p "$ART_DIR"

read_candidate_json() {
  python3 - "$MANIFEST" "$CANDIDATE_ID" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
candidate_id = sys.argv[2]
for candidate in manifest["candidates"]:
    if candidate["id"] == candidate_id:
        print(json.dumps({"manifest": manifest, "candidate": candidate}, ensure_ascii=False))
        raise SystemExit(0)
raise SystemExit(f"candidate not found in manifest: {candidate_id}")
PY
}

CANDIDATE_JSON="$(read_candidate_json)"
CANDIDATE_JSON_FILE="$(mktemp)"
printf '%s\n' "$CANDIDATE_JSON" > "$CANDIDATE_JSON_FILE"
trap 'rm -f "$CANDIDATE_JSON_FILE"' EXIT

cleanup_path="$(python3 - "$CANDIDATE_JSON_FILE" <<'PY'
import json
import sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(d["manifest"]["cleanup_candidate"])
PY
)"
candidate_total_bytes="$(python3 - "$CANDIDATE_JSON_FILE" <<'PY'
import json
import sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(d["candidate"]["total_size_bytes"])
PY
)"
candidate_file_count="$(python3 - "$CANDIDATE_JSON_FILE" <<'PY'
import json
import sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(d["candidate"]["file_count"])
PY
)"
candidate_dest="$DOWNLOAD_ROOT/$CANDIDATE_ID"
first_model_file="$(python3 - "$CANDIDATE_JSON_FILE" "$candidate_dest" <<'PY'
import json
import sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(Path(sys.argv[2]) / d["candidate"]["files"][0]["file"])
PY
)"

available_bytes="$(df -B1 --output=avail "$DOWNLOAD_ROOT" 2>/dev/null | tail -n 1 | tr -d ' ' || true)"
if [[ -z "$available_bytes" ]]; then
  available_bytes="$(df -B1 --output=avail "$(dirname "$DOWNLOAD_ROOT")" | tail -n 1 | tr -d ' ')"
fi
cleanup_bytes=0
if [[ -f "$cleanup_path" ]]; then
  cleanup_bytes="$(stat -c '%s' "$cleanup_path")"
fi
projected_after_cleanup=$((available_bytes + cleanup_bytes))
dry_run_artifact="$ART_DIR/compact-target-after-cleanup-dryrun-${STAMP}.json"

python3 - "$CANDIDATE_JSON_FILE" "$dry_run_artifact" "$REPO_ROOT" "$CANDIDATE_ID" "$candidate_dest" "$cleanup_path" "$available_bytes" "$cleanup_bytes" "$projected_after_cleanup" "$EXECUTE_DOWNLOAD" "$CONFIRM_DELETE" "$SKIP_RUN" "$BINARY" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
artifact = Path(sys.argv[2])
candidate = payload["candidate"]
out = {
    "artifact_id": f"compact-target-after-cleanup-dryrun-{artifact.stem.rsplit('-', 1)[-1]}",
    "repo_root": sys.argv[3],
    "candidate_id": sys.argv[4],
    "candidate": candidate,
    "destination": sys.argv[5],
    "cleanup_candidate": sys.argv[6],
    "available_bytes_now": int(sys.argv[7]),
    "cleanup_candidate_bytes_now": int(sys.argv[8]),
    "projected_available_after_cleanup_bytes": int(sys.argv[9]),
    "candidate_total_size_bytes": int(candidate["total_size_bytes"]),
    "execute_download_requested": sys.argv[10] == "1",
    "confirm_delete_rejected_iq2s": sys.argv[11] == "1",
    "skip_run": sys.argv[12] == "1",
    "binary": sys.argv[13],
    "default_behavior": "dry_run_no_delete_no_download_no_model_execution",
    "required_for_real_execution": [
        "--execute-download",
        "--confirm-delete-rejected-iq2s",
    ],
    "safety_decision": "real_execution_requires_explicit_flags_and_size_validation",
}
artifact.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(out, indent=2, ensure_ascii=False))
PY

if [[ "$EXECUTE_DOWNLOAD" -eq 0 ]]; then
  echo
  echo "dry-run only; no files were deleted, downloaded, or executed"
  echo "artifact: $dry_run_artifact"
  exit 0
fi

if [[ "$CONFIRM_DELETE" -ne 1 ]]; then
  fail "--execute-download requires --confirm-delete-rejected-iq2s"
fi
if [[ ! -x "$BINARY" ]]; then
  fail "missing executable binary: $BINARY"
fi
if [[ "$projected_after_cleanup" -lt "$candidate_total_bytes" ]]; then
  fail "insufficient projected disk after cleanup: need $candidate_total_bytes, have $projected_after_cleanup"
fi
if [[ -e "$cleanup_path" && ! -f "$cleanup_path" ]]; then
  fail "cleanup candidate exists but is not a regular file: $cleanup_path"
fi

run_log="$OUT_ROOT/${STAMP}-compact-target-${CANDIDATE_ID}-after-cleanup"
mkdir -p "$run_log" "$candidate_dest"
cp "$dry_run_artifact" "$run_log/preflight.json"

if [[ -f "$cleanup_path" ]]; then
  echo "deleting already-rejected cleanup candidate: $cleanup_path"
  rm -f -- "$cleanup_path"
else
  echo "cleanup candidate already absent: $cleanup_path"
fi

python3 - "$CANDIDATE_JSON_FILE" "$candidate_dest" > "$run_log/download-list.tsv" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
dest = Path(sys.argv[2])
for item in payload["candidate"]["files"]:
    path = dest / item["file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"{item['url']}\t{path}\t{item['size_bytes']}\t{item.get('etag','')}")
PY

while IFS=$'\t' read -r url path size_bytes etag; do
  if [[ -f "$path" && "$(stat -c '%s' "$path")" == "$size_bytes" ]]; then
    echo "already complete: $path"
    continue
  fi
  echo "downloading: $url"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c --continue=true --max-connection-per-server=8 --split=8 --min-split-size=64M \
      --dir "$(dirname "$path")" --out "$(basename "$path")" "$url" | tee -a "$run_log/download.log"
  else
    curl -L --retry 8 --continue-at - --output "$path" "$url" | tee -a "$run_log/download.log"
  fi
  actual="$(stat -c '%s' "$path")"
  [[ "$actual" == "$size_bytes" ]] || fail "size mismatch for $path: expected $size_bytes got $actual"
done < "$run_log/download-list.tsv"

python3 - "$run_log/download-list.tsv" "$run_log/download-validation.json" <<'PY'
import csv
import json
import sys
from pathlib import Path

rows = []
ok = True
with Path(sys.argv[1]).open() as f:
    for url, path, size, etag in csv.reader(f, delimiter="\t"):
        p = Path(path)
        actual = p.stat().st_size if p.exists() else None
        row = {"url": url, "path": path, "expected_size": int(size), "actual_size": actual, "etag": etag, "size_ok": actual == int(size)}
        ok = ok and row["size_ok"]
        rows.append(row)
out = {"all_size_ok": ok, "files": rows}
Path(sys.argv[2]).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
if not ok:
    raise SystemExit(1)
print(json.dumps(out, indent=2, ensure_ascii=False))
PY

if [[ "$SKIP_RUN" -eq 1 ]]; then
  echo "download complete; validation run skipped by --skip-run"
  exit 0
fi

RUN_NAME="${STAMP}-compact-target-${CANDIDATE_ID}-france-correctness" \
  .Agent/run-tools/strict_ds4_runner.py \
  --binary "$BINARY" \
  --model "$first_model_file" \
  --out-root "$OUT_ROOT" \
  --run-name "${STAMP}-compact-target-${CANDIDATE_ID}-france-correctness" \
  --prompt "Please introduce France in a short paragraph." \
  --case-name "france-${CANDIDATE_ID}" \
  --cpu-moe "$CPU_MOE" \
  --vram-cache-gb 0 \
  --drop-caches-before-case \
  --env GGML_MOE_STREAM=1 \
  --env GGML_MOE_STREAM_DONTNEED=1 \
  --env GGML_MOE_STREAM_ONE_EXPERIMENTAL_DS4=1 \
  --env GGML_MOE_STREAM_ONE_NAME_FILTER=ffn_gate_exps \
  --env GGML_MOE_STREAM_ONE_CACHE_MIB=13568
