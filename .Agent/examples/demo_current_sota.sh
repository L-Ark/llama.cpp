#!/usr/bin/env bash
set -euo pipefail

# User-facing demo for the current prompt-general vendor DeepSeek SOTA path.
#
# The demo accepts arbitrary user prompts and runs them through the current
# no-prompt-specific SOTA configuration under the project constraints:
# - vendor DeepSeek, not ik_llama
# - strict 16 GB host RAM cgroup, including page cache
# - MemorySwapMax=0
# - cold start by default with drop_caches
# - no France-specific expert pack, trace, or prompt-derived admission profile
#
# Usage:
#   .Agent/examples/demo_current_sota.sh --prompt "What does AI infrastructure do?"
#   .Agent/examples/demo_current_sota.sh "今天吃什么？"
#   printf 'Introduce Japan briefly.\n' | .Agent/examples/demo_current_sota.sh --stdin-prompt
#   .Agent/examples/demo_current_sota.sh --prompt-file prompt.txt
#   .Agent/examples/demo_current_sota.sh
#
# Useful options are forwarded to demo_generalized_sota.sh:
#   --n-predict 192      Default comparable decode length.
#   --fast-smoke         Use n_predict=32 only to prove the path runs.
#   --warm               Keep page cache; not a cold-start SOTA metric.
#   --json               Print a final machine-readable JSON line.
#   --print-command      Print the strict runner command and exit.

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
exec "$ROOT/.Agent/examples/demo_generalized_sota.sh" "$@"
