#!/usr/bin/env bash
set -o pipefail
cd /root/lfz/tmp/kimi-stage2m-align || exit 1
RUN=${RUN:?}
set -a
. "$RUN/env.txt"
set +a
PROMPT='<|im_user|>user<|im_middle|>Please introduce France in a short paragraph.<|im_end|><|im_assistant|>assistant<|im_middle|><think></think>'
/usr/bin/time -v build-cuda-batch/bin/llama-speculative-simple --defer-experts --fit off -ngl 99 \
  -md /root/lfz/models/kimi-draft/Kimi-K2-Instruct-DRAFT-0.6B-32k-Q4_0.gguf -ngld 99 \
  --spec-draft-n-max 8 --spec-draft-n-min 1 --spec-draft-p-min 0.0 \
  -m /root/lfz/models/Kimi-K2.7-Code-GGUF-IQ3_S/IQ3_S/Kimi-K2.7-Code-IQ3_S-00001-of-00010.gguf \
  -c 512 -n 32 --temp 0 --top-p 1.0 --top-k 1 --seed 1 -t 32 -tb 32 -p "$PROMPT" \
  > "$RUN/stdout.txt" 2> "$RUN/stderr.txt"
rc=$?
echo $rc > "$RUN/exit.txt"
exit $rc
