# DS4 strict run: matrix-ngl9-n64

- run_dir: `/root/lfz/vendor/llama.cpp-deepseek-v4/.Agent/runs/20260630-ds4-token-rate/matrix-ngl9-n64`
- git_sha: `2769b8a4d086bfc7416a3875e126eaef676a640a`
- n_predict: `64`
- exit_code: `0`
- wall_time_s: `1017.920`
- ttft_s: `905.4901099205017`
- prompt_eval_tok_s: `0.18`
- decode_tok_s: `0.57`
- memory_peak_bytes: `15032385536`
- gpu_memory_peak_mib: `29232`
- output_quality: `pass` - coherent France paragraph; n_predict=64

## Command

```bash
build-a101-nvcc/bin/llama-completion -m /root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf -p 'Question: Please introduce France in a short paragraph.

Answer:' -ngl 9 -c 512 -n 64 --ignore-eos --temp 0 --top-p 1.0 --top-k 1 --seed 1 --no-warmup --no-display-prompt -no-cnv -ub 1 -t 24 -tb 24
```

## Output

```text
France is a country located in Western Europe. It is known for its rich history, diverse culture, and iconic landmarks such as the Eiffel Tower and the Louvre Museum. The country is famous for its cuisine, wine, and fashion. France is also known for its contributions to art, literature, and philosophy. It
```
