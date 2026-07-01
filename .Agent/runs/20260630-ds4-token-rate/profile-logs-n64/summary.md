# DS4 strict run: profile-logs-n64

- run_dir: `/root/lfz/vendor/llama.cpp-deepseek-v4/.Agent/runs/20260630-ds4-token-rate/profile-logs-n64`
- git_sha: `53836d1508d394b6b0d7ab2eac47ab94d6231e4e`
- n_predict: `64`
- exit_code: `0`
- wall_time_s: `1006.882`
- ttft_s: `901.9034945964813`
- prompt_eval_tok_s: `0.15`
- decode_tok_s: `0.62`
- memory_peak_bytes: `15032385536`
- gpu_memory_peak_mib: `25850`
- output_quality: `pass` - coherent France paragraph; n_predict=64

## Command

```bash
build-a101-nvcc/bin/llama-completion -m /root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf -p 'Question: Please introduce France in a short paragraph.

Answer:' -ngl 8 -c 512 -n 64 --ignore-eos --temp 0 --top-p 1.0 --top-k 1 --seed 1 --no-warmup --no-display-prompt -no-cnv -ub 1 -t 24 -tb 24
```

## Output

```text
France is a country located in Western Europe. It is known for its rich history, diverse culture, and iconic landmarks such as the Eiffel Tower and the Louvre Museum. The country is famous for its cuisine, wine, and fashion. France is also known for its contributions to art, literature, and philosophy. It
```
