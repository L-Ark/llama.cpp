# DS4 strict run: candidate-ngl9-n96

- run_dir: `/root/lfz/vendor/llama.cpp-deepseek-v4/.Agent/runs/20260630-ds4-token-rate/candidate-ngl9-n96`
- git_sha: `2769b8a4d086bfc7416a3875e126eaef676a640a`
- n_predict: `96`
- exit_code: `0`
- wall_time_s: `1066.504`
- ttft_s: `898.3368136882782`
- prompt_eval_tok_s: `0.19`
- decode_tok_s: `0.57`
- memory_peak_bytes: `15032385536`
- gpu_memory_peak_mib: `29226`
- output_quality: `pass` - coherent France paragraph; n_predict=96

## Command

```bash
build-a101-nvcc/bin/llama-completion -m /root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf -p 'Question: Please introduce France in a short paragraph.

Answer:' -ngl 9 -c 512 -n 96 --ignore-eos --temp 0 --top-p 1.0 --top-k 1 --seed 1 --no-warmup --no-display-prompt -no-cnv -ub 1 -t 24 -tb 24
```

## Output

```text
France is a country located in Western Europe. It is known for its rich history, diverse culture, and iconic landmarks such as the Eiffel Tower and the Louvre Museum. The country is famous for its cuisine, wine, and fashion. France is also known for its contributions to art, literature, and philosophy. It is a popular tourist destination, attracting millions of visitors each year to its cities, countryside, and coastline. The official language is French, and the currency is the
```
