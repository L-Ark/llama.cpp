# DS4 strict run: baseline-n64

- run_dir: `/root/lfz/vendor/llama.cpp-deepseek-v4/.Agent/runs/20260630-ds4-token-rate/baseline-n64`
- git_sha: `57bded0113152baa8af9a0b06d3b0a108a6b33ee`
- n_predict: `64`
- exit_code: `None`
- wall_time_s: `1265.214`
- ttft_s: `1132.584687948227`
- prompt_eval_tok_s: `None`
- decode_tok_s: `None`
- memory_peak_bytes: `15032385536`
- gpu_memory_peak_mib: `25850`
- output_quality: `pass` - coherent France paragraph; n_predict=64

## Command

```bash
build-a101-nvcc/bin/llama-completion -m /root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf -p 'Question: Please introduce France in a short paragraph.

Answer:' -ngl 8 -c 512 -n 64 --ignore-eos --temp 0 --top-p 1.0 --top-k 1 --seed 1 --no-display-prompt -no-cnv -ub 1 -t 24 -tb 24
```

## Output

```text
France is a country located in Western Europe. It is known for its rich history, diverse culture, and iconic landmarks such as the Eiffel Tower and the Louvre Museum. The country is famous for its cuisine, wine, and fashion. France is also known for its contributions to art, literature, and philosophy
```
