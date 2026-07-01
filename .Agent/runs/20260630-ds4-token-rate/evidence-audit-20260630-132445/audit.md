# Strict n96 evidence audit

- audit_dir: `.Agent/runs/20260630-ds4-token-rate/evidence-audit-20260630-132445`
- TTFT method: harness `ttft_s`, measured as first observed non-empty stdout file byte elapsed from systemd-run launch; same method for baseline and candidate.

## baseline
- summary: `.Agent/runs/20260630-ds4-token-rate/baseline-n96/summary.json`
- raw_stdout: `.Agent/runs/20260630-ds4-token-rate/baseline-n96/stdout.txt`
- raw_stderr: `.Agent/runs/20260630-ds4-token-rate/baseline-n96/stderr.txt`
- TTFT_DETAILS:
  - `ttft_s` = `753.0239360332489`
- EXACT_OUTPUT_SOURCE: `.Agent/runs/20260630-ds4-token-rate/baseline-n96/summary.json:output`
- EXACT_OUTPUT_SHA256: `7f6718bac9e5995a2f6f74cb45d3c37d438ffc4aa1c4f8213dbeeed8150712bb`
- EXACT_OUTPUT_BEGIN
 France is a country located in Western Europe. It is known for its rich history, diverse culture, and iconic landmarks such as the Eiffel Tower and the Louvre Museum. The country is famous for its cuisine, wine, and fashion. France is also known for its contributions to art, literature, and philosophy. It is a popular tourist destination, attracting millions of visitors each year to its cities, countryside, and coastline. The official language is French, and the currency is the
- EXACT_OUTPUT_END
- Acceptance fields:
  - `git_sha` = `5d360bc3dfefb3f6f727ceb6c412bf7a1bf73aee`
  - `n_predict` = `96`
  - `ngl` = `8`
  - `ttft_s` = `753.0239360332489`
  - `decode_tokens_per_second` = `0.51`
  - `prompt_eval_tokens_per_second` = `0.24`
  - `memory_peak_bytes` = `15032385536`
  - `memory_sample_count` = `1761`
  - `gpu_memory_peak_mib` = `25854`
  - `output_quality` = `{'reason': 'coherent France paragraph; n_predict=96', 'status': 'pass', 'tail': 'f visitors each year to its cities, countryside, and coastline. The official language is French, and the currency is the'}`
  - `command` = `['build-a101-nvcc/bin/llama-completion', '-m', '/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf', '-p', 'Question: Please introduce France in a short paragraph.\n\nAnswer:', '-ngl', '8', '-c', '512', '-n', '96', '--ignore-eos', '--temp', '0', '--top-p', '1.0', '--top-k', '1', '--seed', '1', '--no-warmup', '--no-display-prompt', '-no-cnv', '-ub', '1', '-t', '24', '-tb', '24']`
  - `env` = `{'CUDA_VISIBLE_DEVICES': '0', 'GGML_CUDA_NO_PINNED': '1', 'GGML_MOE_VRAM_CACHE_MIB': '16384', 'LLAMA_MMAP_LOW_RAM': '1'}`

## candidate
- summary: `.Agent/runs/20260630-ds4-token-rate/candidate-ngl9-n96/summary.json`
- raw_stdout: `.Agent/runs/20260630-ds4-token-rate/candidate-ngl9-n96/stdout.txt`
- raw_stderr: `.Agent/runs/20260630-ds4-token-rate/candidate-ngl9-n96/stderr.txt`
- TTFT_DETAILS:
  - `ttft_s` = `898.3368136882782`
- EXACT_OUTPUT_SOURCE: `.Agent/runs/20260630-ds4-token-rate/candidate-ngl9-n96/summary.json:output`
- EXACT_OUTPUT_SHA256: `7f6718bac9e5995a2f6f74cb45d3c37d438ffc4aa1c4f8213dbeeed8150712bb`
- EXACT_OUTPUT_BEGIN
 France is a country located in Western Europe. It is known for its rich history, diverse culture, and iconic landmarks such as the Eiffel Tower and the Louvre Museum. The country is famous for its cuisine, wine, and fashion. France is also known for its contributions to art, literature, and philosophy. It is a popular tourist destination, attracting millions of visitors each year to its cities, countryside, and coastline. The official language is French, and the currency is the
- EXACT_OUTPUT_END
- Acceptance fields:
  - `git_sha` = `2769b8a4d086bfc7416a3875e126eaef676a640a`
  - `n_predict` = `96`
  - `ngl` = `9`
  - `ttft_s` = `898.3368136882782`
  - `decode_tokens_per_second` = `0.57`
  - `prompt_eval_tokens_per_second` = `0.19`
  - `memory_peak_bytes` = `15032385536`
  - `memory_sample_count` = `1994`
  - `gpu_memory_peak_mib` = `29226`
  - `output_quality` = `{'reason': 'coherent France paragraph; n_predict=96', 'status': 'pass', 'tail': 'f visitors each year to its cities, countryside, and coastline. The official language is French, and the currency is the'}`
  - `command` = `['build-a101-nvcc/bin/llama-completion', '-m', '/root/lfz/models/DeepSeek-V4-Flash-FP4-FP8-GGUF/DeepSeek-V4-Flash-FP4-FP8-native.gguf', '-p', 'Question: Please introduce France in a short paragraph.\n\nAnswer:', '-ngl', '9', '-c', '512', '-n', '96', '--ignore-eos', '--temp', '0', '--top-p', '1.0', '--top-k', '1', '--seed', '1', '--no-warmup', '--no-display-prompt', '-no-cnv', '-ub', '1', '-t', '24', '-tb', '24']`
  - `env` = `{'CUDA_VISIBLE_DEVICES': '0', 'GGML_CUDA_NO_PINNED': '1', 'GGML_MOE_VRAM_CACHE_MIB': '16384', 'LLAMA_MMAP_LOW_RAM': '1'}`

- candidate_baseline_ttft_ratio: `1.1929724550596665`
- decode_delta_tokens_per_second: `0.05999999999999994`
AUDIT_PASS
