# Kimi general-prompt held-out test set

This test set is sealed for optimization.

- Prompt file: `.Agent/evals/kimi-general-test-prompts.jsonl`
- Allowed during optimization: hash verification and category counts only.
- Not allowed during optimization: route tracing, hotset creation, expert-pack
  design, cache-policy tuning, classifier tuning, compression dictionary
  selection, or any parameter choice based on test prompt behavior.
- Final SOTA may reveal the test prompts and outputs only after the candidate
  is frozen and the held-out run has completed.
- SHA256:
  `8eaa1285f02b94fa94ae7a9f77cc4fc4758e3c1791ba898ff8e4358abbd17b7e`

Categories:

- English factual/general knowledge: 2
- English reasoning/math: 1
- Coding: 1
- Chinese factual/instruction: 1
- Mixed instruction style: 1

Seal hash command:

```bash
sha256sum .Agent/evals/kimi-general-test-prompts.jsonl
```
