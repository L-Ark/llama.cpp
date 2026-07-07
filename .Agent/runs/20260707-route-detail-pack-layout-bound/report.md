# Kimi route-detail pack layout bound

This is a dev-only offline diagnostic. It does not use held-out test prompts and does not generate a runtime pack.

## Summary

- dev prompts: `7`
- route rows: `37562`
- current metadata entries: `69120`

| layout | kind | rows | read GiB | span/read | gap/read | adjacent/job | coalesce rows | missing |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| current | all | 37562 | 1559.78 | 31.013 | 30.013 | 0.023 | 0.0% | 0 |
| current | down | 11508 | 569.77 | 31.166 | 30.166 | 0.023 | 0.0% | 0 |
| current | gate | 13027 | 514.29 | 30.966 | 29.966 | 0.023 | 0.0% | 0 |
| current | up | 13027 | 475.72 | 30.882 | 29.882 | 0.023 | 0.0% | 0 |
| first_use | all | 37562 | 1559.78 | 19.020 | 18.020 | 0.179 | 0.6% | 0 |
| first_use | down | 11508 | 569.77 | 18.949 | 17.949 | 0.179 | 0.6% | 0 |
| first_use | gate | 13027 | 514.29 | 19.056 | 18.056 | 0.179 | 0.6% | 0 |
| first_use | up | 13027 | 475.72 | 19.064 | 18.064 | 0.179 | 0.6% | 0 |
| frequency | all | 37562 | 1559.78 | 20.479 | 19.479 | 0.084 | 0.0% | 0 |
| frequency | down | 11508 | 569.77 | 20.443 | 19.443 | 0.085 | 0.0% | 0 |
| frequency | gate | 13027 | 514.29 | 20.508 | 19.508 | 0.084 | 0.0% | 0 |
| frequency | up | 13027 | 475.72 | 20.491 | 19.491 | 0.084 | 0.0% | 0 |
| greedy_pair | all | 37562 | 1559.78 | 20.228 | 19.228 | 0.353 | 0.1% | 0 |
| greedy_pair | down | 11508 | 569.77 | 20.187 | 19.187 | 0.352 | 0.1% | 0 |
| greedy_pair | gate | 13027 | 514.29 | 20.257 | 19.257 | 0.353 | 0.1% | 0 |
| greedy_pair | up | 13027 | 475.72 | 20.245 | 19.245 | 0.353 | 0.1% | 0 |

Interpretation:

- `span/read` and `gap/read` estimate how much physical locality a layout creates for each active tensor batch.
- Better locality is useful only if runtime can exploit it with larger contiguous/coalesced reads or lower per-read overhead.
- If a layout improves locality but does not reduce read bytes or create fewer wait waves, token-rate gains may be small.
