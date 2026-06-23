# Performance Attempt Workflow

This workflow is mandatory for every future performance, token-rate, profiling,
or SOTA attempt in this repository. It fixes the previous gap where benchmark
metrics were recorded but wall-clock attempt timing was often left as `n/a`.

## Core Rule

No future result may be called `promoted`, `new SOTA`, or `confirmed
performance improvement` unless the attempt has timing metadata and a plan
entry.

Required timing fields:

- `attempt_start_utc`
- `attempt_end_utc`
- `wall_clock_elapsed_seconds`
- `benchmark_runtime` such as `eval_ms`, `total_ms`, `/usr/bin/time`, or
  equivalent runner output
- `result_status`
- `promoted_commit`, or explicit `n/a`

Historical rows that lack reliable start/end timing must stay marked as
`historical timing missing`. Do not backfill elapsed time by guesswork.

## Attempt Lifecycle

1. Update the relevant plan markdown before running the attempt.

Include:

- `attempt_id`
- hypothesis
- exact code/config change to try
- benchmark command or runner
- success metric
- rollback condition
- expected logs

2. Create attempt metadata before code edits, source probes, or benchmarks.

```bash
IK=/root/lfz/ik_llama
RUN_DIR=/root/lfz/runs/ik_llama/<attempt-id>
mkdir -p "$RUN_DIR"
date -u +%FT%TZ | tee "$RUN_DIR/attempt_start_utc.txt" >/dev/null
git -C "$IK" rev-parse HEAD > "$RUN_DIR/git_start_sha.txt"
git -C "$IK" status --short > "$RUN_DIR/git_status_start.txt"

cat > "$RUN_DIR/attempt_meta.env" <<META
attempt_id=<attempt-id>
attempt_goal=<one-line hypothesis>
attempt_kind=<cli-scan|source-probe|benchmark-repeat|analysis|implementation>
baseline_commit=$(git -C "$IK" rev-parse HEAD)
baseline_eval_tok_s=<previous-best-or-n/a>
memory_limit=<MemoryMax-or-unlimited>
model=<model-path-or-n/a>
META
```

3. Run the attempt.

Capture:

- full command and environment;
- stdout/stderr log path;
- systemd unit output when using `systemd-run`;
- `/usr/bin/time` output when available;
- GPU/RAM constraints;
- git diff if source was patched.

4. End the attempt even if it failed.

```bash
date -u +%FT%TZ | tee "$RUN_DIR/attempt_end_utc.txt" >/dev/null
python3 - "$RUN_DIR" <<'PY'
from datetime import datetime
from pathlib import Path
import sys

p = Path(sys.argv[1])
s = datetime.fromisoformat((p / "attempt_start_utc.txt").read_text().strip().replace("Z", "+00:00"))
e = datetime.fromisoformat((p / "attempt_end_utc.txt").read_text().strip().replace("Z", "+00:00"))
(p / "wall_clock_elapsed_seconds.txt").write_text(str(int((e - s).total_seconds())) + "\n")
print({"attempt_start_utc": s.isoformat(), "attempt_end_utc": e.isoformat(), "wall_clock_elapsed_seconds": int((e - s).total_seconds())})
PY
git -C "$IK" status --short > "$RUN_DIR/git_status_end.txt"
```

5. Update the plan before moving on.

Each attempt entry must include:

```text
attempt_id:
attempt_start_utc:
attempt_end_utc:
wall_clock_elapsed:
benchmark_runtime:
result_status: promoted | unpromoted | reverted | failed | needs-timing-audit
metrics:
log_path:
source_changes:
rollback_status:
promoted_commit:
next_step:
```

## Promotion Gate

A performance improvement is promotable only when all of these are true:

- the metric clears the task-specific improvement threshold;
- repeat runs are recorded when the task requires p50/worst validation;
- no required guardrail regresses, including accuracy, RAM cap, VRAM stability,
  read failures, or smoke checks;
- timing metadata exists and is referenced in the plan;
- effective tracked changes and plan updates are committed;
- the commit is pushed immediately after promotion.

If any timing field is missing, set:

```text
result_status: needs-timing-audit
promoted_commit: n/a
```

The result can be promoted only after a reliable audit recovers the missing
fields from files such as `attempt_start_utc.txt`, `attempt_end_utc.txt`,
systemd logs, benchmark logs, or commit timestamps. If the audit source is not
reliable, the result remains unpromoted.

## Failure And Rollback Gate

Failed, neutral, and reverted attempts still need timing. Record them because
they are part of the optimization cost and prevent repeating dead ends.

For reverted source probes:

- save the log path and summary;
- restore the source tree before the next attempt;
- record whether the probe changed performance, correctness, build behavior, or
  diagnostics;
- commit only the plan/documentation update unless the source change itself is
  kept intentionally.

## Final Response Requirements

When reporting a performance session, include:

- best current result and whether it is promoted;
- all attempts made in the session with wall-clock elapsed time;
- benchmark runtime for promoted or near-promoted runs;
- commit and push status;
- any historical timing gaps explicitly labeled as historical, not estimated.
