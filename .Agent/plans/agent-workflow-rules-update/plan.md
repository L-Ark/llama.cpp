# Agent Workflow Rules Update

## Goal

Make the local agent workflow requirements explicit in
`/root/lfz/ik_llama/.Agent/Agent.md`.

## Plan

- Read the existing `Agent.md` before editing.
- Add an explicit rule that every future instruction must start by reading
  `Agent.md`.
- Add an explicit rule that every experiment or attempt needs a plan first.
- Add an explicit rule that every large task needs its own folder under
  `.Agent/plans/`, and all progress must be recorded in that folder's plan
  markdown.

## Progress

- 2026-06-12: Read the existing `Agent.md`.
- 2026-06-12: Created this task plan before editing the workflow notes.
- 2026-06-12: Updated `Agent.md` to make the read-first rule and
  `.Agent/plans/` progress-recording rule explicit.

## Results

- `Agent.md` now explicitly says every future repository instruction must begin
  by reading that file.
- `Agent.md` now explicitly says all experiment/task progress must be recorded
  in the relevant plan markdown file under `.Agent/plans/`.
