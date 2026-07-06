# GP23: lower-byte selected subpack feasibility

Timestamp: `2026-07-07T04:20:00+0800`.

Status: completed non-destructive dry-run; no pack written.

## Rationale

GP21 showed the full selected lower-byte pack would be about `115.220 GiB`,
which does not fit in the current `~86 GiB` free disk. The next question is
whether a split artifact can fit without deleting old packs:

- `up+gate` subpack, because up/gate movement is the dominant byte source;
- `down` subpack, because down has different locality and can be optimized
  separately.

## Method

Remote host:

```text
ssh -p 51056 root@92.180.27.82
```

Input:

- GP21 remote plan and manifest under:
  `/root/lfz/tmp/gp21-remote-range-pack`.

Process:

1. Filter GP21 `plan.tsv` into:
   - `upgate-plan.tsv`, where `kind in {up, gate}`;
   - `down-plan.tsv`, where `kind == down`.
2. Rebuild each manifest from the filtered plan using
   `scripts/kimi-make-remote-pack-manifest.py`, so pack offsets are contiguous.
3. Run `scripts/kimi-build-remote-pack-from-manifest.py --dry-run
   --smoke-entries 2` for each subset.
4. Do not execute pack writing.

## Results

Up/gate subset:

- Entries: `20777`.
- Selected tensors: `120`.
- Payload: `61.797 GiB`.
- Estimated pack: `61.800 GiB`.
- Remote range validity: pass.
- Bad pack offset count: `0`.
- Current remote free space during run: `85.619 GiB`.
- Fits current filesystem: yes.
- Smoke entries: `2`.
- Smoke bytes read: `5734400`.
- Pack written: no.

Down subset:

- Entries: `10822`.
- Selected tensors: `60`.
- Payload: `53.418 GiB`.
- Estimated pack: `53.420 GiB`.
- Remote range validity: pass.
- Bad pack offset count: `0`.
- Current remote free space during run: `85.619 GiB`.
- Fits current filesystem: yes.
- Smoke entries: `2`.
- Smoke bytes read: `8486912`.
- Pack written: no.

Recorded summaries:

- `upgate-manifest-summary.json`
- `upgate-build-dry-run-summary.json`
- `down-manifest-summary.json`
- `down-build-dry-run-summary.json`

## Decision

- A single selected lower-byte subpack can be produced without deleting old
  packs:
  - `up+gate`: about `61.8 GiB`;
  - `down`: about `53.4 GiB`.
- The full selected lower-byte pack still cannot fit without at least about
  `30 GiB` more free space.
- These subpacks are not directly runnable with the current IQ3_S GGUF:
  - all entries have `runtime_nbytes_mismatch_count > 0`;
  - current runtime lookup expects current GGUF expert byte size/type;
  - lower-byte pack entries need either the real lower-byte GGUF loaded, or a
    future explicit lower-byte expert override path.
- Therefore GP23 does not claim a token-rate improvement.

Next implementation options:

1. Approve freeing disk and test the normal lower-byte GGUF path.
2. Build only the `up+gate` subpack and implement a lower-byte expert override
   path for selected experts while dense/non-selected fallback remains IQ3_S.
3. Continue byte-reduction analysis to find a smaller `<=85 GiB` full selected
   pack candidate that is more likely to help random prompts.
