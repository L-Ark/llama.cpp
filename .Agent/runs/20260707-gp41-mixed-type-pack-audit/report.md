# GP41 mixed-type selected expert-pack override source audit

Timestamp: `2026-07-07T14:40:00+0800`.

Branch:
`vendor/kimi-speculative-general-token-rate-16gb`.

Commit audited:
`cdabf98a7`.

Purpose:

- GP40 showed full IQ1_S cannot be stored on the current remote disk without
  deletion approval or external storage.
- This audit checks whether selected IQ1_S expert payloads could be used from
  an expert pack while the main model remains IQ3_S.

## Current v1 pack format

Pack writers:

- `scripts/kimi-build-expert-pack-from-gguf.py:12` defines
  `GGMLMOEPACKv1`.
- `scripts/kimi-build-expert-pack-from-gguf.py:183` writes TSV inventory with
  `type`, `tensor_bytes`, and `expert_bytes`, but the binary pack index does
  not include type.
- `scripts/kimi-build-remote-pack-from-manifest.py:19` defines the same v1
  magic.
- `scripts/kimi-build-remote-pack-from-manifest.py:89` reads
  `remote_nbytes`; `:90` reads `current_nbytes`.
- `scripts/kimi-build-remote-pack-from-manifest.py:106` counts a mismatch if
  `current_nbytes != remote_nbytes` or `runtime_nbytes_match != 1`.
- `scripts/kimi-plan-hotkey-remote-pack.py:373-380` keeps `remote_type`,
  `remote_tensor_bytes`, and `remote_nbytes` in the manifest.

Runtime parser:

- `ggml/src/ggml-cuda/moe_stream_batch.cu:2705` loads v1 pack sources.
- `ggml/src/ggml-cuda/moe_stream_batch.cu:2714-2725` validates only magic,
  version, header size, entry count, and data start.
- `ggml/src/ggml-cuda/moe_stream_batch.cu:2747-2759` reads each entry as:
  `tensor[128]`, `expert_idx`, `reserved`, `offset`, `nbytes`, `source_idx`.
- There is no per-entry `ggml_type`, row stride, block size, logical dims, or
  source quant metadata in v1.

Decision:

- v1 cannot safely carry IQ1_S selected entries for an IQ3_S main tensor.
- Reusing the `reserved` field for type would be ambiguous and would still
  leave row stride and validation gaps.

## Current lookup and copy assumptions

Lookup:

- `expert_pack_key_string()` at
  `ggml/src/ggml-cuda/moe_stream_batch.cu:2642-2648` keys by
  `tensor_name`, `expert_idx`, and `nbytes`.
- `expert_pack_lookup_impl()` at
  `ggml/src/ggml-cuda/moe_stream_batch.cu:3013-3044` requires exact
  `tensor/expert/nbytes` match.
- `expert_pack_lookup_any_size()` at
  `ggml/src/ggml-cuda/moe_stream_batch.cu:3047-3073` is debug/support code,
  not used to execute a different-size payload.

Copy:

- `expert_pack_read_entry_iouring()` at
  `ggml/src/ggml-cuda/moe_stream_batch.cu:3828-3833` rejects if
  `entry->nbytes != sz`.
- `expert_pack_direct_read_entry_to_host()` at
  `ggml/src/ggml-cuda/moe_stream_batch.cu:3920-3924` rejects if
  `entry->nbytes != sz`.
- `expert_pack_read_entry()` at
  `ggml/src/ggml-cuda/moe_stream_batch.cu:3952-3955` also requires
  `entry->nbytes == sz`.
- `batch_cache_copy_h2d()` at
  `ggml/src/ggml-cuda/moe_stream_batch.cu:4517-4524` receives one `sz` and
  copies exactly that many bytes to the selected cache slot.
- `expert_pack_iouring_copy_jobs()` at
  `ggml/src/ggml-cuda/moe_stream_batch.cu:4660-4673` takes one
  `expert_bytes` for the whole batch.
- `expert_pack_iouring_copy_jobs()` at
  `ggml/src/ggml-cuda/moe_stream_batch.cu:4687-4691` rejects any job where
  `job.pack_entry->nbytes != expert_bytes`.

Decision:

- The current copy layer can batch different experts only when they have the
  same byte size.
- A v2 mixed-type path must pass `packed_nbytes` to IO/H2D and must avoid
  using the main GGUF `src0_bytes` as the pack size.

## Current cache and kernel assumptions

Type support:

- `moe_stream_type_supported()` at
  `ggml/src/ggml-cuda/moe_stream_batch.cu:6438-6442` already includes
  `GGML_TYPE_IQ1_S`, `GGML_TYPE_Q2_K`, `GGML_TYPE_IQ3_S`, and related low-bit
  types.
- `ggml/src/ggml-cuda/mmvq.cu:1267-1299` dispatches MMVQ by the passed
  `src0_type_int`.
- `ggml/src/ggml-cuda/mmvq.cu:1290` computes row stride from
  `nb01 / ggml_type_size(t0)`.
- `ggml/src/ggml-cuda/mmvq.cu:1377-1378` recomputes per-row `nb01` from
  `src0_type_int` in the batch helper.

Up/gate path:

- `ggml_cuda_moe_stream_batch_up_gate()` uses caller-provided
  `src0_up_type_int`, `src0_gate_type_int`, `up_nb01`, `gate_nb01`,
  `up_expert_bytes`, and `gate_expert_bytes` at
  `ggml/src/ggml-cuda/moe_stream_batch.cu:7150-7158`.
- It registers the main GGUF tensors at
  `ggml/src/ggml-cuda/moe_stream_batch.cu:7239-7240`.
- It sets `src0_bytes = up_expert_bytes` and `nb01 = up_nb01` at
  `ggml/src/ggml-cuda/moe_stream_batch.cu:7289-7290`.
- `plan_tensor()` at
  `ggml/src/ggml-cuda/moe_stream_batch.cu:7546-7571` looks up pack entries by
  `src0_bytes` and stages that same size.
- `copy_stage_jobs()` at
  `ggml/src/ggml-cuda/moe_stream_batch.cu:7619-7639` assumes one effective
  `copy_bytes` for the iouring fast path.

Down path:

- `ggml_cuda_moe_stream_batch()` registers the main down tensor at
  `ggml/src/ggml-cuda/moe_stream_batch.cu:8912`.
- It computes `src0_bytes = ne01 * nb01` at
  `ggml/src/ggml-cuda/moe_stream_batch.cu:9007`.
- It gets the cache by `src0_bytes` at
  `ggml/src/ggml-cuda/moe_stream_batch.cu:9056`.
- It looks up pack entries by `src0_bytes` at
  `ggml/src/ggml-cuda/moe_stream_batch.cu:9127`.
- The launched kernel uses the main GGUF `src0_type` and `nb01` at
  `ggml/src/ggml-cuda/moe_stream_batch.cu:9215-9219`.

Current-down overlap:

- `start_current_down_overlap()` reads registered tensor metadata at
  `ggml/src/ggml-cuda/moe_stream_batch.cu:7693-7705`.
- It allocates the down cache by `rt.expert_bytes` at
  `ggml/src/ggml-cuda/moe_stream_batch.cu:7712`.
- It looks up pack entries by `rt.expert_bytes` at
  `ggml/src/ggml-cuda/moe_stream_batch.cu:7743`.
- It copies all overlap jobs using one `expert_bytes` at
  `ggml/src/ggml-cuda/moe_stream_batch.cu:7767-7776`.

Decision:

- CUDA kernels can compute IQ1_S or Q2_K if invoked with the correct type and
  row stride.
- The current up/gate, down, and overlap code never switches the effective
  compute type based on pack metadata. It always uses the main GGUF tensor type.
- Therefore a selected IQ1_S pack must not be used until the runtime can carry
  `packed_type`, `packed_nb01`, and `packed_nbytes` from lookup through cache,
  IO, and launch.

## Feasibility

Feasible, but only with a default-off v2 path.

Minimum v2 metadata:

- `packed_type` as a `ggml_type` integer.
- `packed_nbytes` per expert.
- `packed_nb01` or enough dims/type metadata to recompute it.
- Logical dims: `ne00`, `ne01`, `n_experts`.
- Original source tensor name and expert index.
- Optional `source_model_id` / quant family for auditability.

Minimum runtime changes:

- Add `GGMLMOEPACKv2` parser while preserving v1 behavior unchanged.
- Add a lookup that can return `expert_pack_entry_v2` by `tensor/expert` and
  validate logical dims against the main tensor.
- Add a default-off env gate, for example
  `GGML_MOE_EXPERT_PACK_MIXED_TYPE=1`.
- Make pack H2D copy accept `packed_nbytes` instead of caller `src0_bytes`.
- Cache selected low-byte experts in a cache pool sized by `packed_nbytes`.
- Launch MMVQ with `packed_type` and `packed_nb01` when a v2 entry is present.
- For v2 misses or unsupported types, fall back to the existing IQ3_S main GGUF
  path.

Testing path without held-out prompts:

1. Synthetic v2 parser/lookup test with tiny dummy entries.
2. Runtime no-op regression: v1 pack and no v2 env must produce identical
   default behavior.
3. One dev prompt only, `n32`, with selected v2 IQ1_S entries and strict
   France quality gate.
4. Dev n96 only after n32 passes.
5. Held-out test only after a candidate runtime policy is frozen.

Risk:

- Quality risk is real: this produces a mixed-quant model, not the published
  full IQ1_S model.
- Performance risk is moderate: if v2 entries are mixed sizes/types in one
  layer, the current iouring batch fast path needs grouping by
  `packed_nbytes`/`packed_type`, otherwise it falls back to serial copies.
- Implementation risk is contained if v2 is default-off and v1 code remains
  unchanged.

Recommendation:

- Do not force IQ1_S selected payloads into `GGMLMOEPACKv1`.
- Next implementation phase should be a default-off v2 format/parser and
  synthetic lookup test only.
- After that, build a small selected IQ1_S v2 pack from remote ranges and run
  dev `n32` France quality before any broader benchmark.
- No SOTA or token-rate claim is made from this phase.
