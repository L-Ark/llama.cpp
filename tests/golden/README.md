# Golden Output Reference Files

This directory contains golden reference outputs for regression testing of
llama.cpp model architectures.

## What are golden outputs?

Golden outputs are the known-good logit vectors produced by running inference
on deterministic synthetic models with a fixed random seed. They serve as a
regression safety net: if a code change alters the numerical output of any
supported architecture, the test will catch it.

## How it works

The test `test-golden-outputs` (in `tests/test-golden-outputs.cpp`) creates
small synthetic GGUF models using the same infrastructure as
`test-llama-archs.cpp`, runs a forward pass on CPU, and captures the output
logits.

- **Capture mode** (`--capture`): Generates `.bin` golden files in this
  directory for each architecture.
- **Compare mode** (`--compare`): Loads existing golden files and compares
  current outputs against them. Fails if any architecture regresses beyond the
  configured tolerance.
- **Default** (no args): Runs compare mode when golden files exist, otherwise
  runs capture mode.

## File format

Each `.bin` file uses a simple binary format:

| Field        | Type            | Description                          |
|-------------|-----------------|--------------------------------------|
| Magic       | `char[4]`       | `"GOLD"` (no null terminator)        |
| Name length | `uint32_t`      | Length of architecture name          |
| Name        | `char[N]`       | Architecture name (no null)          |
| Seed        | `uint64_t`      | RNG seed used                        |
| Count       | `uint64_t`      | Number of float values               |
| Logits      | `float[Count]`  | Output logits                        |

## Covered architectures

- `llama` — the canonical transformer architecture
- `qwen3` — Qwen 3 family
- `deepseek2` — DeepSeek v2 (MoE)
- `mamba2` — state-space model
- `t5` — encoder-decoder architecture

## Regenerating golden files

```bash
./test-golden-outputs --capture          # all architectures
./test-golden-outputs --capture --arch llama  # single architecture
```

After regenerating, commit the updated `.bin` files so CI can use them.
