#!/usr/bin/env python3
"""Architecture Scaffold CLI Tool for llama.cpp

Generates boilerplate code for adding a new model architecture to llama.cpp.

Usage:
    python scripts/new-arch.py --name mymodel
    python scripts/new-arch.py --name mymodel --moe --activation gelu
    python scripts/new-arch.py --name mymodel --config path/to/config.json

See docs/ARCHITECTURE-CONTRIBUTOR-GUIDE.md for the full walkthrough.
"""

import argparse
import json
import os
import re
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Component definitions
# ---------------------------------------------------------------------------

ATTENTION_TYPES = {
    "gqa": "Grouped Query Attention (default, 89% of models)",
    "mha": "Multi-Head Attention (legacy models)",
}

FFN_TYPES = {
    "parallel": "Gated parallel FFN / SwiGLU (default, 62% of models)",
    "sequential": "Classical MLP (up -> act -> down)",
}

NORM_TYPES = {
    "rmsnorm": "RMSNorm (default, 76% of models)",
    "layernorm": "LayerNorm (older models)",
}

ACTIVATION_TYPES = {
    "silu": "SiLU / SwiGLU (default, 63% of models)",
    "gelu": "GELU (Gemma, Falcon, GPT2, Phi2)",
    "relu": "ReLU (rare)",
}

POSITION_TYPES = {
    "rope": "Rotary Position Embedding (default, 76% of models)",
    "learned": "Learned position embeddings (BERT, GPT2)",
    "none": "No explicit position encoding (SSM models)",
}

# ---------------------------------------------------------------------------
# Template: C++ graph builder (src/models/<name>.cpp)
# ---------------------------------------------------------------------------

def make_graph_builder_cpp(name: str, opts: dict) -> str:
    upper = name.upper()
    lower = name.lower()
    class_name = f"llm_build_{lower}"

    norm_type = "LLM_NORM_RMS" if opts["norm"] == "rmsnorm" else "LLM_NORM"
    norm_bias = "NULL" if opts["norm"] == "rmsnorm" else f"model.layers[il].attn_norm_b"
    ffn_norm_bias = "NULL" if opts["norm"] == "rmsnorm" else f"model.layers[il].ffn_norm_b"

    act_map = {"silu": "LLM_FFN_SILU", "gelu": "LLM_FFN_GELU", "relu": "LLM_FFN_RELU"}
    activation = act_map.get(opts["activation"], "LLM_FFN_SILU")

    ffn_layout = "LLM_FFN_PAR" if opts["ffn"] == "parallel" else "LLM_FFN_SEQ"

    # Gate tensor line (only for parallel/gated FFN)
    if opts["ffn"] == "parallel":
        ffn_gate_line = "model.layers[il].ffn_gate, NULL, NULL,"
    else:
        ffn_gate_line = "NULL, NULL, NULL,"

    # RoPE lines
    if opts["position"] == "rope":
        rope_block = "\n    // Position tensor (for RoPE)\n    ggml_tensor * inp_pos = build_inp_pos();\n"
        rope_factors_line = "            ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);\n"
        rope_apply_lines = (
            "            // Apply RoPE to Q and K\n"
            "            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, rope_factors,\n"
            "                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,\n"
            "                    ext_factor, attn_factor, beta_fast, beta_slow);\n"
            "            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, rope_factors,\n"
            "                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,\n"
            "                    ext_factor, attn_factor, beta_fast, beta_slow);\n"
            "\n"
            "            cb(Qcur, \"Qcur_rope\", il);\n"
            "            cb(Kcur, \"Kcur_rope\", il);\n"
            "\n"
        )
    else:
        rope_block = ""
        rope_factors_line = ""
        rope_apply_lines = ""

    # MoE FFN block
    if opts["moe"]:
        ffn_block = (
            "        // --- Feed-forward (Mixture of Experts) ---\n"
            "        if (model.layers[il].ffn_gate_inp != nullptr) {{\n"
            "            cur = build_moe_ffn(cur,\n"
            "                    model.layers[il].ffn_gate_inp,\n"
            "                    model.layers[il].ffn_up_exps,\n"
            "                    model.layers[il].ffn_gate_exps,\n"
            "                    model.layers[il].ffn_down_exps,\n"
            "                    nullptr,\n"
            f"                    n_expert, n_expert_used,\n"
            f"                    {activation}, false,\n"
            "                    hparams.expert_weights_scale,\n"
            "                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,\n"
            "                    il);\n"
            "            cb(cur, \"ffn_moe_out\", il);\n"
            "        }} else {{\n"
            "            // Dense fallback (some MoE models have dense layers too)\n"
            "            cur = build_ffn(cur,\n"
            "                    model.layers[il].ffn_up,   NULL, NULL,\n"
            f"                    {ffn_gate_line}\n"
            "                    model.layers[il].ffn_down, NULL, NULL,\n"
            "                    NULL,\n"
            f"                    {activation}, {ffn_layout}, il);\n"
            "            cb(cur, \"ffn_out\", il);\n"
            "        }}\n"
        )
    else:
        ffn_block = (
            "        // --- Feed-forward network ---\n"
            "        cur = build_ffn(cur,\n"
            "                model.layers[il].ffn_up,   NULL, NULL,\n"
            f"                {ffn_gate_line}\n"
            "                model.layers[il].ffn_down, NULL, NULL,\n"
            "                NULL,\n"
            f"                {activation}, {ffn_layout}, il);\n"
            "        cb(cur, \"ffn_out\", il);\n"
        )

    return textwrap.dedent(f"""\
// {lower}.cpp -- graph builder for {name}
// Generated by scripts/new-arch.py -- review TODOs before submitting.
#include "models.h"

{class_name}::{class_name}(
        const llama_model & model,
        const llm_graph_params & params) : llm_graph_context(params) {{
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // ==============================
    // 1. Input embeddings
    // ==============================
    inpL = build_inp_embd(model.tok_embd);
{rope_block}
    auto * inp_attn = build_attn_inp_kv();

    const float kq_scale = hparams.f_attention_scale == 0.0f
        ? 1.0f/sqrtf(float(n_embd_head))
        : hparams.f_attention_scale;

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // ==============================
    // 2. Transformer layers
    // ==============================
    for (int il = 0; il < n_layer; ++il) {{
        ggml_tensor * inpSA = inpL;

        // --- Pre-attention norm ---
        cur = build_norm(inpL,
                model.layers[il].attn_norm, {norm_bias},
                {norm_type}, il);
        cb(cur, "attn_norm", il);

        // --- Self-attention ---
        {{
{rope_factors_line}\
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);

            // TODO: If your model uses LoRA scale tensors, use the 3-arg form:
            // ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // TODO: Add bias if your model uses attention bias:
            // if (model.layers[il].bq) {{ Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq); }}

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

{rope_apply_lines}\
            // TODO: Add QK-norm if your model uses it:
            // Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, {norm_type}, il);
            // Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, {norm_type}, il);

            cur = build_attn(inp_attn, model.layers[il].wo, NULL,
                    Qcur, Kcur, Vcur, NULL, NULL, kq_scale, il);
        }}

        // Residual connection
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // --- Pre-FFN norm ---
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, {ffn_norm_bias},
                {norm_type}, il);
        cb(cur, "ffn_norm", il);

{ffn_block}\

        // Residual connection
        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "l_out", il);

        cur = build_cvec(cur, il);
        inpL = cur;
    }}

    // ==============================
    // 3. Output head
    // ==============================
    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, {norm_type}, -1);
    cb(cur, "result_norm", -1);

    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);

    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}}
""")


# ---------------------------------------------------------------------------
# Template: class declaration (for models.h)
# ---------------------------------------------------------------------------

def make_class_declaration(name: str) -> str:
    lower = name.lower()
    return textwrap.dedent(f"""\
    struct llm_build_{lower} : public llm_graph_context {{
        llm_build_{lower}(const llama_model & model, const llm_graph_params & params);
    }};
    """)


# ---------------------------------------------------------------------------
# Tensor set helpers
# ---------------------------------------------------------------------------

def get_tensor_set(opts: dict) -> list[str]:
    tensors = [
        "LLM_TENSOR_TOKEN_EMBD",
        "LLM_TENSOR_OUTPUT_NORM",
        "LLM_TENSOR_OUTPUT",
    ]
    if opts["position"] == "rope":
        tensors.append("LLM_TENSOR_ROPE_FREQS")
    tensors += [
        "LLM_TENSOR_ATTN_NORM",
        "LLM_TENSOR_ATTN_Q",
        "LLM_TENSOR_ATTN_K",
        "LLM_TENSOR_ATTN_V",
        "LLM_TENSOR_ATTN_OUT",
        "LLM_TENSOR_FFN_NORM",
    ]
    if opts["ffn"] == "parallel":
        tensors.append("LLM_TENSOR_FFN_GATE")
    tensors += [
        "LLM_TENSOR_FFN_DOWN",
        "LLM_TENSOR_FFN_UP",
    ]
    if opts["moe"]:
        tensors += [
            "LLM_TENSOR_FFN_GATE_INP",
            "LLM_TENSOR_FFN_GATE_EXPS",
            "LLM_TENSOR_FFN_DOWN_EXPS",
            "LLM_TENSOR_FFN_UP_EXPS",
        ]
    return tensors


def get_python_tensor_set(opts: dict) -> list[str]:
    tensors = [
        "MODEL_TENSOR.TOKEN_EMBD",
        "MODEL_TENSOR.OUTPUT_NORM",
        "MODEL_TENSOR.OUTPUT",
    ]
    if opts["position"] == "rope":
        tensors.append("MODEL_TENSOR.ROPE_FREQS")
    tensors += [
        "MODEL_TENSOR.ATTN_NORM",
        "MODEL_TENSOR.ATTN_Q",
        "MODEL_TENSOR.ATTN_K",
        "MODEL_TENSOR.ATTN_V",
        "MODEL_TENSOR.ATTN_OUT",
        "MODEL_TENSOR.FFN_NORM",
    ]
    if opts["ffn"] == "parallel":
        tensors.append("MODEL_TENSOR.FFN_GATE")
    tensors += [
        "MODEL_TENSOR.FFN_DOWN",
        "MODEL_TENSOR.FFN_UP",
    ]
    if opts["moe"]:
        tensors += [
            "MODEL_TENSOR.FFN_GATE_INP",
            "MODEL_TENSOR.FFN_GATE_EXP",
            "MODEL_TENSOR.FFN_DOWN_EXP",
            "MODEL_TENSOR.FFN_UP_EXP",
        ]
    return tensors


# ---------------------------------------------------------------------------
# Template: load_hparams snippet
# ---------------------------------------------------------------------------

def make_load_hparams(name: str, opts: dict) -> str:
    upper = name.upper()
    norm_key = ("LLM_KV_ATTENTION_LAYERNORM_RMS_EPS" if opts["norm"] == "rmsnorm"
                else "LLM_KV_ATTENTION_LAYERNORM_EPS")
    norm_field = "f_norm_rms_eps" if opts["norm"] == "rmsnorm" else "f_norm_eps"

    lines = [f"            case LLM_ARCH_{upper}:"]
    lines.append("                {")
    lines.append(f"                    ml.get_key({norm_key}, hparams.{norm_field});")
    if opts["moe"]:
        lines.append("")
        lines.append("                    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp, false);")
        lines.append("                    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);")
    lines.append("")
    lines.append("                    // TODO: Update layer-count -> model-size mapping for your model")
    lines.append("                    switch (hparams.n_layer) {")
    lines.append("                        case 24: type = LLM_TYPE_7B;  break;")
    lines.append("                        case 32: type = LLM_TYPE_13B; break;")
    lines.append("                        default: type = LLM_TYPE_UNKNOWN;")
    lines.append("                    }")
    lines.append("                } break;")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Template: load_tensors snippet
# ---------------------------------------------------------------------------

def make_load_tensors(name: str, opts: dict) -> str:
    upper = name.upper()
    lines = [f"            case LLM_ARCH_{upper}:"]
    lines.append("                {")
    lines.append("                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, \"weight\"), {n_embd, n_vocab}, 0);")
    lines.append("")
    lines.append("                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, \"weight\"), {n_embd}, 0);")
    lines.append("                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      \"weight\"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);")
    lines.append("")
    lines.append("                    if (output == NULL) {")
    lines.append("                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, \"weight\"), {n_embd, n_vocab}, TENSOR_DUPLICATED);")
    lines.append("                    }")
    lines.append("")
    lines.append("                    for (int i = 0; i < n_layer; ++i) {")
    lines.append("                        auto & layer = layers[i];")
    lines.append("")
    lines.append("                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, \"weight\", i), {n_embd}, 0);")
    if opts["norm"] == "layernorm":
        lines.append("                        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, \"bias\", i), {n_embd}, TENSOR_NOT_REQUIRED);")
    lines.append("")
    lines.append("                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   \"weight\", i), {n_embd, n_embd_head_k * n_head}, 0);")
    lines.append("                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   \"weight\", i), {n_embd, n_embd_k_gqa}, 0);")
    lines.append("                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   \"weight\", i), {n_embd, n_embd_v_gqa}, 0);")
    lines.append("                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, \"weight\", i), {n_embd_head_k * n_head, n_embd}, 0);")
    lines.append("")
    lines.append("                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, \"weight\", i), {n_embd}, 0);")
    if opts["norm"] == "layernorm":
        lines.append("                        layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, \"bias\", i), {n_embd}, TENSOR_NOT_REQUIRED);")
    lines.append("")

    if opts["moe"]:
        lines.append("                        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, \"weight\", i), {n_embd, n_expert}, TENSOR_NOT_REQUIRED);")
        lines.append("")
        lines.append("                        if (layer.ffn_gate_inp != nullptr) {")
        lines.append("                            // MoE layer")
        lines.append("                            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, \"weight\", i), {n_embd,   n_ff_exp, n_expert}, 0);")
        lines.append("                            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, \"weight\", i), {n_ff_exp, n_embd,   n_expert}, 0);")
        lines.append("                            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   \"weight\", i), {n_embd,   n_ff_exp, n_expert}, 0);")
        lines.append("                        } else {")
        lines.append("                            // Dense layer fallback")

    if opts["ffn"] == "parallel":
        indent = "                            " if opts["moe"] else "                        "
        lines.append(f"{indent}layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, \"weight\", i), {{n_embd, n_ff}}, 0);")
        lines.append(f"{indent}layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, \"weight\", i), {{n_ff,   n_embd}}, 0);")
        lines.append(f"{indent}layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   \"weight\", i), {{n_embd, n_ff}}, 0);")
    else:
        indent = "                            " if opts["moe"] else "                        "
        lines.append(f"{indent}layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, \"weight\", i), {{n_ff,   n_embd}}, 0);")
        lines.append(f"{indent}layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   \"weight\", i), {{n_embd, n_ff}}, 0);")

    if opts["moe"]:
        lines.append("                        }")
    lines.append("                    }")
    lines.append("                } break;")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Template: Python converter class
# ---------------------------------------------------------------------------

def make_converter_class(name: str, opts: dict) -> str:
    upper = name.upper()
    camel = name.capitalize()

    lines = []
    lines.append(f"@ModelBase.register(\"{camel}ForCausalLM\")")
    lines.append(f"class {camel}Model(TextModel):")
    lines.append(f"    model_arch = gguf.MODEL_ARCH.{upper}")
    lines.append("")
    lines.append("    def set_vocab(self):")
    lines.append("        try:")
    lines.append("            self._set_vocab_sentencepiece()")
    lines.append("        except FileNotFoundError:")
    lines.append("            self._set_vocab_gpt2()")
    lines.append("")
    lines.append("    def set_gguf_parameters(self):")
    lines.append("        super().set_gguf_parameters()")
    if opts["moe"]:
        lines.append("        self.gguf_writer.add_expert_count(self.hparams[\"num_experts\"])")
        lines.append("        self.gguf_writer.add_expert_used_count(self.hparams[\"num_experts_per_tok\"])")
    lines.append("        # TODO: Add any architecture-specific GGUF metadata here")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Auto-apply: patch all files in the repo
# ---------------------------------------------------------------------------

def insert_before(filepath: Path, marker: str, new_lines: str, label: str) -> bool:
    """Insert new_lines before the first occurrence of marker in file."""
    content = filepath.read_text(encoding="utf-8")
    if new_lines.strip() in content:
        print(f"  [=] {label}: already present, skipping")
        return True
    idx = content.find(marker)
    if idx == -1:
        print(f"  [!] {label}: marker not found: {marker!r}")
        return False
    filepath.write_text(content[:idx] + new_lines + content[idx:], encoding="utf-8")
    print(f"  [+] {label}")
    return True


def insert_after(filepath: Path, marker: str, new_lines: str, label: str) -> bool:
    """Insert new_lines after the first occurrence of marker in file."""
    content = filepath.read_text(encoding="utf-8")
    if new_lines.strip() in content:
        print(f"  [=] {label}: already present, skipping")
        return True
    idx = content.find(marker)
    if idx == -1:
        print(f"  [!] {label}: marker not found: {marker!r}")
        return False
    end = idx + len(marker)
    filepath.write_text(content[:end] + new_lines + content[end:], encoding="utf-8")
    print(f"  [+] {label}")
    return True


def insert_sorted(filepath: Path, section_pattern: str, entry: str, sort_key: str, label: str) -> bool:
    """Insert entry in alphabetical order within a section of the file."""
    content = filepath.read_text(encoding="utf-8")
    if entry.strip() in content:
        print(f"  [=] {label}: already present, skipping")
        return True
    # Find insertion point using sort_key for alphabetical placement
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if sort_key in line and lines[i].strip() > entry.strip():
            lines.insert(i, entry)
            filepath.write_text("\n".join(lines), encoding="utf-8")
            print(f"  [+] {label}")
            return True
    # Fallback: insert before section end
    print(f"  [!] {label}: could not find alphabetical position")
    return False


def apply_all_patches(repo_root: Path, name: str, opts: dict, hf_arch_name: str, use_std: bool):
    """Apply all code patches to add the new architecture."""
    upper = name.upper()
    lower = name.lower()
    camel = name.capitalize()

    tensors_cpp = get_tensor_set(opts)
    tensors_py = get_python_tensor_set(opts)

    success = True
    print()
    print("Applying patches...")
    print()

    # 1. src/llama-arch.h -- enum entry
    arch_h = repo_root / "src" / "llama-arch.h"
    success &= insert_before(arch_h,
        "    LLM_ARCH_UNKNOWN,",
        f"    LLM_ARCH_{upper},\n",
        "llama-arch.h: enum entry")

    # 2. src/llama-arch.cpp -- name map
    arch_cpp = repo_root / "src" / "llama-arch.cpp"
    success &= insert_before(arch_cpp,
        '    { LLM_ARCH_UNKNOWN,',
        f'    {{ LLM_ARCH_{upper},{" " * max(1, 20 - len(upper))}"{lower}"{" " * max(1, 20 - len(lower))}}},\n',
        "llama-arch.cpp: name map")

    # 3. src/llama-arch.cpp -- tensor set
    tensor_list = ",\n".join(f"                {t}" for t in tensors_cpp)
    tensor_block = (
        f"        case LLM_ARCH_{upper}:\n"
        f"            return {{\n"
        f"{tensor_list},\n"
        f"            }};\n"
    )
    success &= insert_before(arch_cpp,
        "        case LLM_ARCH_GPTJ:\n        case LLM_ARCH_UNKNOWN:",
        tensor_block,
        "llama-arch.cpp: tensor set")

    # 4. src/models/models.h -- class declaration (only if not using std_transformer)
    if not use_std:
        models_h = repo_root / "src" / "models" / "models.h"
        class_decl = (
            f"\nstruct llm_build_{lower} : public llm_graph_context {{\n"
            f"    llm_build_{lower}(const llama_model & model, const llm_graph_params & params);\n"
            f"}};\n"
        )
        content = models_h.read_text(encoding="utf-8")
        if f"llm_build_{lower}" not in content:
            models_h.write_text(content + class_decl, encoding="utf-8")
            print(f"  [+] models/models.h: class declaration")
        else:
            print(f"  [=] models/models.h: already present, skipping")

    # 5 & 6. src/llama-model.cpp -- load_hparams + load_tensors + build_graph
    model_cpp = repo_root / "src" / "llama-model.cpp"
    content = model_cpp.read_text(encoding="utf-8")

    if f"LLM_ARCH_{upper}" not in content:
        hparams_snippet = make_load_hparams(name, opts)
        load_tensors_snippet = make_load_tensors(name, opts)

        # build_graph entry
        if use_std:
            cfg_lines = make_std_transformer_config(opts)
            if cfg_lines:
                graph_snippet = (
                    f"        case LLM_ARCH_{upper}:\n"
                    f"            {{\n"
                    f"{cfg_lines}"
                    f"                llm = std::make_unique<llm_build_std_transformer>(*this, params, cfg);\n"
                    f"            }} break;\n"
                )
            else:
                graph_snippet = (
                    f"        case LLM_ARCH_{upper}:\n"
                    f"            {{\n"
                    f"                llm = std::make_unique<llm_build_std_transformer>(*this, params);\n"
                    f"            }} break;\n"
                )
        else:
            graph_snippet = (
                f"        case LLM_ARCH_{upper}:\n"
                f"            {{\n"
                f"                llm = std::make_unique<llm_build_{lower}>(*this, params);\n"
                f"            }} break;\n"
            )

        # Use unique markers for each function by including surrounding context
        # load_hparams: 8-space "        case LLM_ARCH_XVERSE:\n            {"
        hparams_marker = "        case LLM_ARCH_XVERSE:\n            {\n                ml.get_key"
        idx = content.find(hparams_marker)
        if idx != -1:
            content = content[:idx] + hparams_snippet + "\n" + content[idx:]
        else:
            print(f"  [!] llama-model.cpp: could not find load_hparams insertion point")

        # load_tensors: 12-space "            case LLM_ARCH_XVERSE:"
        tensors_marker = "            case LLM_ARCH_XVERSE:\n                {\n                    tok_embd"
        idx = content.find(tensors_marker)
        if idx != -1:
            content = content[:idx] + load_tensors_snippet + "\n" + content[idx:]
        else:
            print(f"  [!] llama-model.cpp: could not find load_tensors insertion point")

        # build_graph: uses std_transformer or model-specific class
        graph_marker = "        case LLM_ARCH_XVERSE:\n            {\n                llm = std::make_unique<llm_build_std_transformer>"
        # Try the migrated version first, then original
        idx = content.find(graph_marker)
        if idx == -1:
            graph_marker = "        case LLM_ARCH_XVERSE:\n            {\n                llm = std::make_unique<llm_build_xverse>"
            idx = content.find(graph_marker)
        if idx != -1:
            content = content[:idx] + graph_snippet + content[idx:]
        else:
            print(f"  [!] llama-model.cpp: could not find build_graph insertion point")

        model_cpp.write_text(content, encoding="utf-8")
        print(f"  [+] llama-model.cpp: hparams + tensors + build_graph")
    else:
        print(f"  [=] llama-model.cpp: already present, skipping")

    # 7. src/CMakeLists.txt -- build entry (only if not using std_transformer)
    if not use_std:
        cmake = repo_root / "src" / "CMakeLists.txt"
        cmake_content = cmake.read_text(encoding="utf-8")
        entry = f"             models/{lower}.cpp\n"
        if f"models/{lower}.cpp" not in cmake_content:
            # Find alphabetical insertion point
            lines = cmake_content.split("\n")
            inserted = False
            for i, line in enumerate(lines):
                if "models/" in line and line.strip() > f"models/{lower}.cpp":
                    lines.insert(i, entry.rstrip())
                    inserted = True
                    break
            if inserted:
                cmake.write_text("\n".join(lines), encoding="utf-8")
                print(f"  [+] CMakeLists.txt: build entry")
            else:
                print(f"  [!] CMakeLists.txt: could not find insertion point")
                success = False
        else:
            print(f"  [=] CMakeLists.txt: already present, skipping")

    # 8. gguf-py/gguf/constants.py
    constants = repo_root / "gguf-py" / "gguf" / "constants.py"
    const_content = constants.read_text(encoding="utf-8")

    if f"MODEL_ARCH.{upper}" not in const_content and f'{upper}' not in const_content.split("class MODEL_ARCH")[1].split("class ")[0] if "class MODEL_ARCH" in const_content else True:
        # Add to MODEL_ARCH enum -- before the last entry
        # Find the pattern of the last enum entry
        arch_enum_pattern = re.compile(r'(    \w+\s*=\s*auto\(\)\n)((?:\s*\n)*class )')
        match = arch_enum_pattern.search(const_content)
        if match:
            insertion = f"    {upper:<20s} = auto()\n"
            const_content = const_content[:match.start(1)] + match.group(1) + insertion + const_content[match.start(2):]
            print(f"  [+] constants.py: MODEL_ARCH enum")
        else:
            print(f"  [!] constants.py: could not find MODEL_ARCH enum end")

        # Add to MODEL_ARCH_NAMES
        if f'MODEL_ARCH.{upper}' not in const_content:
            const_content = const_content.replace(
                "}\n\nTENSOR_NAMES",
                f'    MODEL_ARCH.{upper}:{" " * max(1, 20 - len(upper))}"{lower}",\n}}\n\nTENSOR_NAMES',
                1)
            print(f"  [+] constants.py: MODEL_ARCH_NAMES")

        # Add tensor list
        tensor_entries = "\n".join(f"        {t}," for t in tensors_py)
        tensor_block = f"    MODEL_ARCH.{upper}: [\n{tensor_entries}\n    ],\n"
        # Insert before the closing of the tensor dict
        const_content = const_content.replace(
            "}\n\n# TODO",
            f"{tensor_block}}}\n\n# TODO",
            1)
        if "# TODO" not in const_content:
            # Fallback: just append before last }
            pass
        print(f"  [+] constants.py: tensor list")

        constants.write_text(const_content, encoding="utf-8")
    else:
        print(f"  [=] constants.py: already present, skipping")

    # 9. convert_hf_to_gguf.py
    converter = repo_root / "convert_hf_to_gguf.py"
    conv_content = converter.read_text(encoding="utf-8")

    if f"MODEL_ARCH.{upper}" not in conv_content:
        converter_class = make_converter_class(name, opts)
        # Add HF arch name to register decorator
        converter_class = converter_class.replace(
            f'@ModelBase.register("{camel}ForCausalLM")',
            f'@ModelBase.register("{hf_arch_name}")')

        # Append before the last class or at the end
        # Find a good insertion point - after the last model class
        conv_content = conv_content.rstrip() + "\n\n\n" + converter_class + "\n"
        converter.write_text(conv_content, encoding="utf-8")
        print(f"  [+] convert_hf_to_gguf.py: converter class")
    else:
        print(f"  [=] convert_hf_to_gguf.py: already present, skipping")

    print()
    if success:
        print("All patches applied successfully!")
    else:
        print("Some patches failed -- check warnings above.")

    return success


def make_std_transformer_config(opts: dict) -> str:
    """Generate llm_transformer_config lines if non-default."""
    lines = []
    if opts["norm"] == "layernorm":
        lines.append("                cfg.norm = LLM_NORM;")
    if opts["activation"] != "silu":
        act_map = {"gelu": "LLM_FFN_GELU", "relu": "LLM_FFN_RELU", "relu_sqr": "LLM_FFN_RELU_SQR", "swiglu": "LLM_FFN_SWIGLU"}
        if opts["activation"] in act_map:
            lines.append(f"                cfg.act = {act_map[opts['activation']]};")
    if opts["ffn"] == "sequential":
        lines.append("                cfg.ffn_type = LLM_FFN_SEQ;")
    if opts.get("attn_bias"):
        lines.append("                cfg.attn_bias = true;")
    if opts.get("ffn_bias"):
        lines.append("                cfg.ffn_bias = true;")
    if opts.get("output_bias"):
        lines.append("                cfg.output_bias = true;")
    if opts.get("qk_norm"):
        lines.append("                cfg.qk_norm = true;")
    if not opts.get("position", "rope") == "rope":
        lines.append("                cfg.use_rope = false;")
    if opts.get("moe"):
        lines.append("                cfg.moe = true;")
        if opts.get("moe_shared"):
            lines.append("                cfg.moe_shared = true;")
    if opts.get("iswa"):
        lines.append("                cfg.iswa = (hparams.swa_type != LLAMA_SWA_TYPE_NONE);")
    if opts.get("combined_qkv"):
        lines.append("                cfg.combined_qkv = true;")
    if opts.get("parallel_ffn"):
        lines.append("                cfg.parallel_ffn = true;")
    if opts.get("no_attn_cache"):
        lines.append("                cfg.no_attn_cache = true;")
    if opts.get("attn_post_norm"):
        lines.append("                cfg.attn_post_norm = true;")
    if opts.get("ffn_post_norm"):
        lines.append("                cfg.ffn_post_norm = true;")
    if opts.get("logit_softcap"):
        lines.append("                cfg.logit_softcap = true;")
    if opts.get("token_embd_scale"):
        lines.append("                cfg.token_embd_scale = true;")

    if not lines:
        return ""
    return "                llm_transformer_config cfg;\n" + "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Config.json inference (enhanced)
# ---------------------------------------------------------------------------

def infer_from_config(config_path: str) -> tuple:
    """Infer architecture components and HF arch name from a HuggingFace config.json."""
    with open(config_path) as f:
        cfg = json.load(f)

    # Extract HF architecture name
    architectures = cfg.get("architectures", [])
    hf_arch_name = architectures[0] if architectures else "UnknownForCausalLM"
    model_type = cfg.get("model_type", "unknown")

    opts = {
        "attention": "gqa",
        "ffn": "parallel",
        "norm": "rmsnorm",
        "activation": "silu",
        "position": "rope",
        "moe": False,
    }

    # Infer GQA vs MHA
    n_head = cfg.get("num_attention_heads", 0)
    n_head_kv = cfg.get("num_key_value_heads", n_head)
    if n_head_kv == n_head:
        opts["attention"] = "mha"
    else:
        opts["attention"] = "gqa"

    # Infer activation
    act = cfg.get("hidden_act", "silu").lower()
    if "gelu" in act:
        opts["activation"] = "gelu"
    elif "relu" in act:
        opts["activation"] = "relu"
    else:
        opts["activation"] = "silu"

    # Infer normalization
    if "rms_norm_eps" in cfg:
        opts["norm"] = "rmsnorm"
    elif "layer_norm_eps" in cfg or "layer_norm_epsilon" in cfg:
        opts["norm"] = "layernorm"

    # Infer position encoding
    if cfg.get("rope_theta") or cfg.get("rope_scaling"):
        opts["position"] = "rope"
    elif cfg.get("position_embedding_type", "").lower() == "absolute":
        opts["position"] = "learned"

    # Infer MoE
    if cfg.get("num_experts") or cfg.get("num_local_experts"):
        opts["moe"] = True

    # Infer bias usage
    attn_bias = cfg.get("attention_bias", cfg.get("add_bias_linear", False))
    if attn_bias:
        opts["attn_bias"] = True
        opts["ffn_bias"] = True

    print(f"  Inferred from {config_path}:")
    print(f"    HF Architecture: {hf_arch_name}")
    print(f"    Model type:      {model_type}")
    print(f"    Attention:  {opts['attention']} (n_head={n_head}, n_head_kv={n_head_kv})")
    print(f"    FFN:        {opts['ffn']}")
    print(f"    Norm:       {opts['norm']}")
    print(f"    Activation: {opts['activation']} (hidden_act={cfg.get('hidden_act', 'N/A')})")
    print(f"    Position:   {opts['position']}")
    print(f"    MoE:        {opts['moe']}")
    if attn_bias:
        print(f"    Biases:     attention + FFN")
    print()

    return opts, hf_arch_name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate boilerplate for a new llama.cpp model architecture.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Add new standard transformer (infers everything from config.json):
              python scripts/new-arch.py --name phonelm --config config.json --std-transformer --apply

              # Add MoE model with ISWA:
              python scripts/new-arch.py --name mymodel --moe --iswa --std-transformer --apply

              # Generate custom builder (not std-transformer):
              python scripts/new-arch.py --name mymodel --config config.json

              # Dry-run to preview changes:
              python scripts/new-arch.py --name mymodel --std-transformer --dry-run
        """),
    )
    parser.add_argument("--name", required=True,
                        help="Architecture name (lowercase, e.g., 'mymodel')")
    parser.add_argument("--config", metavar="PATH",
                        help="Path to HuggingFace config.json to infer components")
    parser.add_argument("--attention", choices=ATTENTION_TYPES, default="gqa",
                        help="Attention type (default: gqa)")
    parser.add_argument("--ffn", choices=FFN_TYPES, default="parallel",
                        help="FFN layout (default: parallel/gated)")
    parser.add_argument("--norm", choices=NORM_TYPES, default="rmsnorm",
                        help="Normalization type (default: rmsnorm)")
    parser.add_argument("--activation", choices=ACTIVATION_TYPES, default="silu",
                        help="Activation function (default: silu)")
    parser.add_argument("--position", choices=POSITION_TYPES, default="rope",
                        help="Position encoding (default: rope)")
    parser.add_argument("--moe", action="store_true",
                        help="Enable Mixture-of-Experts FFN")
    parser.add_argument("--moe-shared", action="store_true",
                        help="MoE with shared expert (DeepSeek, Qwen MoE)")
    parser.add_argument("--iswa", action="store_true",
                        help="Interleaved Sliding Window Attention")
    parser.add_argument("--combined-qkv", action="store_true",
                        help="Combined QKV projection (GPT-2, Bloom, Falcon)")
    parser.add_argument("--no-cache", action="store_true",
                        help="No KV cache (embedding/diffusion models)")
    parser.add_argument("--qk-norm", action="store_true",
                        help="QK normalization (Qwen3, Gemma3)")
    parser.add_argument("--post-norms", action="store_true",
                        help="Attention and FFN post-normalization")
    parser.add_argument("--softcap", action="store_true",
                        help="Final logit softcapping (Gemma)")
    parser.add_argument("--embd-scale", action="store_true",
                        help="Scale token embeddings by sqrt(n_embd) (Gemma)")
    parser.add_argument("--apply", action="store_true",
                        help="Auto-apply all patches to the codebase (instead of printing snippets)")
    parser.add_argument("--std-transformer", action="store_true",
                        help="Use llm_build_std_transformer (generic builder) instead of custom .cpp")
    parser.add_argument("--hf-arch", metavar="NAME",
                        help="HuggingFace architecture name (e.g., 'MyModelForCausalLM')")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print generated code without writing files")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing files")

    args = parser.parse_args()
    name = args.name.lower().replace("-", "_")
    upper = name.upper()
    camel = name.capitalize()

    # Infer options from config.json if provided
    hf_arch_name = args.hf_arch or f"{name.capitalize()}ForCausalLM"
    if args.config:
        opts, inferred_hf_arch = infer_from_config(args.config)
        if not args.hf_arch:
            hf_arch_name = inferred_hf_arch
        # CLI flags override inferred values
        if args.attention != "gqa":
            opts["attention"] = args.attention
        if args.ffn != "parallel":
            opts["ffn"] = args.ffn
        if args.norm != "rmsnorm":
            opts["norm"] = args.norm
        if args.activation != "silu":
            opts["activation"] = args.activation
        if args.position != "rope":
            opts["position"] = args.position
        if args.moe:
            opts["moe"] = True
        if args.moe_shared:
            opts["moe_shared"] = True
        if args.iswa:
            opts["iswa"] = True
        if args.combined_qkv:
            opts["combined_qkv"] = True
        if args.no_cache:
            opts["no_attn_cache"] = True
        if args.qk_norm:
            opts["qk_norm"] = True
        if args.post_norms:
            opts["attn_post_norm"] = True
            opts["ffn_post_norm"] = True
        if args.softcap:
            opts["logit_softcap"] = True
        if args.embd_scale:
            opts["token_embd_scale"] = True
    else:
        opts = {
            "attention": args.attention,
            "ffn": args.ffn,
            "norm": args.norm,
            "activation": args.activation,
            "position": args.position,
            "moe": args.moe,
            "moe_shared": args.moe_shared,
            "iswa": args.iswa,
            "combined_qkv": args.combined_qkv,
            "no_attn_cache": args.no_cache,
            "qk_norm": args.qk_norm,
            "attn_post_norm": args.post_norms,
            "ffn_post_norm": args.post_norms,
            "logit_softcap": args.softcap,
            "token_embd_scale": args.embd_scale,
        }

    # Find repo root
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    if not (repo_root / "src" / "llama-arch.h").exists():
        print("Error: Cannot find llama.cpp repo root. Run from the repo directory.", file=sys.stderr)
        sys.exit(1)

    model_cpp_path = repo_root / "src" / "models" / f"{name}.cpp"

    print("+----------------------------------------------------------+")
    print(f"|  Architecture Scaffold: {name:>30}    |")
    print("+----------------------------------------------------------+")
    print(f"|  Attention:  {opts['attention']:>10}   Norm:       {opts['norm']:>10}  |")
    print(f"|  FFN:        {opts['ffn']:>10}   Activation: {opts['activation']:>10}  |")
    print(f"|  Position:   {opts['position']:>10}   MoE:        {'yes' if opts['moe'] else 'no':>10}  |")
    print("+----------------------------------------------------------+")
    print()

    # ---------------------------------------------------------------
    # AUTO-APPLY mode: patch all files automatically
    # ---------------------------------------------------------------
    if args.apply:
        if not args.std_transformer:
            # Generate model .cpp file
            cpp_content = make_graph_builder_cpp(name, opts)
            if model_cpp_path.exists() and not args.force:
                print(f"  [!] {model_cpp_path} already exists. Use --force to overwrite.")
            else:
                model_cpp_path.write_text(cpp_content, encoding="utf-8")
                print(f"  [+] Created {model_cpp_path.relative_to(repo_root)}")

        apply_all_patches(repo_root, name, opts, hf_arch_name, args.std_transformer)
        return

    # ---------------------------------------------------------------
    # Generate the model .cpp file (non-apply mode)
    # ---------------------------------------------------------------
    cpp_content = make_graph_builder_cpp(name, opts)

    if args.dry_run:
        print(f"=== src/models/{name}.cpp ===")
        print(cpp_content)
    else:
        if model_cpp_path.exists() and not args.force:
            print(f"  [!] {model_cpp_path} already exists. Use --force to overwrite.")
        else:
            model_cpp_path.write_text(cpp_content, encoding="utf-8")
            print(f"  [+] Created {model_cpp_path.relative_to(repo_root)}")

    # ---------------------------------------------------------------
    # Print manual steps
    # ---------------------------------------------------------------
    tensors_cpp = get_tensor_set(opts)
    tensors_py = get_python_tensor_set(opts)
    class_decl = make_class_declaration(name)
    hparams_snippet = make_load_hparams(name, opts)
    load_tensors_snippet = make_load_tensors(name, opts)
    converter_snippet = make_converter_class(name, opts)

    print()
    print("=" * 60)
    print("  MANUAL STEPS (copy-paste into the indicated files)")
    print("=" * 60)
    print()

    print(f"1. src/llama-arch.h -- Add to enum llm_arch (before LLM_ARCH_UNKNOWN):")
    print()
    print(f"       LLM_ARCH_{upper},")
    print()

    print(f"2. src/llama-arch.cpp -- Add to LLM_ARCH_NAMES map:")
    print()
    print(f'       {{ LLM_ARCH_{upper}, "{name}" }},')
    print()

    print(f"3. src/llama-arch.cpp -- Add tensor set case:")
    print()
    print(f"       case LLM_ARCH_{upper}:")
    print(f"           return {{")
    for t in tensors_cpp:
        print(f"               {t},")
    print(f"           }};")
    print()

    print(f"4. src/models/models.h -- Add class declaration:")
    print()
    print(f"   {class_decl}")

    print(f"5. src/llama-model.cpp -- Add to load_hparams() switch:")
    print()
    print(hparams_snippet)
    print()

    print(f"6. src/llama-model.cpp -- Add to load_tensors() switch:")
    print()
    print(load_tensors_snippet)
    print()

    print(f"7. src/llama-model.cpp -- Add to build_graph() switch:")
    print()
    if args.std_transformer:
        cfg_lines = make_std_transformer_config(opts)
        if cfg_lines:
            print(f"           case LLM_ARCH_{upper}:")
            print(f"               {{")
            print(cfg_lines.rstrip())
            print(f"                   llm = std::make_unique<llm_build_std_transformer>(*this, params, cfg);")
            print(f"               }} break;")
        else:
            print(f"           case LLM_ARCH_{upper}:")
            print(f"               {{")
            print(f"                   llm = std::make_unique<llm_build_std_transformer>(*this, params);")
            print(f"               }} break;")
    else:
        print(f"           case LLM_ARCH_{upper}:")
        print(f"               {{")
        print(f"                   llm = std::make_unique<llm_build_{name}>(*this, params);")
        print(f"               }} break;")
    print()

    print(f"8. src/CMakeLists.txt -- Add in alphabetical order:")
    print()
    print(f"              models/{name}.cpp")
    print()

    print(f"9. gguf-py/gguf/constants.py -- Add to MODEL_ARCH enum:")
    print()
    print(f"       {upper} = auto()")
    print()

    print(f"   Add to MODEL_ARCH_NAMES:")
    print()
    print(f'       MODEL_ARCH.{upper}: "{name}",')
    print()

    print(f"   Add tensor list:")
    print()
    print(f"       MODEL_ARCH.{upper}: [")
    for t in tensors_py:
        print(f"           {t},")
    print(f"       ],")
    print()

    print(f"10. convert_hf_to_gguf.py -- Add converter class:")
    print()
    conv = make_converter_class(name, opts).replace(
        f'@ModelBase.register("{camel}ForCausalLM")',
        f'@ModelBase.register("{hf_arch_name}")')
    print(f"    {conv}")
    print()

    print("=" * 60)
    print("  Done! See docs/ARCHITECTURE-CONTRIBUTOR-GUIDE.md for details.")
    print("=" * 60)


if __name__ == "__main__":
    main()

