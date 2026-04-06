#!/usr/bin/env python3
"""Derive graph schema from HuggingFace model files.

Reads config.json and model.safetensors.index.json to produce a
deterministic graph schema. ZERO tribal knowledge — every feature
is derived from explicit data in the source files.

If a feature cannot be determined from the available data, it is
omitted (not guessed). The schema is only as complete as the
source files allow.

Supported architecture categories:
  - Standard transformers (Qwen, LLaMA, Gemma, Mistral, Phi, etc.)
  - MoE transformers (Mixtral, Qwen-MoE, DeepSeek-V2, etc.)
  - SSM / Mamba (pure SSM models)
  - Hybrid SSM+Attention (Jamba, Mamba2-Attn, etc.)
  - Encoder-only (BERT, RoBERTa, etc.)
  - Encoder-decoder (T5, BART, etc.)
  - RWKV (linear attention variants)
  - Multi-head Latent Attention (DeepSeek MLA)

Usage:
    python scripts/derive_schema.py /path/to/hf-model-dir/
"""

import json
import sys
import re
from pathlib import Path
from typing import Any


def load_config(model_dir: Path) -> dict:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json in {model_dir}")
    with open(config_path) as f:
        return json.load(f)


def load_tensor_names(model_dir: Path) -> list[str]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            return sorted(json.load(f).get("weight_map", {}).keys())
    try:
        import safetensors
        for sf_file in model_dir.glob("*.safetensors"):
            with safetensors.safe_open(str(sf_file), framework="pt") as f:
                return sorted(f.keys())
    except ImportError:
        pass
    index_path = model_dir / "pytorch_model.bin.index.json"
    if index_path.exists():
        with open(index_path) as f:
            return sorted(json.load(f).get("weight_map", {}).keys())
    return []


# ── Activation normalization ──

_ACT_MAP = {
    "silu": "silu", "swish": "silu",
    "gelu": "gelu", "gelu_new": "gelu", "gelu_fast": "gelu",
    "gelu_pytorch_tanh": "gelu", "quick_gelu": "gelu",
    "relu": "relu", "relu2": "relu_sqr", "relu_sqr": "relu_sqr",
    "swiglu": "swiglu", "geglu": "geglu",
    "elu": "elu", "tanh": "tanh", "sigmoid": "sigmoid", "mish": "mish",
}

# ── Layer index patterns (how layers are numbered in tensor names) ──

_LAYER_PATTERNS = [
    r'layers\.\d+\.(.*)',
    r'h\.\d+\.(.*)',
    r'transformer\.h\.\d+\.(.*)',
    r'encoder\.layer\.\d+\.(.*)',
    r'encoder\.layers\.\d+\.(.*)',
    r'decoder\.layers?\.\d+\.(.*)',
    r'decoder\.layer\.\d+\.(.*)',
    r'blocks?\.\d+\.(.*)',
    r'backbone\.layers\.\d+\.(.*)',
]


def _get_cfg(text_cfg: dict, config: dict, *keys):
    """Get first matching key from text_config, then root config."""
    for key in keys:
        val = text_cfg.get(key)
        if val is not None:
            return val
        val = config.get(key)
        if val is not None:
            return val
    return None


def _detect_per_layer_variation(tensor_names: list[str]) -> dict:
    """Detect per-layer structural differences (hybrid models).

    Returns a dict mapping layer_index -> set of feature tags.
    """
    layer_features = {}
    for t in tensor_names:
        t_clean = re.sub(r'^(model\.|language_model\.)', '', t)
        m = re.search(r'layers?\.\s*(\d+)', t_clean)
        if not m:
            continue
        idx = int(m.group(1))
        if idx not in layer_features:
            layer_features[idx] = set()

        suffix = t_clean[m.end():]
        if re.search(r'(mixer\.|mamba\.)', suffix):
            layer_features[idx].add('ssm')
        if re.search(r'self_attn\.|SelfAttention\.', suffix):
            layer_features[idx].add('attn')
        if re.search(r'experts\.\d+', suffix):
            layer_features[idx].add('moe')
        if re.search(r'(EncDecAttention|cross_attn)', suffix):
            layer_features[idx].add('cross_attn')

    return layer_features


def derive_schema(config: dict, tensor_names: list[str]) -> dict:
    """Derive schema from ONLY what's in config.json and tensor names.

    Rules:
    - Every decision comes from an explicit key or tensor name pattern
    - If data is missing, the feature is omitted (not guessed)
    - No model-type-name heuristics
    - Tensor names are ground truth for graph structure
    - Config values are ground truth for hyperparameters
    """

    text_cfg = config.get("text_config", config)

    # ── Model type ──
    architectures = config.get("architectures", [])
    arch_str = architectures[0] if architectures else ""
    model_type = "causal_lm"
    if "Encoder" in arch_str and "Decoder" not in arch_str:
        model_type = "encoder"
    elif "Seq2Seq" in arch_str or "Conditional" in arch_str:
        model_type = "encoder_decoder"
    elif "ForMaskedLM" in arch_str:
        model_type = "encoder"
    is_enc_dec = _get_cfg(text_cfg, config, "is_encoder_decoder") is True
    if is_enc_dec:
        model_type = "encoder_decoder"

    # ── Normalize tensor names ──
    normalized = []
    for t in tensor_names:
        normalized.append(re.sub(r'^(model\.|language_model\.|bert\.|transformer\.)', '', t))

    layer_suffixes = set()
    for t in normalized:
        for pattern in _LAYER_PATTERNS:
            m = re.match(pattern, t)
            if m:
                layer_suffixes.add(m.group(1))
                break

    has_tensors = len(layer_suffixes) > 0

    # ═══ FROM CONFIG — explicit keys only ═══

    activation = None
    for key in ["hidden_act", "hidden_activation", "activation_function"]:
        val = _get_cfg(text_cfg, config, key)
        if val:
            activation = _ACT_MAP.get(val.lower(), val.lower())
            break

    has_rms_eps = "rms_norm_eps" in text_cfg or "rms_norm_eps" in config
    has_ln_eps = any(k in text_cfg or k in config
                     for k in ["layer_norm_eps", "layer_norm_epsilon"])
    rms_norm_bool = _get_cfg(text_cfg, config, "rms_norm")
    if rms_norm_bool is True:
        has_rms_eps = True
    norm_type = "rms_norm" if has_rms_eps else ("layer_norm" if has_ln_eps else None)

    tied_embd = _get_cfg(text_cfg, config, "tie_word_embeddings")

    # MoE config
    n_expert = (_get_cfg(text_cfg, config,
                         "num_local_experts", "num_experts", "n_routed_experts") or 0)
    n_expert_used = (_get_cfg(text_cfg, config,
                              "num_experts_per_tok", "num_selected_experts") or 0)
    n_shared_experts = (_get_cfg(text_cfg, config,
                                 "n_shared_experts", "num_shared_experts") or 0)
    first_k_dense = _get_cfg(text_cfg, config, "first_k_dense_replace") or 0
    moe_layer_freq = _get_cfg(text_cfg, config, "moe_layer_freq")
    expert_layer_period = _get_cfg(text_cfg, config, "expert_layer_period")
    expert_layer_offset = _get_cfg(text_cfg, config, "expert_layer_offset")

    # RoPE
    has_rope = any(k in text_cfg or k in config for k in
                   ["rope_theta", "rope_scaling", "rope_parameters", "rotary_emb_base",
                    "partial_rotary_factor", "rotary_dim", "rotary_pct"])

    # Sliding window
    swa_val = _get_cfg(text_cfg, config, "sliding_window")
    has_swa = isinstance(swa_val, (int, float)) and swa_val > 0

    logit_softcap = _get_cfg(text_cfg, config, "final_logit_softcapping")

    use_parallel = _get_cfg(text_cfg, config, "use_parallel_residual", "parallel_attn")

    attn_bias = _get_cfg(text_cfg, config, "attention_bias", "bias")

    config_qk_norm = _get_cfg(text_cfg, config, "use_qk_norm", "use_kq_norm")

    # SSM config (Mamba)
    ssm_conv_kernel = _get_cfg(text_cfg, config, "conv_kernel", "mamba_d_conv")
    ssm_state_size = _get_cfg(text_cfg, config, "state_size", "mamba_d_state")
    ssm_dt_rank = _get_cfg(text_cfg, config, "time_step_rank", "mamba_dt_rank")
    ssm_expand = _get_cfg(text_cfg, config, "expand", "mamba_expand")

    # RWKV config
    rwkv_head_size = _get_cfg(text_cfg, config, "head_size")
    rwkv_rescale_every = _get_cfg(text_cfg, config, "rescale_every")
    rwkv_model_type = (_get_cfg(text_cfg, config, "model_type") or "")
    is_rwkv_config = rwkv_model_type.startswith("rwkv")

    # Hybrid layer patterns (Jamba-style / Nemotron-H)
    attn_layer_period = _get_cfg(text_cfg, config, "attn_layer_period")
    attn_layer_offset = _get_cfg(text_cfg, config, "attn_layer_offset")
    hybrid_override_pattern = _get_cfg(text_cfg, config, "hybrid_override_pattern")

    # DeepSeek MLA
    kv_lora_rank = _get_cfg(text_cfg, config, "kv_lora_rank")
    qk_nope_head_dim = _get_cfg(text_cfg, config, "qk_nope_head_dim")
    qk_rope_head_dim = _get_cfg(text_cfg, config, "qk_rope_head_dim")

    # GLM DSA (Dynamic Sparse Attention)
    n_lora_q = _get_cfg(text_cfg, config, "q_lora_rank")
    n_lora_kv = _get_cfg(text_cfg, config, "kv_lora_rank")
    indexer_n_head = _get_cfg(text_cfg, config, "indexer_n_head",
                               "num_indexer_heads")
    rope_sections = _get_cfg(text_cfg, config, "rope_dimension_sections")

    # LLAMA4 / chunked attention
    swa_period = _get_cfg(text_cfg, config, "swa_period",
                           "sliding_window_pattern")
    n_moe_layer_step = _get_cfg(text_cfg, config, "interleave_moe_layer_step",
                                 "n_moe_layer_step")
    n_ff_exp = _get_cfg(text_cfg, config, "expert_feed_forward_length",
                         "n_ff_exp")

    # Gemma3N (AltUp + Laurel)
    n_altup = _get_cfg(text_cfg, config, "n_altup", "num_altup")
    laurel_rank = _get_cfg(text_cfg, config, "laurel_rank")

    # LFM2 (Short convolution)
    n_shortconv_cache = _get_cfg(text_cfg, config, "n_shortconv_l_cache",
                                  "shortconv_cache_length")

    # Gemma4 (dual-path MoE)
    n_layers_in_expert_block = _get_cfg(text_cfg, config,
                                         "num_layers_in_expert_block",
                                         "n_layers_in_expert_block")

    # Encoder-specific
    position_embedding_type = _get_cfg(text_cfg, config, "position_embedding_type")
    type_vocab_size = _get_cfg(text_cfg, config, "type_vocab_size")
    relative_attention_buckets = _get_cfg(text_cfg, config,
                                          "relative_attention_num_buckets")

    # ═══ FROM TENSORS — pattern matching only ═══

    if has_tensors:
        # Attention QKV style
        has_sep_q = any(re.search(r'\b(q_proj|wq)\b', s) for s in layer_suffixes)
        has_comb_qkv = any(re.search(r'\b(qkv_proj|c_attn|query_key_value|wqkv)\b', s)
                           for s in layer_suffixes)
        has_bert_q = any(re.search(r'attention\.self\.query', s)
                         for s in layer_suffixes)
        has_t5_q = any(re.search(r'SelfAttention\.q', s) for s in layer_suffixes)
        qkv_style = "combined" if has_comb_qkv and not has_sep_q else "separate"

        # DeepSeek MLA (from tensors)
        has_mla_tensors = any(re.search(r'kv_a_proj|kv_b_proj', s)
                              for s in layer_suffixes)

        # QK norm
        t_qk_norm = any(re.search(r'\b(q_norm|q_layernorm|attn_q_norm)\b', s)
                         for s in layer_suffixes)

        # Post-attention norm
        t_attn_post = any(re.search(r'\b(attn_post_norm|post_attention_norm)\b', s)
                           for s in layer_suffixes)
        if (any("post_attention_layernorm" in s for s in layer_suffixes) and
                any("pre_feedforward" in s for s in layer_suffixes)):
            t_attn_post = True

        # Post-FFN norm
        t_ffn_post = any(re.search(r'\b(ffn_post_norm|post_feedforward|post_ffw_norm)\b', s)
                          for s in layer_suffixes)

        # FFN type
        has_gate = any(re.search(r'\b(gate_proj|ffn_gate|gate_up_proj)\b', s)
                       for s in layer_suffixes)
        has_fc1 = any(re.search(r'\bfc1\b', s) for s in layer_suffixes)
        has_wi = any(re.search(r'DenseReluDense\.wi', s) for s in layer_suffixes)
        has_intermediate = any(re.search(r'intermediate\.dense', s)
                               for s in layer_suffixes)
        ffn_type = "gated" if has_gate else "sequential"

        # Pre-attention norm
        has_attn_norm = any(re.search(r'\b(input_layernorm|ln_1|attn_norm)\b', s)
                            for s in layer_suffixes)
        has_bert_attn_norm = any(re.search(r'attention\.output\.LayerNorm', s)
                                  for s in layer_suffixes)
        has_t5_norm = any(re.search(r'\blayer_norm\b', s) for s in layer_suffixes)
        if has_t5_norm:
            has_attn_norm = True

        # FFN norm
        has_ffn_norm = any(
            re.search(r'\b(post_attention_layernorm|ln_2|ffn_norm|'
                      r'pre_feedforward|pre_ff_layernorm)\b', s)
            for s in layer_suffixes)
        has_bert_ffn_norm = any(re.search(r'output\.LayerNorm', s)
                                for s in layer_suffixes)

        # Position embedding tensors
        has_pos_tensor = any(
            re.search(r'\b(position_embedding|wpe|embed_positions|'
                      r'pos_embd|position_embeddings)\b', t)
            for t in normalized)
        has_type_embd = any(re.search(r'token_type_embeddings', t)
                            for t in normalized)
        has_pooler = any(re.search(r'pooler\.dense', t) for t in normalized)

        # SSM tensors
        has_ssm = any(re.search(r'(mixer\.|mamba\.)', s) for s in layer_suffixes)
        has_ssm_conv = any(re.search(r'(mixer\.conv1d|mamba\.conv1d)', s)
                           for s in layer_suffixes)
        has_ssm_dt = any(re.search(r'(mixer\.dt_proj|mamba\.dt_proj)', s)
                         for s in layer_suffixes)
        has_ssm_A = any(re.search(r'(mixer\.A_log|mamba\.A_log)', s)
                        for s in layer_suffixes)

        # RWKV tensors
        has_rwkv = any(re.search(r'(time_mix|channel_mix|time_decay)', s)
                       for s in layer_suffixes)

        # MoE tensors
        has_moe_t = any(re.search(r'experts\.\d+', s) for s in layer_suffixes)
        has_moe_router = any(re.search(r'\b(gate\.weight|router\.weight)\b', s)
                             for s in layer_suffixes)
        has_shared_expert = any(
            re.search(r'shared_expert[s]?\.'
                      r'(gate_proj|up_proj|down_proj|w1|w2|w3)', s)
            for s in layer_suffixes)

        # Cross-attention (encoder-decoder)
        has_cross_attn = any(
            re.search(r'(EncDecAttention|cross_attn|encoder_attn)', s)
            for s in layer_suffixes)
        has_rel_pos_bias = any(re.search(r'relative_attention_bias', s)
                               for s in layer_suffixes)

        # SSM-specific norms (Jamba)
        has_ssm_b_norm = any(re.search(r'mamba\.b_layernorm', s)
                             for s in layer_suffixes)
        has_ssm_dt_norm = any(re.search(r'mamba\.dt_layernorm', s)
                              for s in layer_suffixes)

        # Nemotron-H hybrid: mixer contains both SSM (A_log) and attn (q_proj)
        has_mixer_q = any(re.search(r'mixer\.q_proj', s)
                          for s in layer_suffixes)
        if has_mixer_q:
            has_self_attn = True

        # Self-attention presence (needed for hybrid detection)
        has_self_attn = (has_sep_q or has_comb_qkv or
                         has_bert_q or has_t5_q or has_mixer_q)

        # LFM2 short convolution
        has_shortconv = any(re.search(r'shortconv\.(conv|in_proj|out_proj)', s)
                            for s in layer_suffixes)

        # Gemma3N AltUp + Laurel
        has_altup = any(re.search(r'altup_(proj|unembd|correct|predict|router)', s)
                        for s in layer_suffixes)
        has_laurel = any(re.search(r'laurel_(l|r)\b', s)
                         for s in layer_suffixes)

        # GLM DSA indexer
        has_dsa_indexer = any(re.search(r'indexer', s)
                              for s in layer_suffixes)
    else:
        qkv_style = t_qk_norm = t_attn_post = t_ffn_post = None
        ffn_type = has_attn_norm = has_ffn_norm = None
        has_pos_tensor = has_ssm = has_rwkv = has_moe_t = False
        has_gate = has_fc1 = has_wi = has_intermediate = False
        has_ssm_conv = has_ssm_dt = has_ssm_A = False
        has_moe_router = has_shared_expert = False
        has_cross_attn = has_rel_pos_bias = False
        has_bert_attn_norm = has_bert_ffn_norm = False
        has_bert_q = has_t5_q = has_t5_norm = False
        has_mla_tensors = has_ssm_b_norm = has_ssm_dt_norm = False
        has_type_embd = has_pooler = has_self_attn = False
        has_mixer_q = has_shortconv = has_altup = has_laurel = False
        has_dsa_indexer = False

    # ═══ COMBINE ═══

    has_qk_norm = t_qk_norm or config_qk_norm
    is_moe = n_expert > 0 or has_moe_t
    is_ssm = has_ssm and not has_self_attn
    is_hybrid = (has_ssm and has_self_attn) or hybrid_override_pattern is not None
    is_rwkv = has_rwkv or is_rwkv_config
    is_mla = kv_lora_rank is not None or has_mla_tensors
    is_dsa = indexer_n_head is not None or has_dsa_indexer
    is_bert_style = has_bert_q or has_bert_attn_norm
    is_t5_style = has_t5_q or is_enc_dec
    is_altup = n_altup is not None or has_altup
    is_lfm2 = n_shortconv_cache is not None or has_shortconv
    is_gemma4 = n_layers_in_expert_block is not None

    # Position encoding
    if has_rope:
        position = "rope"
    elif has_rel_pos_bias or relative_attention_buckets:
        position = "relative"
    elif has_pos_tensor or position_embedding_type == "absolute":
        position = "learned"
    elif is_ssm or is_rwkv:
        position = "none"
    else:
        position = "unknown"

    rope_type = "neox" if position == "rope" else "none"

    # Detect per-layer variation for hybrid models
    layer_variation = (_detect_per_layer_variation(tensor_names)
                       if is_hybrid else {})

    # ═══ BUILD SCHEMA ═══

    schema: dict[str, Any] = {"version": 2, "model_type": model_type}

    # ── Attention spec ──
    if is_ssm:
        attn_spec: dict[str, Any] = {"type": "ssm"}
        if ssm_conv_kernel:
            attn_spec["conv_kernel"] = ssm_conv_kernel
        if ssm_state_size:
            attn_spec["state_size"] = ssm_state_size
        if ssm_dt_rank:
            attn_spec["dt_rank"] = ssm_dt_rank
        if ssm_expand:
            attn_spec["expand"] = ssm_expand
    elif is_rwkv:
        attn_spec: dict[str, Any] = {"type": "rwkv"}
        if rwkv_head_size:
            attn_spec["head_size"] = rwkv_head_size
        if rwkv_rescale_every:
            attn_spec["rescale_every"] = rwkv_rescale_every
    elif is_mla:
        attn_spec: dict[str, Any] = {"type": "mla"}
        if kv_lora_rank:
            attn_spec["kv_lora_rank"] = kv_lora_rank
        if qk_nope_head_dim:
            attn_spec["qk_nope_head_dim"] = qk_nope_head_dim
        if qk_rope_head_dim:
            attn_spec["qk_rope_head_dim"] = qk_rope_head_dim
        if qkv_style:
            attn_spec["qkv"] = qkv_style
    else:
        attn_spec: dict[str, Any] = {"type": "sliding_window" if has_swa else "standard"}
        if qkv_style:
            attn_spec["qkv"] = qkv_style
    if rope_type != "none" and not is_ssm:
        attn_spec["rope"] = {"type": rope_type}
    if has_qk_norm:
        attn_spec["qk_norm"] = True
    if attn_bias is True:
        attn_spec["bias"] = True
    if t_attn_post:
        attn_spec["post_norm"] = {"type": norm_type or "rms_norm"}

    # ── FFN spec ──
    ffn_spec: dict[str, Any] = {}
    if is_moe:
        ffn_spec["type"] = "moe"
        moe_detail: dict[str, Any] = {"n_expert": n_expert, "n_expert_used": n_expert_used}
        if n_shared_experts:
            moe_detail["n_shared_experts"] = n_shared_experts
        if has_shared_expert:
            moe_detail["has_shared_expert_gate"] = True
        if first_k_dense:
            moe_detail["first_k_dense"] = first_k_dense
        if moe_layer_freq:
            moe_detail["moe_layer_freq"] = moe_layer_freq
        if expert_layer_period:
            moe_detail["expert_layer_period"] = expert_layer_period
        if expert_layer_offset is not None:
            moe_detail["expert_layer_offset"] = expert_layer_offset
        ffn_spec["moe"] = moe_detail
    elif ffn_type:
        ffn_spec["type"] = ffn_type
    if activation:
        ffn_spec["activation"] = activation
    if t_ffn_post:
        ffn_spec["post_norm"] = {"type": norm_type or "rms_norm"}

    # ── Norm style detection ──
    norm_position = "post" if is_bert_style else "pre"

    # ── Default layer ──
    layer: dict[str, Any] = {}

    if is_ssm:
        if has_attn_norm is not False:
            layer["pre_norm"] = {"type": norm_type or "rms_norm"}
        layer["ssm"] = attn_spec
        layer["residual"] = {"type": "sequential"}
    elif is_rwkv:
        layer["pre_norm"] = {"type": norm_type or "layer_norm"}
        layer["time_mix"] = attn_spec
        layer["residual"] = {"type": "sequential"}
        if has_ffn_norm is not False:
            layer["channel_mix_norm"] = {"type": norm_type or "layer_norm"}
        layer["channel_mix"] = {"type": "rwkv"}
    elif is_bert_style:
        layer["attention"] = attn_spec
        layer["attn_post_norm"] = {"type": "layer_norm"}
        layer["residual"] = {"type": "sequential"}
        layer["ffn"] = ffn_spec
        layer["ffn_post_norm"] = {"type": "layer_norm"}
    else:
        if has_attn_norm is not False:
            layer["pre_norm"] = {"type": norm_type or "rms_norm"}
        else:
            layer["pre_norm"] = {"type": "none"}
        layer["attention"] = attn_spec
        layer["residual"] = {"type": "parallel" if use_parallel else "sequential"}
        if has_ffn_norm is not False and not is_ssm:
            layer["ffn_norm"] = {"type": norm_type or "rms_norm"}
        if ffn_spec and not is_ssm:
            layer["ffn"] = ffn_spec

    # ── Hybrid layer pattern ──
    if is_hybrid:
        ssm_spec: dict[str, Any] = {"type": "ssm"}
        if ssm_conv_kernel:
            ssm_spec["conv_kernel"] = ssm_conv_kernel
        if ssm_state_size:
            ssm_spec["state_size"] = ssm_state_size
        if ssm_dt_rank:
            ssm_spec["dt_rank"] = ssm_dt_rank
        if has_ssm_b_norm:
            ssm_spec["has_b_norm"] = True
        if has_ssm_dt_norm:
            ssm_spec["has_dt_norm"] = True

        ssm_layer: dict[str, Any] = {
            "pre_norm": {"type": norm_type or "rms_norm"},
            "ssm": ssm_spec,
            "residual": {"type": "sequential"},
        }
        if has_ffn_norm:
            ssm_layer["ffn_norm"] = {"type": norm_type or "rms_norm"}
        if ffn_spec:
            ssm_layer["ffn"] = ffn_spec

        attn_layer: dict[str, Any] = {
            "pre_norm": {"type": norm_type or "rms_norm"},
            "attention": attn_spec,
            "residual": {"type": "sequential"},
        }
        if has_ffn_norm:
            attn_layer["ffn_norm"] = {"type": norm_type or "rms_norm"}
        if ffn_spec:
            attn_layer["ffn"] = ffn_spec

        layers_spec: dict[str, Any] = {"ssm_layer": ssm_layer, "attn_layer": attn_layer}
        if attn_layer_period:
            layers_spec["attn_layer_period"] = attn_layer_period
        if attn_layer_offset is not None:
            layers_spec["attn_layer_offset"] = attn_layer_offset
        if hybrid_override_pattern:
            layers_spec["hybrid_override_pattern"] = hybrid_override_pattern
    else:
        layers_spec: dict[str, Any] = {"default": layer}

    # ── Cross-attention for encoder-decoder ──
    if has_cross_attn or is_enc_dec:
        cross_spec: dict[str, Any] = {"type": "cross_attention"}
        if has_rel_pos_bias:
            cross_spec["relative_position_bias"] = True
        layers_spec["cross_attention"] = cross_spec

    # ── Architecture-specific features ──
    features: dict[str, Any] = {}

    if is_altup:
        features["altup"] = {}
        if n_altup:
            features["altup"]["n_altup"] = n_altup
        if laurel_rank:
            features["altup"]["laurel_rank"] = laurel_rank
        if has_laurel:
            features["altup"]["has_laurel"] = True

    if is_dsa:
        features["dsa"] = {}
        if indexer_n_head:
            features["dsa"]["indexer_n_head"] = indexer_n_head
        if rope_sections:
            features["dsa"]["rope_sections"] = rope_sections

    if is_lfm2:
        features["shortconv"] = {}
        if n_shortconv_cache:
            features["shortconv"]["cache_length"] = n_shortconv_cache

    if swa_period:
        features["chunked_attention"] = {"swa_period": swa_period}

    if n_moe_layer_step:
        features["interleaved_moe"] = {"layer_step": n_moe_layer_step}
        if n_ff_exp:
            features["interleaved_moe"]["n_ff_exp"] = n_ff_exp

    if is_gemma4:
        features["dual_path_moe"] = {}
        if n_layers_in_expert_block:
            features["dual_path_moe"]["layers_per_block"] = n_layers_in_expert_block

    # ── Embedding spec ──
    embedding_spec: dict[str, Any] = {"token_embd": {}, "position_embd": position}
    if has_type_embd or (type_vocab_size and type_vocab_size > 0):
        embedding_spec["token_type_embd"] = True
    if has_pooler:
        embedding_spec["pooler"] = True
    if position == "relative" and relative_attention_buckets:
        embedding_spec["relative_attention_buckets"] = relative_attention_buckets

    # ── Tensor manifest ──
    gt: dict[str, Any] = {"token_embd": {"shape": ["n_embd", "n_vocab"]}}
    if has_attn_norm is not False or is_ssm:
        gt["output_norm"] = {"shape": ["n_embd"]}
    if tied_embd is False:
        gt["output"] = {"shape": ["n_embd", "n_vocab"]}
    if is_enc_dec:
        gt["encoder_output_norm"] = {"shape": ["n_embd"]}
        gt["decoder_output_norm"] = {"shape": ["n_embd"]}

    lt: dict[str, Any] = {}
    if has_attn_norm:
        lt["attn_norm"] = {"shape": ["n_embd"]}
    if is_bert_style:
        lt["attn_q"] = {"shape": ["n_embd", "n_embd"]}
        lt["attn_k"] = {"shape": ["n_embd", "n_embd"]}
        lt["attn_v"] = {"shape": ["n_embd", "n_embd"]}
        lt["attn_out"] = {"shape": ["n_embd", "n_embd"]}
        lt["attn_out_norm"] = {"shape": ["n_embd"]}
        lt["ffn_up"] = {"shape": ["n_embd", "n_ff"]}
        lt["ffn_down"] = {"shape": ["n_ff", "n_embd"]}
        lt["ffn_out_norm"] = {"shape": ["n_embd"]}
    elif is_ssm:
        lt["ssm_in"] = {"shape": ["n_embd", "d_inner"]}
        lt["ssm_conv1d"] = {"shape": ["d_inner", "conv_kernel"]}
        lt["ssm_x_proj"] = {"shape": ["d_inner", "dt_rank + state_size * 2"]}
        lt["ssm_dt_proj"] = {"shape": ["dt_rank", "d_inner"]}
        lt["ssm_A_log"] = {"shape": ["d_inner", "state_size"]}
        lt["ssm_D"] = {"shape": ["d_inner"]}
        lt["ssm_out"] = {"shape": ["d_inner", "n_embd"]}
    elif is_rwkv:
        lt["time_mix_w1"] = {"shape": ["n_embd"]}
        lt["time_mix_w2"] = {"shape": ["n_embd"]}
        lt["channel_mix_key"] = {"shape": ["n_embd", "n_ff"]}
        lt["channel_mix_value"] = {"shape": ["n_ff", "n_embd"]}
        lt["channel_mix_receptance"] = {"shape": ["n_embd", "n_embd"]}
    else:
        if qkv_style == "combined":
            lt["attn_qkv"] = {"shape": ["n_embd", "n_embd + 2 * n_embd_gqa"]}
        elif qkv_style == "separate":
            lt["attn_q"] = {"shape": ["n_embd", "n_embd_head_k * n_head"]}
            lt["attn_k"] = {"shape": ["n_embd", "n_embd_k_gqa"]}
            lt["attn_v"] = {"shape": ["n_embd", "n_embd_v_gqa"]}
        if is_mla:
            lt["attn_kv_a_proj"] = {"shape": ["n_embd", "kv_lora_rank"]}
            lt["attn_kv_a_norm"] = {"shape": ["kv_lora_rank"]}
            lt["attn_kv_b_proj"] = {
                "shape": ["kv_lora_rank", "n_embd_k_gqa + n_embd_v_gqa"]}
        lt["attn_out"] = {"shape": ["n_embd_head_k * n_head", "n_embd"]}
        if has_qk_norm:
            lt["attn_q_norm"] = {"shape": ["n_embd_head_k"]}
            lt["attn_k_norm"] = {"shape": ["n_embd_head_k"]}
        if t_attn_post:
            lt["attn_post_norm"] = {"shape": ["n_embd"]}
        if has_ffn_norm:
            lt["ffn_norm"] = {"shape": ["n_embd"]}
        if ffn_type == "gated":
            lt["ffn_gate"] = {"shape": ["n_embd", "n_ff"]}
        lt["ffn_up"] = {"shape": ["n_embd", "n_ff"]}
        lt["ffn_down"] = {"shape": ["n_ff", "n_embd"]}
        if t_ffn_post:
            lt["ffn_post_norm"] = {"shape": ["n_embd"]}

    if is_moe:
        lt["ffn_gate_inp"] = {"shape": ["n_embd", "n_expert"]}
        lt["ffn_gate_exps"] = {"shape": ["n_embd", "n_ff", "n_expert"]}
        lt["ffn_down_exps"] = {"shape": ["n_ff", "n_embd", "n_expert"]}
        lt["ffn_up_exps"] = {"shape": ["n_embd", "n_ff", "n_expert"]}
        if n_shared_experts or has_shared_expert:
            lt["ffn_up_shared"] = {"shape": ["n_embd", "n_ff_shared"]}
            lt["ffn_down_shared"] = {"shape": ["n_ff_shared", "n_embd"]}
            lt["ffn_gate_shared"] = {"shape": ["n_embd", "n_ff_shared"]}

    if has_cross_attn:
        lt["cross_attn_q"] = {"shape": ["n_embd", "n_embd"]}
        lt["cross_attn_k"] = {"shape": ["n_embd", "n_embd"]}
        lt["cross_attn_v"] = {"shape": ["n_embd", "n_embd"]}
        lt["cross_attn_out"] = {"shape": ["n_embd", "n_embd"]}

    # Hybrid models include both SSM and attention tensors
    if is_hybrid:
        lt["ssm_in"] = {"shape": ["n_embd", "d_inner"]}
        lt["ssm_conv1d"] = {"shape": ["d_inner", "conv_kernel"]}
        lt["ssm_dt_proj"] = {"shape": ["dt_rank", "d_inner"]}
        lt["ssm_A_log"] = {"shape": ["d_inner", "state_size"]}
        lt["ssm_D"] = {"shape": ["d_inner"]}
        lt["ssm_out"] = {"shape": ["d_inner", "n_embd"]}

    # LFM2 short convolution tensors
    if is_lfm2:
        lt["shortconv_conv"] = {"shape": ["n_shortconv_cache", "n_embd"]}
        lt["shortconv_in_proj"] = {"shape": ["n_embd", "3 * n_embd"]}
        lt["shortconv_out_proj"] = {"shape": ["n_embd", "n_embd"]}

    # Gemma3N AltUp + Laurel tensors
    if is_altup:
        lt["altup_proj"] = {"shape": ["n_embd", "n_embd", "n_altup - 1"]}
        lt["altup_unembd_proj"] = {"shape": ["n_embd", "n_embd", "n_altup - 1"]}
        lt["altup_correct_coef"] = {"shape": ["n_altup"]}
        lt["altup_predict_coef"] = {"shape": ["n_altup"]}
        if has_laurel:
            lt["laurel_l"] = {"shape": ["n_embd", "laurel_rank"]}
            lt["laurel_r"] = {"shape": ["laurel_rank", "n_embd"]}

    schema["tensors"] = {"global": gt, "per_layer": lt}
    schema["embedding"] = embedding_spec
    schema["layers"] = layers_spec
    if features:
        schema["features"] = features
    output_spec: dict[str, Any] = {
        "norm": ({"type": norm_type or "rms_norm"}
                 if has_attn_norm is not False or is_ssm
                 else {"type": "none"}),
        "projection": {},
    }
    if norm_position == "post":
        schema["norm_position"] = "post"

    if tied_embd is True:
        output_spec["projection"]["tied_to"] = "token_embd"
    if logit_softcap:
        output_spec["logit_softcap"] = float(logit_softcap)
    schema["output"] = output_spec

    return schema


def derive_tokenizer(model_dir: Path) -> dict[str, Any]:
    """Derive tokenizer config from tokenizer.json.

    Extracts pre-tokenizer type, regex patterns, byte encoding mode,
    normalizer, and special token behavior — everything needed to
    reconstruct the tokenizer without per-model hardcoding.
    """
    tok_path = model_dir / "tokenizer.json"
    if not tok_path.exists():
        return {"error": "no tokenizer.json found"}

    with open(tok_path) as f:
        tok = json.load(f)

    result: dict[str, Any] = {}

    # Model type (BPE, Unigram, WordPiece)
    model_section = tok.get("model", {})
    result["model_type"] = model_section.get("type", "unknown")

    # Pre-tokenizer
    pre_tok = tok.get("pre_tokenizer")
    if pre_tok:
        result["pre_tokenizer"] = _extract_pre_tokenizer(pre_tok)

    # Normalizer
    normalizer = tok.get("normalizer")
    if normalizer:
        result["normalizer"] = _extract_normalizer(normalizer)

    # Added tokens summary
    added = tok.get("added_tokens", [])
    special_tokens = [t for t in added if t.get("special")]
    result["added_tokens_count"] = len(added)
    result["special_tokens_count"] = len(special_tokens)

    # Token IDs for key special tokens
    for t in added:
        content = t.get("content", "")
        if content in ("<s>", "<bos>", "[BOS]"):
            result["bos_token"] = content
        elif content in ("</s>", "<eos>", "[EOS]"):
            result["eos_token"] = content
        elif content in ("<pad>", "[PAD]", "<|endoftext|>"):
            result["pad_token"] = content

    # Map to known llama.cpp pre-tokenizer type
    result["llama_pre_type"] = _map_to_llama_pre_type(result)

    # Tokenizer config
    tok_config_path = model_dir / "tokenizer_config.json"
    if tok_config_path.exists():
        with open(tok_config_path) as f:
            tok_config = json.load(f)
        result["add_bos_token"] = tok_config.get("add_bos_token")
        result["add_eos_token"] = tok_config.get("add_eos_token")
        tc = tok_config.get("tokenizer_class", "")
        if tc:
            result["tokenizer_class"] = tc

    return result


def _extract_pre_tokenizer(pre_tok: dict) -> dict[str, Any]:
    """Extract pre-tokenizer info recursively."""
    pt_type = pre_tok.get("type", "unknown")
    result: dict[str, Any] = {"type": pt_type}

    if pt_type == "ByteLevel":
        result["add_prefix_space"] = pre_tok.get("add_prefix_space", False)
        result["use_regex"] = pre_tok.get("use_regex", True)
        result["byte_encode"] = True

    elif pt_type == "Split":
        pattern = pre_tok.get("pattern", {})
        if "Regex" in pattern:
            result["regex"] = pattern["Regex"]
        elif "String" in pattern:
            result["string_pattern"] = pattern["String"]
        result["behavior"] = pre_tok.get("behavior", "removed")
        result["byte_encode"] = False

    elif pt_type == "Sequence":
        steps = []
        has_byte_level = False
        regex_patterns = []
        for step in pre_tok.get("pretokenizers", []):
            step_info = _extract_pre_tokenizer(step)
            steps.append(step_info)
            if step_info.get("byte_encode"):
                has_byte_level = True
            if "regex" in step_info:
                regex_patterns.append(step_info["regex"])
        result["steps"] = steps
        result["byte_encode"] = has_byte_level
        if regex_patterns:
            result["regex"] = regex_patterns[0]  # primary regex

    elif pt_type == "Metaspace":
        result["replacement"] = pre_tok.get("replacement", "▁")
        result["add_prefix_space"] = pre_tok.get("add_prefix_space", True)
        result["byte_encode"] = False

    elif pt_type == "Whitespace":
        result["byte_encode"] = False

    elif pt_type == "WhitespaceSplit":
        result["byte_encode"] = False

    else:
        result["raw"] = pre_tok
        result["byte_encode"] = False

    return result


def _extract_normalizer(normalizer: dict) -> dict[str, Any]:
    """Extract normalizer config."""
    n_type = normalizer.get("type", "unknown")
    result: dict[str, Any] = {"type": n_type}

    if n_type == "Replace":
        pattern = normalizer.get("pattern", {})
        result["pattern"] = pattern.get("String") or pattern.get("Regex", "")
        result["content"] = normalizer.get("content", "")
    elif n_type == "NFC":
        pass  # type is enough
    elif n_type == "NFKC":
        pass
    elif n_type == "Sequence":
        result["steps"] = [_extract_normalizer(s)
                           for s in normalizer.get("normalizers", [])]
    elif n_type == "Lowercase":
        pass

    return result


# ── Map to known llama.cpp pre-tokenizer types ──

_KNOWN_REGEX_PATTERNS: dict[str, str] = {
    # GPT-2 / Phi family
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)":
        "gpt-2",
    # LLaMA3 family
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+":
        "llama-bpe",
    # Qwen2 family (case-insensitive contractions)
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+":
        "qwen2",
}


def _map_to_llama_pre_type(tok_info: dict) -> str:
    """Map extracted tokenizer info to a known llama.cpp pre-tokenizer type."""
    pre_tok = tok_info.get("pre_tokenizer", {})
    pt_type = pre_tok.get("type", "")
    regex = pre_tok.get("regex", "")

    # Pure ByteLevel (GPT-2, Phi, StarCoder)
    if pt_type == "ByteLevel":
        return "gpt-2"

    # SPM-style (Metaspace pre-tokenizer or normalizer with ▁)
    if pt_type == "Metaspace":
        return "default"  # SPM tokenizer
    normalizer = tok_info.get("normalizer", {})
    if normalizer.get("type") == "Replace" and normalizer.get("content") == "▁":
        return "default"  # SPM tokenizer

    # Regex-based — match against known patterns
    if regex:
        for known_regex, pre_type in _KNOWN_REGEX_PATTERNS.items():
            if regex == known_regex:
                return pre_type

    # Sequence with ByteLevel + Split (LLaMA3 pattern)
    if pt_type == "Sequence":
        steps = pre_tok.get("steps", [])
        has_byte = any(s.get("type") == "ByteLevel" for s in steps)
        has_split = any(s.get("type") == "Split" for s in steps)
        if has_byte and has_split:
            # LLaMA3-style: Split regex + ByteLevel encoding
            return "llama-bpe"
        if has_split and not has_byte:
            # Qwen-style: Split regex, no byte encoding
            return "qwen2"

    return "unknown"


def derive_model_config(model_dir: Path) -> dict[str, Any]:
    """Derive complete model config from HuggingFace model directory.

    Produces a unified JSON config containing:
    - graph: layer operations, activation, norm type, features
    - tokenizer: pre-tokenizer type, byte encoding, normalizer
    - tensors: QKV style, FFN type, MoE layout
    - model_info: architecture name, model type, variant info
    """
    config = load_config(model_dir)
    tensor_names = load_tensor_names(model_dir)
    schema = derive_schema(config, tensor_names)
    tokenizer = derive_tokenizer(model_dir)

    text_cfg = config.get("text_config", config)

    # Build the unified config
    model_config: dict[str, Any] = {
        "version": 1,
        "model_info": {
            "architecture": text_cfg.get("model_type", "unknown"),
            "model_type": schema.get("model_type", "causal_lm"),
            "hf_architectures": config.get("architectures", []),
        },
    }

    # Graph config (derived from schema)
    layer = schema.get("layers", {}).get("default", {})
    attn = layer.get("attention", {})
    ffn = layer.get("ffn", {})

    graph: dict[str, Any] = {}
    if layer.get("pre_norm", {}).get("type", "none") != "none":
        ops = ["norm"]
    else:
        ops = []
    if attn.get("qk_norm"):
        ops.extend(["qkv", "qk_norm"])
    if attn.get("rope", {}).get("type", "none") != "none":
        ops.append("rope")
    ops.extend(["attn", "filter"])
    if attn.get("post_norm"):
        ops.append("post_norm")
    ops.append("residual")
    if layer.get("ffn_norm", {}).get("type", "none") != "none":
        ops.append("ffn_norm")
    ops.append("ffn")
    if ffn.get("post_norm"):
        ops.append("ffn_post_norm")
    ops.extend(["residual", "cvec"])
    graph["layer_operations"] = ",".join(ops)
    graph["ffn_activation"] = ffn.get("activation", "silu")
    graph["norm_type"] = layer.get("pre_norm", {}).get("type", "rms_norm")
    model_config["graph"] = graph

    # Tokenizer config
    model_config["tokenizer"] = tokenizer

    # Tensor layout (from schema)
    tensor_layout: dict[str, Any] = {
        "qkv_style": attn.get("qkv", "separate"),
        "ffn_type": ffn.get("type", "gated"),
        "has_bias": attn.get("bias", False),
    }
    if ffn.get("type") == "moe":
        tensor_layout["moe"] = ffn.get("moe", {})
    model_config["tensors"] = tensor_layout

    # Features (from schema)
    if schema.get("features"):
        model_config["features"] = schema["features"]

    # Multimodal configs (vision, audio)
    multimodal = _extract_multimodal(config)
    if multimodal:
        model_config["multimodal"] = multimodal

    return model_config


def _extract_multimodal(config: dict) -> dict[str, Any]:
    """Extract vision and audio encoder configs from HF config.json."""
    result: dict[str, Any] = {}

    # Vision config — multiple naming patterns across model families
    vision_cfg = (config.get("vision_config")
                  or config.get("vision_encoder")
                  or config.get("visual", {}).get("config"))
    if vision_cfg and isinstance(vision_cfg, dict):
        result["vision"] = _extract_encoder_config(vision_cfg, "vision")

    # Audio config
    audio_cfg = (config.get("audio_config")
                 or config.get("whisper_config")
                 or config.get("audio_encoder"))
    if audio_cfg and isinstance(audio_cfg, dict):
        result["audio"] = _extract_encoder_config(audio_cfg, "audio")

    # Special multimodal token IDs
    token_ids: dict[str, Any] = {}
    for key in ["image_token_id", "video_token_id", "audio_token_id",
                "vision_token_id", "vision_start_token_id", "vision_end_token_id",
                "boi_token_id", "eoi_token_id", "boa_token_id", "eoa_token_id"]:
        val = config.get(key)
        if val is not None:
            token_ids[key] = val
    if token_ids:
        result["token_ids"] = token_ids

    # Multimodal integration pattern
    if config.get("vision_lora") or config.get("speech_lora"):
        result["integration"] = "lora"
    elif config.get("embd_layer"):
        result["integration"] = "embedding_layer"
    elif vision_cfg or audio_cfg:
        result["integration"] = "encoder_tower"

    # Vision soft tokens / output length
    for key in ["vision_soft_tokens_per_image", "default_output_length"]:
        val = config.get(key)
        if val is not None:
            result.setdefault("vision", {})
            result["vision"][key] = val

    return result


def _extract_encoder_config(cfg: dict, modality: str) -> dict[str, Any]:
    """Extract key parameters from a vision or audio encoder config."""
    result: dict[str, Any] = {}

    # Architecture
    model_type = cfg.get("model_type")
    if model_type:
        result["model_type"] = model_type

    # Core dimensions
    for key in ["hidden_size", "intermediate_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "head_dim"]:
        val = cfg.get(key) or cfg.get({"num_hidden_layers": "depth",
                                        "num_attention_heads": "num_heads"}.get(key, ""), None)
        if val is not None:
            result[key] = val

    # Vision-specific
    if modality == "vision":
        for key in ["patch_size", "image_size", "in_chans",
                    "spatial_merge_size", "spatial_patch_size",
                    "pooling_kernel_size", "position_embedding_size",
                    "default_output_length", "window_size",
                    "temporal_patch_size", "tokens_per_second"]:
            val = cfg.get(key)
            if val is not None:
                result[key] = val

    # Audio-specific
    if modality == "audio":
        for key in ["conv_kernel_size", "attention_chunk_size",
                    "output_proj_dims", "residual_weight",
                    "attention_context_left", "attention_context_right",
                    "attention_logit_cap"]:
            val = cfg.get(key)
            if val is not None:
                result[key] = val

    # Activation and norm
    hidden_act = cfg.get("hidden_act") or cfg.get("hidden_activation")
    if hidden_act:
        result["hidden_act"] = hidden_act
    rms_eps = cfg.get("rms_norm_eps") or cfg.get("layer_norm_eps")
    if rms_eps is not None:
        result["norm_eps"] = rms_eps

    # RoPE (some vision encoders use it)
    rope_params = cfg.get("rope_parameters")
    if rope_params:
        result["rope"] = rope_params

    return result


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <hf-model-dir> [--config]")
        sys.exit(1)
    model_dir = Path(sys.argv[1])

    if "--config" in sys.argv:
        # Full model config (graph + tokenizer + tensors)
        model_config = derive_model_config(model_dir)
        print(json.dumps(model_config, indent=2))
    else:
        # Graph schema only (backward compatible)
        config = load_config(model_dir)
        tensor_names = load_tensor_names(model_dir)
        if not tensor_names:
            print("WARNING: No tensor names found — schema will be incomplete",
                  file=sys.stderr)
        schema = derive_schema(config, tensor_names)
        print(json.dumps(schema, indent=2))


if __name__ == "__main__":
    main()
