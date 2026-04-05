#!/usr/bin/env python3
"""Generate architecture manifest from HuggingFace model files.

Reads modeling_*.py (via AST) and config.json to produce a complete
architecture manifest — everything llama.cpp needs to construct the
computation graph without hand-coded per-model C++ code.

The manifest can be:
  1. Embedded in GGUF during conversion (for new models)
  2. Compiled into a C++ lookup table (for existing models)

Usage:
    python scripts/generate_arch_manifest.py /path/to/hf-model-dir/
    python scripts/generate_arch_manifest.py /path/to/modeling_mymodel.py --config config.json
"""

import ast
import json
import sys
from pathlib import Path

from extract_layer_ops import (
    find_decoder_forward,
    extract_ops,
    _dedup_consecutive,
    _reclassify_norms,
    ATTR_TO_OP,
)

# ── Init attribute → tensor type mapping ─────────────────────────────
# Maps __init__ attribute assignments to llama.cpp tensor categories.
INIT_ATTR_TO_TENSOR = {
    # Normalization
    'input_layernorm':          'attn_norm',
    'pre_attention_norm':       'attn_norm',
    'pre_attention_layernorm':  'attn_norm',
    'attn_norm':                'attn_norm',
    'ln_1':                     'attn_norm',
    'norm_1':                   'attn_norm',

    'post_attention_layernorm': 'ffn_norm',
    'ln_2':                     'ffn_norm',
    'norm_2':                   'ffn_norm',
    'ffn_norm':                 'ffn_norm',
    'pre_feedforward_norm':     'ffn_norm',
    'pre_feedforward_layernorm':'ffn_norm',

    'attn_post_norm':           'attn_post_norm',
    'post_attention_norm':      'attn_post_norm',

    'ffn_post_norm':            'ffn_post_norm',
    'post_feedforward_layernorm':'ffn_post_norm',
    'post_ffw_norm':            'ffn_post_norm',
    'post_mlp_norm':            'ffn_post_norm',

    # Attention projections
    'q_proj':   'attn_q',
    'k_proj':   'attn_k',
    'v_proj':   'attn_v',
    'o_proj':   'attn_out',
    'qkv_proj': 'attn_qkv',
    'c_attn':   'attn_qkv',
    'query_key_value': 'attn_qkv',
    'wqkv':     'attn_qkv',

    # QK normalization
    'q_norm':       'attn_q_norm',
    'k_norm':       'attn_k_norm',
    'q_layernorm':  'attn_q_norm',
    'k_layernorm':  'attn_k_norm',

    # FFN / MLP sub-modules
    'gate_proj':  'ffn_gate',
    'up_proj':    'ffn_up',
    'down_proj':  'ffn_down',

    # MoE
    'block_sparse_moe': 'moe',
    'moe':              'moe',
    'experts':          'moe',
}


# ── AST extraction functions ─────────────────────────────────────────

def find_decoder_class(source: str) -> ast.ClassDef | None:
    """Find the decoder layer class."""
    tree = ast.parse(source)
    EXCLUDE = ('norm', 'embed', 'attention', 'attn', 'mlp', 'rope', 'rotary')
    candidates = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        name = node.name.lower()
        if any(ex in name for ex in EXCLUDE):
            continue
        priority = 0
        if 'decoder' in name: priority = 4
        elif 'block' in name: priority = 3
        elif 'transformer' in name: priority = 2
        elif 'layer' in name: priority = 1
        if priority > 0:
            candidates.append((priority, len(node.body), node))
    if not candidates:
        return None
    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
    return candidates[0][2]


def extract_init_tensors(cls_node: ast.ClassDef) -> list[str]:
    """Extract tensor/module names from __init__ by matching self.X = ... assignments."""
    tensors = []
    init = None
    for item in cls_node.body:
        if isinstance(item, ast.FunctionDef) and item.name == '__init__':
            init = item
            break
    if not init:
        return tensors

    for stmt in ast.walk(init):
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                if target.value.id == 'self':
                    attr = target.attr.lower()
                    if attr in INIT_ATTR_TO_TENSOR:
                        tensor_type = INIT_ATTR_TO_TENSOR[attr]
                        if tensor_type not in tensors:
                            tensors.append(tensor_type)
    return tensors


def extract_all_tensors(source: str) -> list[str]:
    """Extract tensor types from ALL relevant classes (decoder, attention, MLP)."""
    tree = ast.parse(source)
    tensors = []
    seen_attrs = set()
    attr_to_type = {}  # track which attr name → tensor type for reclassification

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        name = node.name.lower()
        if not any(kw in name for kw in ('decoder', 'layer', 'block', 'attention',
                                          'attn', 'mlp', 'feedforward')):
            continue
        if any(kw in name for kw in ('norm', 'embed', 'rotary', 'rope')):
            continue

        for item in node.body:
            if not isinstance(item, ast.FunctionDef) or item.name != '__init__':
                continue
            for stmt in ast.walk(item):
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                    target = stmt.targets[0]
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                        if target.value.id == 'self':
                            attr = target.attr.lower()
                            if attr in INIT_ATTR_TO_TENSOR and attr not in seen_attrs:
                                seen_attrs.add(attr)
                                tensor_type = INIT_ATTR_TO_TENSOR[attr]
                                attr_to_type[attr] = tensor_type
                                if tensor_type not in tensors:
                                    tensors.append(tensor_type)

    # Context-aware reclassification:
    # If both post_attention_layernorm AND pre_feedforward_layernorm exist,
    # then post_attention_layernorm is a true post-norm (before residual),
    # not the pre-FFN norm.
    has_post_attn_ln = 'post_attention_layernorm' in seen_attrs
    has_pre_ffn_ln = any(a in seen_attrs for a in ('pre_feedforward_layernorm', 'pre_feedforward_norm'))
    if has_post_attn_ln and has_pre_ffn_ln:
        if 'attn_post_norm' not in tensors:
            tensors.append('attn_post_norm')

    return tensors


def detect_combined_qkv(tensors: list[str]) -> bool:
    """Check if the model uses combined QKV projection."""
    return 'attn_qkv' in tensors


def detect_has_bias(source: str) -> bool:
    """Check if attention projections have bias=True."""
    return 'bias=True' in source or 'bias = True' in source


# ── Config extraction ────────────────────────────────────────────────

def extract_from_config(config: dict) -> dict:
    """Extract architecture-relevant fields from config.json."""
    result = {}

    # Activation
    act = config.get('hidden_act', config.get('activation_function', ''))
    if act:
        act_map = {
            'silu': 'silu', 'swish': 'silu',
            'gelu': 'gelu', 'gelu_new': 'gelu', 'gelu_fast': 'gelu',
            'gelu_pytorch_tanh': 'gelu',
            'relu': 'relu', 'relu2': 'relu_sqr', 'relu_sqr': 'relu_sqr',
            'swiglu': 'swiglu', 'geglu': 'geglu',
        }
        result['activation'] = act_map.get(act.lower(), act.lower())

    # Norm type
    if 'rms_norm_eps' in config:
        result['norm_type'] = 'rms_norm'
    elif 'layer_norm_eps' in config or 'layer_norm_epsilon' in config:
        result['norm_type'] = 'layer_norm'

    # Tied embeddings
    result['tied_embeddings'] = config.get('tie_word_embeddings', True)

    # MoE
    n_expert = config.get('num_local_experts', config.get('num_experts', 0))
    if n_expert > 0:
        result['n_expert'] = n_expert
        result['n_expert_used'] = config.get('num_experts_per_tok',
                                              config.get('num_selected_experts', 2))

    # Sliding window
    swa = config.get('sliding_window', None)
    if swa and swa > 0:
        result['sliding_window'] = swa

    # RoPE
    rope_theta = config.get('rope_theta', None)
    if rope_theta:
        result['rope_theta'] = rope_theta

    # Key dimensions
    for key in ['hidden_size', 'num_attention_heads', 'num_key_value_heads',
                'intermediate_size', 'num_hidden_layers', 'vocab_size',
                'max_position_embeddings', 'head_dim']:
        if key in config:
            result[key] = config[key]

    return result


# ── Manifest generation ──────────────────────────────────────────────

def generate_manifest(model_dir: str = None, modeling_file: str = None,
                      config_file: str = None) -> dict:
    """Generate complete architecture manifest."""
    manifest = {
        'version': 1,
        'generator': 'llama.cpp/scripts/generate_arch_manifest.py',
    }

    # Load source
    if model_dir:
        model_path = Path(model_dir)
        modeling_files = list(model_path.glob('modeling_*.py'))
        config_path = model_path / 'config.json'
    else:
        modeling_files = [Path(modeling_file)] if modeling_file else []
        config_path = Path(config_file) if config_file else None

    # Parse modeling file
    if modeling_files:
        source = modeling_files[0].read_text(encoding='utf-8')
        manifest['source_file'] = modeling_files[0].name

        # Layer operations
        forward = find_decoder_forward(source)
        if forward:
            ops = extract_ops(forward)
            ops = _dedup_consecutive(ops)
            ops = _reclassify_norms(ops)
            if 'attn' in ops:
                ops.insert(ops.index('attn') + 1, 'filter')
            if 'cvec' not in ops:
                ops.append('cvec')
            manifest['layer_operations'] = ops

        # Tensor manifest from all relevant classes
        tensors = extract_all_tensors(source)
        manifest['layer_tensors'] = tensors
        manifest['combined_qkv'] = detect_combined_qkv(tensors)
        manifest['has_bias'] = detect_has_bias(source)

    # Parse config.json
    if config_path and config_path.exists():
        with open(config_path, encoding='utf-8') as f:
            config = json.load(f)
        manifest['config'] = extract_from_config(config)

        # Derive flags from config + tensors
        flags = {}
        tensors = manifest.get('layer_tensors', [])
        config_data = manifest.get('config', {})

        flags['use_rope'] = 'rope_theta' in config_data or 'rope' in ' '.join(manifest.get('layer_operations', []))
        flags['use_pos_embd'] = not flags['use_rope'] and config_data.get('max_position_embeddings', 0) > 0
        flags['has_post_norm'] = 'attn_post_norm' in tensors
        flags['has_ffn_post_norm'] = 'ffn_post_norm' in tensors
        flags['has_qk_norm'] = 'attn_q_norm' in tensors or 'attn_k_norm' in tensors
        flags['has_moe'] = config_data.get('n_expert', 0) > 0
        flags['tied_embeddings'] = config_data.get('tied_embeddings', True)

        manifest['flags'] = flags

    return manifest


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <hf-model-dir> [--config config.json]")
        print(f"       {sys.argv[0]} <modeling_*.py> --config <config.json>")
        sys.exit(1)

    path = Path(sys.argv[1])
    config_file = None

    if '--config' in sys.argv:
        idx = sys.argv.index('--config')
        if idx + 1 < len(sys.argv):
            config_file = sys.argv[idx + 1]

    if path.is_dir():
        manifest = generate_manifest(model_dir=str(path))
    elif path.is_file():
        manifest = generate_manifest(modeling_file=str(path), config_file=config_file)
    else:
        print(f"Not found: {path}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
