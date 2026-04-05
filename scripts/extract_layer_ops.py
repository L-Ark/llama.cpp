#!/usr/bin/env python3
"""Extract layer operation sequence from HuggingFace modeling_*.py files.

Uses Python AST to walk the forward() method of the decoder layer class
statement-by-statement, classifying each method call or addition as a
standard operation token.  The resulting token list describes the layer's
computation graph in the order llama.cpp needs to build it.

Usage:
    python scripts/extract-layer-ops.py /path/to/modeling_mymodel.py
    python scripts/extract-layer-ops.py /path/to/hf-model-dir/
"""

import ast
import sys
from pathlib import Path

# ── Leaf attribute name → operation token ────────────────────────────
# When a Call node like self.foo.bar(...) is encountered, `bar` (the leaf
# attribute) is looked up here.  More specific names (q_proj, k_norm) are
# naturally distinct from broader ones (self_attn, mlp).
ATTR_TO_OP: dict[str, str] = {
    # QKV projections
    'q_proj': 'qkv', 'k_proj': 'qkv', 'v_proj': 'qkv',
    'qkv_proj': 'qkv', 'c_attn': 'qkv',
    'query_key_value': 'qkv', 'wqkv': 'qkv',

    # QK normalization
    'q_norm': 'qk_norm', 'k_norm': 'qk_norm',
    'q_layernorm': 'qk_norm', 'k_layernorm': 'qk_norm',

    # Pre-attention normalization
    'input_layernorm': 'norm', 'pre_attention_norm': 'norm',
    'pre_attention_layernorm': 'norm',
    'attn_norm': 'norm', 'ln_1': 'norm', 'norm_1': 'norm',

    # Post-attention normalization (applied BEFORE residual add)
    'attn_post_norm': 'post_norm', 'post_attention_norm': 'post_norm',

    # Pre-FFN normalization
    'post_attention_layernorm': 'ffn_norm', 'ln_2': 'ffn_norm',
    'norm_2': 'ffn_norm', 'ffn_norm': 'ffn_norm',
    'pre_feedforward_norm': 'ffn_norm',
    'pre_feedforward_layernorm': 'ffn_norm',

    # Post-FFN normalization (applied BEFORE residual add)
    'ffn_post_norm': 'ffn_post_norm',
    'post_feedforward_layernorm': 'ffn_post_norm',
    'post_ffw_norm': 'ffn_post_norm', 'post_mlp_norm': 'ffn_post_norm',

    # Attention (module-level call: self.self_attn(...))
    'self_attn': 'attn', 'attention': 'attn',

    # FFN / MLP
    'mlp': 'ffn', 'feed_forward': 'ffn',

    # MoE
    'moe': 'moe', 'moe_layer': 'moe', 'experts': 'moe',
}

# Standalone function name substrings → operation token
FUNC_KEYWORDS: dict[str, str] = {
    'rotary': 'rope',
    'rope': 'rope',
}


# ── AST helpers ──────────────────────────────────────────────────────

def _leaf_attr(node: ast.expr) -> str | None:
    """Return the rightmost attribute name (e.g. 'q_proj' from self.attn.q_proj)."""
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _classify_call(call: ast.Call) -> str | None:
    """Map a Call AST node to an operation token, or None."""
    leaf = _leaf_attr(call.func)
    if leaf:
        lower = leaf.lower()
        if lower in ATTR_TO_OP:
            return ATTR_TO_OP[lower]
        # Standalone function call (e.g. apply_rotary_pos_emb)
        for keyword, token in FUNC_KEYWORDS.items():
            if keyword in lower:
                return token
    return None


def _is_residual_save(stmt: ast.stmt) -> bool:
    """True for `residual = hidden_states` (saving state, not an op)."""
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        t = stmt.targets[0]
        return isinstance(t, ast.Name) and t.id == 'residual' and isinstance(stmt.value, ast.Name)
    return False


def _has_residual_add(node: ast.expr, target_name: str | None = None) -> bool:
    """True if the expression tree contains a residual addition.

    Detects both explicit ``residual + x`` and implicit ``target = ... + target``
    patterns used in different HuggingFace model implementations.
    """
    for n in ast.walk(node):
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
            for operand in (n.left, n.right):
                if isinstance(operand, ast.Name):
                    if operand.id == 'residual':
                        return True
                    if target_name and operand.id == target_name:
                        return True
    return False


# ── Core extraction ──────────────────────────────────────────────────

def find_decoder_forward(source: str) -> ast.FunctionDef | None:
    """Locate the forward() method of the best decoder-layer class candidate.

    Priority: classes with 'decoder' > 'block' > 'transformer' > 'layer'.
    Classes whose names suggest sub-modules (Norm, Embed, Attention, MLP)
    are excluded so that e.g. CohereLayerNorm doesn't steal precedence
    from CohereDecoderLayer.
    """
    tree = ast.parse(source)
    EXCLUDE = ('norm', 'embed', 'attention', 'attn', 'mlp', 'rope', 'rotary')

    candidates: list[tuple[int, int, ast.FunctionDef]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        name = node.name.lower()
        if any(ex in name for ex in EXCLUDE):
            continue
        priority = 0
        if 'decoder' in name:
            priority = 4
        elif 'block' in name:
            priority = 3
        elif 'transformer' in name:
            priority = 2
        elif 'layer' in name:
            priority = 1
        if priority == 0:
            continue
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == 'forward':
                candidates.append((priority, len(item.body), item))

    if not candidates:
        return None
    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
    return candidates[0][2]


def _calls_innermost_first(node: ast.expr) -> list[ast.Call]:
    """Return Call nodes from an expression tree, innermost (executed first) last→first.

    ast.walk is BFS so outer calls appear before inner ones.  Reversing
    gives execution order: ``self.attn(self.norm(x))`` → [norm, attn].
    """
    return list(reversed([n for n in ast.walk(node) if isinstance(n, ast.Call)]))


def _iter_stmts(body: list[ast.stmt]):
    """Yield processable statements, recursing into if/for/while bodies.

    For if/else, only the if-body is processed to avoid mixing
    conflicting operation sequences from different branches.
    """
    for stmt in body:
        if isinstance(stmt, ast.If):
            yield from _iter_stmts(stmt.body)
        elif isinstance(stmt, (ast.For, ast.While)):
            yield from _iter_stmts(stmt.body)
        else:
            yield stmt


def extract_ops(forward: ast.FunctionDef) -> list[str]:
    """Walk forward() body statement-by-statement, emitting op tokens."""
    ops: list[str] = []

    for stmt in _iter_stmts(forward.body):
        if _is_residual_save(stmt):
            continue

        # Detect residual addition: hidden_states = residual + x  OR  x = ... + x
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            tgt = stmt.targets[0]
            tgt_name = tgt.id if isinstance(tgt, ast.Name) else None
            if _has_residual_add(stmt.value, tgt_name):
                ops.append('residual')
                continue

        # Classify calls in execution order (innermost first)
        value = stmt.value if isinstance(stmt, ast.Assign) else stmt
        for call in _calls_innermost_first(value):
            token = _classify_call(call)
            if token:
                ops.append(token)

    return ops


def _dedup_consecutive(ops: list[str]) -> list[str]:
    """Collapse consecutive duplicate tokens (e.g. qkv qkv qkv → qkv)."""
    out: list[str] = []
    for op in ops:
        if not out or out[-1] != op:
            out.append(op)
    return out


def _reclassify_norms(ops: list[str]) -> list[str]:
    """Fix ambiguous norm tokens based on position.

    `post_attention_layernorm` maps to `ffn_norm` by attribute name, but
    in models with post-norms (Gemma2) it appears *before* the residual
    add, making it a true `post_norm`.  This pass corrects the label by
    checking position relative to attn/residual boundaries.
    """
    result: list[str] = []
    saw_attn = False
    for op in ops:
        if op in ('attn',):
            saw_attn = True
        elif op == 'residual':
            saw_attn = False
        elif op == 'ffn_norm' and saw_attn:
            # Between attn and residual → post-attention norm
            result.append('post_norm')
            continue
        result.append(op)
    return result


def extract_layer_operations(source: str) -> list[str]:
    """Full pipeline: source → deduplicated op list with filter + cvec."""
    forward = find_decoder_forward(source)
    if not forward:
        return []

    ops = extract_ops(forward)
    ops = _dedup_consecutive(ops)
    ops = _reclassify_norms(ops)

    # Insert llama.cpp-specific tokens
    if 'attn' in ops:
        ops.insert(ops.index('attn') + 1, 'filter')
    if 'cvec' not in ops:
        ops.append('cvec')

    return ops


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path-to-modeling.py-or-model-dir>")
        sys.exit(1)

    path = Path(sys.argv[1])

    if path.is_dir():
        files = list(path.glob('modeling_*.py'))
        if not files:
            print(f"No modeling_*.py found in {path}", file=sys.stderr)
            sys.exit(1)
        source = files[0].read_text(encoding='utf-8')
        print(f"Parsing {files[0].name}...")
    elif path.is_file():
        source = path.read_text(encoding='utf-8')
        print(f"Parsing {path.name}...")
    else:
        print(f"Not found: {path}", file=sys.stderr)
        sys.exit(1)

    ops = extract_layer_operations(source)
    if ops:
        print(f"\nExtracted layer operations ({len(ops)} ops):")
        print(f"  {ops}")
    else:
        print("Failed to extract operations")
        sys.exit(1)


if __name__ == '__main__':
    main()
