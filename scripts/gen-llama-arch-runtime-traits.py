#!/usr/bin/env python3
"""Generate the llama-arch runtime traits include from JSON config."""

from __future__ import annotations

import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"
ARCH_HEADER_PATH = REPO_ROOT / "src" / "llama-arch.h"
TRAITS_OUTPUT_PATH = REPO_ROOT / "src" / "llama-arch-runtime-traits.inc"
DEFAULT_TRAITS_OUTPUT_PATH = REPO_ROOT / "src" / "llama-arch-runtime-default-traits.inc"

BOOL_FIELDS = [
    "is_recurrent",
    "is_hybrid",
    "is_diffusion",
    "supports_sm_tensor",
    "uses_encoder_pass",
    "prefers_embedding_outputs",
    "uses_sliding_window_metadata",
    "uses_explicit_swa_pattern",
]

INT_FIELDS = [
    "default_full_attention_interval",
    "default_sliding_window_pattern",
]

ENUM_FIELDS = {
    "default_expert_gating_func": {
        "softmax": "LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX",
        "sigmoid": "LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID",
        "softmax_weight": "LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT",
    },
}

OPTIONAL_BOOL_FIELDS = [
    "default_expert_weights_norm",
]

ALLOWED_FIELDS = set(BOOL_FIELDS + INT_FIELDS + list(ENUM_FIELDS) + OPTIONAL_BOOL_FIELDS)


def _bool_literal(value: bool) -> str:
    return "true" if value else "false"


def _validate_bool(arch: str, field: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{arch}.{field} must be a boolean")
    return value


def _validate_uint(arch: str, field: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{arch}.{field} must be a non-negative integer")
    return value


def _validate_enum(arch: str, field: str, value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{arch}.{field} must be a string")

    enum_value = ENUM_FIELDS[field].get(value)
    if enum_value is None:
        choices = ", ".join(sorted(ENUM_FIELDS[field]))
        raise ValueError(f"{arch}.{field} must be one of: {choices}")

    return enum_value


def _load_all_arches() -> list[str]:
    match = re.search(
        r"enum llm_arch\s*\{(?P<body>.*?)\};",
        ARCH_HEADER_PATH.read_text(encoding="utf-8"),
        re.S,
    )
    if match is None:
        raise ValueError(f"failed to parse llm_arch enum from {ARCH_HEADER_PATH}")

    arches = re.findall(r"LLM_ARCH_[A-Z0-9_]+", match.group("body"))
    if not arches:
        raise ValueError(f"no llm_arch entries found in {ARCH_HEADER_PATH}")

    if len(set(arches)) != len(arches):
        raise ValueError(f"duplicate llm_arch entries found in {ARCH_HEADER_PATH}")

    return arches


def _validate_traits(
    path: Path,
    traits: dict[str, object],
    known_arches: set[str],
) -> dict[str, dict[str, object]]:
    validated: dict[str, dict[str, object]] = {}

    for arch, raw_fields in sorted(traits.items()):
        if not isinstance(arch, str) or arch not in known_arches:
            raise ValueError(f"{path}: invalid arch key {arch!r}")
        if not isinstance(raw_fields, dict) or not raw_fields:
            raise ValueError(f"{path}: {arch} must map to a non-empty object")

        unknown_fields = sorted(set(raw_fields) - ALLOWED_FIELDS)
        if unknown_fields:
            raise ValueError(f"{path}: {arch} has unknown fields: {', '.join(unknown_fields)}")

        fields: dict[str, object] = {}

        for field in BOOL_FIELDS:
            if field in raw_fields:
                fields[field] = _validate_bool(arch, field, raw_fields[field])

        for field in INT_FIELDS:
            if field in raw_fields:
                fields[field] = _validate_uint(arch, field, raw_fields[field])

        for field in ENUM_FIELDS:
            if field in raw_fields:
                fields[field] = _validate_enum(arch, field, raw_fields[field])

        for field in OPTIONAL_BOOL_FIELDS:
            if field in raw_fields:
                fields[field] = _validate_bool(arch, field, raw_fields[field])

        validated[arch] = fields

    return validated


def _validate_default_traits_arches(
    path: Path,
    arches: object,
    known_arches: set[str],
) -> list[str]:
    if not isinstance(arches, list):
        raise ValueError(f"{path}: 'runtime.default_traits_arches' must be an array")

    validated: list[str] = []
    seen: set[str] = set()

    for arch in arches:
        if not isinstance(arch, str) or arch not in known_arches:
            raise ValueError(f"{path}: invalid default trait arch {arch!r}")
        if arch in seen:
            raise ValueError(f"{path}: duplicate default trait arch {arch}")
        seen.add(arch)
        validated.append(arch)

    return validated


def _load_runtime_trait_config(
    all_arches: list[str],
) -> tuple[dict[str, dict[str, object]], list[str]]:
    known_arches = set(all_arches)
    validated_traits: dict[str, dict[str, object]] = {}
    default_traits_arches: set[str] = set()
    covered_by_path: dict[str, Path] = {}

    for path in sorted(CONFIGS_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))

        if data.get("version") != 1:
            raise ValueError(f"{path} must declare version 1")

        runtime = data.get("runtime")
        if runtime is None:
            continue
        if not isinstance(runtime, dict):
            raise ValueError(f"{path}: 'runtime' must be an object")

        traits = runtime.get("traits")
        if traits is not None:
            if not isinstance(traits, dict):
                raise ValueError(f"{path}: 'runtime.traits' must be an object")

            for arch, fields in _validate_traits(path, traits, known_arches).items():
                if arch in covered_by_path:
                    raise ValueError(
                        f"Duplicate runtime trait coverage for {arch}: already defined in {covered_by_path[arch]} before {path}"
                    )
                validated_traits[arch] = fields
                covered_by_path[arch] = path

        default_arches = runtime.get("default_traits_arches")
        if default_arches is not None:
            for arch in _validate_default_traits_arches(path, default_arches, known_arches):
                if arch in covered_by_path:
                    raise ValueError(
                        f"Duplicate runtime trait coverage for {arch}: already defined in {covered_by_path[arch]} before {path}"
                    )
                default_traits_arches.add(arch)
                covered_by_path[arch] = path

    missing_arches = [arch for arch in all_arches if arch not in covered_by_path]
    if missing_arches:
        raise ValueError(
            "Missing runtime trait coverage for arches: "
            + ", ".join(missing_arches)
        )

    ordered_default_traits_arches = [arch for arch in all_arches if arch in default_traits_arches]
    return validated_traits, ordered_default_traits_arches


def _render_traits(traits: dict[str, dict[str, object]]) -> str:
    lines = [
        "// This file is generated by scripts/gen-llama-arch-runtime-traits.py",
        "// from runtime.traits entries under configs/*.json. Do not edit by hand.",
        "",
    ]

    for arch, fields in traits.items():
        lines.append(f"    {{ {arch}, []() {{")
        lines.append("        llm_arch_runtime_traits traits;")

        for field in BOOL_FIELDS:
            if field in fields:
                lines.append(f"        traits.{field} = {_bool_literal(fields[field])};")

        for field in INT_FIELDS:
            if field in fields:
                lines.append(f"        traits.{field} = {fields[field]}u;")

        for field in ENUM_FIELDS:
            if field in fields:
                lines.append(f"        traits.{field} = {fields[field]};")

        if "default_expert_weights_norm" in fields:
            lines.append("        traits.has_default_expert_weights_norm = true;")
            lines.append(
                "        traits.default_expert_weights_norm = "
                f"{_bool_literal(fields['default_expert_weights_norm'])};"
            )

        lines.append("        return traits;")
        lines.append("    }() },")

    return "\n".join(lines) + "\n"


def _render_default_traits_arches(default_traits_arches: list[str]) -> str:
    lines = [
        "// This file is generated by scripts/gen-llama-arch-runtime-traits.py",
        "// from runtime.default_traits_arches entries under configs/*.json. Do not edit by hand.",
        "",
    ]

    for arch in default_traits_arches:
        lines.append(f"    {arch},")

    return "\n".join(lines) + "\n"


def main() -> None:
    all_arches = _load_all_arches()
    traits, default_traits_arches = _load_runtime_trait_config(all_arches)
    TRAITS_OUTPUT_PATH.write_text(_render_traits(traits), encoding="utf-8", newline="\n")
    DEFAULT_TRAITS_OUTPUT_PATH.write_text(
        _render_default_traits_arches(default_traits_arches),
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
