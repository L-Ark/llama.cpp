#!/usr/bin/env python3
"""Probe Hugging Face resolver sizes for Kimi lower-quant GGUF candidates.

This is non-destructive: it only reads HF API metadata and resolver headers.
It does not download model payloads.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Candidate:
    label: str
    repo: str
    prefix: str


DEFAULT_CANDIDATES = [
    Candidate("i1-IQ1_S", "mradermacher/Kimi-K2.7-Code-i1-GGUF", "Kimi-K2.7-Code.i1-IQ1_S.gguf.part"),
    Candidate("i1-IQ1_M", "mradermacher/Kimi-K2.7-Code-i1-GGUF", "Kimi-K2.7-Code.i1-IQ1_M.gguf.part"),
    Candidate("i1-IQ2_XXS", "mradermacher/Kimi-K2.7-Code-i1-GGUF", "Kimi-K2.7-Code.i1-IQ2_XXS.gguf.part"),
    Candidate("unsloth UD-IQ1_M", "unsloth/Kimi-K2.7-Code-GGUF", "UD-IQ1_M/"),
    Candidate("unsloth UD-IQ2_XXS", "unsloth/Kimi-K2.7-Code-GGUF", "UD-IQ2_XXS/"),
    Candidate("NullVoider UD-IQ1_M", "NullVoider/Kimi-K2.7-Code-GGUF", "UD-IQ1_M/"),
    Candidate("huihui UD-IQ1_M-MXFP4", "huihui-ai/Huihui-Kimi-K2.7-Code-abliterated-GGUF", "UD-IQ1_M-MXFP4/"),
    Candidate("deep55 pruned", "freakyskittle/kimi-k2.75-code-GGUF", "deep55/"),
    Candidate("pruned compact oxidize q4", "freakyskittle/kimi-k2.75-code-GGUF", "pruned-compact-oxidize-q4/"),
]

SIZE_RE = re.compile(r"^x-linked-size:\s*(\d+)", re.I | re.M)
COMMIT_RE = re.compile(r"^x-repo-commit:\s*(\S+)", re.I | re.M)


def list_gguf_files(repo: str, prefix: str, timeout_s: int) -> list[str]:
    url = f"https://huggingface.co/api/models/{repo}?expand=siblings"
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        data: dict[str, Any] = json.load(response)
    files = []
    for sibling in data.get("siblings", []):
        filename = sibling.get("rfilename") or ""
        lower = filename.lower()
        if filename.startswith(prefix) and (lower.endswith(".gguf") or ".gguf." in lower):
            files.append(filename)
    return sorted(files)


def resolver_header(repo: str, filename: str, timeout_s: int) -> str:
    url = f"https://huggingface.co/{repo}/resolve/main/{urllib.parse.quote(filename)}"
    cmd = [
        "curl",
        "-sS",
        "-D",
        "-",
        "-o",
        "/dev/null",
        "--max-time",
        str(timeout_s),
        "-H",
        "Range: bytes=0-4095",
        url,
    ]
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-s", type=int, default=45)
    parser.add_argument("--out-json")
    args = parser.parse_args()

    results = []
    for candidate in DEFAULT_CANDIDATES:
        row: dict[str, Any] = {
            "label": candidate.label,
            "repo": candidate.repo,
            "prefix": candidate.prefix,
            "files": [],
            "size_bytes": 0,
            "sized_files": 0,
            "commit": None,
            "error": None,
        }
        try:
            files = list_gguf_files(candidate.repo, candidate.prefix, args.timeout_s)
            for filename in files:
                item: dict[str, Any] = {"filename": filename, "size": None}
                try:
                    header = resolver_header(candidate.repo, filename, args.timeout_s)
                    size_match = SIZE_RE.search(header)
                    commit_match = COMMIT_RE.search(header)
                    if commit_match and row["commit"] is None:
                        row["commit"] = commit_match.group(1)
                    if size_match:
                        size = int(size_match.group(1))
                        item["size"] = size
                        row["size_bytes"] += size
                        row["sized_files"] += 1
                    else:
                        item["header_excerpt"] = "\n".join(header.splitlines()[:5])
                except Exception as exc:
                    item["error"] = repr(exc)
                row["files"].append(item)
        except Exception as exc:
            row["error"] = repr(exc)
        row["gib"] = row["size_bytes"] / 1024**3
        row["gb"] = row["size_bytes"] / 1000**3
        results.append(row)

    payload = {"kind": "kimi_hf_candidate_size_probe", "results": results}
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

