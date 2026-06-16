#!/home/wici/lfz/ktransformers/.venv/bin/python
import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect a Hugging Face model repository without downloading weights."
    )
    parser.add_argument("--repo", default="deepseek-ai/DeepSeek-V4-Flash")
    parser.add_argument(
        "--output",
        default="/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-fastllm-ktransformers-ssd/results/hf-model-files.json",
    )
    args = parser.parse_args()

    # Keep all HF cache activity inside the workspace.
    os.environ.setdefault(
        "HF_HOME",
        "/home/wici/lfz/ik_llama/.Agent/plans/deepseek-v4-fastllm-ktransformers-ssd/.hf",
    )
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

    from huggingface_hub import HfApi

    api = HfApi()
    info = api.model_info(args.repo, files_metadata=True)
    files = []
    total_size = 0

    for sibling in info.siblings:
        size = getattr(sibling, "size", None) or 0
        total_size += size
        files.append(
            {
                "path": sibling.rfilename,
                "size_bytes": size,
                "size_gib": round(size / 1024**3, 6),
            }
        )

    payload = {
        "repo": args.repo,
        "sha": info.sha,
        "private": info.private,
        "gated": getattr(info, "gated", None),
        "file_count": len(files),
        "total_size_bytes": total_size,
        "total_size_gib": round(total_size / 1024**3, 3),
        "files": sorted(files, key=lambda item: item["path"]),
        "largest_files": sorted(files, key=lambda item: item["size_bytes"], reverse=True)[
            :40
        ],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in payload if k != "files"}, indent=2))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"inspect_hf_model.py failed: {exc}", file=sys.stderr)
        raise
