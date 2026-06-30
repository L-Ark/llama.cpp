#!/usr/bin/env python3
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--max-memory-bytes", type=int, required=True)
    ap.add_argument("--require-n-predict", type=int)
    ap.add_argument("--require-output-quality", choices=["pass", "fail"])
    args = ap.parse_args()
    data = json.load(open(args.summary))
    errors = []
    if data.get("exit_code") != 0:
        errors.append(f"exit_code={data.get('exit_code')}")
    if args.require_n_predict is not None and data.get("n_predict") != args.require_n_predict:
        errors.append(f"n_predict={data.get('n_predict')} expected {args.require_n_predict}")
    mem = data.get("memory_peak_bytes")
    if mem is None or mem >= args.max_memory_bytes:
        errors.append(f"memory_peak_bytes={mem} limit<{args.max_memory_bytes}")
    if data.get("memory_sample_count", 0) <= 0:
        errors.append("no memory samples")
    if args.require_output_quality and data.get("output_quality", {}).get("status") != args.require_output_quality:
        errors.append(f"output_quality={data.get('output_quality')}")
    if data.get("decode_tokens_per_second") is None:
        errors.append("missing decode token rate")
    if data.get("ttft_s") is None:
        errors.append("missing TTFT")
    if errors:
        print("FAIL " + "; ".join(errors))
        return 1
    print(f"PASS {args.summary}: decode={data.get('decode_tokens_per_second')} tok/s ttft={data.get('ttft_s')}s mem={mem} gpu={data.get('gpu_memory_peak_mib')}MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
