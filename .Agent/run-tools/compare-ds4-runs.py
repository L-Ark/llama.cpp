#!/usr/bin/env python3
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--max-memory-bytes", type=int, required=True)
    ap.add_argument("--max-ttft-ratio", type=float, required=True)
    ap.add_argument("--require-decode-improvement", action="store_true")
    ap.add_argument("--require-output-quality", choices=["pass", "fail"])
    args = ap.parse_args()
    base = json.load(open(args.baseline))
    cand = json.load(open(args.candidate))
    errors = []
    b_dec = base.get("decode_tokens_per_second")
    c_dec = cand.get("decode_tokens_per_second")
    b_ttft = base.get("ttft_s")
    c_ttft = cand.get("ttft_s")
    c_mem = cand.get("memory_peak_bytes")
    if cand.get("exit_code") != 0:
        errors.append(f"candidate exit_code={cand.get('exit_code')}")
    if c_mem is None or c_mem >= args.max_memory_bytes:
        errors.append(f"candidate memory_peak_bytes={c_mem} limit<{args.max_memory_bytes}")
    if args.require_output_quality and cand.get("output_quality", {}).get("status") != args.require_output_quality:
        errors.append(f"candidate output_quality={cand.get('output_quality')}")
    if b_dec is None or c_dec is None:
        errors.append("missing decode token rate")
    elif args.require_decode_improvement and c_dec <= b_dec:
        errors.append(f"decode did not improve: baseline={b_dec} candidate={c_dec}")
    if b_ttft is None or c_ttft is None:
        errors.append("missing TTFT")
    elif c_ttft / b_ttft > args.max_ttft_ratio:
        errors.append(f"TTFT ratio {c_ttft / b_ttft:.3f}>{args.max_ttft_ratio}")
    result = {
        "baseline_decode_tokens_per_second": b_dec,
        "candidate_decode_tokens_per_second": c_dec,
        "decode_delta_tokens_per_second": None if b_dec is None or c_dec is None else c_dec - b_dec,
        "baseline_ttft_s": b_ttft,
        "candidate_ttft_s": c_ttft,
        "ttft_ratio": None if b_ttft in (None, 0) or c_ttft is None else c_ttft / b_ttft,
        "candidate_memory_peak_bytes": c_mem,
        "candidate_gpu_memory_peak_mib": cand.get("gpu_memory_peak_mib"),
        "candidate_output_quality": cand.get("output_quality", {}).get("status"),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        print("FAIL " + "; ".join(errors))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
