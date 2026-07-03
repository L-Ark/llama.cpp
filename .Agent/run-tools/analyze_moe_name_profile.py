#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

LINE_RE = re.compile(
    r"\[kimi_cpu_moe_name_profile\] top(?P<rank>\d+) name=(?P<name>\S+) calls=(?P<calls>\d+) "
    r"total=(?P<total>[0-9.]+) ms/call fallback_t0=(?P<fallback>[0-9.]+) ms/call .*?"
    r"decode_calls=(?P<decode_calls>\d+) decode_total=(?P<decode_total>[0-9.]+) ms/call decode_fallback=(?P<decode_fallback>[0-9.]+) ms/call "
    r"prompt_calls=(?P<prompt_calls>\d+) prompt_total=(?P<prompt_total>[0-9.]+) ms/call prompt_fallback=(?P<prompt_fallback>[0-9.]+) ms/call"
)
LAYER_RE = re.compile(r"blk\.(\d+)\.ffn_(up|down|gate)_exps")

def parse(path: Path):
    rows = []
    for line in path.read_text(errors='ignore').splitlines():
        m = LINE_RE.search(line)
        if not m:
            continue
        d = m.groupdict()
        name = d['name']
        lm = LAYER_RE.search(name)
        layer = int(lm.group(1)) if lm else None
        kind = lm.group(2) if lm else 'other'
        calls = int(d['calls'])
        decode_calls = int(d['decode_calls'])
        prompt_calls = int(d['prompt_calls'])
        row = {
            'rank': int(d['rank']),
            'name': name,
            'layer': layer,
            'kind': kind,
            'calls': calls,
            'total_ms_per_call': float(d['total']),
            'fallback_ms_per_call': float(d['fallback']),
            'fallback_ms_est': calls * float(d['fallback']),
            'decode_calls': decode_calls,
            'decode_fallback_ms_per_call': float(d['decode_fallback']),
            'decode_fallback_ms_est': decode_calls * float(d['decode_fallback']),
            'prompt_calls': prompt_calls,
            'prompt_fallback_ms_per_call': float(d['prompt_fallback']),
            'prompt_fallback_ms_est': prompt_calls * float(d['prompt_fallback']),
        }
        rows.append(row)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stderr')
    ap.add_argument('--json-out')
    args = ap.parse_args()
    rows = parse(Path(args.stderr))
    by_layer = defaultdict(float)
    by_kind = defaultdict(float)
    by_band = defaultdict(float)
    for r in rows:
        if r['kind'] not in ('up','down'):
            continue
        by_kind[r['kind']] += r['fallback_ms_est']
        if r['layer'] is not None:
            by_layer[r['layer']] += r['fallback_ms_est']
            if r['layer'] <= 2:
                by_band['0-2'] += r['fallback_ms_est']
            elif r['layer'] <= 9:
                by_band['3-9'] += r['fallback_ms_est']
            elif r['layer'] <= 19:
                by_band['10-19'] += r['fallback_ms_est']
            elif r['layer'] <= 29:
                by_band['20-29'] += r['fallback_ms_est']
            else:
                by_band['30-39'] += r['fallback_ms_est']
    out = {
        'stderr': args.stderr,
        'rows_parsed': len(rows),
        'top_rows': rows[:20],
        'fallback_ms_top_rows_total': sum(r['fallback_ms_est'] for r in rows if r['kind'] in ('up','down')),
        'fallback_ms_by_kind_top_rows': dict(sorted(by_kind.items())),
        'fallback_ms_by_layer_top_rows': {str(k): v for k, v in sorted(by_layer.items())},
        'fallback_ms_by_band_top_rows': dict(sorted(by_band.items())),
    }
    text = json.dumps(out, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text + '\n')

if __name__ == '__main__':
    main()
