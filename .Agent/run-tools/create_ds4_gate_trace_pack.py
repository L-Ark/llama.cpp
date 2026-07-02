#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, shutil, struct, sys
from pathlib import Path
sys.path.insert(0, '/root/lfz/ik_llama/gguf-py')
from gguf import GGUFReader
MAGIC = b'GGMLMOEPACKv1\0\0\0'
HEADER_STRUCT = struct.Struct('<16sIIQQ')
ENTRY_STRUCT = struct.Struct('<128siIQQ')
COPY_CHUNK = 16 * 1024 * 1024

def align_up(v, a):
    return ((v + a - 1) // a) * a

def copy_slice(src, off, n, out):
    src.seek(off)
    left = n
    while left:
        chunk = src.read(min(COPY_CHUNK, left))
        if not chunk:
            raise RuntimeError(f'short read at {off}')
        out.write(chunk)
        left -= len(chunk)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--trace', required=True)
    ap.add_argument('-o', '--output', required=True)
    ap.add_argument('--align', type=int, default=4096)
    args = ap.parse_args()
    model = Path(args.model)
    trace = Path(args.trace)
    outp = Path(args.output)
    seen = set()
    order = []
    with trace.open() as f:
        for r in csv.DictReader(f):
            if int(r['cache_hit']):
                continue
            key = (r['tensor'], int(r['expert']))
            if key not in seen:
                seen.add(key)
                order.append(key)
    print(f'unique miss pairs: {len(order)}', file=sys.stderr)
    reader = GGUFReader(str(model), 'r')
    tensors = {t.name: t for t in reader.tensors}
    slices = []
    for name, expert in order:
        t = tensors[name]
        shape = [int(x) for x in t.shape.tolist()]
        n_expert = shape[2]
        expert_bytes = int(t.n_bytes) // n_expert
        off = int(t.data_offset) + expert * expert_bytes
        slices.append((name, expert, off, expert_bytes))
    total = sum(s[3] for s in slices)
    print(f'payload bytes={total} gib={total/(1024**3):.3f}', file=sys.stderr)
    index_size = ENTRY_STRUCT.size * len(slices)
    data_start = align_up(HEADER_STRUCT.size + index_size, args.align)
    outp.parent.mkdir(parents=True, exist_ok=True)
    tmp = outp.with_suffix(outp.suffix + '.tmp')
    entries = []
    with model.open('rb') as src, tmp.open('wb') as out:
        out.write(b'\0' * data_start)
        for i, (name, expert, off, nbytes) in enumerate(slices, 1):
            pos = out.tell()
            pad = align_up(pos, args.align) - pos
            if pad:
                out.write(b'\0' * pad)
            dst_off = out.tell()
            copy_slice(src, off, nbytes, out)
            tail = align_up(out.tell(), args.align) - out.tell()
            if tail:
                out.write(b'\0' * tail)
            entries.append((name, expert, dst_off, nbytes))
            if i % 256 == 0:
                print(f'packed {i}/{len(slices)}', file=sys.stderr)
        out.seek(0)
        out.write(HEADER_STRUCT.pack(MAGIC, 1, HEADER_STRUCT.size, len(entries), data_start))
        for name, expert, dst_off, nbytes in entries:
            nb = name.encode('utf-8')
            out.write(ENTRY_STRUCT.pack(nb + b'\0' * (128 - len(nb)), expert, 0, dst_off, nbytes))
    shutil.move(str(tmp), str(outp))
    print(outp)
if __name__ == '__main__':
    main()
