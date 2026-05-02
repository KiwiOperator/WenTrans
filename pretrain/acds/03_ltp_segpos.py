#!/usr/bin/env python3
"""Phase 1.3 — LTP segments + POS-tags modern Chinese; char-splits ancient.

Inputs (from 02_shuffle_subset):
    <io_dir>/src_shuf.txt   - one modern sentence per line
    <io_dir>/tgt_shuf.txt   - one ancient sentence per line

Outputs:
    <io_dir>/src_shuf_seg.txt  - space-joined modern words
    <io_dir>/src_shuf_pos.txt  - space-joined LTP POS tags (aligned with seg)
    <io_dir>/tgt_shuf_seg.txt  - space-joined ancient single chars
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--io_dir", required=True)
    ap.add_argument("--ltp_model", default="LTP/base1",
                    help="HF/LTP model id; 'LTP/base1' is the default in the paper.")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    try:
        import torch
        from ltp import LTP
    except ImportError as e:
        sys.exit(f"Install LTP: pip install ltp\n({e})")

    io = Path(args.io_dir)
    src = (io / "src_shuf.txt").read_text(encoding="utf-8").splitlines()
    tgt = (io / "tgt_shuf.txt").read_text(encoding="utf-8").splitlines()
    assert len(src) == len(tgt)
    print(f"loaded {len(src):,} pairs from {io}")

    ltp = LTP(args.ltp_model)
    if torch.cuda.is_available():
        ltp.to("cuda")
        print("LTP on CUDA")
    else:
        print("LTP on CPU (will be slow)")

    seg_path = io / "src_shuf_seg.txt"
    pos_path = io / "src_shuf_pos.txt"
    tgt_seg_path = io / "tgt_shuf_seg.txt"

    with seg_path.open("w", encoding="utf-8") as fs, \
         pos_path.open("w", encoding="utf-8") as fp, \
         tgt_seg_path.open("w", encoding="utf-8") as ft:
        for start in range(0, len(src), args.batch_size):
            batch = src[start : start + args.batch_size]
            try:
                out = ltp.pipeline(batch, tasks=["cws", "pos"]).to_tuple()
                segs, poss = out[0], out[1]
            except Exception as e:
                # if LTP chokes on a batch, fall back per-sentence
                print(f"batch {start}: {e} — falling back per-sentence", file=sys.stderr)
                segs, poss = [], []
                for s in batch:
                    o = ltp.pipeline([s], tasks=["cws", "pos"]).to_tuple()
                    segs.append(o[0][0])
                    poss.append(o[1][0])

            for i, s in enumerate(batch):
                fs.write(" ".join(segs[i]) + "\n")
                fp.write(" ".join(poss[i]) + "\n")
                ft.write(" ".join(list(tgt[start + i])) + "\n")

            if (start + args.batch_size) % 1000 < args.batch_size:
                done = min(start + args.batch_size, len(src))
                print(f"  processed {done:,}/{len(src):,}")

    print(f"wrote: {seg_path}, {pos_path}, {tgt_seg_path}")


if __name__ == "__main__":
    main()
