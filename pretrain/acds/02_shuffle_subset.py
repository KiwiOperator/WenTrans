#!/usr/bin/env python3
"""Phase 1.2 — Seeded shuffle of (src, tgt) pairs and take a subset.

Reads src.txt + tgt.txt from --in_dir, shuffles them in lockstep, keeps the
first --n pairs, writes src_shuf.txt + tgt_shuf.txt to --out_dir.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src = (in_dir / "src.txt").read_text(encoding="utf-8").splitlines()
    tgt = (in_dir / "tgt.txt").read_text(encoding="utf-8").splitlines()
    assert len(src) == len(tgt), f"src/tgt length mismatch: {len(src)} vs {len(tgt)}"
    print(f"loaded {len(src):,} pairs from {in_dir}")

    pairs = list(zip(src, tgt))
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    pairs = pairs[: args.n]

    sp = out_dir / "src_shuf.txt"
    tp = out_dir / "tgt_shuf.txt"
    with sp.open("w", encoding="utf-8") as fs, tp.open("w", encoding="utf-8") as ft:
        for s, t in pairs:
            fs.write(s + "\n")
            ft.write(t + "\n")
    print(f"wrote {len(pairs):,} pairs (seed={args.seed}) -> {sp}, {tp}")


if __name__ == "__main__":
    main()
