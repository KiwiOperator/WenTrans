#!/usr/bin/env python3
"""Phase 1.1 — Flatten the NiuTrans Classical-Modern bitext into src + tgt files.

The repo's bitext is structured as 3-line blocks per sample:
    line 0: 古文：<ancient sentence>
    line 1: 现代文：<modern sentence>
    line 2: <blank>

We strip the role prefixes, drop non-Chinese-punctuation noise, traditionalise
ancient text, simplify modern text, and keep only pairs where
len(modern) >= len(ancient) - 1 (paper's filter).

Output:
    <out>/src.txt   - one modern sentence per line
    <out>/tgt.txt   - one ancient sentence per line, paired by line number
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

KEEP_CHARS = re.compile(r'[^一-龥。？！，、；：.?!,;:\'"’‘“”—…《》]')
ANCIENT_PREFIX = "古文："
MODERN_PREFIX = "现代文："


def trim_prefix(line: str, prefix: str) -> str:
    s = line.strip()
    if s.startswith(prefix):
        return s[len(prefix):]
    # some files use 3-char prefix without colon; fall back
    return s[3:] if len(s) > 3 else s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bitext", required=True,
                    help="Path to NiuTrans Classical-Modern double_data dir (or any "
                         "ancestor; we recursively glob bitext.txt under it).")
    ap.add_argument("--out", required=True, help="Output dir for src.txt + tgt.txt")
    ap.add_argument("--limit", type=int, default=0,
                    help="Stop after this many kept pairs (0 = no limit).")
    args = ap.parse_args()

    # Lazy traditional/simplified converter: prefer 'opencc' if installed,
    # otherwise pass-through (caller is warned).
    try:
        from opencc import OpenCC
        t2s = OpenCC("t2s").convert
        s2t = OpenCC("s2t").convert
    except Exception:
        print("WARN: opencc not installed; skipping traditional/simplified conversion. "
              "`pip install opencc-python-reimplemented` to enable.", file=sys.stderr)
        t2s = s2t = lambda x: x

    bitext_root = Path(args.bitext).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    src_path = out_dir / "src.txt"
    tgt_path = out_dir / "tgt.txt"

    files = sorted(bitext_root.rglob("bitext.txt"))
    if not files:
        sys.exit(f"No bitext.txt files found under {bitext_root}")
    print(f"found {len(files)} bitext.txt files under {bitext_root}")

    n_in = n_kept = 0
    with src_path.open("w", encoding="utf-8") as fsrc, tgt_path.open("w", encoding="utf-8") as ftgt:
        for fp in files:
            try:
                lines = fp.read_text(encoding="utf-8").splitlines()
            except Exception as e:
                print(f"skip {fp}: {e}", file=sys.stderr)
                continue
            for i in range(0, len(lines), 3):
                if i + 1 >= len(lines):
                    break
                n_in += 1
                ancient = trim_prefix(lines[i], ANCIENT_PREFIX)
                modern = trim_prefix(lines[i + 1], MODERN_PREFIX)
                ancient = KEEP_CHARS.sub("", ancient)
                modern = KEEP_CHARS.sub("", modern)
                if not ancient or not modern:
                    continue
                if len(modern) < len(ancient) - 1:
                    continue
                ancient = s2t(ancient)
                modern = t2s(modern)
                fsrc.write(modern + "\n")
                ftgt.write(ancient + "\n")
                n_kept += 1
                if args.limit and n_kept >= args.limit:
                    break
            if args.limit and n_kept >= args.limit:
                break

    print(f"input pairs:  {n_in:,}")
    print(f"kept pairs:   {n_kept:,}")
    print(f"src -> {src_path}")
    print(f"tgt -> {tgt_path}")


if __name__ == "__main__":
    main()
