#!/usr/bin/env python3
"""Phase 1.5 — Project modern POS tags onto ancient characters via the
GIZA++ alignment. Adjacent ancient chars aligned to the same modern word
are merged into one ancient word. Output format matches cws.txt.

Inputs (in --io_dir):
    src_shuf_seg.txt   - modern words per line
    src_shuf_pos.txt   - modern LTP POS per line
    tgt_shuf_seg.txt   - ancient single-chars per line
    alignment.txt      - GIZA++ output: 'i-j:prob i-j:prob ...'

Output:
    --out (default: <io_dir>/tgt_shuf_segpos.txt)
       e.g.  '蒙武/nr 為/v 秦/ns 裨將軍/n ，/w 與/p ...'
"""
from __future__ import annotations

import argparse
from pathlib import Path

# LTP 863-tag set -> ancient tag set, taken verbatim from
# acds/align-pos_tag_ltp.py. 'null' means: drop the tag (treat as untagged).
LTP2ANCIENT = {
    "a": "a", "b": "a", "c": "c", "d": "d", "e": "y", "h": "null",
    "i": "null", "j": "null", "k": "null", "m": "m", "n": "n", "nd": "f",
    "nh": "nr", "ni": "ns", "nl": "n", "ns": "ns", "nt": "t", "nz": "n",
    "o": "s", "p": "p", "q": "q", "r": "r", "u": "u", "v": "v",
    "wp": "w", "ws": "null", "x": "null", "g": "null", "z": "a",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--io_dir", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()

    io = Path(args.io_dir)
    out_path = Path(args.out) if args.out else (io / "tgt_shuf_segpos.txt")

    src_seg = (io / "src_shuf_seg.txt").read_text(encoding="utf-8").splitlines()
    src_pos = (io / "src_shuf_pos.txt").read_text(encoding="utf-8").splitlines()
    tgt_seg = (io / "tgt_shuf_seg.txt").read_text(encoding="utf-8").splitlines()
    align   = (io / "alignment.txt").read_text(encoding="utf-8").splitlines()
    n = len(align)
    assert len(src_seg) == n == len(src_pos) == len(tgt_seg), (
        f"length mismatch: src_seg={len(src_seg)} src_pos={len(src_pos)} "
        f"tgt_seg={len(tgt_seg)} alignment={n}"
    )
    print(f"projecting labels for {n:,} pairs")

    n_written = 0
    n_chars_total = 0
    n_chars_null = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i in range(n):
            modern_words = src_seg[i].split()
            modern_pos   = src_pos[i].split()
            ancient_chars = tgt_seg[i].split()
            if len(modern_pos) != len(modern_words):
                # LTP very rarely emits a count mismatch; skip those pairs
                continue

            # For each ancient char, keep best alignment (highest prob) to a
            # modern word whose POS maps to a non-'null' ancient tag.
            best = [(-1, -1.0, "null") for _ in ancient_chars]
            for tok in align[i].split():
                # tok format: 'i-j:prob' or 'i-j' (no prob)
                if ":" in tok:
                    pair, prob_s = tok.split(":")
                    prob = float(prob_s)
                else:
                    pair = tok; prob = 1.0
                m_idx_s, a_idx_s = pair.split("-")
                m_idx = int(m_idx_s); a_idx = int(a_idx_s)
                if m_idx >= len(modern_pos) or a_idx >= len(ancient_chars):
                    continue
                anc_tag = LTP2ANCIENT.get(modern_pos[m_idx], "null")
                if anc_tag == "null":
                    continue
                if prob > best[a_idx][1]:
                    best[a_idx] = (m_idx, prob, anc_tag)

            # Now glue adjacent chars sharing the same modern_word index into
            # one ancient word; emit '<word>/<tag>'.
            line_parts = []
            j = 0
            while j < len(ancient_chars):
                m_idx_j, _, tag_j = best[j]
                if m_idx_j == -1:
                    line_parts.append(f"{ancient_chars[j]}/null")
                    n_chars_null += 1
                    j += 1
                else:
                    k = j + 1
                    while k < len(ancient_chars) and best[k][0] == m_idx_j:
                        k += 1
                    word = "".join(ancient_chars[j:k])
                    line_parts.append(f"{word}/{tag_j}")
                    j = k
            n_chars_total += len(ancient_chars)
            if line_parts:
                f.write(" ".join(line_parts) + "\n")
                n_written += 1

    print(f"wrote {n_written:,} sentences -> {out_path}")
    print(f"chars total/null: {n_chars_total:,} / {n_chars_null:,} "
          f"({100*n_chars_null/max(1,n_chars_total):.1f}% unaligned)")


if __name__ == "__main__":
    main()
