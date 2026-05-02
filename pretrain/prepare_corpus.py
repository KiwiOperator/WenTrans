#!/usr/bin/env python3
"""Prepare daizhigev20 ancient-Chinese corpus for MLM pre-training.

Walks the dataset root, subsamples files (file-level, seeded) up to a
character budget, cleans each file, splits into sentences, and writes
train/val files (one sentence per line) ready for HuggingFace's
``run_mlm.py --line_by_line``.
"""
from __future__ import annotations

import argparse
import os
import random
import re
import sys
import unicodedata
from pathlib import Path

SENT_SPLIT = re.compile(r"(?<=[。！？!?])")
INDENT = "　"  # full-width space used for paragraph indent
MIN_LEN = 10
MAX_LEN = 510  # leaves room for [CLS] and [SEP] inside max_seq_length=512


def clean_text(raw: str) -> str:
    text = unicodedata.normalize("NFKC", raw)
    lines = []
    for line in text.splitlines():
        line = line.strip().lstrip(INDENT).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def split_sentences(text: str):
    out = []
    for chunk in text.split("\n"):
        for sent in SENT_SPLIT.split(chunk):
            sent = sent.strip()
            if MIN_LEN <= len(sent) <= MAX_LEN:
                out.append(sent)
    return out


def collect_files(src: Path):
    return sorted(p for p in src.rglob("*.txt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to daizhigev20-master")
    ap.add_argument("--out", required=True, help="Output dir for train.txt / val.txt")
    ap.add_argument(
        "--max_chars",
        type=int,
        default=600_000_000,
        help="Approximate cleaned-character budget (default ~Siku Quanshu scale).",
    )
    ap.add_argument("--val_ratio", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    all_files = collect_files(src)
    if not all_files:
        sys.exit(f"No .txt files found under {src}")

    rng.shuffle(all_files)

    kept_files = []
    kept_chars = 0
    train_path = out / "train.txt"
    val_path = out / "val.txt"
    train_n_sent = val_n_sent = 0
    train_n_chars = val_n_chars = 0

    with open(train_path, "w", encoding="utf-8") as ftrain, open(
        val_path, "w", encoding="utf-8"
    ) as fval:
        for fp in all_files:
            if kept_chars >= args.max_chars:
                break
            try:
                raw = fp.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                print(f"skip {fp}: {e}", file=sys.stderr)
                continue
            cleaned = clean_text(raw)
            if not cleaned:
                continue
            sents = split_sentences(cleaned)
            if not sents:
                continue

            is_val = rng.random() < args.val_ratio
            sink = fval if is_val else ftrain
            for s in sents:
                sink.write(s + "\n")
            n_chars = sum(len(s) for s in sents)
            if is_val:
                val_n_sent += len(sents)
                val_n_chars += n_chars
            else:
                train_n_sent += len(sents)
                train_n_chars += n_chars
            kept_files.append(fp)
            kept_chars += n_chars

    print(f"src:           {src}")
    print(f"seed:          {args.seed}")
    print(f"files scanned: {len(all_files)}")
    print(f"files kept:    {len(kept_files)}")
    print(f"char budget:   {args.max_chars:,}")
    print(f"chars kept:    {kept_chars:,}")
    print(
        f"train: {train_n_sent:,} sentences / {train_n_chars:,} chars -> {train_path}"
    )
    print(f"val:   {val_n_sent:,} sentences / {val_n_chars:,} chars -> {val_path}")


if __name__ == "__main__":
    main()
