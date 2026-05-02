#!/usr/bin/env python3
"""Convert cws.txt (Zuozhuan with /POS tags) into BMES char-level JSONL.

For each input line:
  1. Tokenise on whitespace, drop tokens that have no '/'.
  2. Strip /POS tags -> ordered list of words.
  3. Split this list into sentences at sentence-final word tokens
     (。 ！ ？ ：).
  4. For each sentence: produce (chars, BMES tags). Length filter applied.

Then sentence-level 8/1/1 random split (seed=42) into train/val/test.

Output format (JSONL): {"chars": ["公","及",...], "tags": ["S","S","B","M","E",...]}
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

SENT_END = {"。", "！", "？", "：", "！", "？"}
MIN_LEN = 2     # don't keep one-character "sentences"
MAX_LEN = 510   # leaves room for [CLS] / [SEP] inside max_seq_length=512
POS_RE = re.compile(r"^(.+?)/[a-zA-Z]+$")


def parse_token(tok: str) -> str | None:
    """'公/n' -> '公'. Returns None if the token doesn't fit word/POS shape."""
    m = POS_RE.match(tok)
    if not m:
        return None
    word = m.group(1).strip()
    return word or None


def split_into_sentences(words: list[str]) -> list[list[str]]:
    sents, cur = [], []
    for w in words:
        cur.append(w)
        if w in SENT_END:
            sents.append(cur)
            cur = []
    if cur:
        sents.append(cur)
    return sents


def bmes_for_sentence(words: list[str]) -> tuple[list[str], list[str]]:
    chars, tags = [], []
    for w in words:
        if len(w) == 1:
            chars.append(w)
            tags.append("S")
        else:
            for i, c in enumerate(w):
                chars.append(c)
                if i == 0:
                    tags.append("B")
                elif i == len(w) - 1:
                    tags.append("E")
                else:
                    tags.append("M")
    assert len(chars) == len(tags)
    return chars, tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="cws.txt")
    ap.add_argument("--out", default="data/cws")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    args = ap.parse_args()

    src = Path(args.src)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = []
    n_lines = 0
    n_skipped_no_tokens = 0
    for line in src.read_text(encoding="utf-8").splitlines():
        n_lines += 1
        toks = line.split()
        words = [w for w in (parse_token(t) for t in toks) if w]
        if not words:
            n_skipped_no_tokens += 1
            continue
        for sent_words in split_into_sentences(words):
            chars, tags = bmes_for_sentence(sent_words)
            if MIN_LEN <= len(chars) <= MAX_LEN:
                examples.append({"chars": chars, "tags": tags})

    rng = random.Random(args.seed)
    rng.shuffle(examples)
    n = len(examples)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    splits = {
        "train": examples[:n_train],
        "val": examples[n_train : n_train + n_val],
        "test": examples[n_train + n_val :],
    }

    for name, exs in splits.items():
        path = out_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for ex in exs:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        n_chars = sum(len(e["chars"]) for e in exs)
        n_words = sum(sum(1 for t in e["tags"] if t in ("B", "S")) for e in exs)
        print(f"{name:5s}: {len(exs):>6} sentences, {n_chars:>8} chars, {n_words:>7} words -> {path}")

    print(f"\ninput lines:  {n_lines}")
    print(f"skipped:      {n_skipped_no_tokens} (no word/POS tokens)")
    print(f"sentences:    {n}")
    print(f"split:        {n_train} train / {n_val} val / {n - n_train - n_val} test")
    print(f"seed:         {args.seed}")


if __name__ == "__main__":
    main()
