#!/usr/bin/env python3
"""Phase 1.6 — Convert all word/POS-tagged text files to JSONL with hybrid
BMES+POS labels (88 labels in the paper, ~88 here too — exact set is
derived from the union of D_a + D_p tags, which is what label.py does in
the original ACDS repo).

Inputs (each is a separate text file in the standard 'word/POS word/POS ...' format):
  --d_p   weakly labeled large data from 05_project_labels.py
  --d_a   small clean Zuozhuan training data (zuozhuan_train_utf8.txt OR our cws.txt)
  --test_a  EvaHan_testa_gold.txt
  --test_b  EvaHan_testb_gold.txt

Outputs:
  <out_dir>/labels.json          - id<->label vocabulary built from D_a + D_p
  <out_dir>/d_p.jsonl
  <out_dir>/d_a_train.jsonl      - 90% of D_a
  <out_dir>/d_a_val.jsonl        - 10%  of D_a (used as in-domain dev)
  <out_dir>/test_a.jsonl
  <out_dir>/test_b.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

# Paragraphs in cws.txt-style files can be very long; we cut on sentence-final
# punctuation (acts as a boundary, kept in the prior word).
SENT_END = {"。", "！", "？", "：", "；"}
MIN_LEN = 2
MAX_LEN = 510

POS_RE = re.compile(r"^(.+?)/([a-zA-Z]+|null)$")


def parse_segpos_line(line: str):
    """Return list of (word, pos) — None for malformed tokens."""
    out = []
    for tok in line.split():
        m = POS_RE.match(tok)
        if not m:
            return None
        out.append((m.group(1), m.group(2)))
    return out


def words_to_bmes_pos(words):
    """[(word, pos), ...] -> (chars, hybrid_tags). Hybrid tag = e.g. 'B-n', 'S-w'."""
    chars, tags = [], []
    for w, p in words:
        if len(w) == 1:
            chars.append(w); tags.append(f"S-{p}")
        else:
            chars.append(w[0]); tags.append(f"B-{p}")
            for c in w[1:-1]:
                chars.append(c); tags.append(f"M-{p}")
            chars.append(w[-1]); tags.append(f"E-{p}")
    return chars, tags


def split_long(words):
    """Split a too-long sentence at SENT_END word boundaries; keep the
    punctuation with the prior chunk."""
    cur = []
    out = []
    for w, p in words:
        cur.append((w, p))
        if w in SENT_END:
            out.append(cur); cur = []
    if cur:
        out.append(cur)
    return out


def file_to_examples(path: Path):
    """Walk a 'word/POS word/POS ...' text file -> list of {'chars':..., 'tags':...}."""
    examples = []
    n_skip = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        words = parse_segpos_line(line)
        if not words:
            n_skip += 1
            continue
        for chunk in split_long(words):
            chars, tags = words_to_bmes_pos(chunk)
            if MIN_LEN <= len(chars) <= MAX_LEN:
                examples.append({"chars": chars, "tags": tags})
    return examples, n_skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d_p", required=True)
    ap.add_argument("--d_a", required=True)
    ap.add_argument("--test_a", required=True)
    ap.add_argument("--test_b", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--val_ratio", type=float, default=0.1,
                    help="Fraction of D_a held out as in-domain dev split.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    print(f"reading D_p: {args.d_p}")
    d_p, dp_skip = file_to_examples(Path(args.d_p))
    print(f"  {len(d_p):,} sentences   (skipped {dp_skip})")

    print(f"reading D_a: {args.d_a}")
    d_a, da_skip = file_to_examples(Path(args.d_a))
    print(f"  {len(d_a):,} sentences   (skipped {da_skip})")

    print(f"reading Test-A: {args.test_a}")
    test_a, ta_skip = file_to_examples(Path(args.test_a))
    print(f"  {len(test_a):,} sentences   (skipped {ta_skip})")

    print(f"reading Test-B: {args.test_b}")
    test_b, tb_skip = file_to_examples(Path(args.test_b))
    print(f"  {len(test_b):,} sentences   (skipped {tb_skip})")

    # Build label set from D_a + D_p (so all relabeling targets are in vocab).
    labels = set()
    for ex_set in (d_a, d_p):
        for ex in ex_set:
            labels.update(ex["tags"])
    labels = sorted(labels)
    label2id = {l: i for i, l in enumerate(labels)}
    print(f"\nhybrid labels: {len(labels)}")
    print(f"  sample: {labels[:10]} ...")

    (out / "labels.json").write_text(json.dumps(label2id, ensure_ascii=False, indent=2),
                                     encoding="utf-8")

    # 90/10 D_a split for in-domain dev.
    rng = random.Random(args.seed)
    rng.shuffle(d_a)
    n_val = int(len(d_a) * args.val_ratio)
    d_a_val = d_a[:n_val]
    d_a_train = d_a[n_val:]

    def write(path, exs):
        with open(path, "w", encoding="utf-8") as f:
            for ex in exs:
                # Drop examples that contain a label not in the train vocab
                # (only happens for test sets — we'll filter at eval time).
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  wrote {len(exs):>7} -> {path}")

    print(f"\nwriting JSONLs to {out}/")
    write(out / "d_p.jsonl", d_p)
    write(out / "d_a_train.jsonl", d_a_train)
    write(out / "d_a_val.jsonl", d_a_val)
    write(out / "test_a.jsonl", test_a)
    write(out / "test_b.jsonl", test_b)


if __name__ == "__main__":
    main()
