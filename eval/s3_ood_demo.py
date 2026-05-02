#!/usr/bin/env python3
"""Side-by-side OOD demo: compare M_2 (clean-only) vs M_3 (after distant-sup
relabeling) on Shiji / Zizhitongjian-style sentences.

Picks a few sentences that the two models disagree on and prints them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


def decode_hybrid(chars, tags):
    """BMES+POS hybrid tags -> list of (word, pos)."""
    words = []
    buf_chars = []
    buf_pos = None
    for c, t in zip(chars, tags):
        if "-" in t:
            seg, pos = t.split("-", 1)
        else:
            seg, pos = t, ""
        if seg == "S":
            if buf_chars:
                words.append(("".join(buf_chars), buf_pos)); buf_chars = []; buf_pos = None
            words.append((c, pos))
        elif seg == "B":
            if buf_chars:
                words.append(("".join(buf_chars), buf_pos))
            buf_chars = [c]; buf_pos = pos
        elif seg == "M":
            buf_chars.append(c)
        elif seg == "E":
            buf_chars.append(c)
            words.append(("".join(buf_chars), buf_pos or pos)); buf_chars = []; buf_pos = None
    if buf_chars:
        words.append(("".join(buf_chars), buf_pos))
    return words


@torch.no_grad()
def predict(model, tok, id2label, sent: str):
    chars = list(sent)
    enc = tok(chars, is_split_into_words=True, truncation=True,
              max_length=512, return_tensors="pt").to(model.device)
    pred = model(**enc).logits.argmax(-1)[0].tolist()
    wids = enc.word_ids(0)
    tags = [None] * len(chars); prev = None
    for j, wid in enumerate(wids):
        if wid is None or wid == prev: continue
        if wid < len(tags): tags[wid] = id2label[pred[j]]
        prev = wid
    tags = [t or "S-null" for t in tags]
    return chars, tags


def fmt(words):
    return " / ".join(f"{w}/{p}" if p else w for w, p in words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2", default="output/acds/M2")
    ap.add_argument("--m3", default="output/acds/M3")
    ap.add_argument("--test_jsonl", default="data/acds/jsonl/test_b.jsonl",
                    help="OOD test set (Shiji + Zizhitongjian)")
    ap.add_argument("--n", type=int, default=6, help="how many disagreements to show")
    ap.add_argument("--out", default="report/figures/s3_ood_demo.md")
    args = ap.parse_args()

    print("loading M2/M3...")
    tok2 = AutoTokenizer.from_pretrained(args.m2)
    m2 = AutoModelForTokenClassification.from_pretrained(args.m2)
    tok3 = AutoTokenizer.from_pretrained(args.m3)
    m3 = AutoModelForTokenClassification.from_pretrained(args.m3)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    m2 = m2.to(device).eval(); m3 = m3.to(device).eval()
    id2_2 = {int(k): v for k, v in m2.config.id2label.items()}
    id2_3 = {int(k): v for k, v in m3.config.id2label.items()}

    examples = []
    with open(args.test_jsonl, encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))

    rows = ["# Stage 3 — OOD comparison (Test-B, Shiji + Zizhitongjian)\n"]
    rows.append(f"_Model directories: M2 = `{args.m2}`, M3 = `{args.m3}`._\n")
    rows.append(f"_Picking up to {args.n} sentences where M3 differs from M2._\n")

    n_shown = 0
    for ex in examples:
        if n_shown >= args.n:
            break
        sent = "".join(ex["chars"])
        _, t2 = predict(m2, tok2, id2_2, sent)
        _, t3 = predict(m3, tok3, id2_3, sent)
        if t2 == t3:
            continue
        gold_words = decode_hybrid(ex["chars"], ex["tags"])
        m2_words = decode_hybrid(ex["chars"], t2)
        m3_words = decode_hybrid(ex["chars"], t3)
        rows.append(f"```\ninput: {sent}\nM2  : {fmt(m2_words)}\nM3  : {fmt(m3_words)}\ngold: {fmt(gold_words)}\n```\n")
        n_shown += 1

    if n_shown == 0:
        rows.append("\n*M2 and M3 agreed on every sentence in this test split — try `--n` larger or a different split.*\n")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(rows), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
