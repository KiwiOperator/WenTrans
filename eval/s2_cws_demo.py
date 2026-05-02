#!/usr/bin/env python3
"""Live demo of the Stage-2 CWS model.

Segments a panel of unlabelled classical Chinese sentences and prints the
result alongside the gold (where available). Writes a markdown snippet for
inclusion in the report.

Usage:
    python eval/s2_cws_demo.py --model output/cws-wentrans \\
        --gold_jsonl data/cws/test.jsonl --n 8 \\
        --out report/figures/s2_cws_demo.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


# A few sentences from different classical works, kept short for legibility.
PROBES_NO_GOLD = [
    "公子重耳奔狄狐偃從之",
    "孔子曰學而時習之不亦說乎",
    "莊子釣於濮水楚王使大夫二人往先焉",
    "孫子曰兵者國之大事死生之地存亡之道不可不察也",
]


def decode_bmes(chars, tags):
    """BMES tag sequence -> list of words."""
    words, buf = [], ""
    for c, t in zip(chars, tags):
        if t == "S":
            if buf:
                words.append(buf); buf = ""
            words.append(c)
        elif t == "B":
            if buf:
                words.append(buf)
            buf = c
        elif t == "M":
            buf += c
        elif t == "E":
            buf += c
            words.append(buf); buf = ""
    if buf:
        words.append(buf)
    return words


@torch.no_grad()
def segment(model, tok, id2label, sentence: str):
    chars = list(sentence)
    enc = tok(chars, is_split_into_words=True, return_tensors="pt").to(model.device)
    pred_ids = model(**enc).logits.argmax(-1)[0].tolist()
    wids = enc.word_ids(0)
    tags = [None] * len(chars)
    prev = None
    for j, wid in enumerate(wids):
        if wid is None or wid == prev:
            continue
        tags[wid] = id2label[pred_ids[j]]
        prev = wid
    tags = [t or "S" for t in tags]
    return chars, tags, decode_bmes(chars, tags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="output/cws-wentrans")
    ap.add_argument("--gold_jsonl", help="Optional: pull additional probes (chars+tags) from this JSONL")
    ap.add_argument("--n", type=int, default=8, help="How many gold examples to add")
    ap.add_argument("--out", default="report/figures/s2_cws_demo.md")
    args = ap.parse_args()

    print(f"loading {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model)
    if torch.cuda.is_available():
        model = model.to("cuda")
    elif torch.backends.mps.is_available():
        model = model.to("mps")
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    out_lines = ["# Stage 2 — CWS demo\n"]

    out_lines.append("## A. unseen classical sentences (no gold)\n")
    for sent in PROBES_NO_GOLD:
        _, _, words = segment(model, tok, id2label, sent)
        out_lines.append(f"```\ninput : {sent}\nours  : {' / '.join(words)}\n```\n")

    if args.gold_jsonl and Path(args.gold_jsonl).exists():
        out_lines.append(f"## B. test-set examples vs gold (from {args.gold_jsonl})\n")
        examples = []
        with open(args.gold_jsonl, encoding="utf-8") as f:
            for line in f:
                examples.append(json.loads(line))
                if len(examples) >= args.n:
                    break
        n_match = 0
        for ex in examples:
            sent = "".join(ex["chars"])
            _, _, ours_words = segment(model, tok, id2label, sent)
            gold_words = decode_bmes(ex["chars"], ex["tags"])
            ok = ours_words == gold_words
            n_match += int(ok)
            mark = "✓" if ok else "✗"
            out_lines.append(
                f"```\ninput : {sent}\nours  : {' / '.join(ours_words)}\n"
                f"gold  : {' / '.join(gold_words)}    {mark}\n```\n"
            )
        out_lines.append(f"\n*Exact-match on first {len(examples)} test sentences: {n_match}/{len(examples)}.*\n")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(out_lines), encoding="utf-8")
    print(f"wrote {args.out}")
    # also echo to stdout
    print("\n".join(out_lines))


if __name__ == "__main__":
    main()
