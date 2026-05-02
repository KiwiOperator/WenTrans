#!/usr/bin/env python3
"""Quick post-training smoke test: load the pretrained model, run MLM
on a few masked classical Chinese sentences, and print top-k predictions.

Run from repo root:
    python pretrain/smoke_test.py --ckpt pretrain/results
"""
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


PROBES = [
    # From cws.txt line 7 (Zuozhuan): "生桓公而惠公薨"
    ("生[MASK]公而惠公薨", "桓"),
    # Idiomatic: "學而時習之"
    ("學而[MASK]習之", "時"),
    # From cws.txt line 25: "蔓草猶不可除"
    ("蔓草[MASK]不可除", "猶"),
    # Common: "天下[MASK]公"
    ("天下[MASK]公", "為"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="pretrain/results", help="Path to pretrained model dir")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    model = AutoModelForMaskedLM.from_pretrained(args.ckpt)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded: {args.ckpt}")
    print(f"params: {n_params:,} ({n_params/1e6:.1f} M)")
    print(f"vocab:  {len(tok)}")
    print()

    mask_id = tok.mask_token_id
    for sent, gold in PROBES:
        enc = tok(sent, return_tensors="pt")
        ids = enc["input_ids"][0]
        mask_pos = (ids == mask_id).nonzero(as_tuple=True)[0]
        if len(mask_pos) == 0:
            print(f"[skip] no [MASK] in: {sent}")
            continue
        with torch.no_grad():
            logits = model(**enc).logits  # [1, T, V]
        probs = logits[0, mask_pos[0]].softmax(-1)
        top_p, top_i = probs.topk(args.topk)
        preds = [tok.convert_ids_to_tokens(int(i)) for i in top_i]
        gold_id = tok.convert_tokens_to_ids(gold)
        gold_p = float(probs[gold_id])
        gold_rank = int((probs > gold_p).sum()) + 1

        print(f"input : {sent}")
        print(f"gold  : {gold}  P={gold_p:.4f}  rank={gold_rank}")
        print(f"top{args.topk}  : " + "  ".join(f"{t}({p:.3f})" for t, p in zip(preds, top_p.tolist())))
        print()


if __name__ == "__main__":
    main()
