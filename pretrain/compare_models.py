#!/usr/bin/env python3
"""Side-by-side: base hfl/chinese-roberta-wwm-ext vs our continued-pretrain.

Two evaluations:
  1. Top-k MLM predictions on a few masked classical sentences.
  2. Pseudo-perplexity (mask each token one at a time, average NLL of the
     gold token, exp at the end) on held-out classical + modern sentences.

Lower pseudo-PPL = better domain fit. We expect:
  - Ours BETTER on classical Chinese
  - Ours slightly WORSE on modern Chinese (domain shift cost)
"""
from __future__ import annotations

import math

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

BASE = "hfl/chinese-roberta-wwm-ext"
OURS = "pretrain/results"

# (sentence, position-of-interest, gold-char) — the gold char is what we mask.
PROBES = [
    ("生[MASK]公而惠公薨", "桓"),       # Zuozhuan — name slot
    ("學而[MASK]習之", "時"),            # Analects opening
    ("蔓草[MASK]不可除", "猶"),         # Zuozhuan — function word
    ("天下[MASK]公", "為"),              # famous Confucian phrase
]

# For pseudo-PPL: held-out plain sentences (not masked).
# We expect classical lower-PPL for ours, modern higher-PPL.
PASSAGES = {
    "classical_zz": "公及邾儀父盟于蔑——邾子克也。",
    "classical_lunyu": "君子不器。多聞闕疑，慎言其餘，則寡尤。",
    "classical_mengzi": "天時不如地利，地利不如人和。",
    "modern_zh": "今天天氣很好，我打算去公園散步看看花。",
}


@torch.no_grad()
def topk_for_probe(model, tok, sentence: str, gold: str, k: int = 5):
    enc = tok(sentence, return_tensors="pt")
    ids = enc["input_ids"][0]
    mask_pos = (ids == tok.mask_token_id).nonzero(as_tuple=True)[0][0].item()
    logits = model(**enc).logits[0, mask_pos]
    probs = logits.softmax(-1)
    top_p, top_i = probs.topk(k)
    preds = [tok.convert_ids_to_tokens(int(i)) for i in top_i]
    gold_id = tok.convert_tokens_to_ids(gold)
    gold_p = float(probs[gold_id])
    gold_rank = int((probs > gold_p).sum()) + 1
    return preds, top_p.tolist(), gold_p, gold_rank


@torch.no_grad()
def pseudo_ppl(model, tok, text: str) -> float:
    """Mask each non-special token one at a time, average -log P(gold)."""
    enc = tok(text, return_tensors="pt")
    ids = enc["input_ids"][0]
    mask_id = tok.mask_token_id
    special = set(tok.all_special_ids)
    nlls = []
    for i in range(len(ids)):
        if int(ids[i]) in special:
            continue
        masked = ids.clone()
        gold = int(masked[i])
        masked[i] = mask_id
        logits = model(input_ids=masked.unsqueeze(0)).logits[0, i]
        logp = torch.log_softmax(logits, -1)[gold]
        nlls.append(-float(logp))
    if not nlls:
        return float("nan")
    return math.exp(sum(nlls) / len(nlls))


def main():
    print(f"Loading base : {BASE}")
    tok_b = AutoTokenizer.from_pretrained(BASE)
    m_b = AutoModelForMaskedLM.from_pretrained(BASE).eval()

    print(f"Loading ours : {OURS}")
    tok_o = AutoTokenizer.from_pretrained(OURS)
    m_o = AutoModelForMaskedLM.from_pretrained(OURS).eval()

    # ----- 1. top-k probes -----
    print("\n" + "=" * 78)
    print("Top-5 MLM predictions (lower gold rank = better)")
    print("=" * 78)
    for sent, gold in PROBES:
        print(f"\ninput: {sent}   gold: {gold}")
        for label, model, tok in [("base", m_b, tok_b), ("ours", m_o, tok_o)]:
            preds, ps, gp, gr = topk_for_probe(model, tok, sent, gold)
            top = "  ".join(f"{t}({p:.3f})" for t, p in zip(preds, ps))
            print(f"  {label}: gold P={gp:.4f} rank={gr:<5}  top5: {top}")

    # ----- 2. pseudo-PPL -----
    print("\n" + "=" * 78)
    print("Pseudo-perplexity (lower = better)")
    print("=" * 78)
    print(f"\n{'passage':<25} {'base':>10} {'ours':>10} {'Δ (ours-base)':>16}")
    print("-" * 65)
    for name, text in PASSAGES.items():
        ppl_b = pseudo_ppl(m_b, tok_b, text)
        ppl_o = pseudo_ppl(m_o, tok_o, text)
        delta = ppl_o - ppl_b
        marker = "  ✓ better" if delta < 0 else ("  ✗ worse" if delta > 0 else "")
        print(f"{name:<25} {ppl_b:>10.3f} {ppl_o:>10.3f} {delta:>+16.3f}{marker}")
    print("\n(passage text:)")
    for name, text in PASSAGES.items():
        print(f"  {name}: {text}")


if __name__ == "__main__":
    main()
