#!/usr/bin/env python3
"""Three-way pseudo-perplexity comparison: base vs ours vs SikuRoBERTa.

Pseudo-PPL = exp(average over tokens of -log P(gold) when each token is masked
in turn). Lower = the model is more confident in the genuine sequence.

Usage:
    python eval/s1_pseudo_ppl.py \\
        --ours pretrain/results \\
        --out_table report/figures/s1_pseudo_ppl_table.md \\
        --out_fig   report/figures/s1_pseudo_ppl.png
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


PASSAGES = {
    "classical_zz":     "公及邾儀父盟于蔑——邾子克也。",
    "classical_lunyu":  "君子不器。多聞闕疑，慎言其餘，則寡尤。",
    "classical_mengzi": "天時不如地利，地利不如人和。",
    "modern_zh":        "今天天氣很好，我打算去公園散步看看花。",
}


@torch.no_grad()
def pseudo_ppl(model, tok, text: str) -> float:
    enc = tok(text, return_tensors="pt").to(model.device)
    ids = enc["input_ids"][0]
    mask_id = tok.mask_token_id
    special = set(tok.all_special_ids)
    nlls = []
    for i in range(len(ids)):
        if int(ids[i]) in special:
            continue
        masked = ids.clone()
        gold = int(masked[i]); masked[i] = mask_id
        logits = model(input_ids=masked.unsqueeze(0)).logits[0, i]
        logp = torch.log_softmax(logits, -1)[gold]
        nlls.append(-float(logp))
    if not nlls:
        return float("nan")
    return math.exp(sum(nlls) / len(nlls))


def load(name_or_path):
    print(f"  loading {name_or_path}")
    tok = AutoTokenizer.from_pretrained(name_or_path)
    m = AutoModelForMaskedLM.from_pretrained(name_or_path)
    if torch.cuda.is_available():
        m = m.to("cuda")
    elif torch.backends.mps.is_available():
        m = m.to("mps")
    return tok, m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="pretrain/results")
    ap.add_argument("--base", default="hfl/chinese-roberta-wwm-ext")
    ap.add_argument("--siku", default="SIKU-BERT/sikuroberta")
    ap.add_argument("--out_table", default="report/figures/s1_pseudo_ppl_table.md")
    ap.add_argument("--out_fig",   default="report/figures/s1_pseudo_ppl.png")
    ap.add_argument("--out_json",  default="report/figures/s1_pseudo_ppl.json")
    args = ap.parse_args()

    print("Models:")
    base_tok, base = load(args.base)
    ours_tok, ours = load(args.ours)
    siku_tok, siku = load(args.siku)

    rows = {}
    for name, text in PASSAGES.items():
        rows[name] = {
            "text": text,
            "base": pseudo_ppl(base, base_tok, text),
            "ours": pseudo_ppl(ours, ours_tok, text),
            "siku": pseudo_ppl(siku, siku_tok, text),
        }
        b, o, s = rows[name]["base"], rows[name]["ours"], rows[name]["siku"]
        print(f"  {name:<20} base={b:>10.3f}  ours={o:>10.3f}  siku={s:>10.3f}")

    # markdown table
    Path(args.out_table).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_table, "w", encoding="utf-8") as f:
        f.write("| passage | base | ours | SikuRoBERTa | best |\n")
        f.write("|---|---:|---:|---:|---|\n")
        for name, r in rows.items():
            best_label = min(("base", r["base"]), ("ours", r["ours"]),
                             ("siku", r["siku"]), key=lambda x: x[1])[0]
            f.write(f"| {name} | {r['base']:.2f} | {r['ours']:.2f} | "
                    f"{r['siku']:.2f} | {best_label} |\n")

    # bar chart (grouped, log y)
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed; skipping figure")
        Path(args.out_json).write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
        return

    names = list(rows.keys())
    x = np.arange(len(names))
    w = 0.27
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, [rows[n]["base"] for n in names], w, label="chinese-roberta-wwm-ext")
    ax.bar(x,     [rows[n]["ours"] for n in names], w, label="WenTrans (ours)")
    ax.bar(x + w, [rows[n]["siku"] for n in names], w, label="SikuRoBERTa")
    ax.set_yscale("log")
    ax.set_ylabel("Pseudo-perplexity (log scale, lower = better)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_title("Stage 1 — pseudo-PPL on classical & modern Chinese passages")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_fig, dpi=150)
    print(f"figure -> {args.out_fig}")
    print(f"table  -> {args.out_table}")

    Path(args.out_json).write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                   encoding="utf-8")


if __name__ == "__main__":
    main()
