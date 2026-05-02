#!/usr/bin/env python3
"""Per-tag confusion matrix for the Stage-2 BMES classifier on the test set.

Also prints the most common error patterns.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

LABELS = ["B", "M", "E", "S"]


@torch.no_grad()
def predict_batch(model, tok, id2label, examples, device):
    chars_batch = [ex["chars"] for ex in examples]
    enc = tok(
        chars_batch, is_split_into_words=True, truncation=True,
        max_length=512, padding=True, return_tensors="pt",
    ).to(device)
    pred_ids = model(**enc).logits.argmax(-1).cpu().tolist()

    all_pred_tags, all_gold_tags = [], []
    for i, ex in enumerate(examples):
        wids = enc.word_ids(batch_index=i)
        pt = [None] * len(ex["chars"])
        prev = None
        for j, wid in enumerate(wids):
            if wid is None or wid == prev:
                continue
            if wid < len(pt):
                pt[wid] = id2label[pred_ids[i][j]]
            prev = wid
        pt = [t or "S" for t in pt]
        all_pred_tags.extend(pt)
        all_gold_tags.extend(ex["tags"])
    return all_pred_tags, all_gold_tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="output/cws-wentrans")
    ap.add_argument("--test_jsonl", default="data/cws/test.jsonl")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out", default="report/figures/s2_confusion_matrix.png")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = model.to(device).eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    examples = []
    with open(args.test_jsonl, encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"test examples: {len(examples)}")

    all_pred, all_gold = [], []
    for s in range(0, len(examples), args.batch_size):
        p, g = predict_batch(model, tok, id2label,
                             examples[s : s + args.batch_size], device)
        all_pred.extend(p); all_gold.extend(g)
    print(f"total tokens: {len(all_pred):,}")

    cm = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for p, g in zip(all_pred, all_gold):
        if g in LABELS and p in LABELS:
            cm[LABELS.index(g), LABELS.index(p)] += 1
    accuracy = cm.trace() / cm.sum()
    print(f"per-tag accuracy: {accuracy:.4f}")

    # top-5 error patterns
    err = []
    for i, gold in enumerate(LABELS):
        for j, pred in enumerate(LABELS):
            if i != j:
                err.append((cm[i, j], gold, pred))
    err.sort(reverse=True)
    print("\ntop confusion patterns (gold -> pred, count):")
    for c, g, p in err[:5]:
        print(f"  {g} -> {p}: {c:,}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS)
    ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS)
    ax.set_xlabel("predicted"); ax.set_ylabel("gold")
    ax.set_title(f"Stage 2 — BMES confusion (per-tag acc {accuracy:.3f})")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]:,}\n({cm_norm[i, j]:.2f})",
                    ha="center", va="center", color=color, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.045)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"figure -> {args.out}")


if __name__ == "__main__":
    main()
