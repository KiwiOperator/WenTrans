#!/usr/bin/env python3
"""Use M_2 to relabel D_p, producing D_r (large + cleaner).

For each example in --in_jsonl, runs the model in eval mode, takes argmax,
and writes a new JSONL line with the predicted hybrid tags. The 'chars'
field is preserved verbatim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to M_2 checkpoint dir")
    ap.add_argument("--in_jsonl", required=True, help="D_p JSONL to relabel")
    ap.add_argument("--out_jsonl", required=True, help="D_r JSONL to write")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_seq_length", type=int, default=512)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    print(f"loaded {args.model}: {len(id2label)} labels, device={device}")

    in_path = Path(args.in_jsonl); out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    examples = []
    with in_path.open(encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"relabeling {len(examples):,} sentences")

    n_done = 0
    with out_path.open("w", encoding="utf-8") as fout, torch.no_grad():
        for start in range(0, len(examples), args.batch_size):
            batch = examples[start : start + args.batch_size]
            chars_batch = [ex["chars"] for ex in batch]
            enc = tokenizer(
                chars_batch,
                is_split_into_words=True,
                truncation=True,
                max_length=args.max_seq_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            preds = logits.argmax(-1).cpu().tolist()

            for i, ex in enumerate(batch):
                # Map subtokens back to per-character tag (only the first
                # subtoken per word_id receives the prediction).
                wids = enc.word_ids(batch_index=i)
                cur_tags = [None] * len(ex["chars"])
                prev = None
                for j, wid in enumerate(wids):
                    if wid is None or wid == prev:
                        continue
                    if wid < len(cur_tags):
                        cur_tags[wid] = id2label[preds[i][j]]
                    prev = wid
                # truncated tail (very rare with max_len 512): fall back to S-null
                cur_tags = [t if t is not None else "S-null" for t in cur_tags]
                fout.write(json.dumps({"chars": ex["chars"], "tags": cur_tags},
                                      ensure_ascii=False) + "\n")
            n_done += len(batch)
            if n_done % 2000 < args.batch_size:
                print(f"  {n_done:,} / {len(examples):,}")

    print(f"wrote D_r -> {out_path}")


if __name__ == "__main__":
    main()
