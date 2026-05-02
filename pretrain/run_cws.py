#!/usr/bin/env python3
"""Stage 2 — fine-tune the WenTrans pretrained encoder for Zuozhuan CWS.

Replicates Wang et al. 2021 §3.3 (SikuRoBERTa CWS task):
  - 4-label BMES token classification on top of AutoModelForTokenClassification
  - Word-level P / R / F evaluation (a predicted word counts only on exact span)

Usage:
    python pretrain/run_cws.py \
        --model_name_or_path pretrain/results \
        --data_dir data/cws \
        --output_dir output/cws-wentrans \
        --do_train --do_eval --do_predict
"""
from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)

LABELS = ["B", "M", "E", "S"]
LABEL2ID = {lab: i for i, lab in enumerate(LABELS)}
ID2LABEL = {i: lab for i, lab in enumerate(LABELS)}


# --------------------------- args ---------------------------

@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained encoder (e.g. pretrain/results)."})


@dataclass
class DataArguments:
    data_dir: str = field(default="data/cws", metadata={"help": "Dir with train.jsonl / val.jsonl / test.jsonl"})
    max_seq_length: int = field(default=512)


# --------------------------- data ---------------------------

def load_jsonl(path: Path):
    examples = {"chars": [], "tags": []}
    with open(path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            examples["chars"].append(ex["chars"])
            examples["tags"].append(ex["tags"])
    return Dataset.from_dict(examples)


def make_tokenize_fn(tokenizer, max_len: int):
    def fn(batch):
        # is_split_into_words=True so each Chinese char becomes one wordpiece slot.
        enc = tokenizer(
            batch["chars"],
            is_split_into_words=True,
            truncation=True,
            max_length=max_len,
            return_tensors=None,
        )
        labels_batch = []
        for i, tags in enumerate(batch["tags"]):
            word_ids = enc.word_ids(batch_index=i)
            ids = []
            prev = None
            for wid in word_ids:
                if wid is None:                     # [CLS]/[SEP]/[PAD]
                    ids.append(-100)
                elif wid != prev:                   # first sub-token of a char
                    ids.append(LABEL2ID[tags[wid]])
                    prev = wid
                else:                               # subword continuation (rare with char tokenizer)
                    ids.append(-100)
            labels_batch.append(ids)
        enc["labels"] = labels_batch
        return enc
    return fn


# --------------------------- word-level metric ---------------------------

def tags_to_word_spans(tags: list[str]) -> set[tuple[int, int]]:
    """Decode a BMES tag sequence into a set of (start, end_inclusive) word spans."""
    spans = set()
    start = None
    for i, t in enumerate(tags):
        if t == "S":
            if start is not None:
                spans.add((start, i - 1))   # close any dangling word
            spans.add((i, i))
            start = None
        elif t == "B":
            if start is not None:
                spans.add((start, i - 1))
            start = i
        elif t == "M":
            if start is None:
                start = i                    # tolerate stray M
        elif t == "E":
            if start is None:
                start = i
            spans.add((start, i))
            start = None
    if start is not None:
        spans.add((start, len(tags) - 1))
    return spans


def compute_metrics_factory():
    def compute_metrics(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=-1)
        labels = eval_pred.label_ids

        tp = fp = fn = 0
        for p_seq, l_seq in zip(preds, labels):
            keep = l_seq != -100
            pred_tags = [ID2LABEL[int(t)] for t in p_seq[keep]]
            gold_tags = [ID2LABEL[int(t)] for t in l_seq[keep]]
            pred_spans = tags_to_word_spans(pred_tags)
            gold_spans = tags_to_word_spans(gold_tags)
            tp += len(pred_spans & gold_spans)
            fp += len(pred_spans - gold_spans)
            fn += len(gold_spans - pred_spans)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}
    return compute_metrics


# --------------------------- main ---------------------------

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO if training_args.local_rank in (-1, 0) else logging.WARN,
    )
    logger.info(f"Training args: {training_args}")
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    data_dir = Path(data_args.data_dir)
    raw = {
        "train": load_jsonl(data_dir / "train.jsonl"),
        "validation": load_jsonl(data_dir / "val.jsonl"),
        "test": load_jsonl(data_dir / "test.jsonl"),
    }
    tok_fn = make_tokenize_fn(tokenizer, data_args.max_seq_length)
    tok = {k: v.map(tok_fn, batched=True, remove_columns=["chars", "tags"]) for k, v in raw.items()}

    collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)

    last_ckpt = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_ckpt = get_last_checkpoint(training_args.output_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok["train"] if training_args.do_train else None,
        eval_dataset=tok["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics_factory(),
    )

    if training_args.do_train:
        resume = training_args.resume_from_checkpoint or last_ckpt
        result = trainer.train(resume_from_checkpoint=resume)
        trainer.save_model()
        trainer.log_metrics("train", result.metrics)
        trainer.save_metrics("train", result.metrics)
        trainer.save_state()

    if training_args.do_eval:
        metrics = trainer.evaluate(eval_dataset=tok["validation"], metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        metrics = trainer.evaluate(eval_dataset=tok["test"], metric_key_prefix="test")
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    main()
