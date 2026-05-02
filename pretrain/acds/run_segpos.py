#!/usr/bin/env python3
"""Joint WSG+POS fine-tuner using a hybrid BMES+POS tag set (88 labels).

Reuses the Stage-2 architecture (AutoModelForTokenClassification) — the
only change vs run_cws.py is the larger label set (~88 hybrid tags) and
the metric (separate WSG-F1 and POS-F1 like the paper's tables).

Usage examples (see run_*.sbatch):
  # Stage-1 train M_0 (pretrain/results) on D_p
  python run_segpos.py --model_name_or_path pretrain/results \\
      --train_jsonl data/acds/jsonl/d_p.jsonl \\
      --val_jsonl   data/acds/jsonl/d_a_val.jsonl \\
      --labels_json data/acds/jsonl/labels.json \\
      --output_dir output/acds-stage1 ...

  # Stage-2 continue M_1 on D_a
  python run_segpos.py --model_name_or_path output/acds-stage1 \\
      --train_jsonl data/acds/jsonl/d_a_train.jsonl \\
      --val_jsonl   data/acds/jsonl/d_a_val.jsonl ...
"""
from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
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


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Encoder path or HF id."})


@dataclass
class DataArguments:
    train_jsonl: str = field(metadata={"help": "Training JSONL (chars/tags)."})
    val_jsonl: str = field(metadata={"help": "Validation JSONL."})
    labels_json: str = field(metadata={"help": "JSON file mapping hybrid label -> id."})
    test_a_jsonl: str = field(default="", metadata={"help": "Optional Test-A JSONL"})
    test_b_jsonl: str = field(default="", metadata={"help": "Optional Test-B JSONL"})
    max_seq_length: int = field(default=512)


# ---------- data ----------

def load_jsonl(path: Path):
    chars, tags = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            chars.append(ex["chars"])
            tags.append(ex["tags"])
    return Dataset.from_dict({"chars": chars, "tags": tags})


def make_tokenize_fn(tokenizer, max_len, label2id):
    unk_id = -100
    def fn(batch):
        enc = tokenizer(
            batch["chars"],
            is_split_into_words=True,
            truncation=True,
            max_length=max_len,
            return_tensors=None,
        )
        labels_batch = []
        for i, tags in enumerate(batch["tags"]):
            wids = enc.word_ids(batch_index=i)
            ids = []
            prev = None
            for wid in wids:
                if wid is None:
                    ids.append(unk_id)
                elif wid != prev:
                    tag = tags[wid]
                    ids.append(label2id.get(tag, unk_id))
                    prev = wid
                else:
                    ids.append(unk_id)
            labels_batch.append(ids)
        enc["labels"] = labels_batch
        return enc
    return fn


# ---------- metrics: separate WSG-F1 and POS-F1 ----------

def label_to_seg_pos(label: str):
    """'B-n' -> ('B', 'n'). Tolerant of stray '-' inside POS by splitting only the first."""
    if "-" in label:
        seg, pos = label.split("-", 1)
        return seg, pos
    return label, ""  # shouldn't happen given our generator


def tags_to_word_spans_with_pos(tags):
    """Decode hybrid BMES+POS tag sequence to {(start, end_inclusive, pos)}."""
    spans = set()
    start = None
    cur_pos = None
    for i, t in enumerate(tags):
        seg, pos = label_to_seg_pos(t)
        if seg == "S":
            if start is not None:
                spans.add((start, i - 1, cur_pos))
            spans.add((i, i, pos))
            start = None; cur_pos = None
        elif seg == "B":
            if start is not None:
                spans.add((start, i - 1, cur_pos))
            start = i; cur_pos = pos
        elif seg == "M":
            if start is None:
                start = i; cur_pos = pos
        elif seg == "E":
            if start is None:
                start = i; cur_pos = pos
            spans.add((start, i, cur_pos if cur_pos is not None else pos))
            start = None; cur_pos = None
    if start is not None:
        spans.add((start, len(tags) - 1, cur_pos))
    return spans


def make_compute_metrics(id2label):
    def compute(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=-1)
        labels = eval_pred.label_ids

        # word-level: span (start,end) must match exactly
        # POS-level: span AND pos must match exactly
        wsg_tp = wsg_fp = wsg_fn = 0
        pos_tp = pos_fp = pos_fn = 0
        for p_seq, l_seq in zip(preds, labels):
            keep = l_seq != -100
            pred_tags = [id2label.get(int(t), "S-null") for t in p_seq[keep]]
            gold_tags = [id2label.get(int(t), "S-null") for t in l_seq[keep]]
            pred_spans = tags_to_word_spans_with_pos(pred_tags)
            gold_spans = tags_to_word_spans_with_pos(gold_tags)
            # WSG: ignore pos in match
            pred_w = {(a, b) for a, b, _ in pred_spans}
            gold_w = {(a, b) for a, b, _ in gold_spans}
            wsg_tp += len(pred_w & gold_w)
            wsg_fp += len(pred_w - gold_w)
            wsg_fn += len(gold_w - pred_w)
            # POS: full triple match
            pos_tp += len(pred_spans & gold_spans)
            pos_fp += len(pred_spans - gold_spans)
            pos_fn += len(gold_spans - pred_spans)

        def prf(tp, fp, fn):
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f = 2 * p * r / (p + r) if (p + r) else 0.0
            return p, r, f

        wp, wr, wf = prf(wsg_tp, wsg_fp, wsg_fn)
        pp, pr, pf = prf(pos_tp, pos_fp, pos_fn)
        return {
            "wsg_precision": wp, "wsg_recall": wr, "wsg_f1": wf,
            "pos_precision": pp, "pos_recall": pr, "pos_f1": pf,
            # for load_best_model_at_end:
            "f1": wf,  # default best-metric is WSG F1
        }
    return compute


# ---------- main ----------

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO if training_args.local_rank in (-1, 0) else logging.WARN,
    )
    set_seed(training_args.seed)

    label2id = json.loads(Path(data_args.labels_json).read_text(encoding="utf-8"))
    id2label = {int(v): k for k, v in label2id.items()}
    logger.info(f"hybrid labels: {len(label2id)}")

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,  # in case --model_name_or_path had a different head
    )

    raw_train = load_jsonl(Path(data_args.train_jsonl))
    raw_val   = load_jsonl(Path(data_args.val_jsonl))
    tok_fn = make_tokenize_fn(tokenizer, data_args.max_seq_length, label2id)
    train_ds = raw_train.map(tok_fn, batched=True, remove_columns=["chars", "tags"])
    val_ds   = raw_val.map(tok_fn, batched=True, remove_columns=["chars", "tags"])

    collator = DataCollatorForTokenClassification(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )
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
        train_dataset=train_ds if training_args.do_train else None,
        eval_dataset=val_ds if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=make_compute_metrics(id2label),
    )

    if training_args.do_train:
        resume = training_args.resume_from_checkpoint or last_ckpt
        result = trainer.train(resume_from_checkpoint=resume)
        trainer.save_model()
        trainer.log_metrics("train", result.metrics)
        trainer.save_metrics("train", result.metrics)
        trainer.save_state()

    if training_args.do_eval:
        m = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix="eval")
        trainer.log_metrics("eval", m)
        trainer.save_metrics("eval", m)

    if training_args.do_predict:
        for name, path_str in (("test_a", data_args.test_a_jsonl),
                               ("test_b", data_args.test_b_jsonl)):
            if not path_str:
                continue
            ds = load_jsonl(Path(path_str)).map(tok_fn, batched=True, remove_columns=["chars", "tags"])
            m = trainer.evaluate(eval_dataset=ds, metric_key_prefix=name)
            trainer.log_metrics(name, m)
            trainer.save_metrics(name, m)


if __name__ == "__main__":
    main()
