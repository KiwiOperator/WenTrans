#!/usr/bin/env python3
"""Continue MLM pre-training of a HuggingFace masked-LM model.

Minimal HF Trainer wrapper for the WenTrans Stage-1 pretrain. Mirrors the
methodology in Wang et al. 2021 (SikuRoBERTa): MLM only (NSP removed),
15% random masking, perplexity as the eval metric.

Reads one sentence per line from --train_file / --validation_file.
"""
from __future__ import annotations

import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import datasets
import transformers
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "HF model id or local path (e.g. hfl/chinese-roberta-wwm-ext)."}
    )
    cache_dir: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    train_file: str = field(metadata={"help": "Path to one-sentence-per-line train file."})
    validation_file: str = field(metadata={"help": "Path to one-sentence-per-line val file."})
    max_seq_length: int = field(default=512)
    mlm_probability: float = field(default=0.15)
    line_by_line: bool = field(
        default=True,
        metadata={"help": "Treat each non-empty input line as one example (default)."},
    )
    preprocessing_num_workers: int = field(default=4)
    overwrite_cache: bool = field(default=False)


@dataclass
class RuntimeArguments:
    max_train_seconds: float = field(
        default=0.0,
        metadata={
            "help": (
                "Soft wall-clock budget. After this many seconds of training, the "
                "Trainer is asked to stop at the next step boundary, which lets the "
                "final save and eval complete cleanly before SLURM kills the job. "
                "0 disables the budget."
            )
        },
    )


class TimeBudgetCallback(TrainerCallback):
    """Stop training gracefully once max_seconds of wall-clock elapsed."""

    def __init__(self, max_seconds: float):
        self.max_seconds = max_seconds
        self._start = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._start = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        if self._start is None:
            return control
        if time.monotonic() - self._start >= self.max_seconds:
            control.should_training_stop = True
            control.should_save = True
        return control


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, RuntimeArguments, TrainingArguments))
    model_args, data_args, runtime_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO if training_args.local_rank in (-1, 0) else logging.WARN,
    )
    transformers.utils.logging.set_verbosity_info()
    logger.info(f"Training/eval parameters {training_args}")

    set_seed(training_args.seed)

    # ---------- model + tokenizer ----------
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_fast=True
    )
    model = AutoModelForMaskedLM.from_pretrained(
        model_args.model_name_or_path, config=config, cache_dir=model_args.cache_dir
    )
    model.resize_token_embeddings(len(tokenizer))

    # ---------- data ----------
    raw = datasets.load_dataset(
        "text",
        data_files={"train": data_args.train_file, "validation": data_args.validation_file},
        cache_dir=model_args.cache_dir,
    )

    max_len = min(data_args.max_seq_length, tokenizer.model_max_length)

    if data_args.line_by_line:
        def tokenize_fn(examples):
            texts = [t for t in examples["text"] if t and not t.isspace()]
            return tokenizer(
                texts,
                padding=False,
                truncation=True,
                max_length=max_len,
                return_special_tokens_mask=True,
            )

        tokenized = raw.map(
            tokenize_fn,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=["text"],
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Tokenizing",
        )
    else:
        # Concatenate-and-chunk path (kept for completeness; not the default).
        def tokenize_fn(examples):
            return tokenizer(examples["text"], return_special_tokens_mask=True)

        tokenized = raw.map(
            tokenize_fn,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=["text"],
            load_from_cache_file=not data_args.overwrite_cache,
        )

        def group(examples):
            concat = {k: sum(examples[k], []) for k in examples}
            total = (len(concat["input_ids"]) // max_len) * max_len
            return {k: [v[i : i + max_len] for i in range(0, total, max_len)] for k, v in concat.items()}

        tokenized = tokenized.map(
            group,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Chunking",
        )

    train_ds = tokenized["train"]
    eval_ds = tokenized["validation"]

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=data_args.mlm_probability,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # ---------- resume detection ----------
    last_ckpt = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_ckpt = get_last_checkpoint(training_args.output_dir)
        if last_ckpt is None and os.listdir(training_args.output_dir):
            logger.info("Output dir is non-empty but contains no checkpoint; continuing.")
        elif last_ckpt is not None:
            logger.info(f"Found checkpoint: {last_ckpt}. Resuming.")

    callbacks = []
    if runtime_args.max_train_seconds > 0:
        logger.info(
            f"Soft wall-clock budget: {runtime_args.max_train_seconds:.0f}s "
            f"(~{runtime_args.max_train_seconds / 3600:.2f}h). "
            "Trainer will stop gracefully at next step boundary once exceeded."
        )
        callbacks.append(TimeBudgetCallback(runtime_args.max_train_seconds))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds if training_args.do_train else None,
        eval_dataset=eval_ds if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=callbacks,
    )

    # ---------- train ----------
    if training_args.do_train:
        resume = training_args.resume_from_checkpoint or last_ckpt
        train_result = trainer.train(resume_from_checkpoint=resume)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # ---------- eval ----------
    if training_args.do_eval:
        metrics = trainer.evaluate()
        try:
            metrics["perplexity"] = math.exp(metrics["eval_loss"])
        except OverflowError:
            metrics["perplexity"] = float("inf")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
