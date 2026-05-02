"""Three-stage fine-tuning entry point for Erya + soft tag adapters.

Usage::

    python -m Erya.finetune.train --stage 1 --config Erya/configs/stage1_adapters.yaml
    python -m Erya.finetune.train --stage 2 --config Erya/configs/stage2_encoder.yaml \
        --resume_adapter checkpoints/stage1/best/adapter.pt
    python -m Erya.finetune.train --stage 3 --config Erya/configs/stage3_full.yaml \
        --resume checkpoints/stage2/best/full.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

try:
    from transformers import AutoModelForSeq2SeqLM, BertTokenizer
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "transformers is required. CPTForConditionalGeneration is loaded via "
        "AutoModelForSeq2SeqLM with trust_remote_code=True; install a compatible "
        "transformers version (e.g. fnlp's CPT fork or upstream when available)."
    ) from e

from . import losses as losses_mod
from . import utils
from .aux_dataset import (
    LengthBalancedSampler,
    MixedSampler,
    MonoDenoiseDataset,
    PairedCollator,
    TaggedConcat,
    build_parallel_concat,
)
from .tag_adapter import TagAdapterConfig, TaggedErya


# ---------- config --------------------------------------------------------------

@dataclass
class TrainConfig:
    # data
    finetune_root: str = "Erya/dataset/finetune_aux"
    monolingual_path: Optional[str] = None
    tokenizer_path: str = "Erya"
    model_path: str = "Erya"

    # training
    stage: int = 1
    batch_size: int = 16
    grad_accum: int = 1
    num_steps: int = 20000
    warmup_steps: int = 500
    valid_every: int = 1000
    save_every: int = 1000
    log_every: int = 50

    # learning rates per stage
    lr_adapter: float = 5e-4
    lr_encoder: float = 5e-5
    lr_decoder: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # losses
    kl_weight: float = 0.1
    denoise_weight: float = 1.0
    denoise_ratio: float = 0.0
    label_smoothing: float = 0.0

    # dataset shape
    max_src_chars: int = 1022
    max_tgt_tokens: int = 1024

    # misc
    output_dir: str = "Erya/checkpoints/stage1"
    seed: int = 42
    bf16: bool = True
    grad_checkpointing: bool = False
    resume: Optional[str] = None
    resume_adapter: Optional[str] = None
    dry_run: bool = False
    num_workers: int = 2
    # "sdpa" (default), "eager" (slowest, most compatible), "flash_attention_2"
    attn_implementation: Optional[str] = None
    # If set, validation scores at most this many batches per pass instead of
    # the full valid set. The same fixed random subset is reused at every
    # validation pass so CE numbers are comparable across steps.
    valid_max_batches: Optional[int] = None


def load_yaml_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------- core --------------------------------------------------------------

def cosine_lr(step: int, warmup: int, total: int, base: float, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return base * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return base * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def apply_lr(optimizer: torch.optim.Optimizer, step: int, cfg: TrainConfig) -> None:
    """Cosine schedule with shared warmup, per-group base LR."""
    for group in optimizer.param_groups:
        base = group.get("base_lr", group["lr"])
        group["base_lr"] = base
        group["lr"] = cosine_lr(step, cfg.warmup_steps, cfg.num_steps, base)


def identity_sanity_check(tagged: TaggedErya, batch, device) -> None:
    """With alpha=0 and E_*=0, the wrapped model must equal the vanilla forward."""
    tagged.eval()
    saved = {
        "alpha_seg": tagged.alpha_seg.detach().clone(),
        "alpha_pos": tagged.alpha_pos.detach().clone(),
        "E_seg": tagged.E_seg.weight.detach().clone(),
        "E_pos": tagged.E_pos.weight.detach().clone(),
    }
    with torch.no_grad():
        tagged.alpha_seg.zero_()
        tagged.alpha_pos.zero_()
        tagged.E_seg.weight.zero_()
        tagged.E_pos.weight.zero_()
        wrapped = tagged(
            input_ids=batch.input_ids.to(device),
            attention_mask=batch.attention_mask.to(device),
            seg_probs=batch.seg_probs.to(device),
            pos_probs=batch.pos_probs.to(device),
            labels=batch.labels.to(device),
        )
        vanilla = tagged.model(
            input_ids=batch.input_ids.to(device),
            attention_mask=batch.attention_mask.to(device),
            labels=batch.labels.to(device),
        )
        diff = (wrapped.loss - vanilla.loss).abs().item()
        if diff > 1e-3:
            raise RuntimeError(
                f"identity sanity check failed: |loss_tagged - loss_vanilla| = {diff:.6f}"
            )
        with torch.no_grad():
            tagged.alpha_seg.copy_(saved["alpha_seg"])
            tagged.alpha_pos.copy_(saved["alpha_pos"])
            tagged.E_seg.weight.copy_(saved["E_seg"])
            tagged.E_pos.weight.copy_(saved["E_pos"])
    tagged.train()


def build_data(cfg: TrainConfig, tokenizer):
    parallel_train, train_names = build_parallel_concat(
        Path(cfg.finetune_root),
        split="train",
        tokenizer=tokenizer,
        max_src_chars=cfg.max_src_chars,
        max_tgt_tokens=cfg.max_tgt_tokens,
    )
    parallel_valid, valid_names = build_parallel_concat(
        Path(cfg.finetune_root),
        split="valid",
        tokenizer=tokenizer,
        max_src_chars=cfg.max_src_chars,
        max_tgt_tokens=cfg.max_tgt_tokens,
    )
    # shapes for tag adapter
    sample_ds = parallel_train.datasets[0]
    num_seg = sample_ds.num_seg
    num_pos = sample_ds.num_pos
    seg_labels = sample_ds.seg_label_names
    pos_labels = sample_ds.pos_label_names

    mono = None
    if cfg.denoise_ratio > 0 and cfg.monolingual_path:
        mono = MonoDenoiseDataset(
            cfg.monolingual_path,
            tokenizer=tokenizer,
            num_seg=num_seg,
            num_pos=num_pos,
            max_src_chars=cfg.max_src_chars,
            max_tgt_tokens=cfg.max_tgt_tokens,
            seed=cfg.seed,
        )

    return parallel_train, parallel_valid, mono, num_seg, num_pos, seg_labels, pos_labels, train_names, valid_names


def make_loaders(cfg: TrainConfig, parallel_train, parallel_valid, mono, tokenizer):
    pad_id = tokenizer.pad_token_id
    collator = PairedCollator(pad_id=pad_id)

    parallel_sampler = LengthBalancedSampler(
        parallel_train,
        num_samples=cfg.num_steps * cfg.batch_size * cfg.grad_accum,
        seed=cfg.seed,
    )

    train_dataset = TaggedConcat(parallel_train, mono)

    if mono is not None and cfg.denoise_ratio > 0:
        from torch.utils.data import RandomSampler
        mono_sampler = RandomSampler(
            mono,
            replacement=True,
            num_samples=cfg.num_steps * cfg.batch_size * cfg.grad_accum,
        )
        sampler = MixedSampler(
            parallel_sampler=parallel_sampler,
            mono_sampler=mono_sampler,
            denoise_ratio=cfg.denoise_ratio,
            num_samples=cfg.num_steps * cfg.batch_size * cfg.grad_accum,
            seed=cfg.seed,
        )

        def _collate(items):
            return collator(items)

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=_BatchifyMixed(sampler, cfg.batch_size),
            collate_fn=_collate,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            parallel_train,
            sampler=parallel_sampler,
            batch_size=cfg.batch_size,
            collate_fn=collator,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    if cfg.valid_max_batches is not None and cfg.valid_max_batches > 0:
        from torch.utils.data import Subset
        rng = np.random.default_rng(cfg.seed + 7)
        n_full = len(parallel_valid)
        n_take = min(n_full, cfg.valid_max_batches * cfg.batch_size)
        indices = sorted(rng.choice(n_full, size=n_take, replace=False).tolist())
        valid_dataset = Subset(parallel_valid, indices)
    else:
        valid_dataset = parallel_valid

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=max(1, cfg.num_workers // 2),
        pin_memory=True,
    )
    return train_loader, valid_loader


class _BatchifyMixed:
    """Group a mixed (ds_id, idx) iterator into batches that are homogeneous in
    ds_id within a batch (we just yield batches of mixed ids and rely on the
    dataset to dispatch). The dataset returns PairedExample either way, so the
    collator handles any mix uniformly."""

    def __init__(self, sampler, batch_size: int):
        self.sampler = sampler
        self.batch_size = batch_size

    def __iter__(self):
        batch = []
        for key in self.sampler:
            batch.append(key)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def __len__(self):
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size


@torch.no_grad()
def eval_loss(tagged: TaggedErya, loader, device, log=None, label: str = "valid") -> float:
    tagged.eval()
    total = 0.0
    n = 0
    n_batches = len(loader) if hasattr(loader, "__len__") else None
    last_print = time.time()
    for batch in loader:
        out = tagged(
            input_ids=batch.input_ids.to(device),
            attention_mask=batch.attention_mask.to(device),
            seg_probs=batch.seg_probs.to(device),
            pos_probs=batch.pos_probs.to(device),
            labels=batch.labels.to(device),
        )
        total += float(out.loss.item())
        n += 1
        if log is not None and time.time() - last_print > 15:
            running = total / n
            tag = f"{n}/{n_batches}" if n_batches else f"{n}"
            log.info("[%s] %s batches | running CE = %.4f", label, tag, running)
            last_print = time.time()
    tagged.train()
    return total / max(1, n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume_adapter", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg_dict = load_yaml_config(Path(args.config))
    if args.stage is not None:
        cfg_dict["stage"] = args.stage
    if args.resume is not None:
        cfg_dict["resume"] = args.resume
    if args.resume_adapter is not None:
        cfg_dict["resume_adapter"] = args.resume_adapter
    if args.output_dir is not None:
        cfg_dict["output_dir"] = args.output_dir
    cfg_dict["dry_run"] = bool(args.dry_run or cfg_dict.get("dry_run", False))
    cfg = TrainConfig(**cfg_dict)

    log = utils.get_logger("erya_finetune.train")
    utils.set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("config: %s", json.dumps(asdict(cfg), indent=2, default=str))
    log.info("device: %s", device)

    # cuDNN's SDPA backend has had recurring "no valid execution plans" failures
    # on H100/A100 with mismatched cuDNN wheels. Force SDPA to use the flash /
    # memory-efficient / math backends instead — all stable on H100.
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.enable_cudnn_sdp(False)
            log.info("disabled cuDNN SDPA backend (using flash/mem-efficient/math)")
        except AttributeError:
            pass

    tokenizer = BertTokenizer.from_pretrained(cfg.tokenizer_path)
    log.info("loaded tokenizer: vocab=%d cls=%d sep=%d pad=%d mask=%d",
             tokenizer.vocab_size, tokenizer.cls_token_id, tokenizer.sep_token_id,
             tokenizer.pad_token_id, tokenizer.mask_token_id)

    log.info("loading base model from %s ...", cfg.model_path)
    load_kwargs = {"trust_remote_code": True}
    if cfg.attn_implementation:
        load_kwargs["attn_implementation"] = cfg.attn_implementation
        log.info("forcing attn_implementation=%s", cfg.attn_implementation)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_path, **load_kwargs)
    if cfg.grad_checkpointing:
        try:
            base_model.gradient_checkpointing_enable()
        except Exception as e:
            log.warning("could not enable gradient checkpointing: %s", e)
    d_model = base_model.config.d_model

    (parallel_train, parallel_valid, mono,
     num_seg, num_pos, seg_labels, pos_labels,
     train_names, valid_names) = build_data(cfg, tokenizer)
    log.info("data: train_corpora=%s valid_corpora=%s", train_names, valid_names)
    log.info("seg labels (%d): %s", num_seg, seg_labels)
    log.info("pos labels (%d): %s", num_pos, pos_labels)

    tagged = TaggedErya(
        base_model,
        TagAdapterConfig(num_seg_labels=num_seg, num_pos_labels=num_pos, d_model=d_model),
    ).to(device)

    kl_head: Optional[nn.Module] = None
    if cfg.kl_weight > 0:
        kl_head = losses_mod.KLDistillHead(d_model=d_model, num_seg=num_seg, num_pos=num_pos).to(device)

    if cfg.resume:
        log.info("resuming from full checkpoint: %s", cfg.resume)
        meta = utils.load_full(Path(cfg.resume), tagged, kl_head=kl_head)
        utils.assert_label_tables_match(meta, seg_labels, pos_labels)
    elif cfg.resume_adapter:
        log.info("resuming adapter from: %s", cfg.resume_adapter)
        meta = utils.load_adapter(Path(cfg.resume_adapter), tagged, kl_head=kl_head)
        utils.assert_label_tables_match(meta, seg_labels, pos_labels)

    param_groups = utils.build_param_groups(
        tagged, stage=cfg.stage,
        lr_adapter=cfg.lr_adapter, lr_encoder=cfg.lr_encoder, lr_decoder=cfg.lr_decoder,
        weight_decay=cfg.weight_decay, kl_head=kl_head,
    )
    n_train = utils.count_trainable(tagged) + (utils.count_trainable(kl_head) if kl_head is not None else 0)
    log.info("trainable params: %d (stage %d)", n_train, cfg.stage)
    optimizer = torch.optim.AdamW(param_groups)
    for g in optimizer.param_groups:
        g["base_lr"] = g["lr"]

    train_loader, valid_loader = make_loaders(cfg, parallel_train, parallel_valid, mono, tokenizer)
    if cfg.valid_max_batches is not None and cfg.valid_max_batches > 0:
        log.info(
            "valid_max_batches=%d -> capped valid pass at ~%d examples (full valid set has %d)",
            cfg.valid_max_batches, cfg.valid_max_batches * cfg.batch_size, len(parallel_valid),
        )

    # ---- identity sanity check (only meaningful before any training)
    sanity_iter = iter(valid_loader)
    sanity_batch = next(sanity_iter)
    if cfg.resume is None and cfg.resume_adapter is None:
        log.info("running identity sanity check ...")
        identity_sanity_check(tagged, sanity_batch, device)
        log.info("identity sanity check passed")

    if cfg.dry_run:
        log.info("dry-run complete; exiting before training")
        return

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_dir = out_dir / "best"
    last_dir = out_dir / "last"

    autocast_dtype = torch.bfloat16 if (cfg.bf16 and torch.cuda.is_available()) else torch.float32

    log.info(
        "starting stage %d: %d optimizer steps (batch=%d, grad_accum=%d) -> ~%d examples seen",
        cfg.stage, cfg.num_steps, cfg.batch_size, cfg.grad_accum,
        cfg.num_steps * cfg.batch_size * cfg.grad_accum,
    )

    # ---- training loop
    tagged.train()
    if kl_head is not None:
        kl_head.train()
    optimizer.zero_grad(set_to_none=True)
    step = 0
    micro = 0
    best_valid = float("inf")
    start = time.time()
    last_log_time = start
    last_log_step = 0
    ema_ce = None
    ema_kl = None
    HEARTBEAT_SECONDS = 30
    train_iter = iter(train_loader)

    while step < cfg.num_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=autocast_dtype, enabled=(autocast_dtype != torch.float32)):
            inputs_embeds = tagged.compute_inputs_embeds(
                batch.input_ids.to(device),
                batch.seg_probs.to(device),
                batch.pos_probs.to(device),
            )
            out = tagged.model(
                inputs_embeds=inputs_embeds,
                attention_mask=batch.attention_mask.to(device),
                labels=batch.labels.to(device),
                output_hidden_states=(kl_head is not None),
                return_dict=True,
            )
            ce_loss = out.loss
            total_loss = ce_loss

            kl_loss_val = torch.zeros((), device=device)
            if kl_head is not None and cfg.kl_weight > 0:
                hidden = out.encoder_last_hidden_state
                if hidden is None:
                    hidden = out.encoder_hidden_states[-1]
                seg_logits, pos_logits = kl_head(hidden)
                valid_mask = losses_mod.build_valid_kl_mask(
                    attention_mask=batch.attention_mask.to(device),
                    cls_id=tokenizer.cls_token_id,
                    sep_id=tokenizer.sep_token_id,
                    input_ids=batch.input_ids.to(device),
                    teacher_seg=batch.seg_probs.to(device),
                )
                kl_loss_val = losses_mod.kl_distill_loss(
                    student_seg_logits=seg_logits,
                    student_pos_logits=pos_logits,
                    teacher_seg_probs=batch.seg_probs.to(device),
                    teacher_pos_probs=batch.pos_probs.to(device),
                    valid_mask=valid_mask,
                )
                total_loss = total_loss + cfg.kl_weight * kl_loss_val

            total_loss = total_loss / cfg.grad_accum

        total_loss.backward()
        micro += 1
        if micro % cfg.grad_accum != 0:
            continue

        torch.nn.utils.clip_grad_norm_([p for p in tagged.parameters() if p.requires_grad], cfg.max_grad_norm)
        if kl_head is not None:
            torch.nn.utils.clip_grad_norm_(kl_head.parameters(), cfg.max_grad_norm)
        apply_lr(optimizer, step, cfg)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        ce_now = float(ce_loss.detach().item())
        kl_now = float(kl_loss_val.detach().item())
        ema_ce = ce_now if ema_ce is None else 0.98 * ema_ce + 0.02 * ce_now
        ema_kl = kl_now if ema_kl is None else 0.98 * ema_kl + 0.02 * kl_now

        now = time.time()
        time_since_log = now - last_log_time
        due_by_step = (step % cfg.log_every == 0)
        due_by_time = time_since_log >= HEARTBEAT_SECONDS
        if due_by_step or due_by_time:
            elapsed = now - start
            steps_since = max(1, step - last_log_step)
            sec_per_step = time_since_log / steps_since
            remaining = cfg.num_steps - step
            eta = utils.format_eta(remaining * sec_per_step)
            elapsed_str = utils.format_eta(elapsed)
            pct = 100.0 * step / cfg.num_steps
            current_lr = optimizer.param_groups[0]["lr"]
            log.info(
                "step %d/%d (%5.1f%%) | ce=%.4f (ema %.4f) kl=%.4f (ema %.4f) | "
                "lr=%.2e alpha_s=%+.3f alpha_p=%+.3f | %.2fs/step | %s elapsed | ETA %s",
                step, cfg.num_steps, pct,
                ce_now, ema_ce, kl_now, ema_kl,
                current_lr,
                float(tagged.alpha_seg.detach().item()),
                float(tagged.alpha_pos.detach().item()),
                sec_per_step, elapsed_str, eta,
            )
            last_log_time = now
            last_log_step = step

        if step % cfg.valid_every == 0:
            log.info("running validation at step %d ...", step)
            v = eval_loss(tagged, valid_loader, device, log=log, label="valid")
            log.info("step %d valid CE = %.4f (best so far %.4f)",
                     step, v, min(best_valid, v))
            meta = {
                "stage": cfg.stage,
                "step": step,
                "valid_loss": v,
                "seg_label_names": seg_labels,
                "pos_label_names": pos_labels,
            }
            if v < best_valid:
                best_valid = v
                if cfg.stage == 1:
                    utils.save_adapter(best_dir / "adapter.pt", tagged, kl_head, meta)
                else:
                    utils.save_full(best_dir / "full.pt", tagged, kl_head, meta)
                utils.write_meta_json(best_dir / "meta.json", meta)
                log.info("new best at step %d: %.4f -> saved to %s", step, v, best_dir)

        if step % cfg.save_every == 0:
            meta = {
                "stage": cfg.stage, "step": step,
                "seg_label_names": seg_labels, "pos_label_names": pos_labels,
            }
            if cfg.stage == 1:
                utils.save_adapter(last_dir / "adapter.pt", tagged, kl_head, meta)
            else:
                utils.save_full(last_dir / "full.pt", tagged, kl_head, meta)
            utils.write_meta_json(last_dir / "meta.json", meta)

    # final save
    meta = {
        "stage": cfg.stage, "step": step,
        "seg_label_names": seg_labels, "pos_label_names": pos_labels,
    }
    if cfg.stage == 1:
        utils.save_adapter(last_dir / "adapter.pt", tagged, kl_head, meta)
    else:
        utils.save_full(last_dir / "full.pt", tagged, kl_head, meta)
    utils.write_meta_json(last_dir / "meta.json", meta)
    log.info("training complete.")


if __name__ == "__main__":
    main()
