"""Param-group builders, checkpoint I/O, label-table sanity checks, logging."""
from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn


def get_logger(name: str = "erya_finetune", level: int = logging.INFO) -> logging.Logger:
    """Stdout logger so SLURM sends logs to %j.out (not %j.err).

    A line-buffered StreamHandler also lets ``tail -f`` see records in
    real time on PACE.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    try:  # ensure stdout is line-buffered so SLURM sees logs as they happen
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    return logger


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def unfreeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = True


def build_param_groups(
    tagged_model,
    stage: int,
    lr_adapter: float,
    lr_encoder: float,
    lr_decoder: float,
    weight_decay: float,
    kl_head: Optional[nn.Module] = None,
):
    """Configure ``requires_grad`` and produce optimizer param groups for a stage.

    Stage 1: adapters only (and KL head).
    Stage 2: adapters + KL head + encoder. Decoder frozen.
    Stage 3: everything trainable, low LR everywhere.
    """
    base = tagged_model.model
    encoder = tagged_model.encoder
    decoder = tagged_model.decoder

    freeze(base)
    for p in (tagged_model.E_seg.parameters(), tagged_model.E_pos.parameters()):
        for q in p:
            q.requires_grad = True
    tagged_model.alpha_seg.requires_grad = True
    tagged_model.alpha_pos.requires_grad = True
    if kl_head is not None:
        unfreeze(kl_head)

    if stage >= 2:
        unfreeze(encoder)
    if stage >= 3:
        unfreeze(decoder)
        # LM head / shared embeddings
        for p in base.parameters():
            p.requires_grad = True

    adapter_params = [
        tagged_model.E_seg.weight,
        tagged_model.E_pos.weight,
        tagged_model.alpha_seg,
        tagged_model.alpha_pos,
    ]
    if kl_head is not None:
        adapter_params.extend([p for p in kl_head.parameters() if p.requires_grad])

    groups = [
        {"params": adapter_params, "lr": lr_adapter, "weight_decay": 0.0},
    ]
    if stage >= 2:
        enc_params = [p for p in encoder.parameters() if p.requires_grad]
        if enc_params:
            groups.append({"params": enc_params, "lr": lr_encoder, "weight_decay": weight_decay})
    if stage >= 3:
        # Everything else (decoder + LM head + shared embeds, excluding encoder
        # which is already in its own group).
        encoder_param_ids = {id(p) for p in encoder.parameters()}
        adapter_param_ids = {id(p) for p in adapter_params}
        rest = [
            p for p in base.parameters()
            if p.requires_grad
            and id(p) not in encoder_param_ids
            and id(p) not in adapter_param_ids
        ]
        if rest:
            groups.append({"params": rest, "lr": lr_decoder, "weight_decay": weight_decay})

    return groups


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_adapter(path: Path, tagged_model, kl_head: Optional[nn.Module], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "adapter": tagged_model.adapter_state_dict(),
        "kl_head": (kl_head.state_dict() if kl_head is not None else None),
        "meta": meta,
    }
    torch.save(payload, path)


def load_adapter(path: Path, tagged_model, kl_head: Optional[nn.Module] = None) -> dict:
    payload = torch.load(path, map_location="cpu")
    tagged_model.load_adapter_state_dict(payload["adapter"], strict=True)
    if kl_head is not None and payload.get("kl_head") is not None:
        kl_head.load_state_dict(payload["kl_head"])
    return payload.get("meta", {})


def save_full(path: Path, tagged_model, kl_head: Optional[nn.Module], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": tagged_model.model.state_dict(),
        "adapter": tagged_model.adapter_state_dict(),
        "kl_head": (kl_head.state_dict() if kl_head is not None else None),
        "meta": meta,
    }
    torch.save(payload, path)


def load_full(path: Path, tagged_model, kl_head: Optional[nn.Module] = None) -> dict:
    payload = torch.load(path, map_location="cpu")
    if "model" in payload:
        tagged_model.model.load_state_dict(payload["model"])
    tagged_model.load_adapter_state_dict(payload["adapter"], strict=True)
    if kl_head is not None and payload.get("kl_head") is not None:
        kl_head.load_state_dict(payload["kl_head"])
    return payload.get("meta", {})


def assert_label_tables_match(meta: dict, seg_labels: list, pos_labels: list) -> None:
    saved_seg = list(meta.get("seg_label_names", []))
    saved_pos = list(meta.get("pos_label_names", []))
    if saved_seg and saved_seg != list(seg_labels):
        raise ValueError(
            f"seg label table mismatch between checkpoint and aux data:\n"
            f"  ckpt: {saved_seg}\n  data: {list(seg_labels)}"
        )
    if saved_pos and saved_pos != list(pos_labels):
        raise ValueError(
            f"pos label table mismatch between checkpoint and aux data:\n"
            f"  ckpt: {saved_pos}\n  data: {list(pos_labels)}"
        )


def write_meta_json(path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
