"""KL distillation head + helpers.

The encoder must retain enough information to reproduce the SikuRoBERTa
seg/pos distributions. We mount a tiny linear head on the encoder's last
hidden state, predict (num_seg + num_pos) logits, and compute KL divergence
against the teacher distributions.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDistillHead(nn.Module):
    """Linear projection from ``d_model`` to seg + pos logits."""

    def __init__(self, d_model: int, num_seg: int, num_pos: int, init_std: float = 0.02):
        super().__init__()
        self.num_seg = num_seg
        self.num_pos = num_pos
        self.proj = nn.Linear(d_model, num_seg + num_pos)
        nn.init.normal_(self.proj.weight, std=init_std)
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.proj(hidden)  # (B, T, num_seg + num_pos)
        seg_logits = logits[..., : self.num_seg]
        pos_logits = logits[..., self.num_seg :]
        return seg_logits, pos_logits


def kl_distill_loss(
    student_seg_logits: torch.Tensor,
    student_pos_logits: torch.Tensor,
    teacher_seg_probs: torch.Tensor,
    teacher_pos_probs: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """KL(teacher || student) summed over valid positions, mean-reduced.

    ``valid_mask`` is (B, T) bool: True at character positions to score on.
    Special tokens ([CLS], [SEP], [PAD]) should be False so we ignore them.
    """
    if valid_mask.sum() == 0:
        return student_seg_logits.new_zeros(())

    student_seg_logp = F.log_softmax(student_seg_logits, dim=-1)
    student_pos_logp = F.log_softmax(student_pos_logits, dim=-1)
    teacher_seg = teacher_seg_probs.clamp(min=eps)
    teacher_pos = teacher_pos_probs.clamp(min=eps)

    seg_kl = (teacher_seg * (teacher_seg.log() - student_seg_logp)).sum(dim=-1)  # (B, T)
    pos_kl = (teacher_pos * (teacher_pos.log() - student_pos_logp)).sum(dim=-1)

    mask_f = valid_mask.to(seg_kl.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    return ((seg_kl + pos_kl) * mask_f).sum() / denom


def build_valid_kl_mask(
    attention_mask: torch.Tensor,
    cls_id: int,
    sep_id: int,
    input_ids: torch.Tensor,
    teacher_seg: torch.Tensor,
) -> torch.Tensor:
    """KL is computed only on real character positions, not specials or pads,
    and not on positions where the teacher distribution is all-zero (used by
    monolingual denoising batches).
    """
    is_special = (input_ids == cls_id) | (input_ids == sep_id)
    has_teacher = teacher_seg.abs().sum(dim=-1) > 0
    return attention_mask.bool() & ~is_special & has_teacher
