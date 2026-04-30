"""TaggedErya: wrap CPTForConditionalGeneration with soft POS/Seg tag adapters.

Injection point: encoder ``inputs_embeds`` argument. We replicate the encoder's
internal token-embedding step ourselves, add the soft tag contributions, and
hand the result to the model so the rest of BART (``embed_positions``, encoder
layers, decoder, LM head) runs unchanged.

At init, ``alpha_seg = alpha_pos = 0`` so ``tanh(alpha) == 0`` and the wrapper
forward is bit-identical to vanilla Erya.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class TagAdapterConfig:
    num_seg_labels: int
    num_pos_labels: int
    d_model: int
    init_std: float = 0.02


class TaggedErya(nn.Module):
    """Wrap an encoder-decoder model with soft seg/pos tag embeddings."""

    def __init__(self, base_model: nn.Module, cfg: TagAdapterConfig):
        super().__init__()
        self.model = base_model
        self.cfg = cfg

        self.E_seg = nn.Embedding(cfg.num_seg_labels, cfg.d_model)
        self.E_pos = nn.Embedding(cfg.num_pos_labels, cfg.d_model)
        nn.init.normal_(self.E_seg.weight, mean=0.0, std=cfg.init_std)
        nn.init.normal_(self.E_pos.weight, mean=0.0, std=cfg.init_std)

        # Trainable scalar gates. tanh(0) = 0, so model is identity at init.
        self.alpha_seg = nn.Parameter(torch.zeros(()))
        self.alpha_pos = nn.Parameter(torch.zeros(()))

    @property
    def encoder(self):
        return self.model.get_encoder()

    @property
    def decoder(self):
        return self.model.get_decoder()

    @property
    def config(self):
        return self.model.config

    def get_input_embeddings(self):
        return self.encoder.embed_tokens

    def compute_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        seg_probs: torch.Tensor,
        pos_probs: torch.Tensor,
    ) -> torch.Tensor:
        """token_embed * embed_scale + tanh(alpha_seg) * (seg_probs @ E_seg)
        + tanh(alpha_pos) * (pos_probs @ E_pos)."""
        embed_tokens = self.encoder.embed_tokens
        embed_scale = getattr(self.encoder, "embed_scale", 1.0)

        tok = embed_tokens(input_ids) * embed_scale  # (B, T, d_model)
        # (B, T, num_*) @ (num_*, d_model) -> (B, T, d_model)
        seg = seg_probs.to(tok.dtype) @ self.E_seg.weight
        pos = pos_probs.to(tok.dtype) @ self.E_pos.weight
        return tok + torch.tanh(self.alpha_seg) * seg + torch.tanh(self.alpha_pos) * pos

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        seg_probs: Optional[torch.Tensor] = None,
        pos_probs: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        if seg_probs is None or pos_probs is None:
            B, T = input_ids.shape
            zero_seg = input_ids.new_zeros((B, T, self.cfg.num_seg_labels), dtype=torch.float32)
            zero_pos = input_ids.new_zeros((B, T, self.cfg.num_pos_labels), dtype=torch.float32)
            seg_probs = zero_seg
            pos_probs = zero_pos

        inputs_embeds = self.compute_inputs_embeds(input_ids, seg_probs, pos_probs)

        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        seg_probs: Optional[torch.Tensor] = None,
        pos_probs: Optional[torch.Tensor] = None,
        **gen_kwargs,
    ):
        """generate() with soft tags injected into the encoder embeddings.

        Strategy: precompute encoder_outputs with our adapted inputs_embeds,
        then call model.generate(encoder_outputs=...) so HF re-uses the
        encoder pass we already did.
        """
        if seg_probs is None or pos_probs is None:
            B, T = input_ids.shape
            seg_probs = input_ids.new_zeros((B, T, self.cfg.num_seg_labels), dtype=torch.float32)
            pos_probs = input_ids.new_zeros((B, T, self.cfg.num_pos_labels), dtype=torch.float32)

        inputs_embeds = self.compute_inputs_embeds(input_ids, seg_probs, pos_probs)
        encoder = self.encoder
        encoder_outputs = encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return self.model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

    def adapter_state_dict(self) -> dict:
        """Adapter-only weights (small) for stage-1 checkpoints."""
        return {
            "E_seg.weight": self.E_seg.weight.detach().cpu(),
            "E_pos.weight": self.E_pos.weight.detach().cpu(),
            "alpha_seg": self.alpha_seg.detach().cpu(),
            "alpha_pos": self.alpha_pos.detach().cpu(),
        }

    def load_adapter_state_dict(self, sd: dict, strict: bool = True):
        missing = []
        for key in ("E_seg.weight", "E_pos.weight", "alpha_seg", "alpha_pos"):
            if key not in sd:
                missing.append(key)
        if strict and missing:
            raise KeyError(f"adapter checkpoint missing keys: {missing}")
        with torch.no_grad():
            if "E_seg.weight" in sd:
                self.E_seg.weight.copy_(sd["E_seg.weight"].to(self.E_seg.weight.device))
            if "E_pos.weight" in sd:
                self.E_pos.weight.copy_(sd["E_pos.weight"].to(self.E_pos.weight.device))
            if "alpha_seg" in sd:
                self.alpha_seg.copy_(sd["alpha_seg"].to(self.alpha_seg.device))
            if "alpha_pos" in sd:
                self.alpha_pos.copy_(sd["alpha_pos"].to(self.alpha_pos.device))
