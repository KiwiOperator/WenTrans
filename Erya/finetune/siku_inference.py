"""Single-sentence SikuRoBERTa annotator for inference time.

Reuses the imports/plumbing from ``augment_finetune_with_siku.SikuAnnotator``
so the seg/pos label tables produced at inference are byte-identical to the
ones the model was fine-tuned on.
"""
from __future__ import annotations

import contextlib
import importlib
import io
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer


def _load_siku_modules(siku_dir: Path):
    sys.path.insert(0, str(siku_dir))
    for name in ["config", "label", "model", "data_process"]:
        sys.modules.pop(name, None)
    siku_config = importlib.import_module("config")
    siku_config.berta_model = str((siku_dir / "sikuRoberta_model").resolve())
    siku_config.save_checkpoint = str((siku_dir / "sikuRoberta_model_crf0.pth").resolve())
    siku_config.resume_checkpoint = siku_config.save_checkpoint
    siku_config.data_dir = str((siku_dir / "zuozhuan_train_utf8.txt").resolve())
    siku_config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with contextlib.redirect_stdout(io.StringIO()):
        siku_label = importlib.import_module("label")
    siku_model = importlib.import_module("model")
    return siku_config, siku_label, siku_model


def _normalize_char(token: str) -> str:
    if token in {"“", "”", "「", "」"}:
        return '"'
    if token in {"‘", "’", "『", "』", "（", "）", "(", ")"}:
        return "'"
    return token


class SingleSentenceAnnotator:
    """Annotate one sentence at a time. Loads Siku once and persists."""

    def __init__(
        self,
        siku_dir: Path,
        device: torch.device | str | None = None,
        dtype: np.dtype = np.float32,
        chunk_size: int = 511,
        batch_size: int = 8,
    ):
        self.siku_dir = Path(siku_dir).resolve()
        self.dtype = dtype
        self.chunk_size = chunk_size
        self.batch_size = batch_size

        self.config, self.label, model_module = _load_siku_modules(self.siku_dir)
        if device is not None:
            self.config.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.berta_model)
        self.model = model_module.BertSegPos(self.config, None).to(self.config.device)
        state_dict = torch.load(self.config.save_checkpoint, map_location=self.config.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.seg_label_names: List[str] = [
            self.label.id_seg2label[i] for i in range(self.label.num_seglabels)
        ]
        self.pos_label_names: List[str] = [
            self.label.id_pos2label[i] for i in range(self.label.num_poslabels)
        ]
        self.num_seg = len(self.seg_label_names)
        self.num_pos = len(self.pos_label_names)

    @torch.no_grad()
    def annotate(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return (seg_probs, pos_probs) of shape (len(text), num_*)."""
        if not text:
            return (
                np.zeros((0, self.num_seg), dtype=self.dtype),
                np.zeros((0, self.num_pos), dtype=self.dtype),
            )

        chars = list(text)
        norm = [_normalize_char(c) for c in chars]
        chunks: List[Tuple[int, int, List[int]]] = []
        for s in range(0, len(chars), self.chunk_size):
            e = min(len(chars), s + self.chunk_size)
            ids = [self.tokenizer.cls_token_id] + [
                self.tokenizer.convert_tokens_to_ids(ch) for ch in norm[s:e]
            ]
            chunks.append((s, e, ids))

        seg_probs = np.zeros((len(chars), self.num_seg), dtype=self.dtype)
        pos_probs = np.zeros((len(chars), self.num_pos), dtype=self.dtype)

        for batch_start in range(0, len(chunks), self.batch_size):
            batch = chunks[batch_start : batch_start + self.batch_size]
            tensors = [torch.LongTensor(ids) for _, _, ids in batch]
            data = pad_sequence(tensors, batch_first=True, padding_value=0).to(self.config.device)
            attention_mask = data.gt(0).to(self.config.device)
            lengths = torch.tensor([e - s for s, e, _ in batch], device=self.config.device)
            max_len = int(lengths.max().item())
            decode_mask = (
                torch.arange(max_len, device=self.config.device).unsqueeze(0) < lengths.unsqueeze(1)
            )
            logits_seg, logits_pos = self.model(
                data, token_type_ids=None, attention_mask=attention_mask
            )[0]
            ps = torch.softmax(logits_seg, dim=-1).detach().cpu().numpy()
            pp = torch.softmax(logits_pos, dim=-1).detach().cpu().numpy()
            for i, (s, e, _) in enumerate(batch):
                length = e - s
                seg_probs[s:e] = ps[i, :length].astype(self.dtype, copy=False)
                pos_probs[s:e] = pp[i, :length].astype(self.dtype, copy=False)

        return seg_probs, pos_probs
