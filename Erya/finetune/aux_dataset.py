"""Datasets and collators for Erya soft-tag fine-tuning.

The augmenter (``augment_finetune_with_siku.py``) produces, for every ``.src``
file in ``finetune.tgz``, a sibling ``.src.siku_aux.npz`` with arrays:

  - ``row_lengths`` (int32, [N]):       characters per source line
  - ``row_offsets`` (int64, [N+1]):     cumulative character offsets
  - ``seg_probs`` / ``pos_probs`` (float16, [total_chars, num_*_labels])
  - ``seg_emissions`` / ``pos_emissions``: raw CRF logits (unused by default)
  - ``seg_decode_ids`` / ``pos_decode_ids`` (int16, [total_chars])
  - ``seg_label_names`` / ``pos_label_names`` (object): label string tables

Alignment is per-character. For each source line ``i`` we slice
``[row_offsets[i] : row_offsets[i+1]]`` to recover its prob array.

We tokenize the source character-by-character (mirroring the augmenter's
``convert_tokens_to_ids(normalize_char(ch))``), prepend ``[CLS]`` and append
``[SEP]``, and pad the prob tensors with a zero row at each special-token
position so alignment is exactly 1:1 with ``input_ids``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, Sampler


# ---------- normalization (mirror augmenter) -----------------------------------

_NORM_DOUBLE_QUOTES = {"“", "”", "「", "」"}      # “ ” 「 」
_NORM_SINGLE_QUOTES = {
    "‘", "’", "『", "』", "（", "）", "(", ")",
}                                                                    # ‘ ’ 『 』 （ ）

def normalize_char(token: str) -> str:
    if token in _NORM_DOUBLE_QUOTES:
        return '"'
    if token in _NORM_SINGLE_QUOTES:
        return "'"
    return token


# ---------- line-offset index for memory-efficient line lookup -----------------

def build_line_offsets(path: Path) -> np.ndarray:
    """Return cumulative byte offsets to the start of each line, plus EOF.

    Cached as ``<path>.idx.npy``. Avoids loading large files into memory.
    """
    idx_path = path.with_suffix(path.suffix + ".idx.npy")
    if idx_path.exists() and idx_path.stat().st_mtime >= path.stat().st_mtime:
        return np.load(idx_path)

    offsets = [0]
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            offsets.append(offsets[-1] + len(line))
    arr = np.asarray(offsets, dtype=np.int64)
    np.save(idx_path, arr)
    return arr


def read_line(path: Path, offsets: np.ndarray, idx: int) -> str:
    """Return line ``idx`` (0-based) without trailing newline."""
    start = int(offsets[idx])
    end = int(offsets[idx + 1])
    with open(path, "rb") as f:
        f.seek(start)
        raw = f.read(end - start)
    return raw.decode("utf-8").rstrip("\n").rstrip("\r")


# ---------- paired aux dataset --------------------------------------------------

@dataclass
class PairedExample:
    input_ids: torch.LongTensor       # (T,) including [CLS] and [SEP]
    attention_mask: torch.LongTensor  # (T,)
    seg_probs: torch.FloatTensor      # (T, num_seg)
    pos_probs: torch.FloatTensor      # (T, num_pos)
    labels: torch.LongTensor          # (L,) target ids; -100 will be applied at collation
    src_len: int                      # original character count
    tgt_len: int


class PairedAuxDataset(Dataset):
    """One (src, tgt, npz) tuple = one split of one sub-corpus."""

    def __init__(
        self,
        src_path: os.PathLike,
        tgt_path: os.PathLike,
        npz_path: os.PathLike,
        tokenizer,
        max_src_chars: int = 1022,
        max_tgt_tokens: int = 1024,
        prob_source: str = "probs",
        emissions_temperature: float = 1.0,
        check_alignment: bool = True,
    ):
        self.src_path = Path(src_path)
        self.tgt_path = Path(tgt_path)
        self.npz_path = Path(npz_path)
        self.tokenizer = tokenizer
        self.max_src_chars = max_src_chars
        self.max_tgt_tokens = max_tgt_tokens
        self.prob_source = prob_source
        self.emissions_temperature = emissions_temperature
        self.check_alignment = check_alignment

        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id

        self.aux = np.load(self.npz_path, mmap_mode="r", allow_pickle=True)
        self.row_offsets = np.asarray(self.aux["row_offsets"])  # small, load fully
        self.row_lengths = np.asarray(self.aux["row_lengths"])
        self.seg_label_names = list(self.aux["seg_label_names"])
        self.pos_label_names = list(self.aux["pos_label_names"])
        self.num_seg = len(self.seg_label_names)
        self.num_pos = len(self.pos_label_names)

        if prob_source == "probs":
            self._seg_arr = self.aux["seg_probs"]
            self._pos_arr = self.aux["pos_probs"]
        elif prob_source == "emissions":
            self._seg_arr = self.aux["seg_emissions"]
            self._pos_arr = self.aux["pos_emissions"]
        elif prob_source == "decode":
            self._seg_arr = self.aux["seg_decode_ids"]
            self._pos_arr = self.aux["pos_decode_ids"]
        else:
            raise ValueError(f"unknown prob_source={prob_source!r}")

        self.src_offsets = build_line_offsets(self.src_path)
        self.tgt_offsets = build_line_offsets(self.tgt_path)
        n_src = len(self.src_offsets) - 1
        n_tgt = len(self.tgt_offsets) - 1
        n_aux = len(self.row_lengths)
        if not (n_src == n_tgt == n_aux):
            raise ValueError(
                f"line count mismatch: src={n_src} tgt={n_tgt} aux={n_aux} "
                f"(in {self.src_path})"
            )
        self._n = n_src

    def __len__(self) -> int:
        return self._n

    def _get_source_probs(self, idx: int, src_chars: int) -> tuple[np.ndarray, np.ndarray]:
        start = int(self.row_offsets[idx])
        end = int(self.row_offsets[idx + 1])
        if end - start != src_chars:
            # The augmenter wrote per-character outputs from the line as it
            # appeared on disk. If our re-read length differs, there is a real
            # alignment problem (e.g. the file was modified after augmenting).
            raise ValueError(
                f"row_offsets disagree with re-read line {idx} of "
                f"{self.src_path}: aux={end - start} chars vs file={src_chars}"
            )
        seg = np.asarray(self._seg_arr[start:end])
        pos = np.asarray(self._pos_arr[start:end])
        if self.prob_source == "emissions":
            seg = _softmax(seg.astype(np.float32) / self.emissions_temperature, axis=-1)
            pos = _softmax(pos.astype(np.float32) / self.emissions_temperature, axis=-1)
        elif self.prob_source == "decode":
            seg = _onehot(seg.astype(np.int64), self.num_seg)
            pos = _onehot(pos.astype(np.int64), self.num_pos)
        else:
            seg = seg.astype(np.float32, copy=False)
            pos = pos.astype(np.float32, copy=False)
        return seg, pos

    def __getitem__(self, idx: int) -> PairedExample:
        src_line = read_line(self.src_path, self.src_offsets, idx)
        tgt_line = read_line(self.tgt_path, self.tgt_offsets, idx)

        chars = list(src_line)
        if len(chars) > self.max_src_chars:
            chars = chars[: self.max_src_chars]
        norm_chars = [normalize_char(ch) for ch in chars]
        src_ids = [self.tokenizer.convert_tokens_to_ids(ch) for ch in norm_chars]

        seg_full, pos_full = self._get_source_probs(idx, int(self.row_lengths[idx]))
        if len(chars) < int(self.row_lengths[idx]):
            seg_full = seg_full[: len(chars)]
            pos_full = pos_full[: len(chars)]

        if self.check_alignment and len(src_ids) != seg_full.shape[0]:
            raise AssertionError(
                f"alignment check failed at line {idx} of {self.src_path}: "
                f"len(src_ids)={len(src_ids)} vs seg_rows={seg_full.shape[0]}"
            )

        # Wrap with [CLS] ... [SEP] and pad probs with zero rows at specials.
        input_ids = [self.cls_id] + src_ids + [self.sep_id]
        seg_padded = np.zeros((len(input_ids), self.num_seg), dtype=np.float32)
        pos_padded = np.zeros((len(input_ids), self.num_pos), dtype=np.float32)
        seg_padded[1:-1] = seg_full
        pos_padded[1:-1] = pos_full

        attention_mask = [1] * len(input_ids)

        tgt_enc = self.tokenizer(
            tgt_line,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_tgt_tokens,
            return_tensors=None,
        )
        labels = tgt_enc["input_ids"]

        return PairedExample(
            input_ids=torch.as_tensor(input_ids, dtype=torch.long),
            attention_mask=torch.as_tensor(attention_mask, dtype=torch.long),
            seg_probs=torch.from_numpy(seg_padded),
            pos_probs=torch.from_numpy(pos_padded),
            labels=torch.as_tensor(labels, dtype=torch.long),
            src_len=len(chars),
            tgt_len=len(labels),
        )


# ---------- monolingual denoising dataset --------------------------------------

class MonoDenoiseDataset(Dataset):
    """Read raw ancient lines, BART-mask the source, target = original.

    Soft tag distributions are returned as zero tensors so the same forward
    path (with seg/pos probs in the input) works without special-casing.
    """

    def __init__(
        self,
        path: os.PathLike,
        tokenizer,
        num_seg: int,
        num_pos: int,
        max_src_chars: int = 1022,
        max_tgt_tokens: int = 1024,
        mask_ratio: float = 0.30,
        poisson_lambda: float = 3.0,
        seed: int = 0,
    ):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.num_seg = num_seg
        self.num_pos = num_pos
        self.max_src_chars = max_src_chars
        self.max_tgt_tokens = max_tgt_tokens
        self.mask_ratio = mask_ratio
        self.poisson_lambda = poisson_lambda
        self.rng = np.random.default_rng(seed)

        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id
        self.mask_id = tokenizer.mask_token_id

        self.offsets = build_line_offsets(self.path)
        self._n = len(self.offsets) - 1

    def __len__(self) -> int:
        return self._n

    def _mask_spans(self, ids: List[int]) -> List[int]:
        n = len(ids)
        if n == 0:
            return ids
        n_to_mask = int(round(self.mask_ratio * n))
        out = list(ids)
        masked = 0
        guard = 0
        while masked < n_to_mask and guard < 32:
            guard += 1
            span_len = max(1, int(self.rng.poisson(self.poisson_lambda)))
            if span_len > n_to_mask - masked:
                span_len = n_to_mask - masked
            start = int(self.rng.integers(0, max(1, n - span_len + 1)))
            # collapse span into a single [MASK]
            out = out[:start] + [self.mask_id] + out[start + span_len :]
            masked += span_len
            n -= span_len - 1
            if n <= 1:
                break
        return out

    def __getitem__(self, idx: int) -> PairedExample:
        line = read_line(self.path, np.asarray(self.offsets), idx)
        chars = list(line)[: self.max_src_chars]
        norm_chars = [normalize_char(ch) for ch in chars]
        src_ids = [self.tokenizer.convert_tokens_to_ids(ch) for ch in norm_chars]

        # target is the un-masked sequence (with specials)
        target = self.tokenizer(
            line[: self.max_src_chars],
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_tgt_tokens,
            return_tensors=None,
        )["input_ids"]

        masked_src_ids = self._mask_spans(src_ids)
        input_ids = [self.cls_id] + masked_src_ids + [self.sep_id]
        attention_mask = [1] * len(input_ids)

        T = len(input_ids)
        seg_zero = np.zeros((T, self.num_seg), dtype=np.float32)
        pos_zero = np.zeros((T, self.num_pos), dtype=np.float32)

        return PairedExample(
            input_ids=torch.as_tensor(input_ids, dtype=torch.long),
            attention_mask=torch.as_tensor(attention_mask, dtype=torch.long),
            seg_probs=torch.from_numpy(seg_zero),
            pos_probs=torch.from_numpy(pos_zero),
            labels=torch.as_tensor(target, dtype=torch.long),
            src_len=len(chars),
            tgt_len=len(target),
        )


# ---------- collator ------------------------------------------------------------

@dataclass
class Batch:
    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    seg_probs: torch.FloatTensor
    pos_probs: torch.FloatTensor
    labels: torch.LongTensor       # -100 on pads
    teacher_seg: torch.FloatTensor # alias of seg_probs (same content; named for KL)
    teacher_pos: torch.FloatTensor


def _pad_2d_long(seqs: List[torch.Tensor], pad_value: int) -> torch.Tensor:
    max_len = max(s.numel() for s in seqs)
    out = torch.full((len(seqs), max_len), pad_value, dtype=seqs[0].dtype)
    for i, s in enumerate(seqs):
        out[i, : s.numel()] = s
    return out


def _pad_3d_float(seqs: List[torch.Tensor]) -> torch.Tensor:
    max_len = max(s.shape[0] for s in seqs)
    last = seqs[0].shape[1]
    out = torch.zeros((len(seqs), max_len, last), dtype=torch.float32)
    for i, s in enumerate(seqs):
        out[i, : s.shape[0]] = s
    return out


class PairedCollator:
    def __init__(self, pad_id: int, label_pad_id: int = -100):
        self.pad_id = pad_id
        self.label_pad_id = label_pad_id

    def __call__(self, batch: Sequence[PairedExample]) -> Batch:
        input_ids = _pad_2d_long([b.input_ids for b in batch], self.pad_id)
        attention_mask = _pad_2d_long([b.attention_mask for b in batch], 0)
        seg_probs = _pad_3d_float([b.seg_probs for b in batch])
        pos_probs = _pad_3d_float([b.pos_probs for b in batch])
        labels = _pad_2d_long([b.labels for b in batch], self.label_pad_id)
        return Batch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            seg_probs=seg_probs,
            pos_probs=pos_probs,
            labels=labels,
            teacher_seg=seg_probs,
            teacher_pos=pos_probs,
        )


# ---------- length-balanced sampler --------------------------------------------

class LengthBalancedSampler(Sampler[int]):
    """Sample indices from a ConcatDataset with per-corpus weights.

    Default weighting: per-sample weight = 1 / sqrt(corpus_size). Sampling is
    with replacement so every "epoch" of size ``num_samples`` mixes corpora
    fairly regardless of relative size.
    """

    def __init__(
        self,
        concat: ConcatDataset,
        num_samples: int,
        weights: Optional[Sequence[float]] = None,
        replacement: bool = True,
        seed: int = 0,
    ):
        self.concat = concat
        self.num_samples = num_samples
        self.replacement = replacement
        self.seed = seed
        self.epoch = 0

        sizes = [len(d) for d in concat.datasets]
        if weights is None:
            weights = [1.0 / np.sqrt(max(1, s)) for s in sizes]
        if len(weights) != len(sizes):
            raise ValueError("len(weights) must equal len(concat.datasets)")

        per_sample = np.empty(sum(sizes), dtype=np.float64)
        offset = 0
        for w, s in zip(weights, sizes):
            per_sample[offset : offset + s] = w
            offset += s
        per_sample /= per_sample.sum()
        self._probs = per_sample

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch)
        idx = rng.choice(len(self._probs), size=self.num_samples, replace=self.replacement, p=self._probs)
        return iter(idx.tolist())

    def __len__(self) -> int:
        return self.num_samples


class MixedSampler(Sampler[tuple]):
    """Yield ``(dataset_id, index)`` mixing parallel and denoising at a fixed ratio.

    Used by an outer Dataset that demuxes on ``dataset_id``.
    """

    def __init__(
        self,
        parallel_sampler: Sampler[int],
        mono_sampler: Sampler[int],
        denoise_ratio: float,
        num_samples: int,
        seed: int = 0,
    ):
        self.parallel_sampler = parallel_sampler
        self.mono_sampler = mono_sampler
        self.denoise_ratio = denoise_ratio
        self.num_samples = num_samples
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        if hasattr(self.parallel_sampler, "set_epoch"):
            self.parallel_sampler.set_epoch(epoch)
        if hasattr(self.mono_sampler, "set_epoch"):
            self.mono_sampler.set_epoch(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        p_iter = iter(self.parallel_sampler)
        m_iter = iter(self.mono_sampler)
        for _ in range(self.num_samples):
            if rng.random() < self.denoise_ratio:
                yield (1, next(m_iter))
            else:
                yield (0, next(p_iter))

    def __len__(self) -> int:
        return self.num_samples


class TaggedConcat(Dataset):
    """Wrap (parallel_concat, mono_dataset) and dispatch by ``(ds_id, idx)``."""

    def __init__(self, parallel: ConcatDataset, mono: Optional[Dataset]):
        self.parallel = parallel
        self.mono = mono

    def __len__(self) -> int:
        n = len(self.parallel)
        if self.mono is not None:
            n += len(self.mono)
        return n

    def __getitem__(self, key):
        if isinstance(key, tuple):
            ds_id, idx = key
        else:
            ds_id, idx = 0, key
        if ds_id == 0:
            return self.parallel[idx]
        if self.mono is None:
            raise IndexError("mono dataset not configured")
        return self.mono[idx]


# ---------- helpers -------------------------------------------------------------

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    np.exp(x, out=x)
    x /= x.sum(axis=axis, keepdims=True)
    return x


def _onehot(ids: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((ids.shape[0], n), dtype=np.float32)
    valid = (ids >= 0) & (ids < n)
    out[np.arange(ids.shape[0])[valid], ids[valid]] = 1.0
    return out


def discover_corpora(root: Path, split: str) -> List[tuple[Path, Path, Path]]:
    """Walk an extracted ``finetune_with_siku_aux`` tree and find all triples
    ``(.src, .tgt, .src.siku_aux.npz)`` for the requested split."""
    triples: List[tuple[Path, Path, Path]] = []
    for src in sorted(root.rglob(f"{split}.src")):
        tgt = src.with_suffix(".tgt")
        npz = src.with_name(src.name + ".siku_aux.npz")
        if not tgt.exists() or not npz.exists():
            continue
        triples.append((src, tgt, npz))
    return triples


def build_parallel_concat(
    root: Path,
    split: str,
    tokenizer,
    **dataset_kwargs,
) -> tuple[ConcatDataset, List[str]]:
    """Build a ConcatDataset over all sub-corpora for a split."""
    triples = discover_corpora(root, split)
    if not triples:
        raise FileNotFoundError(f"no '{split}' triples under {root}")
    datasets = [
        PairedAuxDataset(s, t, n, tokenizer=tokenizer, **dataset_kwargs)
        for s, t, n in triples
    ]
    names = [str(s.relative_to(root)) for s, _, _ in triples]
    return ConcatDataset(datasets), names
