#!/usr/bin/env python3
import argparse
import contextlib
import importlib
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Augment Erya finetune.tgz with character-aligned SikuRoBERTa "
            "segmentation/POS emissions and probabilities."
        )
    )
    parser.add_argument(
        "--input",
        default="dataset/finetune.tgz",
        help="Path to the original finetune.tgz relative to Erya/ or absolute.",
    )
    parser.add_argument(
        "--output",
        default="dataset/finetune_with_siku_aux.tgz",
        help="Path for the augmented tarball.",
    )
    parser.add_argument(
        "--siku-dir",
        default="../The-first-ancient-Chinese-word-segmentation-and-part-of-speech-tagging-code-and-analysis-main",
        help="Path to the SikuRoBERTa segmentation/POS project directory.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size over text chunks.",
    )
    parser.add_argument(
        "--save-dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Floating-point dtype used when writing emissions/probabilities.",
    )
    return parser.parse_args()


def resolve_path(base_dir: Path, path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (base_dir / path).resolve()


def normalize_char(token: str) -> str:
    if token in {"“", "”", "「", "」"}:
        return '"'
    if token in {"‘", "’", "『", "』", "（", "）", "(", ")"}:
        return "'"
    return token


def chunk_chars(text: str, max_chars: int = 511):
    chars = list(text)
    norm_chars = [normalize_char(ch) for ch in chars]
    for start in range(0, len(chars), max_chars):
        end = start + max_chars
        yield start, chars[start:end], norm_chars[start:end]


def load_siku_modules(siku_dir: Path):
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


class SikuAnnotator:
    def __init__(self, siku_dir: Path, batch_size: int, save_dtype: str):
        self.siku_dir = siku_dir
        self.batch_size = batch_size
        self.save_dtype = np.float16 if save_dtype == "float16" else np.float32
        self.config, self.label, model_module = load_siku_modules(siku_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.berta_model)
        self.model = model_module.BertSegPos(self.config, None)
        self.model.to(self.config.device)
        state_dict = torch.load(self.config.save_checkpoint, map_location=self.config.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.seg_label_names = np.array(
            [self.label.id_seg2label[i] for i in range(self.label.num_seglabels)],
            dtype=object,
        )
        self.pos_label_names = np.array(
            [self.label.id_pos2label[i] for i in range(self.label.num_poslabels)],
            dtype=object,
        )

    def annotate_file(self, src_path: Path, file_label=None):
        row_lengths = []
        total_chunks = 0
        with open(src_path, "r", encoding="utf-8") as f:
            for raw_text in f:
                text = raw_text.rstrip("\n")
                length = len(text)
                row_lengths.append(length)
                if length:
                    total_chunks += (length + 510) // 511

        row_lengths = np.asarray(row_lengths, dtype=np.int32)
        row_offsets = np.zeros(row_lengths.shape[0] + 1, dtype=np.int64)
        row_offsets[1:] = np.cumsum(row_lengths, dtype=np.int64)
        total_chars = int(row_offsets[-1])
        total_batches = (total_chunks + self.batch_size - 1) // self.batch_size

        seg_dim = self.label.num_seglabels
        pos_dim = self.label.num_poslabels

        seg_emissions = np.zeros((total_chars, seg_dim), dtype=self.save_dtype)
        seg_probs = np.zeros((total_chars, seg_dim), dtype=self.save_dtype)
        pos_emissions = np.zeros((total_chars, pos_dim), dtype=self.save_dtype)
        pos_probs = np.zeros((total_chars, pos_dim), dtype=self.save_dtype)
        seg_decode_ids = np.zeros((total_chars,), dtype=np.int16)
        pos_decode_ids = np.zeros((total_chars,), dtype=np.int16)

        if file_label is not None:
            print(
                f"[annotate] {file_label}: rows={row_lengths.shape[0]}, chars={total_chars}, "
                f"chunks={total_chunks}, batches={total_batches}",
                flush=True,
            )

        batch_token_ids = []
        batch_meta = []
        batch_index = 0
        chunk_done = 0

        def flush_batch():
            nonlocal batch_index, chunk_done, batch_token_ids, batch_meta
            if not batch_token_ids:
                return

            batch_index += 1
            chunk_done += len(batch_token_ids)
            if file_label is not None and (
                batch_index == 1 or batch_index == total_batches or batch_index % 100 == 0
            ):
                print(
                    f"[annotate] {file_label}: batch {batch_index}/{total_batches} "
                    f"chunks={chunk_done}/{total_chunks}",
                    flush=True,
                )

            batch_tensors = [torch.LongTensor(token_ids) for token_ids in batch_token_ids]
            batch_data = pad_sequence(batch_tensors, batch_first=True, padding_value=0).to(self.config.device)
            attention_mask = batch_data.gt(0).to(self.config.device)
            lengths = torch.tensor([meta[3] for meta in batch_meta], device=self.config.device)
            max_len = int(lengths.max().item())
            decode_mask = (
                torch.arange(max_len, device=self.config.device).unsqueeze(0) < lengths.unsqueeze(1)
            )

            with torch.no_grad():
                logits_seg, logits_pos = self.model(
                    batch_data,
                    token_type_ids=None,
                    attention_mask=attention_mask,
                )[0]
                probs_seg = torch.softmax(logits_seg, dim=-1)
                probs_pos = torch.softmax(logits_pos, dim=-1)
                pred_seg = self.model.crf_seg.decode(logits_seg, mask=decode_mask)
                pred_pos = self.model.crf_pos.decode(logits_pos, mask=decode_mask)

            logits_seg = logits_seg.detach().cpu().numpy()
            logits_pos = logits_pos.detach().cpu().numpy()
            probs_seg = probs_seg.detach().cpu().numpy()
            probs_pos = probs_pos.detach().cpu().numpy()

            for i, (row_idx, start, end, length) in enumerate(batch_meta):
                row_start = int(row_offsets[row_idx]) + start
                row_end = int(row_offsets[row_idx]) + end
                seg_emissions[row_start:row_end] = logits_seg[i, :length].astype(self.save_dtype, copy=False)
                seg_probs[row_start:row_end] = probs_seg[i, :length].astype(self.save_dtype, copy=False)
                pos_emissions[row_start:row_end] = logits_pos[i, :length].astype(self.save_dtype, copy=False)
                pos_probs[row_start:row_end] = probs_pos[i, :length].astype(self.save_dtype, copy=False)
                seg_decode_ids[row_start:row_end] = np.asarray(pred_seg[i][:length], dtype=np.int16)
                pos_decode_ids[row_start:row_end] = np.asarray(pred_pos[i][:length], dtype=np.int16)

            batch_token_ids = []
            batch_meta = []

        with open(src_path, "r", encoding="utf-8") as f:
            for row_idx, raw_text in enumerate(f):
                text = raw_text.rstrip("\n")
                if not text:
                    continue
                normalized_text = "".join(normalize_char(ch) for ch in text)
                text_len = len(text)
                for start in range(0, text_len, 511):
                    end = min(start + 511, text_len)
                    norm_chunk = normalized_text[start:end]
                    token_ids = [self.tokenizer.cls_token_id]
                    token_ids.extend(self.tokenizer.convert_tokens_to_ids(ch) for ch in norm_chunk)
                    batch_token_ids.append(token_ids)
                    batch_meta.append((row_idx, start, end, end - start))
                    if len(batch_token_ids) >= self.batch_size:
                        flush_batch()

        flush_batch()

        return {
            "row_lengths": row_lengths,
            "row_offsets": row_offsets,
            "seg_emissions": seg_emissions,
            "seg_probs": seg_probs,
            "pos_emissions": pos_emissions,
            "pos_probs": pos_probs,
            "seg_decode_ids": seg_decode_ids,
            "pos_decode_ids": pos_decode_ids,
            "seg_label_names": self.seg_label_names,
            "pos_label_names": self.pos_label_names,
        }


def save_npz(path: Path, arrays: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def repack_directory(source_root: Path, output_tgz: Path):
    with tarfile.open(output_tgz, "w:gz") as tar:
        for path in sorted(source_root.rglob("*")):
            arcname = path.relative_to(source_root)
            tar.add(path, arcname=str(arcname))


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    input_tgz = resolve_path(base_dir, args.input)
    output_tgz = resolve_path(base_dir, args.output)
    siku_dir = resolve_path(base_dir, args.siku_dir)

    if not input_tgz.exists():
        raise FileNotFoundError(f"Input tarball not found: {input_tgz}")
    if not siku_dir.exists():
        raise FileNotFoundError(f"Siku model directory not found: {siku_dir}")

    annotator = SikuAnnotator(siku_dir=siku_dir, batch_size=args.batch_size, save_dtype=args.save_dtype)

    with tempfile.TemporaryDirectory(prefix="erya_finetune_aux_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        extract_root = tmp_dir / "extract"
        extract_root.mkdir(parents=True, exist_ok=True)

        print(f"Extracting {input_tgz} ...")
        with tarfile.open(input_tgz, "r:gz") as tar:
            tar.extractall(extract_root)

        src_files = sorted(extract_root.rglob("*.src"))
        if not src_files:
            raise RuntimeError("No .src files found inside finetune.tgz")

        manifest = {
            "source_tarball": str(input_tgz),
            "siku_dir": str(siku_dir),
            "checkpoint": str((siku_dir / "sikuRoberta_model_crf0.pth").resolve()),
            "save_dtype": args.save_dtype,
            "files": [],
        }

        total_files = len(src_files)
        for file_idx, src_file in enumerate(src_files, start=1):
            rel_src = src_file.relative_to(extract_root)
            print(f"[file {file_idx}/{total_files}] Annotating {rel_src} ...", flush=True)
            arrays = annotator.annotate_file(src_file, file_label=str(rel_src))
            aux_path = src_file.with_name(src_file.name + ".siku_aux.npz")
            print(f"[file {file_idx}/{total_files}] Saving {aux_path.relative_to(extract_root)} ...", flush=True)
            save_npz(aux_path, arrays)
            manifest["files"].append(
                {
                    "src": str(rel_src),
                    "aux": str(aux_path.relative_to(extract_root)),
                    "rows": int(arrays["row_lengths"].shape[0]),
                    "chars": int(arrays["row_offsets"][-1]),
                }
            )
            print(
                f"[file {file_idx}/{total_files}] Done {rel_src}: "
                f"rows={int(arrays['row_lengths'].shape[0])}, chars={int(arrays['row_offsets'][-1])}",
                flush=True,
            )

        manifest_path = extract_root / "dataset" / "siku_aux_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        if output_tgz.exists():
            output_tgz.unlink()
        output_tgz.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing augmented tarball to {output_tgz} ...")
        repack_directory(extract_root, output_tgz)
        print("Done.")


if __name__ == "__main__":
    main()
