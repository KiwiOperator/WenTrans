"""Translation evaluation harness for the tagged Erya model.

Two input modes for the soft tag distributions:

  --bundled                  : run SikuRoBERTa at inference (default)
  --aux_npz <path>           : use pre-computed siku_aux npz (companion to .src)

Metrics: sacreBLEU and chrF, character-tokenized for zh.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, BertTokenizer

from . import utils
from .aux_dataset import normalize_char
from .siku_inference import SingleSentenceAnnotator
from .tag_adapter import TagAdapterConfig, TaggedErya


def load_model(model_path: str, ckpt_path: Optional[str], device: torch.device,
               num_seg: int, num_pos: int, label_meta: dict) -> TaggedErya:
    base = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)
    tagged = TaggedErya(
        base,
        TagAdapterConfig(num_seg_labels=num_seg, num_pos_labels=num_pos, d_model=base.config.d_model),
    ).to(device)
    if ckpt_path is not None:
        ckpt = Path(ckpt_path)
        meta = {}
        if ckpt.is_dir():
            full = ckpt / "full.pt"
            adapter = ckpt / "adapter.pt"
            if full.exists():
                meta = utils.load_full(full, tagged)
            elif adapter.exists():
                meta = utils.load_adapter(adapter, tagged)
            else:
                raise FileNotFoundError(f"no full.pt or adapter.pt under {ckpt}")
        else:
            payload = torch.load(ckpt, map_location="cpu")
            if "model" in payload:
                meta = utils.load_full(ckpt, tagged)
            else:
                meta = utils.load_adapter(ckpt, tagged)
        utils.assert_label_tables_match(
            meta,
            label_meta.get("seg_label_names", []),
            label_meta.get("pos_label_names", []),
        )
    tagged.eval()
    return tagged


def encode_with_tags(
    tokenizer: BertTokenizer,
    text: str,
    seg_probs: np.ndarray,
    pos_probs: np.ndarray,
    max_src_chars: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    chars = list(text)[:max_src_chars]
    if seg_probs.shape[0] > len(chars):
        seg_probs = seg_probs[: len(chars)]
        pos_probs = pos_probs[: len(chars)]
    norm = [normalize_char(c) for c in chars]
    src_ids = [tokenizer.convert_tokens_to_ids(c) for c in norm]
    input_ids = [tokenizer.cls_token_id] + src_ids + [tokenizer.sep_token_id]
    attn = [1] * len(input_ids)

    T = len(input_ids)
    seg_padded = np.zeros((T, seg_probs.shape[1]), dtype=np.float32)
    pos_padded = np.zeros((T, pos_probs.shape[1]), dtype=np.float32)
    seg_padded[1:-1] = seg_probs.astype(np.float32, copy=False)
    pos_padded[1:-1] = pos_probs.astype(np.float32, copy=False)

    return (
        torch.as_tensor([input_ids], dtype=torch.long),
        torch.as_tensor([attn], dtype=torch.long),
        torch.from_numpy(seg_padded).unsqueeze(0),
        torch.from_numpy(pos_padded).unsqueeze(0),
    )


def iter_aux_lines(npz_path: Path) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    aux = np.load(npz_path, mmap_mode="r", allow_pickle=True)
    row_offsets = np.asarray(aux["row_offsets"])
    seg = aux["seg_probs"]
    pos = aux["pos_probs"]
    for i in range(len(row_offsets) - 1):
        s, e = int(row_offsets[i]), int(row_offsets[i + 1])
        yield np.asarray(seg[s:e]), np.asarray(pos[s:e])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Erya")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to checkpoint dir (with full.pt / adapter.pt) or single .pt file")
    parser.add_argument("--src", type=str, required=True, help="Source .src file")
    parser.add_argument("--ref", type=str, default=None, help="Reference .tgt file (for BLEU)")
    parser.add_argument("--output", type=str, default=None, help="Where to write hypotheses")
    parser.add_argument("--bundled", action="store_true", default=True,
                        help="Use bundled SikuRoBERTa for tag inference (default).")
    parser.add_argument("--aux_npz", type=str, default=None,
                        help="Pre-computed .src.siku_aux.npz; overrides --bundled.")
    parser.add_argument("--siku_dir", type=str,
                        default="The-first-ancient-Chinese-word-segmentation-and-part-of-speech-tagging-code-and-analysis-main")
    parser.add_argument("--max_src_chars", type=int, default=1022)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    log = utils.get_logger("erya_finetune.eval")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(args.model_path)

    # Determine label tables and dims.
    if args.aux_npz:
        aux = np.load(args.aux_npz, allow_pickle=True)
        seg_labels = list(aux["seg_label_names"])
        pos_labels = list(aux["pos_label_names"])
        annotator = None
    else:
        annotator = SingleSentenceAnnotator(siku_dir=Path(args.siku_dir), device=device)
        seg_labels = annotator.seg_label_names
        pos_labels = annotator.pos_label_names

    tagged = load_model(
        args.model_path, args.ckpt, device,
        num_seg=len(seg_labels), num_pos=len(pos_labels),
        label_meta={"seg_label_names": seg_labels, "pos_label_names": pos_labels},
    )

    sources: List[str] = []
    with open(args.src, "r", encoding="utf-8") as f:
        for line in f:
            sources.append(line.rstrip("\n"))
    if args.limit:
        sources = sources[: args.limit]

    refs: Optional[List[str]] = None
    if args.ref:
        with open(args.ref, "r", encoding="utf-8") as f:
            refs = [l.rstrip("\n") for l in f]
        if args.limit:
            refs = refs[: args.limit]

    aux_iter = iter_aux_lines(Path(args.aux_npz)) if args.aux_npz else None
    hypotheses: List[str] = []

    for i, src in enumerate(sources):
        if aux_iter is not None:
            seg_p, pos_p = next(aux_iter)
        else:
            assert annotator is not None
            seg_p, pos_p = annotator.annotate(src)

        input_ids, attn, seg_t, pos_t = encode_with_tags(
            tokenizer, src, seg_p, pos_p, args.max_src_chars
        )
        out_ids = tagged.generate(
            input_ids=input_ids.to(device),
            attention_mask=attn.to(device),
            seg_probs=seg_t.to(device),
            pos_probs=pos_t.to(device),
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
        hyp = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]
        hypotheses.append(hyp.replace(" ", ""))

        if (i + 1) % 50 == 0:
            log.info("translated %d/%d", i + 1, len(sources))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for h in hypotheses:
                f.write(h + "\n")
        log.info("wrote hypotheses -> %s", args.output)

    if refs is not None:
        try:
            import sacrebleu
        except ImportError:
            log.warning("sacrebleu not installed; skipping BLEU/chrF")
            return
        bleu = sacrebleu.corpus_bleu(hypotheses, [refs], tokenize="zh")
        chrf = sacrebleu.corpus_chrf(hypotheses, [refs])
        log.info("BLEU = %.2f", bleu.score)
        log.info("chrF = %.2f", chrf.score)
        print(json.dumps({"bleu": bleu.score, "chrf": chrf.score}, indent=2))


if __name__ == "__main__":
    main()
