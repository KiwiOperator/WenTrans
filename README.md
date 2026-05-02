# WenTrans — Soft Tag Adapters for Ancient → Modern Chinese Translation

Fine-tune the [Erya](https://huggingface.co/RUCAIBox/Erya) encoder–decoder translation model with **soft POS-tag and word-segmentation embeddings** distilled from a [SikuRoBERTa](https://github.com/SIKU-BERT/SikuBERT) tagger. Curriculum training in three stages keeps the pretrained model intact at first and only gradually integrates the new tag features.

```
ancient Chinese tokens
        │
   token embeddings
        +  α_seg · (seg_probs @ E_seg)        ← soft segmentation embedding
        +  α_pos · (pos_probs @ E_pos)        ← soft POS embedding
        ↓
     encoder (12-layer)
        ↓
     decoder ──→ modern Chinese
```

`α_seg`, `α_pos` are trainable scalars initialised to 0; the model is bit-identical to vanilla Erya at step 0.

## Quick results

Stage-3 model on six EvaHan test splits, evaluated twice — once with the trained adapter active, once with the gates patched to zero so the wrapper is identity to plain encoder forward.

| Corpus  | α = 0 BLEU | trained-α BLEU | ΔBLEU | ΔchrF |
| ------- | :--------: | :------------: | :---: | :---: |
| `xint`  |   27.54    |   **35.31**    | **+7.77** | +6.36 |
| `mings` |   30.18    |   **36.69**    | +6.51 | +5.43 |
| `shij`  |   20.24    |     26.19      | +5.95 | +4.64 |
| `xux`   |   27.14    |     33.03      | +5.89 | +4.66 |
| `hans`  |   22.99    |     28.48      | +5.49 | +4.30 |
| `taip`  |   17.96    |     22.00      | +4.04 | +3.12 |
| **mean** | **24.34**  |   **30.28**    | **+5.94** | **+4.75** |

The adapter contributes a **mean +5.94 BLEU / +4.75 chrF** lift across all six sub-corpora — about **+24 % relative BLEU** — and helps every corpus (no negative or marginal deltas). Full breakdown and discussion in [report.md](report.md) § 5.

## Architecture

The wrapper [Erya/finetune/tag_adapter.py:TaggedErya](Erya/finetune/tag_adapter.py) wraps `CPTForConditionalGeneration` (loaded as BART-compatible) and computes encoder inputs as

```python
inputs_embeds = embed_tokens(input_ids) * embed_scale \
              + tanh(α_seg) · (seg_probs @ E_seg) \
              + tanh(α_pos) · (pos_probs @ E_pos)
out = base_model(inputs_embeds=..., attention_mask=..., labels=...)
```

`E_seg` is `(num_seg_labels, d_model)` (BIES, 4 labels), `E_pos` is `(num_pos_labels, d_model)` (~30 labels). Tag distributions come from the SikuRoBERTa CRF tagger via [augment_finetune_with_siku.py](Erya/augment_finetune_with_siku.py) and are stored per-character per-line in `*.src.siku_aux.npz` files alongside each `.src` file.

## Three-stage training

| Stage | What's trainable | Loss | Why |
| --- | --- | --- | --- |
| **1** — adapters only | `E_seg`, `E_pos`, `α_*`, KL head | translation CE + 0.10 · KL | Find a useful tag-embed subspace without disturbing pretrained Erya |
| **2** — encoder + adapters | + 12-layer encoder | + 0.05 · KL + 1.0 · denoising on monolingual | Encoder learns to *use* the tag-conditioned input |
| **3** — full model (optional) | everything | + 0.02 · KL | Decoder gently re-aligns to the new encoder representations |

KL distillation pulls a tiny linear head on the encoder's last hidden state to reproduce the SikuRoBERTa tag distributions, encouraging the encoder to retain tag information.

Configs: [Erya/configs/stage{1,2,3}*.yaml](Erya/configs/). Training entry point: [Erya/finetune/train.py](Erya/finetune/train.py).

## Repository layout

```
Erya/
  pytorch_model.bin             # Erya weights (downloaded from HF; gitignored)
  vocab.txt, config.json, ...   # Erya tokenizer + config
  augment_finetune_with_siku.py # Pre-compute SikuRoBERTa tag distributions
  finetune/                     # ← all our code lives here
    tag_adapter.py              # TaggedErya wrapper
    aux_dataset.py              # data loaders, sampler, denoise dataset
    losses.py                   # KL distillation head + loss
    siku_inference.py           # single-sentence Siku annotator (for eval)
    train.py                    # 3-stage training entry point
    eval.py                     # generation + BLEU + chrF + BERTScore
    utils.py, denoise.py        # param groups, ckpt I/O, span masking
  configs/                      # stage1/2/3 YAML configs
  scripts/                      # extract, runners, PACE-ICE + RunPod helpers
  dataset/                      # large data files (gitignored)
The-first-ancient-Chinese-word-segmentation-and-part-of-speech-tagging-code-and-analysis-main/
                                # vendored SikuRoBERTa tagger code + checkpoint
ACDS-master/                    # vendored ACDS reference
WordSeg-main/                   # vendored WordSeg reference
```

## Running it

You need a GPU with ≥ 24 GB VRAM (H100/A100 ideal), CUDA-enabled PyTorch ≥ 2.6, and a recent transformers.

### A. Local development (small smoke test)

```bash
pip install "torch>=2.6" torchvision "transformers>=4.30" "huggingface_hub[cli]>=0.30" \
            pytorch-crf safetensors sacrebleu pyyaml numpy bert_score
hf download RUCAIBox/Erya --local-dir Erya
bash Erya/scripts/extract_finetune_aux.sh
python -m Erya.finetune.train --config Erya/configs/stage1_adapters.yaml --stage 1 --dry_run
```

### B. PACE-ICE (Georgia Tech)

```bash
sbatch --export=ALL,STAGE=1 Erya/scripts/train_ice.sbatch
sbatch --export=ALL,STAGE=2,RESUME_ADAPTER=Erya/checkpoints/stage1/best/adapter.pt \
       --time=24:00:00 --mem=64G \
       Erya/scripts/train_ice.sbatch
sbatch --export=ALL,STAGE=3,RESUME=Erya/checkpoints/stage2/best/full.pt \
       --time=12:00:00 --mem=64G \
       Erya/scripts/train_ice.sbatch
```

### C. RunPod

```bash
# from your laptop
rsync -rlhtvP --no-owner --no-group --no-perms \
      --exclude='.git' --exclude='*.tgz' --exclude='*.bin' \
      --exclude='*.pth' --exclude='checkpoints*' \
      ./ runpod:/workspace/WenTrans/
rsync -avhP --partial --inplace --append-verify --info=progress2 \
      --no-owner --no-group --no-perms \
      Erya/dataset/finetune_with_siku_aux.tgz \
      runpod:/workspace/WenTrans/Erya/dataset/

# on the pod
ssh runpod
cd /workspace/WenTrans
bash Erya/scripts/runpod_setup.sh        # installs deps + downloads Erya + extracts data
tmux new -s erya
bash Erya/scripts/run_all_stages.sh      # all three stages, fail-fast, idempotent
```

Single-shot resumable runner: [Erya/scripts/run_all_stages.sh](Erya/scripts/run_all_stages.sh) — skips stages whose `best/*.pt` already exists, auto-passes `--resume_adapter` / `--resume`.

## Evaluation

```bash
python -m Erya.finetune.eval \
    --ckpt Erya/checkpoints/stage3/best \
    --src Erya/dataset/finetune_aux/dataset/hans/test.src \
    --ref Erya/dataset/finetune_aux/dataset/hans/test.tgt \
    --aux_npz Erya/dataset/finetune_aux/dataset/hans/test.src.siku_aux.npz \
    --output Erya/eval_outputs/hans.hyp.txt \
    --bertscore --bertscore_model bert-base-chinese
```

`--aux_npz` reuses the pre-computed SikuRoBERTa tag distributions for the test set (no Siku model loaded at inference). For ad-hoc inputs, drop `--aux_npz` and the script will load SikuRoBERTa via [Erya/finetune/siku_inference.py](Erya/finetune/siku_inference.py) on demand.

## Caveats and known issues

1. **`AutoModelForSeq2SeqLM` falls back to BART for CPT.** Erya's `config.json` declares `architectures: ["CPTForConditionalGeneration"]` but `model_type: "bart"`. Without [fnlp's CPT fork](https://github.com/fastnlp/CPT), HuggingFace loads the checkpoint as `BartForConditionalGeneration` — the **encoder** weights load (Bart-shaped) but the **decoder** weights don't (CPT uses a BERT-style decoder). Decoder is left randomly initialised. Stages 2 + 3 train it from scratch on the EvaHan parallel data, so the final model still works, but a vanilla `--ckpt None` zero-shot evaluation through this pipeline is meaningless. To compare honestly against published Erya numbers, vendor the CPT modeling file.
2. **`α_seg` and `α_pos` are domain-sensitive.** SikuRoBERTa was trained on the *Zuozhuan* corpus. Tag distributions on out-of-domain test sets (e.g. Ming-era texts) may shift; consider scaling α at inference or per-corpus-fine-tuning the adapter.
3. **Validation is BLEU-blind.** The trainer selects best checkpoints by validation cross-entropy on a 200-batch random subset, not BLEU. CE-best and BLEU-best can disagree.
4. **Stage 3 is risky.** Unfreezing the (BART-from-scratch) decoder at low LR after 24 k steps of frozen training can drift. Stage 2 best is usually a safer ship target if stage 3 BLEU regresses.

## Components and credits

- **[Erya](https://huggingface.co/RUCAIBox/Erya)** — pretrained Ancient → Modern Chinese translation model (RUCAIBox).
- **[SikuRoBERTa segmentation/POS tagger](The-first-ancient-Chinese-word-segmentation-and-part-of-speech-tagging-code-and-analysis-main/)** — character-level CRF tagger trained on the *Zuozhuan*; produces the soft labels we distil into the encoder.
- **[CPT (fnlp)](https://github.com/fastnlp/CPT)** — Chinese pretrained transformer with hybrid BART encoder + BERT decoder used by Erya.
- This repo wires the three together with a soft-tag adapter and a three-stage curriculum.

## License

Code in `Erya/finetune/` and `Erya/scripts/` is released under MIT. Model weights and vendored components retain their original licenses (Apache-2.0 for Erya; see component subdirectories).
