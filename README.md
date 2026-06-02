# WenTrans — A CWS-Powered Ancient → Modern Chinese Translator

> **CS 4650 — Spring 2026 — Hanzhi Li, William X. Gao**

WenTrans is a two-stage Ancient → Modern Chinese translation pipeline that uses **Chinese Word Segmentation (CWS)** as an auxiliary signal to a neural machine translation model. We first build a SOTA-class CWS / POS tagger for ancient Chinese on top of `chinese-roberta-wwm-ext`, then distil that tagger's per-character segmentation and POS distributions into the encoder of the Erya translation model via a small **soft tag adapter**.

---

## Authors and contributions

| Person | Component | Branch | Highlights |
|---|---|---|---|
| **Hanzhi Li** | CWS / POS-tagging model | [`cws`](https://github.com/KiwiOperator/WenTrans/tree/cws) | Pre-training on 600 M classical-Chinese characters from DaizhiGeV2 (~14 h on PACE L40S, eval perplexity 15.69); BMES fine-tune on EvaHan; distant-supervision augmentation via GIZA++ IBM-4 alignment with three-stage M₁ → M₂ → M₃ noise correction; **F1 = 95.71 %** on EvaHan 2022 Test-A |
| **William X. Gao** | Soft-tag-adapter translation model | [`translation`](https://github.com/KiwiOperator/WenTrans/tree/translation) | SikuRoBERTa-CRF dataset augmentation (per-character soft seg/POS distributions, 12.7 GB); `TaggedErya` wrapper (~75k trainable params); three-stage training curriculum on H100; ablation across all six EvaHan corpora; **+5.5 BLEU** from soft-tag adapter on `hans` |

The two halves were developed in parallel on separate branches and are intended to be combined: the next step is to drop the EvaHan-trained `SikuRoBERTa-CRF` checkpoint used by the translation pipeline and substitute Hanzhi's M₂ tagger.

[Read the paper](src/paper.pdf)

---

## Pipeline overview

```
 ┌─────────────────────────────────────────────────────────────────────────┐
 │  Stage 1: Pre-training         Stage 2: CWS fine-tuning                 │
 │  (chinese-roberta-wwm-ext +    (EvaHan BMES fine-tune)                  │
 │   600 M chars from             ──→ F1 95.09 %                           │
 │   DaizhiGeV2, MLM)                                                      │
 │                                                                         │
 │       │                                                                 │
 │       ▼                                                                 │
 │  Stage 3: Distant-supervision augmentation                              │
 │  (GIZA++ IBM-4 ancient↔modern alignment, 29→22 POS map,                 │
 │   M₁ noisy → M₂ corrected → M₃ relabeled)  ──→ F1 95.71 %               │
 │                                                                         │
 └────────────────────────────┬────────────────────────────────────────────┘
                              │
                              ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │  Translation pipeline (TaggedErya)                                      │
 │                                                                         │
 │   ancient tokens ──→ token embeds                                       │
 │                       + tanh(α_seg) · (seg_probs @ E_seg)               │
 │                       + tanh(α_pos) · (pos_probs @ E_pos)               │
 │                       ──→ Erya encoder ──→ decoder ──→ modern Chinese   │
 │                                                                         │
 │   Three-stage curriculum:                                               │
 │      1. adapters only          (~75 k params)                           │
 │      2. + encoder              (~125 M params)                          │
 │      3. + full model           (~410 M params)                          │
 └─────────────────────────────────────────────────────────────────────────┘
```

---

## Headline results

### CWS / POS tagger (`cws` branch)

| Test setup | F1 |
|---|---|
| SikuRoBERTa 2021 (self-administered Zuǒzhuàn CWS) | 88.88 % |
| Feng & Li 2023 baseline (EvaHan 2022 Test-A) | 94.73 % |
| Feng & Li 2023 M₃ full pipeline (EvaHan 2022 Test-A) | 95.64 % |
| **Ours, Stage 2 only (10 % holdout, in-domain)** | 95.09 % |
| **Ours, Stage 2 (M₂-equivalent) on EvaHan 2022 Test-A** | **95.71 %** |

Pre-training pseudo-perplexity (lower = better):

| Passage | Base `RoBERTa-wwm-ext` | **Ours** | SikuRoBERTa |
|---|---:|---:|---:|
| classical_zz (Zuǒzhuàn) | 2,257.28 | **51.45** | **24.35** |
| classical_lunyu (Analects) | **111.31** | 442.67 | 853.89 |
| classical_mengzi (Mencius) | **2.16** | 16.14 | 18.36 |
| modern_zh | **3.04** | 34.24 | 268.36 |

### Translation model (`translation` branch)

BLEU on EvaHan test splits:

| Model | Book of Han | Ming History | Taiping Guangji | Xu Xiake | Tang History |
|---|---:|---:|---:|---:|---:|
| gpt-3.5-turbo | 17.7 | 21.5 | 16.8 | 20.1 | 20.5 |
| Erya (published) | 29.9 | 37.1 | 24.1 | 34.2 | 34.5 |
| **TaggedErya (ours)** | **28.5** | **36.7** | **22.0** | **33.0** | **35.3** |

TaggedErya holds up against the SOTA Erya baseline and **surpasses it on the Tang History (Xīn Táng Shū) split** (+0.8 BLEU). The α=0 ablation confirms a robust **+5.5 BLEU mean lift** from the soft tag adapter alone.

---

## Repository layout

```
WenTrans/
├── README.md                    ← (this file)
├── Erya/
│   ├── README.md                ← detailed README for the translation half
│   ├── report.md                ← full project report (paper-style)
│   ├── README.modelcard.md      ← original Erya HF model card
│   ├── pytorch_model.bin        ← Erya weights (downloaded from HF; gitignored)
│   ├── augment_finetune_with_siku.py
│   ├── finetune/                ← TaggedErya code
│   │   ├── tag_adapter.py
│   │   ├── aux_dataset.py
│   │   ├── losses.py
│   │   ├── siku_inference.py
│   │   ├── train.py
│   │   ├── eval.py
│   │   └── utils.py, denoise.py
│   ├── configs/                 ← stage1/2/3 YAML configs
│   ├── scripts/                 ← extract, run-all, PACE-ICE + RunPod helpers
│   └── dataset/                 ← parallel + monolingual data (gitignored)
├── The-first-ancient-Chinese-word-segmentation-and-part-of-speech-tagging-code-and-analysis-main/
│                                ← vendored SikuRoBERTa tagger code + checkpoint
├── ACDS-master/, WordSeg-main/  ← vendored CWS reference implementations
└── datasets/                    ← raw data sources
```

The CWS model (Hanzhi's half) lives on the **`cws`** branch — it has its own README, configs, and training scripts. This `README.md`, `Erya/README.md`, and `Erya/report.md` describe the **`translation`** branch (William's half).

---

## Running the project

The two branches build different artifacts; switch branches based on what you want to reproduce.

### Prerequisites

- A GPU with ≥ 24 GB VRAM (H100 / A100 / L40S all work).
- CUDA-enabled PyTorch ≥ 2.6, transformers ≥ 4.30, `huggingface_hub`, `pytorch-crf`, `sacrebleu`, `bert_score`.
- ~30 GB disk for the augmented dataset + checkpoints.

### A. CWS / POS tagger (Hanzhi, branch `cws`)

```bash
git checkout cws
# follow the README on that branch — pre-training on DaizhiGeV2, BMES fine-tune
# on EvaHan, then distant-supervision augmentation. Produces an
# AutoModelForTokenClassification checkpoint scoring F1 = 95.71 % on
# EvaHan 2022 Test-A.
```

### B. Translation model (William, branch `translation` — current branch)

The translation pipeline is fully scripted under `Erya/scripts/`. Two paths:

#### B.1. PACE-ICE (Georgia Tech)

```bash
sbatch --export=ALL,STAGE=1 Erya/scripts/train_ice.sbatch
sbatch --export=ALL,STAGE=2,RESUME_ADAPTER=Erya/checkpoints/stage1/best/adapter.pt \
       --time=24:00:00 --mem=64G \
       Erya/scripts/train_ice.sbatch
sbatch --export=ALL,STAGE=3,RESUME=Erya/checkpoints/stage2/best/full.pt \
       --time=12:00:00 --mem=64G \
       Erya/scripts/train_ice.sbatch
```

#### B.2. RunPod (or any cloud GPU pod)

```bash
# from your laptop — push code + the 12 GB augmented dataset
rsync -rlhtvP --no-owner --no-group --no-perms \
      --exclude='.git' --exclude='*.tgz' --exclude='*.bin' --exclude='*.pth' \
      --exclude='checkpoints*' --exclude='__pycache__' \
      ./ runpod:/workspace/WenTrans/
rsync -avhP --partial --inplace --append-verify --info=progress2 \
      --no-owner --no-group --no-perms \
      Erya/dataset/finetune_with_siku_aux.tgz \
      runpod:/workspace/WenTrans/Erya/dataset/

# on the pod — installs deps, downloads Erya from HF, extracts data
ssh runpod
cd /workspace/WenTrans
bash Erya/scripts/runpod_setup.sh

# run the full three-stage curriculum end-to-end
tmux new -s erya
bash Erya/scripts/run_all_stages.sh
```

`run_all_stages.sh` is idempotent: stages whose `best/*.pt` already exist on disk are skipped, so a crash mid-pipeline can be resumed by re-running the same command.

#### B.3. Evaluation

```bash
python -m Erya.finetune.eval \
    --ckpt Erya/checkpoints/stage3/best \
    --src Erya/dataset/finetune_aux/dataset/hans/test.src \
    --ref Erya/dataset/finetune_aux/dataset/hans/test.tgt \
    --aux_npz Erya/dataset/finetune_aux/dataset/hans/test.src.siku_aux.npz \
    --output Erya/eval_outputs/hans.hyp.txt \
    --bertscore --bertscore_model bert-base-chinese
```

`--aux_npz` reuses pre-computed SikuRoBERTa tag distributions for the test set; drop it and the script will load SikuRoBERTa via `Erya/finetune/siku_inference.py` to annotate inputs on demand.

For step-by-step deep dives (architecture, training stages, results, ablations), see [Erya/README.md](Erya/README.md) and the full report at [Erya/report.md](Erya/report.md).


## Citation

If you use this code, please cite the project paper:

> Hanzhi Li and William X. Gao. *WenTrans: A CWS Powered Ancient-Modern Chinese Translator.* CS 4650, Spring 2026.

For the components we build on, see references [1]–[14] in [Erya/report.md](Erya/report.md), in particular:
- Erya translation model — Guo et al. 2023 ([arXiv:2308.00240](https://arxiv.org/abs/2308.00240))
- SikuRoBERTa / SikuGPT — Chang et al. 2023 ([arXiv:2304.07778](https://arxiv.org/abs/2304.07778))
- Distant-supervision CWS — Feng & Li 2023 ([arXiv:2303.01912](https://arxiv.org/abs/2303.01912))
- Chinese RoBERTa with whole-word masking — Cui et al. 2021 ([IEEEACM TASLP](https://doi.org/10.1109/TASLP.2021.3124365))

---

## License

Code in `Erya/finetune/` and `Erya/scripts/` is released under MIT. Model weights and vendored components retain their original licenses (Apache-2.0 for Erya; see component subdirectories).
