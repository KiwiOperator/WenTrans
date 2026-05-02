# Stage 3 — ACDS: distant-supervision augmentation for ancient Chinese WSG+POS

Replicates Feng & Li, ICASSP 2023 ([arXiv 2303.01912](https://arxiv.org/abs/2303.01912))
with one deliberate simplification: hybrid **88-tag classifier head** instead of
the paper's dual-head + 2-CRF architecture. The paper's main contribution is the
**3-stage relabeling**, which we faithfully reproduce.

## Pipeline at a glance

```
NiuTrans Classical-Modern bitext  -->  src/tgt pairs  -->  shuffle subset (20K)
                                                                |
                          [LTP cws+pos on modern] +  [char-split ancient]
                                                                |
                                            GIZA++ IBM-4 alignment
                                                                |
                                  POS map + adjacency merge -->  D_p (large noisy)
                                                                |
                              -----------------------------------
                              |          |          |          |
                              v          v          v          v
                       D_a (clean)   Test-A    Test-B     all -> JSONL

  M_0 (Stage-1 WenTrans encoder)
     |
     |---> train on D_p (3 ep)            ===>   M_1
                                                  |
                                                  |---> continue on D_a (3 ep)  ===> M_2
                                                                                       |
                                                            relabel D_p with M_2 ===>  D_r
                                                                                       |
     |---> train on D_r (3 ep)                                              ===>   M_3 (final)
```

## Time budget (~4 h total on 1×L40S, 20 K subset)

| Phase | Time |
|---|---|
| `setup.sh` (one-time mgiza build, LTP install) | ~15 min |
| `run_data_prep.sbatch` | ~75 min |
| `run_train_all.sbatch` (Stage 1+2+relabel+Stage 3+eval) | ~120 min |

## Files in this folder

| File | Phase |
|---|---|
| `setup.sh` | one-time install: LTP, opencc, giza-py, mgiza |
| `01_make_data.py` | NiuTrans `bitext.txt` blocks → `src.txt` + `tgt.txt` |
| `02_shuffle_subset.py` | seeded shuffle, take 20 K pairs |
| `03_ltp_segpos.py` | LTP segments + POS-tags modern; char-splits ancient |
| `04_align_giza.sh` | giza-py IBM-4 alignment (`--include-probs`) |
| `05_project_labels.py` | POS map + adjacency merge → `tgt_shuf_segpos.txt` (D_p) |
| `06_prepare_jsonl.py` | text files → JSONL with hybrid BMES+POS tags + label vocab |
| `run_segpos.py` | HF Trainer wrapper, hybrid 88-tag classifier, separate WSG / POS F1 |
| `relabel.py` | M_2 predicts D_p → D_r |
| `run_data_prep.sbatch` | SLURM wrapper for steps 01–06 |
| `run_train_all.sbatch` | SLURM wrapper for the 3 training stages + relabel + eval |

## Workflow on PACE

### 0. One-time setup
```bash
bash pretrain/acds/setup.sh
# verify
ls $HOME/giza-py/mgiza/mgizapp/bin/mgiza
python -c "from ltp import LTP; print('LTP ok')"
python -c "from opencc import OpenCC; print('opencc ok')"
```

### 1. Data
You said the parallel corpus is already on PACE. Tell the sbatch where:
```bash
# defaults assume:
#   $WORK_DIR/Classical-Modern/双语数据   (NiuTrans bitext)
#   $WORK_DIR/acds-repo/                 (clone of farlit/acds — has EvaHan + zuozhuan files)
# if your paths differ, override at submit time (see below)

# clone the small acds repo just for its test/zuozhuan files (~2 MB)
git clone https://github.com/farlit/acds.git acds-repo

mkdir -p data/acds
```

### 2. Run data prep (~75 min)
```bash
sbatch pretrain/acds/run_data_prep.sbatch
# or with overrides:
sbatch --export=ALL,BITEXT=/path/to/Classical-Modern/双语数据,ACDS_REPO=/path/to/acds,N_PAIRS=20000 \
       pretrain/acds/run_data_prep.sbatch
```
Watch `logs/wentrans-acds-prep.<jobid>.out`. It logs each phase header in order.

### 3. Run training + relabel + eval (~2 h)
```bash
sbatch pretrain/acds/run_train_all.sbatch
```
Final summary section at the bottom of the log prints WSG-F1 / POS-F1 for each
of M_1, M_2, M_3 on **eval** (in-domain D_a held-out), **test_a** (Zuozhuan
out-of-distribution), and **test_b** (Shiji + Zizhitongjian out-of-distribution).

## What "success" looks like

The paper reports (Table 2):

| Model | Test-A WSG | Test-A POS | Test-B WSG | Test-B POS |
|---|---|---|---|---|
| baseline (D_a only) | 94.73 | 90.93 | 89.19 | 83.48 |
| **M_3 (theirs, 967 K)** | **95.64** | 90.55 | **93.64** | **86.21** |

We're using **20 K pairs** (~2% of theirs) and a simpler head. Realistic
expectations:

- **M_2 ≥ Stage-2 CWS baseline** on Test-A (the paper's small-clean-data ceiling)
- **M_3 > M_2 on Test-B** — biggest gain expected on out-of-distribution data,
  since that's where distant-supervision augmentation provides the most lift.
  Even a 1–2 F point lift confirms the relabeling idea is working on our scale.

## Hyperparameters (vs paper)

| | Paper | This repo |
|---|---|---|
| Subset of D_p | 967,257 | 20,000 |
| Encoder | SikuRoBERTa | our `pretrain/results` |
| Classifier head | dual (seg + pos) + 2 CRFs | single hybrid (88 labels), no CRF |
| LR / opt / wd | 1e-5 / AdamW / 0.01 | 3e-5 / AdamW / 0.01 |
| Batch | 24 | 32 |
| Epochs (each stage) | 20 | 3 |
| Max seq len | 512 | 512 |
| Aligner | GIZA++ IBM-4 | GIZA++ IBM-4 (paper-faithful) |

## Common gotchas

- **mgiza build fails on PACE**: typically a missing `boost` dev header. Try
  `module load gcc/12.1.0 boost/1.79.0` (names vary) before re-running `setup.sh`.
- **LTP downloads ~500 MB on first run**. Cached under `~/.cache/torch/ltp/`.
- **Empty `alignment.txt` lines**: a few sentences may yield no alignments
  (e.g. modern len ≪ ancient). `05_project_labels.py` writes them as
  all-`null` and `06_prepare_jsonl.py` filters too-short ones via `MIN_LEN=2`.
- **Test set has labels not in the train vocab**: `run_segpos.py` maps unknown
  tag IDs to `-100` (ignored in loss). Eval works fine; very rare token-level
  losses at most.

## Comparing apples-to-apples with the paper

If after the 20 K-subset run M_3 is still well below the paper, the most
likely fixes (in order of effort):
1. Increase `N_PAIRS` to 100K–200K (re-submit `run_data_prep.sbatch` with
   `--export=ALL,N_PAIRS=200000` and bump SLURM time to 4–5 h).
2. Add a CRF layer on top of the classifier head (drop `pytorch-crf`, ~50 lines
   change in `run_segpos.py`).
3. Switch to dual-head (separate seg + pos classifiers + 2 CRFs).
