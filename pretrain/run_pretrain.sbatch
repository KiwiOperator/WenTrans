#!/bin/bash
# WenTrans Stage-1: continue MLM pretraining of chinese-roberta-wwm-ext on
# the daizhigev20 ancient-Chinese corpus. Target cluster: Georgia Tech PACE.
#
# === EDIT THESE THREE LINES BEFORE FIRST SUBMISSION ===
#   1. -A <charge-account>      (your PACE charge account, e.g. gts-yourPI)
#   2. -p <partition>           (whichever GPU partition you can access)
#   3. --mail-user              (or remove the mail lines)
# ======================================================
#
#SBATCH -J wentrans-mlm
#SBATCH -A coc
#SBATCH -p coc-gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia     # any NVIDIA card on coc-gpu (L40S/V100/A40/A100/RTX6000); excludes MI210
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 16:00:00
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ivany.mrinsect@gmail.com

# For multi-GPU on the same node:
#   #SBATCH --gres=gpu:A100:4
#   and change --nproc_per_node below to match.

set -euo pipefail

mkdir -p logs

# Initialise the user's own conda (NOT the system anaconda3 module — that
# would point at a different conda root with no 'wentrans' env).
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "[wentrans] cannot locate conda init script under \$HOME" >&2
  exit 1
fi
conda activate wentrans

# CUDA: take whatever the partition exposes; only load a module if needed.
module load cuda/12.1 2>/dev/null || true

cd "$SLURM_SUBMIT_DIR"

OUT=output/wentrans-roberta-ancient-zh
mkdir -p "$OUT"

# Auto-resume: if any checkpoint-* dir exists under $OUT, the trainer will
# pick up the latest one. We pass the output dir; HF resolves it.
RESUME=""
if compgen -G "$OUT/checkpoint-*" > /dev/null; then
  RESUME="--resume_from_checkpoint $OUT"
  echo "[wentrans] resuming from latest checkpoint in $OUT"
else
  echo "[wentrans] no checkpoint found; starting from hfl/chinese-roberta-wwm-ext"
fi

# Make sure data has been prepared (run prepare_corpus.py once on a login node):
#   python pretrain/prepare_corpus.py --src daizhigev20-master --out data --max_chars 600000000
if [[ ! -s data/train.txt || ! -s data/val.txt ]]; then
  echo "[wentrans] data/train.txt or data/val.txt missing. Run prepare_corpus.py first." >&2
  exit 2
fi

# Wall-clock budget: SLURM kills at 16:00:00. We tell the Trainer to stop
# gracefully at 15:30:00 (55800s) so final save + eval has 30 min of headroom.
torchrun --nproc_per_node=1 pretrain/run_mlm.py \
  --model_name_or_path hfl/chinese-roberta-wwm-ext \
  --train_file data/train.txt \
  --validation_file data/val.txt \
  --line_by_line \
  --max_seq_length 512 \
  --mlm_probability 0.15 \
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.06 \
  --num_train_epochs 1 \
  --save_strategy steps \
  --save_steps 2000 \
  --save_total_limit 5 \
  --evaluation_strategy steps \
  --eval_steps 2000 \
  --logging_steps 200 \
  --fp16 \
  --gradient_checkpointing \
  --max_train_seconds 55800 \
  --output_dir "$OUT" \
  --do_train --do_eval $RESUME
