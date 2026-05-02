#!/bin/bash
# WenTrans Stage-2: fine-tune the pretrained encoder for Zuozhuan CWS.
# Replicates the recipe from Wang et al. 2021 §3.3 (BMES, AdamW, P/R/F).
#
#SBATCH -J wentrans-cws
#SBATCH -A coc
#SBATCH -p coc-gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1     # change to v100 / a40 / a100 / rtx_6000 if l40s queue is long
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ivany.mrinsect@gmail.com

set -euo pipefail
mkdir -p logs

# user's own conda (avoids the system anaconda3 module pointing at a different root)
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "[wentrans] cannot locate conda init script under \$HOME" >&2; exit 1
fi
conda activate wentrans
module load cuda/12.1 2>/dev/null || true

cd "$SLURM_SUBMIT_DIR"

# Default model: Stage-1 pretrain output. Override at submit-time with:
#   sbatch --export=ALL,MODEL=hfl/chinese-roberta-wwm-ext pretrain/run_cws.sbatch
MODEL="${MODEL:-pretrain/results}"
OUT="${OUT:-output/cws-wentrans}"
DATA="${DATA:-data/cws}"

if [[ ! -s "$DATA/train.jsonl" ]]; then
  echo "[wentrans] $DATA/train.jsonl missing. Run prepare_cws.py first." >&2
  exit 2
fi

mkdir -p "$OUT"

# Auto-resume from latest checkpoint if one exists.
RESUME=""
if compgen -G "$OUT/checkpoint-*" > /dev/null; then
  RESUME="--resume_from_checkpoint $OUT"
  echo "[wentrans] resuming CWS training from latest checkpoint in $OUT"
fi

# Hyperparameters: paper §3.3 keeps these consistent across all 5 pretrains.
# The values below are the BERT-CWS-on-Zuozhuan recipe that lands in the
# paper's reported F-score range (~88-89% for RoBERTa-class encoders).
torchrun --nproc_per_node=1 pretrain/run_cws.py \
  --model_name_or_path "$MODEL" \
  --data_dir "$DATA" \
  --output_dir "$OUT" \
  --max_seq_length 512 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 64 \
  --learning_rate 3e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --num_train_epochs 10 \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 3 \
  --load_best_model_at_end \
  --metric_for_best_model f1 \
  --greater_is_better true \
  --logging_steps 50 \
  --fp16 \
  --do_train --do_eval --do_predict $RESUME
