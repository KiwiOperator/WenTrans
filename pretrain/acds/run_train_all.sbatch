#!/bin/bash
# Phase 2 — three-stage training + relabel + eval, all in one SLURM job.
#
# Stage-1: M_0 -> M_1 on D_p   (~50 min on L40S, 20K sentences, 3 epochs)
# Stage-2: M_1 -> M_2 on D_a   (~10 min, 8K sentences, 3 epochs)
# Relabel: M_2 predicts D_p   (~10 min)
# Stage-3: M_0 -> M_3 on D_r   (~50 min)
# Eval:    on Test-A + Test-B
# Total budget: ~2 h.
#
#SBATCH -J wentrans-acds-train
#SBATCH -A coc
#SBATCH -p coc-gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ivany.mrinsect@gmail.com

set -euo pipefail
mkdir -p logs

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate wentrans
module load cuda/12.1 2>/dev/null || true

cd "$SLURM_SUBMIT_DIR"

M0="${M0:-pretrain/results}"             # WenTrans Stage-1 pretrained encoder
JSONL="${JSONL:-data/acds/jsonl}"
OUT_BASE="${OUT_BASE:-output/acds}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-3e-5}"
BS="${BS:-32}"

mkdir -p "$OUT_BASE"

# Common args used in all 3 training stages.
COMMON=(
  --labels_json "$JSONL/labels.json"
  --val_jsonl   "$JSONL/d_a_val.jsonl"
  --test_a_jsonl "$JSONL/test_a.jsonl"
  --test_b_jsonl "$JSONL/test_b.jsonl"
  --max_seq_length 512
  --per_device_train_batch_size "$BS"
  --per_device_eval_batch_size 64
  --learning_rate "$LR"
  --weight_decay 0.01
  --warmup_ratio 0.1
  --num_train_epochs "$EPOCHS"
  --evaluation_strategy epoch
  --save_strategy epoch
  --save_total_limit 1
  --load_best_model_at_end
  --metric_for_best_model wsg_f1
  --greater_is_better true
  --logging_steps 100
  --fp16
  --do_train --do_eval --do_predict
  --overwrite_output_dir
)

echo "===== Stage 1: M_0 -> M_1 on D_p ====="
torchrun --nproc_per_node=1 pretrain/acds/run_segpos.py \
  --model_name_or_path "$M0" \
  --train_jsonl "$JSONL/d_p.jsonl" \
  --output_dir  "$OUT_BASE/M1" \
  "${COMMON[@]}"

echo "===== Stage 2: M_1 -> M_2 on D_a ====="
torchrun --nproc_per_node=1 pretrain/acds/run_segpos.py \
  --model_name_or_path "$OUT_BASE/M1" \
  --train_jsonl "$JSONL/d_a_train.jsonl" \
  --output_dir  "$OUT_BASE/M2" \
  "${COMMON[@]}"

echo "===== Relabel: M_2 predicts D_p -> D_r ====="
python pretrain/acds/relabel.py \
  --model "$OUT_BASE/M2" \
  --in_jsonl "$JSONL/d_p.jsonl" \
  --out_jsonl "$JSONL/d_r.jsonl"

echo "===== Stage 3: M_0 -> M_3 on D_r (the final model) ====="
torchrun --nproc_per_node=1 pretrain/acds/run_segpos.py \
  --model_name_or_path "$M0" \
  --train_jsonl "$JSONL/d_r.jsonl" \
  --output_dir  "$OUT_BASE/M3" \
  "${COMMON[@]}"

echo "===== summary ====="
for stage in M1 M2 M3; do
  echo "--- $stage ---"
  for split in eval test_a test_b; do
    f="$OUT_BASE/$stage/${split}_results.json"
    if [[ -f "$f" ]]; then
      python -c "import json,sys; d=json.load(open('$f'));
print(' $split:', {k:round(v,4) for k,v in d.items() if k.endswith(('_f1','_precision','_recall'))})"
    fi
  done
done
