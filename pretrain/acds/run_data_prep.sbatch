#!/bin/bash
# Phase 1 — data prep on PACE.
# Pipeline: bitext -> src/tgt -> shuffle/subset -> LTP -> GIZA++ -> project -> JSONL.
#
# Time budget on 1 GPU (L40S/V100/A40):
#   LTP on 20K modern sentences:    ~30 min
#   giza-py IBM-4 on 20K pairs:     ~30-45 min  (depends on CPU speed)
#   projection + JSONL:             ~5 min
# Total target: ~75 min.
#
#SBATCH -J wentrans-acds-prep
#SBATCH -A coc
#SBATCH -p coc-gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 02:30:00
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

# === paths you can override at sbatch time via --export=ALL,VAR=value ===
BITEXT="${BITEXT:-Classical-Modern/双语数据}"   # NiuTrans repo's parallel-data dir
ACDS_REPO="${ACDS_REPO:-acds-repo}"             # cloned farlit/acds (for test files + zuozhuan)
WORK="${WORK:-data/acds/work}"                   # intermediate artefacts go here
JSONL="${JSONL:-data/acds/jsonl}"                # final JSONL outputs
N_PAIRS="${N_PAIRS:-20000}"
GIZAPY="${GIZAPY:-$HOME/giza-py}"

mkdir -p "$WORK" "$JSONL"

echo "===== 01: bitext -> src/tgt ====="
python pretrain/acds/01_make_data.py --bitext "$BITEXT" --out "$WORK"

echo "===== 02: shuffle + subset (n=$N_PAIRS) ====="
python pretrain/acds/02_shuffle_subset.py --in_dir "$WORK" --out_dir "$WORK" --n "$N_PAIRS"

echo "===== 03: LTP segment + POS ====="
python pretrain/acds/03_ltp_segpos.py --io_dir "$WORK"

echo "===== 04: GIZA++ alignment ====="
IO_DIR="$WORK" GIZAPY="$GIZAPY" bash pretrain/acds/04_align_giza.sh

echo "===== 05: project labels -> tgt_shuf_segpos.txt (D_p) ====="
python pretrain/acds/05_project_labels.py --io_dir "$WORK"

echo "===== 06: build JSONLs ====="
python pretrain/acds/06_prepare_jsonl.py \
  --d_p    "$WORK/tgt_shuf_segpos.txt" \
  --d_a    "$ACDS_REPO/zuozhuan_train_utf8.txt" \
  --test_a "$ACDS_REPO/EvaHan_testa_gold.txt" \
  --test_b "$ACDS_REPO/EvaHan_testb_gold.txt" \
  --out_dir "$JSONL"

echo "===== done ====="
ls -lh "$JSONL"
