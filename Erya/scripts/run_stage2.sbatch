#!/bin/bash
#SBATCH --job-name=erya-stage2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/stage2_%j.out
#SBATCH --error=logs/stage2_%j.err

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

mkdir -p logs

python -m Erya.finetune.train \
  --config Erya/configs/stage2_encoder.yaml \
  --stage 2 \
  --resume_adapter Erya/checkpoints/stage1/best/adapter.pt \
  --output_dir Erya/checkpoints/stage2
