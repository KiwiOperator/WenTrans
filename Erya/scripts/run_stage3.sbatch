#!/bin/bash
#SBATCH --job-name=erya-stage3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/stage3_%j.out
#SBATCH --error=logs/stage3_%j.err

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

mkdir -p logs

python -m Erya.finetune.train \
  --config Erya/configs/stage3_full.yaml \
  --stage 3 \
  --resume Erya/checkpoints/stage2/best/full.pt \
  --output_dir Erya/checkpoints/stage3
