#!/bin/bash
#SBATCH --job-name=erya-stage1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/stage1_%j.out
#SBATCH --error=logs/stage1_%j.err

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

mkdir -p logs

python -m Erya.finetune.train \
  --config Erya/configs/stage1_adapters.yaml \
  --stage 1 \
  --output_dir Erya/checkpoints/stage1
