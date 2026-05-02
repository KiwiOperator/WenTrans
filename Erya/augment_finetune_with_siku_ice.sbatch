#!/bin/bash
#SBATCH -J erya_siku_aux
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:H100:1
#SBATCH -t 04:00:00
#SBATCH -o erya_siku_aux-%j.out
#SBATCH -e erya_siku_aux-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wgao305@gatech.edu

set -euo pipefail

if [[ -f "$SLURM_SUBMIT_DIR/augment_finetune_with_siku.py" ]]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR="$SLURM_SUBMIT_DIR/Erya"
fi
cd "$PROJECT_DIR"

module purge
module load pytorch/2.1.0

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=0

echo "Job started on $(hostname) at $(date)"
echo "Working directory: $PWD"
nvidia-smi

echo "Installing Python requirements into user site-packages"
python -m pip install --user \
  numpy \
  transformers \
  pytorch-crf

python - <<'PY'
import importlib
missing = []
for name in ("torch", "numpy", "transformers", "torchcrf"):
    try:
        importlib.import_module(name)
    except Exception:
        missing.append(name)
if missing:
    raise SystemExit("Missing Python packages: " + ", ".join(missing))
print("Python dependencies look available.")
PY

python augment_finetune_with_siku.py \
  --input dataset/finetune.tgz \
  --output dataset/finetune_with_siku_aux.tgz \
  --siku-dir ../The-first-ancient-Chinese-word-segmentation-and-part-of-speech-tagging-code-and-analysis-main \
  --batch-size 64 \
  --save-dtype float16

echo "Job finished at $(date)"
