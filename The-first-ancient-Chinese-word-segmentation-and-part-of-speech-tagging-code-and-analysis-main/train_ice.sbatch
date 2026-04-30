#!/bin/bash
#SBATCH -J anc_posseg
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:H100:1
#SBATCH -t 04:00:00
#SBATCH -o anc_posseg-%j.out
#SBATCH -e anc_posseg-%j.err

set -euo pipefail

if [[ -f "$SLURM_SUBMIT_DIR/train.py" ]]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR="$SLURM_SUBMIT_DIR/The-first-ancient-Chinese-word-segmentation-and-part-of-speech-tagging-code-and-analysis-main"
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
echo "CUDA devices:"
nvidia-smi

echo "Installing Python requirements into user site-packages"
python -m pip install --user -r requirements.txt

python - <<'PY'
import importlib
missing = []
for name in ("torch", "transformers", "torchcrf", "numpy", "tqdm"):
    try:
        importlib.import_module(name)
    except Exception:
        missing.append(name)
if missing:
    raise SystemExit("Missing Python packages: " + ", ".join(missing))
print("Python dependencies look available.")
PY

python train.py

echo "Job finished at $(date)"
