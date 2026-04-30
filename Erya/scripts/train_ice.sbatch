#!/bin/bash
# Generic PACE-ICE launcher for the three-stage Erya tag-adapter fine-tune.
#
# Submit a stage by overriding STAGE/CONFIG/RESUME via --export. Examples:
#
#   # Stage 1 (adapters only)
#   sbatch --export=ALL,STAGE=1 \
#          Erya/scripts/train_ice.sbatch
#
#   # Stage 2 (adapters + encoder), resuming from stage 1 best
#   sbatch --export=ALL,STAGE=2,RESUME_ADAPTER=Erya/checkpoints/stage1/best/adapter.pt \
#          --time=24:00:00 --mem=64G \
#          Erya/scripts/train_ice.sbatch
#
#   # Stage 3 (full model), resuming from stage 2 best
#   sbatch --export=ALL,STAGE=3,RESUME=Erya/checkpoints/stage2/best/full.pt \
#          --time=12:00:00 --mem=64G \
#          Erya/scripts/train_ice.sbatch
#
# Per-stage SLURM resource overrides (time, mem, gres) can be appended to sbatch
# directly, as shown above; the defaults below are sized for stage 1.

#SBATCH -J erya_tag_ft
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:H100:1
#SBATCH -t 12:00:00
#SBATCH -o erya_tag_ft-%j.out
#SBATCH -e erya_tag_ft-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wgao305@gatech.edu

set -euo pipefail

# ---------------- arguments (via --export=ALL,STAGE=...,...) ----------------
STAGE="${STAGE:-1}"

case "${STAGE}" in
    1) DEFAULT_CONFIG="Erya/configs/stage1_adapters.yaml" ;;
    2) DEFAULT_CONFIG="Erya/configs/stage2_encoder.yaml"  ;;
    3) DEFAULT_CONFIG="Erya/configs/stage3_full.yaml"     ;;
    *) echo "STAGE must be 1, 2, or 3 (got '${STAGE}')" >&2; exit 1 ;;
esac

CONFIG="${CONFIG:-${DEFAULT_CONFIG}}"
OUTPUT_DIR="${OUTPUT_DIR:-Erya/checkpoints/stage${STAGE}}"
RESUME="${RESUME:-}"
RESUME_ADAPTER="${RESUME_ADAPTER:-}"

# ---------------- locate project root ----------------
# Submit from either the repo root (which contains Erya/) or from inside Erya/.
if [[ -d "${SLURM_SUBMIT_DIR}/Erya/finetune" ]]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
elif [[ -d "${SLURM_SUBMIT_DIR}/finetune" ]]; then
    PROJECT_ROOT="$(dirname "${SLURM_SUBMIT_DIR}")"
else
    echo "ERROR: cannot find Erya/finetune relative to SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}" >&2
    exit 1
fi
cd "${PROJECT_ROOT}"

# ---------------- environment ----------------
module purge
module load pytorch/2.1.0

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=0
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

echo "============================================================"
echo "Job:          ${SLURM_JOB_ID:-?} on $(hostname) at $(date)"
echo "Project root: ${PROJECT_ROOT}"
echo "Stage:        ${STAGE}"
echo "Config:       ${CONFIG}"
echo "Output dir:   ${OUTPUT_DIR}"
echo "Resume:       ${RESUME:-<none>}"
echo "Resume adapt: ${RESUME_ADAPTER:-<none>}"
echo "GPU:"
nvidia-smi
echo "============================================================"

# ---------------- python deps ----------------
echo "Installing Python requirements into user site-packages ..."
python -m pip install --user --quiet \
    numpy \
    pyyaml \
    transformers \
    pytorch-crf \
    sacrebleu

python - <<'PY'
import importlib
missing = []
for name in ("torch", "numpy", "yaml", "transformers", "torchcrf", "sacrebleu"):
    try:
        importlib.import_module(name)
    except Exception:
        missing.append(name)
if missing:
    raise SystemExit("Missing Python packages: " + ", ".join(missing))
print("Python dependencies look available.")
PY

# ---------------- data preflight ----------------
EXTRACT_DIR="Erya/dataset/finetune_aux"
if [[ ! -d "${EXTRACT_DIR}" ]] || [[ -z "$(ls -A "${EXTRACT_DIR}" 2>/dev/null)" ]]; then
    echo "Extracting finetune_with_siku_aux.tgz -> ${EXTRACT_DIR} (one-off) ..."
    bash Erya/scripts/extract_finetune_aux.sh
else
    echo "${EXTRACT_DIR} already populated; skipping extraction."
fi

# ---------------- launch ----------------
RESUME_ARGS=()
if [[ -n "${RESUME}" ]];          then RESUME_ARGS+=("--resume" "${RESUME}"); fi
if [[ -n "${RESUME_ADAPTER}" ]];  then RESUME_ARGS+=("--resume_adapter" "${RESUME_ADAPTER}"); fi

set -x
python -m Erya.finetune.train \
    --config "${CONFIG}" \
    --stage "${STAGE}" \
    --output_dir "${OUTPUT_DIR}" \
    "${RESUME_ARGS[@]}"
set +x

echo "Job finished at $(date)"
