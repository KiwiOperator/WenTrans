#!/usr/bin/env bash
# One-shot setup for running Erya tag-adapter fine-tuning on a RunPod pod.
#
# Idempotent: re-running is safe; each step is skipped if already done.
#
# Steps:
#   1. install Python deps (transformers, pytorch-crf, sacrebleu, pyyaml, huggingface_hub)
#   2. fetch Erya weights from HuggingFace (RUCAIBox/Erya) if pytorch_model.bin missing
#   3. fetch finetune_with_siku_aux.tgz from a private HF dataset (optional, gated by env)
#   4. extract finetune_with_siku_aux.tgz -> Erya/dataset/finetune_aux/
#   5. import-check torch/transformers/sacrebleu/torchcrf
#   6. print next steps
#
# Optional environment variables:
#   WORKDIR             : repo root (default: parent of this script)
#   HF_DATASET_REPO     : private HF dataset to pull aux tar from
#                         (e.g. "your-username/erya-finetune-siku-aux")
#   HF_DATASET_FILENAME : default "finetune_with_siku_aux.tgz"
#   HF_TOKEN            : HuggingFace API token; needed for private datasets
#   SKIP_DEPS=1         : skip pip install
#   SKIP_DOWNLOAD=1     : skip Erya / aux dataset downloads
#   SKIP_EXTRACT=1      : skip tar extraction
set -euo pipefail

# ---- locate repo root --------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_WORKDIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKDIR="${WORKDIR:-${DEFAULT_WORKDIR}}"
ERYA_DIR="${WORKDIR}/Erya"

cd "${WORKDIR}"
echo "============================================================"
echo "RunPod setup for Erya tag-adapter fine-tuning"
echo "============================================================"
echo "Workdir:   ${WORKDIR}"
echo "Erya dir:  ${ERYA_DIR}"
echo "Started:   $(date)"
echo

if [[ ! -d "${ERYA_DIR}/finetune" ]]; then
    echo "ERROR: ${ERYA_DIR}/finetune not found. Did you upload the code?" >&2
    echo "Expected layout: ${WORKDIR}/Erya/finetune/, ${WORKDIR}/Erya/configs/, ..." >&2
    exit 1
fi

# ---- 1. python deps ----------------------------------------------------------
if [[ "${SKIP_DEPS:-0}" != "1" ]]; then
    echo "[1/5] Installing Python dependencies ..."
    python -m pip install --upgrade --quiet pip
    # transformers >= 4.45 refuses torch.load on .bin files unless torch >= 2.6
    # (CVE-2025-32434). Bumping torch can leave torchvision out of sync, which
    # breaks transformers' image_utils import; upgrade torchvision in lockstep.
    python -m pip install --upgrade --quiet \
        "torch>=2.6" \
        "torchvision" \
        "transformers>=4.30" \
        "huggingface_hub[cli]>=0.30" \
        pytorch-crf \
        safetensors \
        sacrebleu \
        pyyaml \
        numpy
    echo "    ok."
else
    echo "[1/5] SKIP_DEPS=1, skipping pip install."
fi
echo

# Pick the HuggingFace CLI binary: newer hub renamed `huggingface-cli` -> `hf`.
if command -v hf >/dev/null 2>&1; then
    HF_CLI="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_CLI="huggingface-cli"
else
    HF_CLI=""
fi

# ---- 2. Erya weights ---------------------------------------------------------
ERYA_WEIGHTS="${ERYA_DIR}/pytorch_model.bin"
if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
    if [[ -f "${ERYA_WEIGHTS}" ]]; then
        echo "[2/5] ${ERYA_WEIGHTS} already present ($(du -h "${ERYA_WEIGHTS}" | cut -f1)), skipping."
    elif [[ -z "${HF_CLI}" ]]; then
        echo "[2/5] ERROR: no huggingface CLI found (neither 'hf' nor 'huggingface-cli'); cannot download." >&2
        echo "       Install with:  python -m pip install --upgrade 'huggingface_hub[cli]'" >&2
        exit 1
    else
        echo "[2/5] Downloading RUCAIBox/Erya -> ${ERYA_DIR} (using ${HF_CLI}) ..."
        ${HF_CLI} download RUCAIBox/Erya --local-dir "${ERYA_DIR}"
        echo "    ok."
    fi
else
    echo "[2/5] SKIP_DOWNLOAD=1, skipping Erya download."
fi
echo

# ---- 3. augmented dataset (optional, via HF) ---------------------------------
AUX_TGZ="${ERYA_DIR}/dataset/finetune_with_siku_aux.tgz"
HF_DATASET_FILENAME="${HF_DATASET_FILENAME:-finetune_with_siku_aux.tgz}"

if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
    if [[ -f "${AUX_TGZ}" ]]; then
        echo "[3/5] ${AUX_TGZ} already present ($(du -h "${AUX_TGZ}" | cut -f1)), skipping."
    elif [[ -n "${HF_DATASET_REPO:-}" ]]; then
        if [[ -z "${HF_CLI}" ]]; then
            echo "[3/5] ERROR: no huggingface CLI found; cannot download dataset." >&2
            exit 1
        fi
        echo "[3/5] Downloading ${HF_DATASET_FILENAME} from HF dataset ${HF_DATASET_REPO} (using ${HF_CLI}) ..."
        mkdir -p "${ERYA_DIR}/dataset"
        if [[ -n "${HF_TOKEN:-}" ]]; then
            export HF_TOKEN
            export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
        fi
        ${HF_CLI} download "${HF_DATASET_REPO}" "${HF_DATASET_FILENAME}" \
            --repo-type=dataset \
            --local-dir "${ERYA_DIR}/dataset"
        echo "    ok."
    else
        cat <<MSG
[3/5] WARNING: ${AUX_TGZ} is missing AND HF_DATASET_REPO is unset.
      Either:
        a) rsync it from your laptop:
             rsync -avzP -e "ssh -p <port> -i <key>" \\
               local:/path/to/finetune_with_siku_aux.tgz \\
               root@<pod-ip>:${ERYA_DIR}/dataset/
        b) Re-run this script with HF_DATASET_REPO=<user>/<repo> set.
      Skipping for now; extraction step will also be skipped.
MSG
    fi
else
    echo "[3/5] SKIP_DOWNLOAD=1, skipping aux dataset download."
fi
echo

# ---- 4. extract --------------------------------------------------------------
EXTRACT_DIR="${ERYA_DIR}/dataset/finetune_aux"
if [[ "${SKIP_EXTRACT:-0}" != "1" ]]; then
    if [[ ! -f "${AUX_TGZ}" ]]; then
        echo "[4/5] No tarball at ${AUX_TGZ}, skipping extraction."
    elif [[ -d "${EXTRACT_DIR}" && -n "$(ls -A "${EXTRACT_DIR}" 2>/dev/null || true)" ]]; then
        echo "[4/5] ${EXTRACT_DIR} already populated, skipping extraction."
    else
        echo "[4/5] Extracting ${AUX_TGZ} -> ${EXTRACT_DIR} ..."
        bash "${ERYA_DIR}/scripts/extract_finetune_aux.sh"
        echo "    ok."
    fi
else
    echo "[4/5] SKIP_EXTRACT=1, skipping extraction."
fi
echo

# ---- 5. import sanity check --------------------------------------------------
echo "[5/5] Import sanity check ..."
python - <<'PY'
import importlib, sys
checks = ["torch", "numpy", "yaml", "transformers", "torchcrf", "sacrebleu", "huggingface_hub"]
missing = []
for name in checks:
    try:
        m = importlib.import_module(name)
        ver = getattr(m, "__version__", "?")
        print(f"  ok  {name:18s} {ver}")
    except Exception as e:
        missing.append((name, str(e)))
        print(f"  ERR {name:18s} {e}")
import torch
print(f"  cuda available    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  cuda device       : {torch.cuda.get_device_name(0)}")
    print(f"  cuda mem (GB)     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}")
if missing:
    print("\nMissing/broken modules:", [m for m,_ in missing])
    sys.exit(1)
PY
echo

# ---- summary -----------------------------------------------------------------
echo "============================================================"
echo "Setup complete at $(date)."
echo
if [[ -d "${EXTRACT_DIR}" && -n "$(ls -A "${EXTRACT_DIR}" 2>/dev/null || true)" ]]; then
    echo "Augmented dataset is ready at:"
    echo "  ${EXTRACT_DIR}"
fi
if [[ -f "${ERYA_WEIGHTS}" ]]; then
    echo "Erya weights are ready at:"
    echo "  ${ERYA_WEIGHTS}"
fi
cat <<NEXT

Next steps:

  # (recommended) run inside tmux so you can detach safely
  tmux new -s erya
  cd ${WORKDIR}

  # All three stages, sequentially, fail-fast on error
  bash Erya/scripts/run_all_stages.sh

  # Or pick a subset:  STAGES="1 2"   STAGES="3"   FORCE=1 STAGES="2 3"
  #
  # Per-stage equivalents (if you want to run them one at a time):
  python -m Erya.finetune.train --config Erya/configs/stage1_adapters.yaml \\
      --stage 1 --output_dir ${ERYA_DIR}/checkpoints/stage1
  python -m Erya.finetune.train --config Erya/configs/stage2_encoder.yaml \\
      --stage 2 --resume_adapter ${ERYA_DIR}/checkpoints/stage1/best/adapter.pt \\
      --output_dir ${ERYA_DIR}/checkpoints/stage2
  python -m Erya.finetune.train --config Erya/configs/stage3_full.yaml \\
      --stage 3 --resume ${ERYA_DIR}/checkpoints/stage2/best/full.pt \\
      --output_dir ${ERYA_DIR}/checkpoints/stage3

Detach from tmux: Ctrl-b d   |   Reattach: tmux attach -t erya
============================================================
NEXT
