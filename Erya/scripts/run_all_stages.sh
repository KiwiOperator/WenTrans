#!/usr/bin/env bash
# Run stages 1 -> 2 -> 3 sequentially, resuming each from the previous.
#
# Idempotent: if a stage's "best" checkpoint already exists on disk, it is
# skipped (so a re-run after a crash picks up at the next stage). To force a
# rerun of a completed stage, delete its best/ directory or set FORCE=1.
#
# Optional environment variables:
#   WORKDIR        : repo root (default: parent of this script's parent dir)
#   STAGES         : space-separated list, e.g. "1 2", "2 3", "3" (default: "1 2 3")
#   FORCE          : "1" to retrain stages whose best/ exists (default: 0)
#   STAGE1_CONFIG  : override config for stage 1 (default: Erya/configs/stage1_adapters.yaml)
#   STAGE2_CONFIG  : override config for stage 2
#   STAGE3_CONFIG  : override config for stage 3
#   CKPT_ROOT      : root for checkpoints (default: $WORKDIR/Erya/checkpoints)
#   LOG_DIR        : log destination (default: $WORKDIR/logs)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_WORKDIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKDIR="${WORKDIR:-${DEFAULT_WORKDIR}}"
cd "${WORKDIR}"

STAGES="${STAGES:-1 2 3}"
FORCE="${FORCE:-0}"
CKPT_ROOT="${CKPT_ROOT:-${WORKDIR}/Erya/checkpoints}"
LOG_DIR="${LOG_DIR:-${WORKDIR}/logs}"
mkdir -p "${LOG_DIR}"

STAGE1_CONFIG="${STAGE1_CONFIG:-Erya/configs/stage1_adapters.yaml}"
STAGE2_CONFIG="${STAGE2_CONFIG:-Erya/configs/stage2_encoder.yaml}"
STAGE3_CONFIG="${STAGE3_CONFIG:-Erya/configs/stage3_full.yaml}"

CKPT1="${CKPT_ROOT}/stage1"
CKPT2="${CKPT_ROOT}/stage2"
CKPT3="${CKPT_ROOT}/stage3"

run_stage () {
    local stage="$1"
    local config="$2"
    local out_dir="$3"
    local log_file="$4"
    shift 4
    local extra_args=("$@")

    echo "============================================================"
    echo "Stage ${stage}"
    echo "  config:   ${config}"
    echo "  output:   ${out_dir}"
    echo "  log:      ${log_file}"
    echo "  resume:   ${extra_args[*]:-<none>}"
    echo "  started:  $(date)"
    echo "============================================================"

    # tee to a log AND print to terminal; preserve python's exit code via PIPESTATUS.
    set -o pipefail
    python -m Erya.finetune.train \
        --config "${config}" \
        --stage "${stage}" \
        --output_dir "${out_dir}" \
        "${extra_args[@]}" 2>&1 | tee "${log_file}"
    local rc="${PIPESTATUS[0]}"
    if [[ "${rc}" -ne 0 ]]; then
        echo "Stage ${stage} FAILED (exit ${rc})." >&2
        exit "${rc}"
    fi
    echo "Stage ${stage} done at $(date)."
    echo
}

echo "Pipeline configuration:"
echo "  workdir:    ${WORKDIR}"
echo "  stages:     ${STAGES}"
echo "  force:      ${FORCE}"
echo "  ckpt root:  ${CKPT_ROOT}"
echo "  log dir:    ${LOG_DIR}"
echo "  python:     $(which python)"
echo

# ---------------- Stage 1 ----------------
if [[ " ${STAGES} " == *" 1 "* ]]; then
    if [[ "${FORCE}" != "1" && -f "${CKPT1}/best/adapter.pt" ]]; then
        echo "[skip] Stage 1 already complete (${CKPT1}/best/adapter.pt exists). Set FORCE=1 to redo."
        echo
    else
        run_stage 1 "${STAGE1_CONFIG}" "${CKPT1}" "${LOG_DIR}/stage1.log"
    fi
fi

# ---------------- Stage 2 ----------------
if [[ " ${STAGES} " == *" 2 "* ]]; then
    if [[ "${FORCE}" != "1" && -f "${CKPT2}/best/full.pt" ]]; then
        echo "[skip] Stage 2 already complete (${CKPT2}/best/full.pt exists). Set FORCE=1 to redo."
        echo
    else
        if [[ ! -f "${CKPT1}/best/adapter.pt" ]]; then
            echo "Stage 2 needs ${CKPT1}/best/adapter.pt but it is missing." >&2
            echo "Run stage 1 first, or set STAGES='1 2 3'." >&2
            exit 1
        fi
        run_stage 2 "${STAGE2_CONFIG}" "${CKPT2}" "${LOG_DIR}/stage2.log" \
            --resume_adapter "${CKPT1}/best/adapter.pt"
    fi
fi

# ---------------- Stage 3 ----------------
if [[ " ${STAGES} " == *" 3 "* ]]; then
    if [[ "${FORCE}" != "1" && -f "${CKPT3}/best/full.pt" ]]; then
        echo "[skip] Stage 3 already complete (${CKPT3}/best/full.pt exists). Set FORCE=1 to redo."
        echo
    else
        if [[ ! -f "${CKPT2}/best/full.pt" ]]; then
            echo "Stage 3 needs ${CKPT2}/best/full.pt but it is missing." >&2
            echo "Run stage 2 first, or set STAGES='1 2 3'." >&2
            exit 1
        fi
        run_stage 3 "${STAGE3_CONFIG}" "${CKPT3}" "${LOG_DIR}/stage3.log" \
            --resume "${CKPT2}/best/full.pt"
    fi
fi

echo "============================================================"
echo "All requested stages complete at $(date)."
echo
echo "Best checkpoints:"
[[ -e "${CKPT1}/best/adapter.pt" ]] && echo "  stage 1:  ${CKPT1}/best/adapter.pt"
[[ -e "${CKPT2}/best/full.pt"   ]] && echo "  stage 2:  ${CKPT2}/best/full.pt"
[[ -e "${CKPT3}/best/full.pt"   ]] && echo "  stage 3:  ${CKPT3}/best/full.pt"
echo
echo "Logs: ${LOG_DIR}/stage{1,2,3}.log"
echo "============================================================"
