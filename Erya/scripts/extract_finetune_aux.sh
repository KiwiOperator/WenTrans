#!/usr/bin/env bash
# Extract Erya/dataset/finetune_with_siku_aux.tgz into Erya/dataset/finetune_aux/
# (one-off step before training).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${HERE}/dataset/finetune_with_siku_aux.tgz"
DST="${HERE}/dataset/finetune_aux"

if [[ ! -f "${SRC}" ]]; then
  echo "ERROR: ${SRC} not found" >&2
  exit 1
fi

mkdir -p "${DST}"
echo "Extracting ${SRC} -> ${DST} ..."
tar -xzf "${SRC}" -C "${DST}"
echo "Done. Top-level entries:"
ls -la "${DST}"
