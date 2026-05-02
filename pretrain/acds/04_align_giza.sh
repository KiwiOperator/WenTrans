#!/bin/bash
# Phase 1.4 — IBM-4 word alignment via giza-py (paper-faithful).
#
# Reads:   <IO_DIR>/src_shuf_seg.txt  (modern, segmented)
#          <IO_DIR>/tgt_shuf_seg.txt  (ancient, char-split)
# Writes:  <IO_DIR>/alignment.txt     (one line per pair: 'i-j:prob i-j:prob ...')
#
# Requires giza-py (https://github.com/sillsdev/giza-py) cloned alongside,
# with mgiza built. Path passed via $GIZAPY (default: $HOME/giza-py).
#
# Usage:
#   IO_DIR=data/acds/work bash pretrain/acds/04_align_giza.sh

set -euo pipefail

IO_DIR="${IO_DIR:?set IO_DIR (the dir with src_shuf_seg.txt + tgt_shuf_seg.txt)}"
GIZAPY="${GIZAPY:-$HOME/giza-py}"

if [[ ! -x "$GIZAPY/giza.py" && ! -f "$GIZAPY/giza.py" ]]; then
  echo "ERROR: giza-py not found at $GIZAPY"
  echo "Install once with:"
  echo "  cd \$HOME && git clone https://github.com/sillsdev/giza-py.git"
  echo "  cd giza-py && pip install -r requirements.txt"
  echo "  # then build mgiza per giza-py's README (uses cmake)"
  exit 1
fi

cd "$GIZAPY"

# Paper's invocation. --include-probs is what makes the per-pair confidence
# scores show up in the alignment file; align-pos_tag_ltp.py needs them.
python giza.py \
  --source "$OLDPWD/$IO_DIR/src_shuf_seg.txt" \
  --target "$OLDPWD/$IO_DIR/tgt_shuf_seg.txt" \
  --alignments "$OLDPWD/$IO_DIR/alignment.txt" \
  --model ibm4 \
  --m1 10 --mh 10 --m3 10 --m4 10 \
  --include-probs

echo "alignment written to $IO_DIR/alignment.txt"
wc -l "$OLDPWD/$IO_DIR/alignment.txt"
