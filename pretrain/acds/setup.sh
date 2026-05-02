#!/bin/bash
# One-time setup for Stage 3 (ACDS): install LTP, opencc, build mgiza/giza-py.
# Run on a PACE login node (CPU is fine; mgiza build needs cmake + g++).

set -euo pipefail

# 1. Python deps inside the existing wentrans conda env.
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate wentrans

pip install ltp==4.2.13 opencc-python-reimplemented
pip install "numpy<2"   # ltp's deps upgrade numpy; torch 2.2.2 needs numpy<2
# 2. giza-py and mgiza. We install under $HOME so it persists across jobs.
GIZAPY_DIR="${GIZAPY_DIR:-$HOME/giza-py}"
if [[ ! -d "$GIZAPY_DIR" ]]; then
  git clone https://github.com/sillsdev/giza-py.git "$GIZAPY_DIR"
fi
cd "$GIZAPY_DIR"
pip install -r requirements.txt

# Build mgiza if not already present.
if [[ ! -x "$GIZAPY_DIR/mgiza/mgizapp/bin/mgiza" ]]; then
  echo "Building mgiza..."
  if [[ ! -d "$GIZAPY_DIR/mgiza" ]]; then
    git clone https://github.com/moses-smt/mgiza.git "$GIZAPY_DIR/mgiza"
  fi
  cd "$GIZAPY_DIR/mgiza/mgizapp"
  cmake .
  make -j8 || make    # serial fallback
fi

echo
echo "All set. Verify giza-py points at the mgiza binaries:"
echo "  $GIZAPY_DIR/giza.py --help | head"
