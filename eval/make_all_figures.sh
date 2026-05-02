#!/bin/bash
# Regenerate every figure / table referenced from report/report.md.
# Run from the repo root, after model artefacts exist.

set -euo pipefail

# pick the python in the wentrans env (override if your path is different)
PY="${PY:-$HOME/miniconda3/envs/wentrans/bin/python}"
[[ -x "$PY" ]] || PY=python

# ---- Stage 1 ----
echo "===== Stage 1 ====="
$PY eval/s1_training_curve.py \
  --state output/wentrans-roberta-ancient-zh/trainer_state.json \
  --out report/figures/s1_training_curve.png \
  || echo "  skipped: trainer_state.json not found"

$PY eval/s1_pseudo_ppl.py \
  --ours pretrain/results \
  --base hfl/chinese-roberta-wwm-ext \
  --siku SIKU-BERT/sikuroberta \
  --out_table report/figures/s1_pseudo_ppl_table.md \
  --out_fig   report/figures/s1_pseudo_ppl.png \
  --out_json  report/figures/s1_pseudo_ppl.json

# ---- Stage 2 ----
echo "===== Stage 2 ====="
$PY eval/s2_training_curve.py \
  --state output/cws-wentrans/trainer_state.json \
  --out report/figures/s2_training_curve.png

$PY eval/s2_confusion_matrix.py \
  --model output/cws-wentrans \
  --test_jsonl data/cws/test.jsonl \
  --out report/figures/s2_confusion_matrix.png

$PY eval/s2_cws_demo.py \
  --model output/cws-wentrans \
  --gold_jsonl data/cws/test.jsonl --n 8 \
  --out report/figures/s2_cws_demo.md

# ---- Stage 3 ----
echo "===== Stage 3 ====="
$PY eval/s3_stages_chart.py \
  --root output/acds \
  --out_fig   report/figures/s3_stages.png \
  --out_table report/figures/s3_stages_table.md \
  || echo "  skipped: ACDS results not yet present"

$PY eval/s3_ood_demo.py \
  --m2 output/acds/M2 \
  --m3 output/acds/M3 \
  --test_jsonl data/acds/jsonl/test_b.jsonl \
  --n 6 \
  --out report/figures/s3_ood_demo.md \
  || echo "  skipped: M2/M3 not yet trained"

echo "done. report/figures/ updated."
ls -la report/figures/
