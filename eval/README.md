# `eval/` — report figures & tables

Scripts that consume the trained-model artefacts in `output/` and produce the
charts and markdown snippets referenced from the two report files:
- [`../report/report.md`](../report/report.md) — Stages 1 & 2
- [`../report/report_stage3.md`](../report/report_stage3.md) — Stage 3

## Outputs

Everything lands in `report/figures/`:

| File | Source | What it shows |
|---|---|---|
| `s1_training_curve.png` | `s1_training_curve.py` | Stage-1 MLM train loss + eval PPL vs step |
| `s1_pseudo_ppl.png` + `.md` table | `s1_pseudo_ppl.py` | base vs ours vs SikuRoBERTa pseudo-PPL on 4 passages |
| `s2_training_curve.png` | `s2_training_curve.py` | Stage-2 train loss + per-epoch P/R/F1 |
| `s2_confusion_matrix.png` | `s2_confusion_matrix.py` | per-tag BMES confusion on test set |
| `s2_cws_demo.md` | `s2_cws_demo.py` | live segmentation examples (with gold) |
| `s3_stages.png` + `.md` table | `s3_stages_chart.py` | M_1/M_2/M_3 WSG + POS F1 across splits |
| `s3_ood_demo.md` | `s3_ood_demo.py` | M_2 vs M_3 disagreements on Test-B (Shiji/Zizhitongjian) |

## One-shot regeneration

After artefacts exist:
```bash
bash eval/make_all_figures.sh
```

Skips any step whose inputs aren't yet on disk (so it's safe to run before
Stage 3 finishes).

## Running individual scripts

Each script has sensible defaults assuming the standard repo layout. Override
with CLI flags when your paths differ:

```bash
# Pseudo-PPL on a different model triple
python eval/s1_pseudo_ppl.py --ours output/some-other-checkpoint

# CWS demo with more examples from a different test file
python eval/s2_cws_demo.py --gold_jsonl data/cws/val.jsonl --n 12

# Stage-3 chart against an alternate run dir
python eval/s3_stages_chart.py --root output/acds-200k
```

## Dependencies

In the existing `wentrans` conda env, plus:
```bash
pip install matplotlib
```

(Pseudo-PPL also pulls SikuRoBERTa from HF Hub on first run — ~400 MB cached
under `~/.cache/huggingface/`.)

## Where each figure goes in the report

`report/report.md` references everything by relative path. After regenerating
the figures, just open `report.md` in a markdown viewer and the images update
automatically.
