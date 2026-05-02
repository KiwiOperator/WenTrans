#!/usr/bin/env python3
"""Build the Stage-3 P/R/F comparison chart from M_1 / M_2 / M_3 metrics.

Reads:
    output/acds/M1/eval_results.json + test_a_results.json + test_b_results.json
    output/acds/M2/...
    output/acds/M3/...

Produces a grouped bar chart per test split (eval / test_a / test_b) showing
WSG-F1 and POS-F1 across stages, plus a markdown table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Paper's reported scores for context (Wang et al. ACDS 2023, Table 2).
PAPER = {
    "test_a": {"wsg_f1_baseline": 0.9473, "wsg_f1_M3": 0.9564,
               "pos_f1_baseline": 0.9093, "pos_f1_M3": 0.9055},
    "test_b": {"wsg_f1_baseline": 0.8919, "wsg_f1_M3": 0.9364,
               "pos_f1_baseline": 0.8348, "pos_f1_M3": 0.8621},
}


def load_metrics(model_dir: Path):
    out = {}
    for split, fname in (("eval", "eval_results.json"),
                         ("test_a", "test_a_results.json"),
                         ("test_b", "test_b_results.json")):
        f = model_dir / fname
        if f.exists():
            d = json.loads(f.read_text(encoding="utf-8"))
            out[split] = {
                "wsg_f1": d.get(f"{split}_wsg_f1", d.get("eval_wsg_f1") if split=="eval" else None),
                "pos_f1": d.get(f"{split}_pos_f1", d.get("eval_pos_f1") if split=="eval" else None),
                "wsg_p":  d.get(f"{split}_wsg_precision", d.get("eval_wsg_precision") if split=="eval" else None),
                "wsg_r":  d.get(f"{split}_wsg_recall",    d.get("eval_wsg_recall")    if split=="eval" else None),
                "pos_p":  d.get(f"{split}_pos_precision", d.get("eval_pos_precision") if split=="eval" else None),
                "pos_r":  d.get(f"{split}_pos_recall",    d.get("eval_pos_recall")    if split=="eval" else None),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output/acds")
    ap.add_argument("--out_fig", default="report/figures/s3_stages.png")
    ap.add_argument("--out_table", default="report/figures/s3_stages_table.md")
    args = ap.parse_args()

    root = Path(args.root)
    metrics = {stage: load_metrics(root / stage) for stage in ("M1", "M2", "M3")}

    print("loaded metrics:")
    for s, m in metrics.items():
        print(f"  {s}: splits = {sorted(m.keys())}")

    # markdown table
    Path(args.out_table).parent.mkdir(parents=True, exist_ok=True)
    rows = ["| model | Test-A WSG-F1 | Test-A POS-F1 | Test-B WSG-F1 | Test-B POS-F1 |",
            "|---|---:|---:|---:|---:|"]
    rows.append(f"| Tina baseline (paper) | {PAPER['test_a']['wsg_f1_baseline']*100:.2f} | "
                f"{PAPER['test_a']['pos_f1_baseline']*100:.2f} | "
                f"{PAPER['test_b']['wsg_f1_baseline']*100:.2f} | "
                f"{PAPER['test_b']['pos_f1_baseline']*100:.2f} |")
    rows.append(f"| **Paper M_3 (theirs)** | **{PAPER['test_a']['wsg_f1_M3']*100:.2f}** | "
                f"{PAPER['test_a']['pos_f1_M3']*100:.2f} | "
                f"**{PAPER['test_b']['wsg_f1_M3']*100:.2f}** | "
                f"**{PAPER['test_b']['pos_f1_M3']*100:.2f}** |")
    for stage in ("M1", "M2", "M3"):
        m = metrics[stage]
        ta_w = (m.get("test_a") or {}).get("wsg_f1")
        ta_p = (m.get("test_a") or {}).get("pos_f1")
        tb_w = (m.get("test_b") or {}).get("wsg_f1")
        tb_p = (m.get("test_b") or {}).get("pos_f1")
        def fmt(x): return f"{x*100:.2f}" if x is not None else "—"
        em = "**" if stage == "M3" else ""
        rows.append(f"| {em}{stage} (ours){em} | {em}{fmt(ta_w)}{em} | {fmt(ta_p)} | "
                    f"{em}{fmt(tb_w)}{em} | {em}{fmt(tb_p)}{em} |")
    Path(args.out_table).write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"table -> {args.out_table}")

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed; skipping figure")
        return

    splits = ["eval", "test_a", "test_b"]
    stages = ["M1", "M2", "M3"]
    metric_kinds = [("wsg_f1", "WSG-F1"), ("pos_f1", "POS-F1")]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, (key, title) in zip(axes, metric_kinds):
        x = np.arange(len(splits))
        w = 0.27
        for i, stage in enumerate(stages):
            vals = [(metrics[stage].get(s) or {}).get(key) for s in splits]
            vals = [v if v is not None else 0 for v in vals]
            ax.bar(x + (i - 1) * w, vals, w, label=stage)
        # paper baselines (dashed lines on test_a / test_b)
        if key == "wsg_f1":
            ax.axhline(PAPER["test_a"]["wsg_f1_M3"], ls="--", lw=1, alpha=0.5,
                       color="gray", label="paper M3 (Test-A)")
        ax.set_xticks(x); ax.set_xticklabels(splits)
        ax.set_title(title); ax.set_ylim(0.5, 1.0)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("Stage 3 — WSG / POS F1 across stages and test sets")
    fig.tight_layout()
    Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, dpi=150)
    print(f"figure -> {args.out_fig}")


if __name__ == "__main__":
    main()
