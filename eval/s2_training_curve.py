#!/usr/bin/env python3
"""Plot Stage-2 CWS training curves: per-epoch loss and val P / R / F1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="output/cws-wentrans/trainer_state.json")
    ap.add_argument("--out", default="report/figures/s2_training_curve.png")
    args = ap.parse_args()

    state = json.loads(Path(args.state).read_text(encoding="utf-8"))
    log = state["log_history"]

    train_steps, train_loss = [], []
    eval_epochs, eval_p, eval_r, eval_f, eval_loss = [], [], [], [], []
    for entry in log:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(entry["step"])
            train_loss.append(entry["loss"])
        if "eval_f1" in entry:
            eval_epochs.append(entry.get("epoch"))
            eval_p.append(entry["eval_precision"])
            eval_r.append(entry["eval_recall"])
            eval_f.append(entry["eval_f1"])
            eval_loss.append(entry["eval_loss"])

    print(f"train logs: {len(train_steps)} | eval logs: {len(eval_epochs)}")
    if eval_f:
        print(f"final eval F1 = {eval_f[-1]:.4f}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; pip install matplotlib")
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(train_steps, train_loss, lw=0.8, label="train loss")
    if eval_loss:
        # secondary axis-friendly: scatter eval losses near corresponding step
        # (we approximate step from epoch since trainer logs epoch but not step
        # for all eval entries — fall back to evenly-spaced if needed)
        ax1.plot([s for s in train_steps[::max(1, len(train_steps)//len(eval_loss))]][:len(eval_loss)],
                 eval_loss, marker="o", lw=0, ms=5, label="eval loss")
    ax1.set_xlabel("optimization step")
    ax1.set_ylabel("loss")
    ax1.legend()
    ax1.set_title("Stage 2 — CWS training & eval loss")
    ax1.grid(alpha=0.3)

    if eval_epochs:
        ax2.plot(eval_epochs, eval_f, marker="o", lw=1.4, label="F1")
        ax2.plot(eval_epochs, eval_p, marker="s", lw=1.0, ls="--", label="precision")
        ax2.plot(eval_epochs, eval_r, marker="^", lw=1.0, ls="--", label="recall")
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("score")
        ax2.set_ylim(0.85, 1.0)
        ax2.set_title("Stage 2 — validation P / R / F1")
        ax2.legend()
        ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"figure -> {args.out}")


if __name__ == "__main__":
    main()
