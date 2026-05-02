#!/usr/bin/env python3
"""Plot Stage-1 MLM training curves from trainer_state.json.

Two panels: training loss (rolling) and validation perplexity vs step.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="output/wentrans-roberta-ancient-zh/trainer_state.json")
    ap.add_argument("--out", default="report/figures/s1_training_curve.png")
    args = ap.parse_args()

    state = json.loads(Path(args.state).read_text(encoding="utf-8"))
    log = state["log_history"]

    train_steps, train_loss = [], []
    eval_steps,  eval_ppl   = [], []
    for entry in log:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(entry["step"])
            train_loss.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry["step"])
            eval_ppl.append(math.exp(entry["eval_loss"]))

    print(f"train points: {len(train_steps)} | eval points: {len(eval_steps)}")
    if eval_ppl:
        print(f"final eval PPL = {eval_ppl[-1]:.3f}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; pip install matplotlib")
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(train_steps, train_loss, lw=0.8)
    ax1.set_xlabel("optimization step")
    ax1.set_ylabel("training loss")
    ax1.set_title("Stage 1 — MLM training loss")
    ax1.grid(alpha=0.3)

    ax2.plot(eval_steps, eval_ppl, marker="o", lw=1.2)
    ax2.set_xlabel("optimization step")
    ax2.set_ylabel("validation perplexity")
    ax2.set_yscale("log")
    ax2.set_title("Stage 1 — validation perplexity (log scale)")
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"figure -> {args.out}")


if __name__ == "__main__":
    main()
