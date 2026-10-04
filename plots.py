"""Training curves and model comparison figures."""
import os

import matplotlib.pyplot as plt
import numpy as np

from .config import Config


def plot_curves(history, title="CMUPF", save=None):
    panels = [("total_loss", "total loss"), ("task_loss", "task loss"),
              ("accuracy", "accuracy"), ("auc", "AUC"), ("ece", "ECE"),
              ("balanced_acc", "balanced accuracy")]
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    for a, (key, name) in zip(ax.ravel(), panels):
        col = f"val_{key}"
        if col not in history:
            a.set_visible(False)
            continue
        for f, g in history.groupby("fold"):
            a.plot(g.epoch, g[col], alpha=.85, label=f"fold {f}")
        a.set_title(f"{name} (validation)")
        a.set_xlabel("epoch")
        a.grid(alpha=.3)
        a.legend(fontsize=7)
    fig.suptitle(title)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=140)
    plt.show()


def plot_comparison(tables: dict, metric="auc", save=None):
    """Pooled metric per model with the fold spread shown."""
    names = list(tables)
    means = [tables[n][f"{metric}_mean"].mean() for n in names]
    stds = [tables[n][f"{metric}_mean"].std() for n in names]

    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(names)), 5))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=5, color="#5B8FF9", alpha=.85)
    for i, n in enumerate(names):
        pts = tables[n][f"{metric}_mean"].values
        ax.scatter(np.full_like(pts, i, dtype=float), pts, color="#333",
                   zorder=3, s=18, alpha=.8)
    ax.axhline(0.5, ls="--", c="grey", lw=1, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} by model (bars: mean ± std over folds; dots: folds)")
    ax.grid(axis="y", alpha=.3)
    ax.legend()
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=140)
    plt.show()
