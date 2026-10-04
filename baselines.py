"""Non-deep baseline: logistic regression on volume intensity summaries.

This is the floor. If CMUPF cannot clear it, the problem is not the fusion
strategy, and no amount of architecture tuning will help.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import Config
from .data import tabular_features
from .engine import make_folds
from .metrics import compute_metrics, fit_threshold


def logistic_cv(cohort: pd.DataFrame, modalities=("mri", "pet"), C=1.0,
                tune_threshold=None):
    """Same folds as the deep models, so the comparison is like for like."""
    tune = Config.TUNE_THRESHOLD if tune_threshold is None else tune_threshold
    rows = []
    for fold, (tr_df, va_df, te_df) in enumerate(make_folds(cohort), 1):
        Xtr, _ = tabular_features(tr_df, modalities)
        Xva, _ = tabular_features(va_df, modalities)
        Xte, names = tabular_features(te_df, modalities)
        ytr, yva, yte = tr_df.label.values, va_df.label.values, te_df.label.values

        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=C, max_iter=5000))
        clf.fit(Xtr, ytr)

        def logits(X):
            p = clf.predict_proba(X)
            return np.log(np.clip(p, 1e-9, 1))

        thr = fit_threshold(logits(Xva), yva) if tune else 0.5
        m = compute_metrics(logits(Xte), yte, T=1.0, thr=thr)
        m.update({"model": f"logreg[{'+'.join(modalities)}]", "fold": fold,
                  "threshold": thr})
        rows.append(m)
        print(f"  fold {fold}: acc {m['accuracy']:.3f}  bacc {m['balanced_acc']:.3f}  "
              f"auc {m['auc']:.3f}  sens {m['sensitivity']:.3f}  spec {m['specificity']:.3f}")

    df = pd.DataFrame(rows)
    # match the column naming used by the deep-model tables
    for k in ("accuracy", "balanced_acc", "auc", "f1", "precision",
              "sensitivity", "specificity", "ece", "brier"):
        df[f"{k}_mean"] = df[k]
        df[f"{k}_std"] = 0.0
    return df


def compare(tables: dict):
    """Side-by-side pooled comparison of every model that has been run."""
    keys = ("auc", "balanced_acc", "accuracy", "sensitivity", "specificity", "ece")
    print(f"{'model':28s} " + " ".join(f"{k:>13s}" for k in keys))
    print("-" * (28 + 14 * len(keys)))
    for name, df in tables.items():
        cells = []
        for k in keys:
            v = df[f"{k}_mean"]
            cells.append(f"{v.mean():.3f}±{v.std():.3f}")
        print(f"{name:28s} " + " ".join(f"{c:>13s}" for c in cells))
