"""Metrics, calibration (temperature) and operating point (threshold)."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score,
                             roc_auc_score, roc_curve)

from .config import Config


def expected_calibration_error(probs, labels, n_bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    acc = (pred == labels).astype(float)
    e, edges = 0.0, np.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def compute_metrics(logits, labels, T: float = 1.0, thr: float = 0.5):
    """AUC is threshold-free; everything else uses `thr` on P(class 1)."""
    labels = np.asarray(labels)
    probs = torch.softmax(torch.as_tensor(logits).float() / T, 1).numpy()
    p1 = probs[:, 1]
    pred = (p1 >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    two = len(set(labels.tolist())) > 1
    return {
        "accuracy": float((pred == labels).mean()),
        "balanced_acc": float(balanced_accuracy_score(labels, pred)),
        "auc": float(roc_auc_score(labels, p1)) if two else float("nan"),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "ece": expected_calibration_error(probs, labels),
        "brier": float(brier_score_loss(labels, p1)) if two else float("nan"),
        "threshold": float(thr),
    }


def fit_temperature(logits, labels, lo=0.05, hi=10.0):
    """Bounded grid search for temperature.

    An unconstrained optimiser drives T -> infinity on a small or poorly
    separated validation set, flattening probabilities to 1/C and reporting a
    spurious ECE of 0.
    """
    labels = np.asarray(labels)
    if len(labels) < Config.MIN_VAL_FOR_CALIB or len(set(labels.tolist())) < 2:
        return 1.0
    lg = torch.as_tensor(logits).float()
    y = torch.as_tensor(labels).long()

    def nll(T):
        return float(F.cross_entropy(lg / float(T), y))

    best = min(np.geomspace(lo, hi, 60), key=nll)
    best = min(np.linspace(max(lo, best * 0.7), min(hi, best * 1.4), 40), key=nll)
    return float(np.clip(best, lo, hi))


def fit_threshold(logits, labels, T: float = 1.0):
    """Operating point by Youden's J on validation.

    A fixed 0.5 is arbitrary: it is only optimal when the classifier is
    calibrated *and* the prior is 1:1. Ours is neither.
    """
    labels = np.asarray(labels)
    if len(labels) < Config.MIN_VAL_FOR_CALIB or len(set(labels.tolist())) < 2:
        return 0.5
    p1 = torch.softmax(torch.as_tensor(logits).float() / T, 1).numpy()[:, 1]
    fpr, tpr, thr = roc_curve(labels, p1)
    j = tpr - fpr
    best = thr[int(np.argmax(j))]
    if not np.isfinite(best):
        return 0.5
    return float(np.clip(best, 0.01, 0.99))


def aggregate(runs):
    """mean / std across a list of metric dicts."""
    keys = runs[0].keys()
    return ({k: float(np.nanmean([r[k] for r in runs])) for k in keys},
            {k: float(np.nanstd([r[k] for r in runs])) for k in keys})
