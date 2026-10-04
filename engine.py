"""Training / evaluation loops and 5-fold cross-validation."""
import copy
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import (GroupShuffleSplit, StratifiedGroupKFold,
                                     StratifiedKFold, train_test_split)
from tqdm.auto import tqdm

from .config import Config, set_seed
from .data import make_loader
from .metrics import aggregate, compute_metrics, fit_temperature, fit_threshold
from .models import CMUPF, CMUPFLoss, SingleModalityCNN, SingleModalityLoss


def enable_mc_dropout(model):
    """Eval mode for BatchNorm, but dropout stays on so MC sampling works."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout3d)):
            m.train()


def _forward(model, batch, modalities, missing):
    if len(modalities) == 1:
        return model(batch[modalities[0]].to(Config.DEVICE))
    return model(batch["mri"].to(Config.DEVICE), batch["pet"].to(Config.DEVICE),
                 missing_modality=missing)


def run_epoch(model, loader, criterion, optimizer=None, missing=None,
              modalities=("mri", "pet"), desc="", thr=0.5):
    train = optimizer is not None
    if train:
        model.train()
    else:
        enable_mc_dropout(model)

    sums, nb, L, Y = {}, 0, [], []
    for b in tqdm(loader, desc=desc, leave=False, disable=not desc):
        if b is None:
            continue
        y = b["label"].to(Config.DEVICE)
        with torch.set_grad_enabled(train):
            out = _forward(model, b, modalities, missing)
            losses = criterion(out, y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            losses["total_loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
            optimizer.step()
        for k, v in losses.items():
            sums[k] = sums.get(k, 0.0) + float(v.detach())
        nb += 1
        L.append(out["logits"].detach().float().cpu())
        Y.append(y.cpu())

    logits, labels = torch.cat(L).numpy(), torch.cat(Y).numpy()
    stats = {k: v / max(nb, 1) for k, v in sums.items()}
    stats.update(compute_metrics(logits, labels, thr=thr))
    return stats, logits, labels


@torch.no_grad()
def collect_logits(model, loader, modalities=("mri", "pet"), missing=None):
    enable_mc_dropout(model)
    L, Y = [], []
    for b in loader:
        if b is None:
            continue
        out = _forward(model, b, modalities, missing)
        L.append(out["logits"].float().cpu())
        Y.append(b["label"])
    return torch.cat(L).numpy(), torch.cat(Y).numpy()


def repeated_eval(model, loader, modalities=("mri", "pet"), missing=None,
                  repeats=None, T=1.0, thr=0.5):
    """Repeat the stochastic (MC dropout) evaluation -> mean ± std."""
    R = repeats or Config.EVAL_REPEATS
    runs = []
    for _ in range(R):
        lg, y = collect_logits(model, loader, modalities, missing)
        runs.append(compute_metrics(lg, y, T, thr))
    return aggregate(runs)


# ------------------------------------------------------------------ folds
_DEFAULT = object()


def make_folds(cohort: pd.DataFrame, verbose: bool = False, group_by=_DEFAULT):
    """Outer CV folds, optionally keeping each acquisition site whole.

    With plain StratifiedKFold, subjects from one site land in both train and
    test. Since site predicts diagnosis at AUC 0.640 here, a model can be
    rewarded for recognising scanner signature rather than disease. Pass
    group_by="site" to remove that path; the inner train/val split is grouped
    the same way. group_by=None forces the plain scheme.

    If omitted, Config.GROUP_CV_BY is used — but prefer passing it explicitly.
    `importlib.reload(cmupf.config)` rebinds Config to a new class object, so a
    value set on the old one is silently ignored and the grouping never
    happens.
    """
    n = len(cohort)
    y = cohort.label.values
    gcol = Config.GROUP_CV_BY if group_by is _DEFAULT else group_by

    if gcol == "site" and "site" not in cohort.columns and "subject" in cohort.columns:
        # a cohort built before `site` was added to load_cohort(); the site is
        # the first field of the ADNI id (SSS_S_NNNN), so just derive it
        cohort = cohort.copy()
        cohort["site"] = cohort["subject"].astype(str).str[:3]

    grouped = bool(gcol) and gcol in cohort.columns
    if bool(gcol) and not grouped:
        raise KeyError(f"group_by={gcol!r} requested but that column is missing "
                       f"from the cohort (have {list(cohort.columns)[:10]}...)")

    if grouped:
        groups = cohort[gcol].values
        splitter = StratifiedGroupKFold(n_splits=Config.N_FOLDS, shuffle=True,
                                        random_state=Config.SEED)
        outer = splitter.split(np.arange(n), y, groups=groups)
    else:
        splitter = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True,
                                   random_state=Config.SEED)
        outer = splitter.split(np.arange(n), y)

    folds = []
    for k, (tr_idx, te_idx) in enumerate(outer, 1):
        tr_full = cohort.iloc[tr_idx].reset_index(drop=True)
        te = cohort.iloc[te_idx].reset_index(drop=True)

        if grouped:
            gss = GroupShuffleSplit(n_splits=1, test_size=Config.INNER_VAL,
                                    random_state=Config.SEED)
            i_tr, i_va = next(gss.split(tr_full, tr_full.label.values,
                                        groups=tr_full[gcol].values))
            tr, va = tr_full.iloc[i_tr], tr_full.iloc[i_va]
        else:
            tr, va = train_test_split(tr_full, test_size=Config.INNER_VAL,
                                      stratify=tr_full.label, random_state=Config.SEED)

        tr, va = tr.reset_index(drop=True), va.reset_index(drop=True)
        if verbose:
            msg = f"  fold {k}: train {len(tr)} val {len(va)} test {len(te)}"
            if grouped:
                leak = set(tr[gcol]) & set(te[gcol])
                msg += (f" | {gcol}s train {tr[gcol].nunique()} test {te[gcol].nunique()}"
                        f" | overlap {len(leak)}")
            print(msg)
        folds.append((tr, va, te))
    return folds


def _train_fold(model, criterion, tr_df, va_df, modalities, fold, epochs, tag):
    opt = torch.optim.AdamW(model.parameters(), lr=Config.LR,
                            weight_decay=Config.WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                     patience=8)
    tl = make_loader(tr_df, True, modalities)
    vl = make_loader(va_df, False, modalities)

    best_auc, best_state, bad, hist = -1.0, None, 0, []
    hdr = (f"{'ep':>3} | {'tr loss':>8} {'tr acc':>7} | {'va loss':>8} "
           f"{'va acc':>7} {'auc':>6} {'ece':>6} | {'lr':>8}   t")
    print(hdr)
    print("-" * len(hdr))

    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr, _, _ = run_epoch(model, tl, criterion, opt, modalities=modalities,
                             desc=f"{tag}f{fold} train {ep}")
        va, _, _ = run_epoch(model, vl, criterion, None, modalities=modalities,
                             desc=f"{tag}f{fold} val {ep}")
        auc = va["auc"] if np.isfinite(va["auc"]) else 0.0
        sch.step(auc)
        lr = opt.param_groups[0]["lr"]
        print(f"{ep:3d} | {tr['total_loss']:8.4f} {tr['accuracy']:7.3f} | "
              f"{va['total_loss']:8.4f} {va['accuracy']:7.3f} {va['auc']:6.3f} "
              f"{va['ece']:6.3f} | {lr:8.2e} {time.time() - t0:5.1f}s")
        hist.append({"model": tag or "cmupf", "fold": fold, "epoch": ep, "lr": lr,
                     **{f"train_{k}": v for k, v in tr.items()},
                     **{f"val_{k}": v for k, v in va.items()}})

        if auc > best_auc:
            best_auc, bad = auc, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= Config.EARLY_STOP:
                print(f"  early stop at epoch {ep} (best val AUC {best_auc:.4f})")
                break

    model.load_state_dict(best_state)
    return hist, best_auc, vl


def cross_validate(cohort, model_name="cmupf", modalities=("mri", "pet"),
                   epochs=None, out_dir=None):
    """Run N-fold CV for CMUPF or a single-modality baseline."""
    epochs = epochs or (Config.NUM_EPOCHS if model_name == "cmupf"
                        else Config.BASELINE_EPOCHS)
    out_dir = out_dir or Config.OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    all_hist, fold_rows, miss_rows = [], [], []
    for fold, (tr_df, va_df, te_df) in enumerate(make_folds(cohort), 1):
        set_seed(Config.SEED + fold)
        if model_name == "cmupf":
            model, crit = CMUPF().to(Config.DEVICE), CMUPFLoss()
        else:
            model, crit = SingleModalityCNN().to(Config.DEVICE), SingleModalityLoss()

        print(f"\n{'=' * 78}\n{model_name.upper()}  FOLD {fold}/{Config.N_FOLDS}  "
              f"train={len(tr_df)} val={len(va_df)} test={len(te_df)}\n{'=' * 78}")
        hist, best_auc, vl = _train_fold(model, crit, tr_df, va_df, modalities,
                                         fold, epochs, f"{model_name} ")
        all_hist += hist

        # calibrate + choose operating point on validation only
        val_lg, val_y = collect_logits(model, vl, modalities)
        T = fit_temperature(val_lg, val_y)
        thr = fit_threshold(val_lg, val_y, T) if Config.TUNE_THRESHOLD else 0.5

        tel = make_loader(te_df, False, modalities)
        mean, std = repeated_eval(model, tel, modalities, None, T=T, thr=thr)
        fixed, _ = repeated_eval(model, tel, modalities, None, T=T, thr=0.5)
        raw, _ = repeated_eval(model, tel, modalities, None, T=1.0, thr=0.5)

        print(f"  T={T:.3f} thr={thr:.3f} | test AUC {mean['auc']:.3f}±{std['auc']:.3f}"
              f" | bacc {fixed['balanced_acc']:.3f} (thr .5) -> "
              f"{mean['balanced_acc']:.3f} (tuned) | ECE {raw['ece']:.3f} -> {mean['ece']:.3f}")

        row = {"model": model_name, "fold": fold, "temperature": T, "threshold": thr,
               "best_val_auc": best_auc, "ece_raw": raw["ece"],
               "balanced_acc_thr05": fixed["balanced_acc"],
               "sensitivity_thr05": fixed["sensitivity"]}
        row.update({f"{k}_mean": v for k, v in mean.items()})
        row.update({f"{k}_std": v for k, v in std.items()})
        fold_rows.append(row)

        if model_name == "cmupf":
            for miss in ("mri", "pet"):
                m, s = repeated_eval(model, tel, modalities, miss, T=T, thr=thr)
                miss_rows.append({"model": model_name, "fold": fold, "missing": miss,
                                  **{f"{k}_mean": v for k, v in m.items()}})

        torch.save({"fold": fold, "model_state_dict": model.state_dict(),
                    "temperature": T, "threshold": thr, "test_mean": mean},
                   os.path.join(out_dir, f"{model_name}_fold{fold}.pth"))

        pd.DataFrame(all_hist).to_csv(os.path.join(out_dir, f"{model_name}_history.csv"),
                                      index=False)
        pd.DataFrame(fold_rows).to_csv(os.path.join(out_dir, f"{model_name}_folds.csv"),
                                       index=False)
        if miss_rows:
            pd.DataFrame(miss_rows).to_csv(os.path.join(out_dir, f"{model_name}_missing.csv"),
                                           index=False)

    return pd.DataFrame(fold_rows), pd.DataFrame(miss_rows), pd.DataFrame(all_hist)


# ------------------------------------------------------------- reporting
def report(folds_df: pd.DataFrame, miss_df=None, title="RESULTS"):
    print(f"\n=== {title} — per fold (± over repeated MC evaluations) ===")
    for _, r in folds_df.iterrows():
        print(f"fold {int(r.fold)}  acc {r.accuracy_mean:.3f}±{r.accuracy_std:.3f}  "
              f"bacc {r.balanced_acc_mean:.3f}  auc {r.auc_mean:.3f}±{r.auc_std:.3f}  "
              f"sens {r.sensitivity_mean:.3f}  spec {r.specificity_mean:.3f}  "
              f"ece {r.ece_mean:.3f}  thr {r.threshold:.2f}")

    print(f"\n=== {title} — pooled (mean ± std over folds) ===")
    for k in ("accuracy", "balanced_acc", "auc", "f1", "precision",
              "sensitivity", "specificity", "ece", "brier"):
        v = folds_df[f"{k}_mean"]
        print(f"  {k:14s} {v.mean():.4f} ± {v.std():.4f}   [{v.min():.3f}, {v.max():.3f}]")
    if "balanced_acc_thr05" in folds_df:
        print(f"  {'bacc @thr=0.5':14s} {folds_df.balanced_acc_thr05.mean():.4f}"
              f"   (tuned threshold gives {folds_df.balanced_acc_mean.mean():.4f})")

    if miss_df is not None and len(miss_df):
        print(f"\n=== {title} — missing modality (mean over folds) ===")
        print(f"{'condition':16s} {'acc':>7} {'bacc':>7} {'auc':>7} {'f1':>7}")
        print(f"{'both present':16s} {folds_df.accuracy_mean.mean():7.3f} "
              f"{folds_df.balanced_acc_mean.mean():7.3f} "
              f"{folds_df.auc_mean.mean():7.3f} {folds_df.f1_mean.mean():7.3f}")
        for name in ("mri", "pet"):
            s = miss_df[miss_df.missing == name]
            if len(s):
                print(f"{name.upper() + ' missing':16s} {s.accuracy_mean.mean():7.3f} "
                      f"{s.balanced_acc_mean.mean():7.3f} {s.auc_mean.mean():7.3f} "
                      f"{s.f1_mean.mean():7.3f}")


def save_summary(tables: dict, path=None):
    path = path or os.path.join(Config.OUT_DIR, "summary.json")
    out = {}
    for name, df in tables.items():
        out[name] = {k: [float(df[f"{k}_mean"].mean()), float(df[f"{k}_mean"].std())]
                     for k in ("accuracy", "balanced_acc", "auc", "f1",
                               "sensitivity", "specificity", "ece", "brier")}
    json.dump(out, open(path, "w"), indent=2)
    print("saved ->", path)
    return out
