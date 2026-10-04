"""CMUPF over derived features, cross-validated the same way as everything else.

This is the fair test of the framework. On raw 96^3 voxels the MRI encoder had
nothing to learn from (CNN-MRI scored 0.494, i.e. chance), so the cross-modal
machinery was being asked to fuse signal with noise. Here both encoders see
representations that demonstrably carry signal — 44 FastSurfer volumes and 12
PET intensity summaries — and the question becomes answerable:

    does uncertainty-guided fusion beat plain concatenation
    into a logistic regression (AUC 0.963)?

Protocol matches the logistic models exactly: site-grouped folds, ComBat fitted
on the training fold only and applied to the structural block, scaler fitted on
train, temperature and decision threshold fitted on validation, and the test
fold evaluated with repeated MC-dropout passes reported as mean +/- std.
"""
import copy
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from .config import Config, set_seed
from .engine import enable_mc_dropout, make_folds
from .metrics import aggregate, compute_metrics, fit_temperature, fit_threshold
from .models import CMUPFLoss, CMUPFTabular
from .structural import combat_fit_apply



# --------------------------------------------------------------- fold tensors
def _prepare_fold(tr, va, te, table, mri_cols, pet_cols, batch_col,
                  harmonize, add_demo, quiet=True):
    """Slice the feature table, harmonise and scale — all fitted on train."""
    t = table.set_index("subject")
    parts = []
    for split in (tr, va, te):
        keep = [s for s in split.subject if s in t.index]
        parts.append(t.loc[keep].reset_index())
    tr_v, va_v, te_v = parts

    Xm = [d[mri_cols].values.astype(np.float64) for d in (tr_v, va_v, te_v)]
    if harmonize and mri_cols:
        try:
            import contextlib
            import io
            ctx = (contextlib.redirect_stdout(io.StringIO()) if quiet
                   else contextlib.nullcontext())
            with ctx:
                h_tr, (h_va, h_te) = combat_fit_apply(tr_v, [va_v, te_v], mri_cols,
                                                      batch_col, verbose=not quiet)
            cand = [h_tr, h_va, h_te]
            if all(np.isfinite(c).all() for c in cand):
                Xm = cand
            else:
                print("    ComBat gave non-finite values; using raw volumes")
        except Exception as e:
            print(f"    ComBat failed ({e}); using raw volumes")

    Xp = [d[pet_cols].values.astype(np.float64) for d in (tr_v, va_v, te_v)]
    if add_demo:
        for i, d in enumerate((tr_v, va_v, te_v)):
            demo = np.column_stack([d.age.values,
                                    (d.sex == "M").astype(float).values])
            Xm[i] = np.column_stack([Xm[i], demo])

    sm, sp = StandardScaler().fit(Xm[0]), StandardScaler().fit(Xp[0])
    Xm = [sm.transform(x) for x in Xm]
    Xp = [sp.transform(x) for x in Xp]

    dev = Config.DEVICE
    out = []
    for i, d in enumerate((tr_v, va_v, te_v)):
        out.append((torch.tensor(Xm[i], dtype=torch.float32, device=dev),
                    torch.tensor(Xp[i], dtype=torch.float32, device=dev),
                    torch.tensor(d.label.values, dtype=torch.long, device=dev)))
    return out


# -------------------------------------------------------------------- eval
@torch.no_grad()
def _logits(model, m, p, missing=None):
    enable_mc_dropout(model)
    return model(m, p, missing_modality=missing)["logits"].float().cpu().numpy()


def _repeated(model, m, p, y, missing=None, repeats=None, T=1.0, thr=0.5):
    R = repeats or getattr(Config, "EVAL_REPEATS", 10)
    yc = y.cpu().numpy()
    runs = [compute_metrics(_logits(model, m, p, missing), yc, T, thr)
            for _ in range(R)]
    return aggregate(runs)


# ------------------------------------------------------------------- training
def _train_fold(model, crit, tr, va, fold, epochs, verbose):
    m_tr, p_tr, y_tr = tr
    m_va, p_va, y_va = va
    opt = torch.optim.AdamW(model.parameters(), lr=getattr(Config, "TAB_LR", 1e-3),
                            weight_decay=getattr(Config, "TAB_WEIGHT_DECAY", 1e-4))
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                     patience=20)
    n = len(y_tr)
    bs = min(getattr(Config, "TAB_BATCH", 32), max(8, n // 4))
    best_auc, best_state, bad, hist = -1.0, copy.deepcopy(model.state_dict()), 0, []

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=y_tr.device)
        tot, nb = 0.0, 0
        for i in range(0, n - bs + 1, bs):        # drop last partial batch: BatchNorm
            idx = perm[i:i + bs]
            out = model(m_tr[idx], p_tr[idx])
            losses = crit(out, y_tr[idx])
            opt.zero_grad(set_to_none=True)
            losses["total_loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), getattr(Config, "GRAD_CLIP", 1.0))
            opt.step()
            tot += float(losses["total_loss"].detach())
            nb += 1

        va_m = compute_metrics(_logits(model, m_va, p_va), y_va.cpu().numpy())
        auc = va_m["auc"] if np.isfinite(va_m["auc"]) else 0.0
        sch.step(auc)
        hist.append({"model": "cmupf_tab", "fold": fold, "epoch": ep,
                     "train_total_loss": tot / max(nb, 1),
                     "val_auc": auc, "val_accuracy": va_m["accuracy"],
                     "val_balanced_acc": va_m["balanced_acc"], "val_ece": va_m["ece"],
                     "lr": opt.param_groups[0]["lr"]})

        if auc > best_auc:
            best_auc, bad = auc, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= getattr(Config, "TAB_EARLY_STOP", 60):
                break
        if verbose and ep % 50 == 0:
            print(f"    ep {ep:4d}  loss {tot/max(nb,1):.4f}  val auc {auc:.3f}"
                  f"  best {best_auc:.3f}")

    model.load_state_dict(best_state)
    return hist, best_auc, len(hist)


def cross_validate_tabular(cohort, table, mri_cols, pet_cols,
                           batch_col="manufacturer", harmonize=True,
                           add_demo=True, group_by="site", epochs=None,
                           tag="CMUPF-tab", verbose=True, out_dir=None,
                           quiet_combat=True):
    """Site-grouped CV for CMUPFTabular. Minutes, not hours."""
    epochs = epochs or getattr(Config, "TAB_EPOCHS", 400)
    out_dir = out_dir or Config.OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    all_hist, fold_rows, miss_rows = [], [], []
    for fold, (tr, va, te) in enumerate(make_folds(cohort, group_by=group_by), 1):
        set_seed(Config.SEED + fold)
        t0 = time.time()
        TR, VA, TE = _prepare_fold(tr, va, te, table, mri_cols, pet_cols,
                                   batch_col, harmonize, add_demo, quiet_combat)
        model = CMUPFTabular(TR[0].shape[1], TR[1].shape[1]).to(Config.DEVICE)
        crit = CMUPFLoss()

        hist, best_auc, n_ep = _train_fold(model, crit, TR, VA, fold, epochs, verbose)
        all_hist += hist

        val_lg = _logits(model, VA[0], VA[1])
        val_y = VA[2].cpu().numpy()
        T = fit_temperature(val_lg, val_y)
        thr = fit_threshold(val_lg, val_y, T) if getattr(Config, "TUNE_THRESHOLD", True) else 0.5

        mean, std = _repeated(model, TE[0], TE[1], TE[2], None, T=T, thr=thr)
        fixed, _ = _repeated(model, TE[0], TE[1], TE[2], None, T=T, thr=0.5)
        raw, _ = _repeated(model, TE[0], TE[1], TE[2], None, T=1.0, thr=0.5)

        if verbose:
            print(f"  fold {fold}: n={len(TE[2]):3d}  {n_ep:4d} ep  "
                  f"{time.time()-t0:5.1f}s  T={T:.2f} thr={thr:.2f} | "
                  f"acc {mean['accuracy']:.3f}  bacc {mean['balanced_acc']:.3f}  "
                  f"auc {mean['auc']:.3f}±{std['auc']:.3f}  "
                  f"sens {mean['sensitivity']:.3f}  spec {mean['specificity']:.3f}")

        row = {"model": tag, "fold": fold, "n_test": int(len(TE[2])),
               "temperature": T, "threshold": thr, "best_val_auc": best_auc,
               "epochs": n_ep, "ece_raw": raw["ece"],
               "balanced_acc_thr05": fixed["balanced_acc"]}
        row.update({f"{k}_mean": v for k, v in mean.items()})
        row.update({f"{k}_std": v for k, v in std.items()})
        fold_rows.append(row)

        for miss in ("mri", "pet"):
            m, _ = _repeated(model, TE[0], TE[1], TE[2], miss, T=T, thr=thr)
            miss_rows.append({"model": tag, "fold": fold, "missing": miss,
                              **{f"{k}_mean": v for k, v in m.items()}})

        torch.save({"fold": fold, "model_state_dict": model.state_dict(),
                    "temperature": T, "threshold": thr, "test_mean": mean},
                   os.path.join(out_dir, f"{tag}_fold{fold}.pth"))

    F = pd.DataFrame(fold_rows)
    M = pd.DataFrame(miss_rows)
    H = pd.DataFrame(all_hist)
    F.to_csv(os.path.join(out_dir, f"{tag}_folds.csv"), index=False)
    M.to_csv(os.path.join(out_dir, f"{tag}_missing.csv"), index=False)
    H.to_csv(os.path.join(out_dir, f"{tag}_history.csv"), index=False)
    return F, M, H


# ------------------------------------------------------------------ ablation
def _config_objects():
    """Every distinct Config class object the loaded cmupf modules hold.

    `%autoreload` rebuilds the class when config.py changes, and modules that
    already imported it keep their old reference — so `setattr(Config, ...)`
    in one module can be invisible to another. That is how an ablation ends up
    changing nothing and every configuration reports the same number.
    """
    import sys
    objs, seen = [], set()
    for name, mod in list(sys.modules.items()):
        if not name.startswith("cmupf"):
            continue
        C = getattr(mod, "Config", None)
        if isinstance(C, type) and id(C) not in seen:
            seen.add(id(C))
            objs.append(C)
    return objs or [Config]


def _set_everywhere(name, value):
    for C in _config_objects():
        setattr(C, name, value)


def _read_effective(name, default=None):
    """What the model will actually see — read through models.py's Config."""
    import sys
    m = sys.modules.get("cmupf.models")
    C = getattr(m, "Config", Config) if m else Config
    return getattr(C, name, default)


def ablate(cohort, table, mri_cols, pet_cols, **kw):
    """Turn each framework component off in turn, one at a time.

    Only meaningful if the full model beats the logistic baseline. If it does
    not, this tells you which part is responsible for the shortfall.
    """
    keys = ("ATTENTION_MODE", "IMPUTE_FROM_TRANSFER", "CONSISTENCY_SPACE",
            "HETEROSCEDASTIC", "MODALITY_DROPOUT", "CONSISTENCY_WEIGHT",
            "TRANSFER_WEIGHT", "AUX_WEIGHT")
    defaults = {"ATTENTION_MODE": "per_sample", "IMPUTE_FROM_TRANSFER": True,
                "CONSISTENCY_SPACE": "shared", "HETEROSCEDASTIC": True,
                "MODALITY_DROPOUT": 0.25, "CONSISTENCY_WEIGHT": 0.1,
                "TRANSFER_WEIGHT": 0.1, "AUX_WEIGHT": 1.0}
    saved = {k: _read_effective(k, defaults[k]) for k in keys}

    n_cfg = len(_config_objects())
    if n_cfg > 1:
        print(f"note: {n_cfg} distinct Config objects are live (autoreload); "
              f"settings will be applied to all of them")
    settings = {
        "full":                {},
        "no cross-attention":  {"ATTENTION_MODE": "off"},
        "attn, no unc. guide": {"ATTENTION_MODE": "no_guidance"},
        "no consistency loss": {"CONSISTENCY_WEIGHT": 0.0},
        "no transfer loss":    {"TRANSFER_WEIGHT": 0.0},
        "no aux heads":        {"AUX_WEIGHT": 0.0},
        "no heteroscedastic":  {"HETEROSCEDASTIC": False},
        "no modality dropout": {"MODALITY_DROPOUT": 0.0},
    }
    out = {}
    try:
        for name, over in settings.items():
            for k, v in saved.items():
                _set_everywhere(k, v)
            for k, v in over.items():
                _set_everywhere(k, v)

            # confirm the change reached the module the model actually reads,
            # otherwise every configuration silently returns the same number
            effective = {k: _read_effective(k) for k in over}
            if any(effective[k] != v for k, v in over.items()):
                raise RuntimeError(
                    f"ablation '{name}' did not take effect: asked for {over}, "
                    f"model sees {effective}. Restart the kernel and re-run.")

            print(f"\n--- ablation: {name} ---")
            if over:
                print(f"    effective: {effective}")
            F, M, _ = cross_validate_tabular(cohort, table, mri_cols, pet_cols,
                                             tag=f"CMUPF-tab [{name}]", **kw)
            out[name] = F
            miss = {r: M[M.missing == r].auc_mean.mean() for r in ("mri", "pet")}
            print(f"  {name:22s} auc {F.auc_mean.mean():.3f}±{F.auc_mean.std():.3f}"
                  f"  bacc {F.balanced_acc_mean.mean():.3f}"
                  f"  | missing MRI {miss['mri']:.3f}  missing PET {miss['pet']:.3f}")
    finally:
        for k, v in saved.items():
            _set_everywhere(k, v)
    return out
