"""Structural MRI pipeline: DICOM -> N4 -> FastSurfer segmentation -> volumes.

Why this exists: a 3D CNN on raw 96^3 volumes scored 0.494 AUC (chance) on
MRI. The signal in AD is hippocampal / medial-temporal atrophy — small,
localised, and unlearnable from 203 training subjects without being told
where to look. This pipeline measures those structures directly, which is
the same "derived features + linear model" pattern that already beat the
CNN on PET.

Order of operations:
    convert_cohort()     DICOM -> NIfTI, N4 bias correction       ~10 s/subject
    segment_subject()    FastSurfer VINN whole-brain segmentation ~90 s/subject
    extract_volumes()    label map -> structure volumes           instant
    build_volume_table() the whole cohort as a feature table
    harmonize()          ComBat by scanner vendor
"""
import os
import subprocess
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd

FASTSURFER = "/mnt/External/Seagate/dawood/tools/FastSurfer"
SEG_DIR = "/mnt/External/Seagate/dawood/datasets/ADNI_new/fastsurfer"

# FreeSurfer aseg label ids -> names. These are the structures with
# established AD effects, plus the normalisers.
ASEG_LABELS = {
    17: "L_Hippocampus", 53: "R_Hippocampus",
    18: "L_Amygdala", 54: "R_Amygdala",
    4: "L_LatVent", 43: "R_LatVent",
    5: "L_InfLatVent", 44: "R_InfLatVent",
    10: "L_Thalamus", 49: "R_Thalamus",
    11: "L_Caudate", 50: "R_Caudate",
    12: "L_Putamen", 51: "R_Putamen",
    13: "L_Pallidum", 52: "R_Pallidum",
    26: "L_Accumbens", 58: "R_Accumbens",
    28: "L_VentralDC", 60: "R_VentralDC",
    2: "L_WM", 41: "R_WM",
    3: "L_Cortex", 42: "R_Cortex",
    7: "L_CerebellumWM", 46: "R_CerebellumWM",
    8: "L_CerebellumCortex", 47: "R_CerebellumCortex",
    14: "Vent3rd", 15: "Vent4th", 16: "BrainStem", 24: "CSF",
}

# structures whose volume is expected to fall (atrophy) or rise (ex-vacuo) in AD
AD_SENSITIVE = ["Hippocampus", "Amygdala", "InfLatVent", "LatVent", "Cortex"]


# ------------------------------------------------------------- segmentation
def segment_subject(t1_nii: str, subject: str, seg_dir: str = SEG_DIR,
                    device: str = "cuda:0", vox_size: float = 1.0,
                    batch_size: int = 4, overwrite: bool = False,
                    verbose: bool = False) -> Optional[str]:
    """Run FastSurfer VINN. Returns the path to aseg, or None on failure."""
    aseg = os.path.join(seg_dir, subject, "mri", "aseg.auto_noCCseg.mgz")
    if os.path.exists(aseg) and not overwrite:
        return aseg
    os.makedirs(seg_dir, exist_ok=True)

    env = dict(os.environ, PYTHONPATH=FASTSURFER)
    cmd = ["python3", os.path.join(FASTSURFER, "FastSurferCNN", "run_prediction.py"),
           "--t1", t1_nii, "--sid", subject, "--sd", seg_dir,
           "--vox_size", str(vox_size), "--device", device,
           "--viewagg_device", "cpu", "--batch_size", str(batch_size)]
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        print(f"  {subject}: segmentation timed out")
        return None
    if not os.path.exists(aseg):
        print(f"  {subject}: segmentation failed (rc={r.returncode})")
        if verbose:
            print(r.stderr[-1500:])
        return None
    return aseg


# ------------------------------------------------------------------ volumes
def extract_volumes(aseg_path: str) -> Dict[str, float]:
    """Voxel counts x voxel volume, in mm^3, plus normalised ratios."""
    import nibabel as nib
    img = nib.load(aseg_path)
    a = np.asarray(img.dataobj)
    vox = float(np.prod(img.header.get_zooms()))

    out = {}
    for lid, name in ASEG_LABELS.items():
        out[name] = float((a == lid).sum() * vox)

    tbv = float((a > 0).sum() * vox)          # total segmented volume
    out["TotalBrainVolume"] = tbv

    # bilateral sums, which is what the AD literature reports
    for base in ("Hippocampus", "Amygdala", "LatVent", "InfLatVent", "Thalamus",
                 "Caudate", "Putamen", "Pallidum", "Cortex", "WM"):
        out[f"bi_{base}"] = out.get(f"L_{base}", 0.0) + out.get(f"R_{base}", 0.0)

    # head-size normalisation: absolute volumes track skull size as much as
    # disease, so every structure also gets a fraction-of-brain version
    if tbv > 0:
        for k in list(out):
            if k != "TotalBrainVolume":
                out[f"{k}_frac"] = out[k] / tbv

    # left-right asymmetry — AD atrophy is often asymmetric early on
    for base in ("Hippocampus", "Amygdala"):
        l, r = out.get(f"L_{base}", 0.0), out.get(f"R_{base}", 0.0)
        out[f"asym_{base}"] = abs(l - r) / max(l + r, 1e-6)

    return out


def qc_stats(vols: Dict[str, float]) -> Dict[str, float]:
    """A few numbers to eyeball for plausibility."""
    return {
        "TBV_litres": vols["TotalBrainVolume"] / 1e6,
        "hippocampus_cm3": vols["bi_Hippocampus"] / 1e3,
        "ventricles_cm3": vols["bi_LatVent"] / 1e3,
        "cortex_cm3": vols["bi_Cortex"] / 1e3,
        "hippo_frac_pct": 100 * vols["bi_Hippocampus_frac"],
    }


PLAUSIBLE = {
    "TBV_litres": (0.85, 1.65),
    "hippocampus_cm3": (3.0, 11.0),
    "ventricles_cm3": (5.0, 150.0),
    "cortex_cm3": (250.0, 700.0),
}


def qc_flags(stats: Dict[str, float]) -> list:
    """Return the names of any measurement outside a physiological range."""
    bad = []
    for k, (lo, hi) in PLAUSIBLE.items():
        if k in stats and not (lo <= stats[k] <= hi):
            bad.append(f"{k}={stats[k]:.2f} outside [{lo}, {hi}]")
    return bad


# ------------------------------------------------------------------- cohort
def build_volume_table(cohort: pd.DataFrame, seg_dir: str = SEG_DIR,
                       verbose: bool = True) -> pd.DataFrame:
    """Collect volumes for every subject that has been segmented."""
    rows, missing = [], []
    for r in cohort.to_dict("records"):
        s = r["subject"]
        aseg = os.path.join(seg_dir, s, "mri", "aseg.auto_noCCseg.mgz")
        if not os.path.exists(aseg):
            missing.append(s)
            continue
        v = extract_volumes(aseg)
        rows.append({"subject": s, "group": r["group"], "label": r["label"],
                     "sex": r["sex"], "age": r["age"],
                     "manufacturer_raw": r.get("manufacturer", ""),
                     "manufacturer": normalize_vendor(r.get("manufacturer", "")),
                     "site": s[:3], **v})
    df = pd.DataFrame(rows)
    if verbose:
        print(f"volumes for {len(df)} / {len(cohort)} subjects")
        if missing:
            print(f"  not yet segmented: {len(missing)}  e.g. {missing[:5]}")
    return df


def build_feature_table(cohort: pd.DataFrame, verbose=True):
    """Everything the tabular models need, rebuilt from disk.

    Returns (fused, vol_df, mri_cols, pet_cols). Cheap — the segmentations and
    the PET cache are already on disk — so notebook cells can call this instead
    of depending on variables left over from an earlier cell.
    """
    from .data import tabular_features

    vol_df = build_volume_table(cohort, verbose=verbose)
    mri_cols = [c for c in vol_df.columns
                if c.endswith("_frac") or c.startswith("asym_")]

    X, names = tabular_features(cohort, ("pet",))
    pet = pd.DataFrame(X, columns=names)
    pet["subject"] = cohort["subject"].values
    # tabular_features appends age and sex_M, which vol_df already carries;
    # leaving both in makes the merge emit age_x / age_y
    pet = pet.drop(columns=[c for c in ("age", "sex_M") if c in pet.columns])

    fused = vol_df.merge(pet, on="subject", how="inner")
    pet_cols = [c for c in names if c.startswith("pet_")]
    if verbose:
        print(f"fused: {len(fused)} subjects | {len(mri_cols)} MRI + "
              f"{len(pet_cols)} PET features")
    return fused, vol_df, mri_cols, pet_cols


def segment_cohort(cohort: pd.DataFrame, nii_col="mri_nii_n4",
                   seg_dir: str = SEG_DIR, device: str = "cuda:0",
                   overwrite: bool = False, limit: Optional[int] = None):
    """Segment every subject. ~90 s each — 254 subjects is roughly 6 hours."""
    recs = cohort.to_dict("records")[:limit]
    done, failed, t0 = 0, [], time.time()
    for i, r in enumerate(recs, 1):
        s, nii = r["subject"], r.get(nii_col, "")
        if not nii or not os.path.exists(nii):
            failed.append(s)
            continue
        out = segment_subject(nii, s, seg_dir, device, overwrite=overwrite)
        if out:
            done += 1
        else:
            failed.append(s)
        el = time.time() - t0
        print(f"[{i}/{len(recs)}] {s}  ok={done} failed={len(failed)}  "
              f"elapsed {el/60:.1f}m  eta {(el/i)*(len(recs)-i)/60:.1f}m", flush=True)
    print(f"\nsegmented {done}, failed {len(failed)}")
    if failed:
        print("failed:", failed[:20])
    return done, failed


# -------------------------------------------------------------- harmonisation
def normalize_vendor(s: str) -> str:
    """Collapse DICOM manufacturer strings to the actual vendor.

    ADNI carries several spellings per vendor ("Philips Medical Systems" vs
    "Philips Healthcare", "SIEMENS" vs "Siemens Healthineers"). Treating them
    as separate ComBat batches creates batches of n=1, which have no estimable
    variance and produce NaN.
    """
    u = str(s).upper()
    if "GE " in u or u.startswith("GE"):
        return "GE"
    if "PHILIPS" in u:
        return "Philips"
    if "SIEMENS" in u:
        return "Siemens"
    return "other"


def _combat_covars(df: pd.DataFrame, batch_col: str, covars) -> pd.DataFrame:
    cov = pd.DataFrame({"batch": df[batch_col].values})
    for c in covars:
        if c == "sex":
            cov["sex"] = (df["sex"] == "M").astype(int).values
        elif c in df:
            cov[c] = df[c].values
    return cov


def combat_fit_apply(train_df: pd.DataFrame, other_dfs, feature_cols,
                     batch_col="manufacturer", covars=("age", "sex", "label"),
                     min_batch=5, verbose=False):
    """Fit ComBat on the training fold only, then apply to held-out folds.

    Fitting on the whole cohort — which is what `harmonize()` below does — lets
    the transform see test subjects *and* their labels, so the folds are no
    longer independent. Here the estimates come from training data only, and
    `neuroCombatFromTraining` applies them without needing held-out labels.

    Batches with fewer than `min_batch` training subjects are left
    unharmonised: their location/scale cannot be estimated, and forcing it
    yields NaN or wild corrections. Same for any batch a held-out fold contains
    that training never saw.

    Returns (train_out, [other_out, ...]) as feature matrices.
    """
    from neuroCombat import neuroCombat, neuroCombatFromTraining

    counts = train_df[batch_col].value_counts()
    good = set(counts[counts >= min_batch].index)
    if verbose and set(counts.index) - good:
        skipped = {b: int(counts[b]) for b in set(counts.index) - good}
        print(f"    batches too small to harmonise, left raw: {skipped}")
    if len(good) < 2:
        if verbose:
            print("    fewer than 2 usable batches; skipping ComBat")
        return (train_df[feature_cols].values,
                [d[feature_cols].values if d is not None and len(d) else
                 np.empty((0, len(feature_cols))) for d in other_dfs])

    m_tr = train_df[batch_col].isin(good).values
    sub = train_df[m_tr]
    cov = _combat_covars(sub, batch_col, covars)
    cats = [c for c in ("sex", "label") if c in cov.columns]
    cons = [c for c in ("age",) if c in cov.columns]

    fitted = neuroCombat(dat=sub[feature_cols].values.T, covars=cov,
                         batch_col="batch", categorical_cols=cats,
                         continuous_cols=cons)
    est = fitted["estimates"]

    train_out = train_df[feature_cols].values.copy()
    adj = fitted["data"].T
    if np.isfinite(adj).all():
        train_out[m_tr] = adj
    elif verbose:
        print("    ComBat produced non-finite values on train; keeping raw")

    outs = []
    for df in other_dfs:
        if df is None or len(df) == 0:
            outs.append(np.empty((0, len(feature_cols))))
            continue
        X = df[feature_cols].values.copy()
        m = df[batch_col].isin(good).values
        if m.any():
            res = neuroCombatFromTraining(dat=df.loc[m, feature_cols].values.T,
                                          batch=df.loc[m, batch_col].values,
                                          estimates=est)
            a = res["data"].T
            if np.isfinite(a).all():
                X[m] = a
            elif verbose:
                print("    ComBat produced non-finite values on held-out; keeping raw")
        outs.append(X)
    return train_out, outs


def volume_cv(cohort: pd.DataFrame, vol_df: pd.DataFrame, feature_cols=None,
              harmonize_folds=True, batch_col="manufacturer", add_demo=True,
              C=1.0, tag="logreg MRI-vol", verbose=True, group_by="site",
              harmonize_cols=None, quiet_combat=True):
    """Volumetric logistic regression with fold-wise ComBat.

    Everything that learns anything — ComBat, the scaler, the classifier and
    the decision threshold — is fitted inside the training/validation part of
    each fold. The test fold contributes nothing but its predictions.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from .engine import make_folds
    from .metrics import compute_metrics, fit_threshold

    if feature_cols is None:
        feature_cols = [c for c in vol_df.columns
                        if c.endswith("_frac") or c.startswith("asym_")]
    feature_cols = [c for c in feature_cols
                    if vol_df[c].std() > 1e-12 and vol_df[c].notna().all()]

    # ComBat applies to the structural block only by default; PET intensity
    # summaries have no vendor correction to inherit.
    if harmonize_cols is None:
        harmonize_cols = [c for c in feature_cols
                          if c.endswith("_frac") or c.startswith("asym_")]
    harmonize_cols = [c for c in harmonize_cols if c in feature_cols]
    h_idx = [feature_cols.index(c) for c in harmonize_cols]

    vol = vol_df.set_index("subject")
    rows = []
    for fold, (tr, va, te) in enumerate(make_folds(cohort, group_by=group_by), 1):
        parts = []
        for split in (tr, va, te):
            keep = [s for s in split.subject if s in vol.index]
            parts.append(vol.loc[keep].reset_index())
        tr_v, va_v, te_v = parts
        if min(len(tr_v), len(va_v), len(te_v)) == 0 or tr_v.label.nunique() < 2:
            continue

        raw = tuple(d[feature_cols].values for d in (tr_v, va_v, te_v))
        Xtr, Xva, Xte = (r.copy() for r in raw)
        if harmonize_folds and h_idx:
            try:
                import contextlib
                import io
                sink = io.StringIO() if quiet_combat else None
                ctx = (contextlib.redirect_stdout(sink) if quiet_combat
                       else contextlib.nullcontext())
                with ctx:
                    Htr, (Hva, Hte) = combat_fit_apply(
                        tr_v, [va_v, te_v], harmonize_cols, batch_col,
                        verbose=not quiet_combat)
                Xtr[:, h_idx], Xva[:, h_idx], Xte[:, h_idx] = Htr, Hva, Hte
            except Exception as e:
                print(f"    fold {fold}: ComBat failed ({e}); using raw features")
                Xtr, Xva, Xte = raw

        # last line of defence: never hand NaN/inf to the classifier
        if not all(np.isfinite(X).all() for X in (Xtr, Xva, Xte)):
            n_bad = sum(int((~np.isfinite(X)).sum()) for X in (Xtr, Xva, Xte))
            print(f"    fold {fold}: {n_bad} non-finite values after "
                  f"harmonisation; falling back to raw features")
            Xtr, Xva, Xte = raw

        if add_demo:
            def demo(d):
                return np.column_stack([d.age.values,
                                        (d.sex == "M").astype(float).values])
            Xtr = np.column_stack([Xtr, demo(tr_v)])
            Xva = np.column_stack([Xva, demo(va_v)])
            Xte = np.column_stack([Xte, demo(te_v)])

        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
        clf.fit(Xtr, tr_v.label.values)

        def lg(X):
            return np.log(np.clip(clf.predict_proba(X), 1e-9, 1))

        thr = fit_threshold(lg(Xva), va_v.label.values)
        m = compute_metrics(lg(Xte), te_v.label.values, thr=thr)
        m.update({"fold": fold, "model": tag, "n_test": len(te_v)})
        rows.append(m)
        if verbose:
            print(f"  fold {fold}: n={len(te_v):3d}  acc {m['accuracy']:.3f}  "
                  f"bacc {m['balanced_acc']:.3f}  auc {m['auc']:.3f}  "
                  f"sens {m['sensitivity']:.3f}  spec {m['specificity']:.3f}")

    df = pd.DataFrame(rows)
    for k in ("accuracy", "balanced_acc", "auc", "f1", "precision",
              "sensitivity", "specificity", "ece", "brier"):
        df[f"{k}_mean"] = df[k]
        df[f"{k}_std"] = 0.0
    return df


def harmonize(vol_df: pd.DataFrame, batch_col="manufacturer",
              feature_cols=None, covars=("age", "sex", "label"),
              preserve_label=True) -> pd.DataFrame:
    """ComBat across scanner vendors.

    Site is too fine-grained here — many sites have 1-7 subjects and ComBat's
    per-batch location/scale estimates become unstable. Vendor gives three
    reasonably sized batches (GE 70, Siemens 138, Philips 46).

    `label` is passed as a covariate to preserve so harmonisation removes
    scanner effects without also removing the disease effect.

    WARNING — leaky. This fits on the whole cohort, so the transform sees
    test subjects and their labels. Kept only for the leaky-vs-clean
    comparison; use `volume_cv(..., harmonize_folds=True)` for anything
    you intend to report.
    """
    try:
        from neuroCombat import neuroCombat
    except ImportError:
        print("neuroCombat not installed — returning data unharmonised.\n"
              "  pip install --no-deps neuroCombat")
        return vol_df.copy()

    if feature_cols is None:
        feature_cols = [c for c in vol_df.columns
                        if c.endswith("_frac") or c.startswith("asym_")]
    feature_cols = [c for c in feature_cols
                    if vol_df[c].std() > 1e-12 and vol_df[c].notna().all()]

    batches = vol_df[batch_col].fillna("unknown")
    keep = batches.map(batches.value_counts()) >= 10
    if (~keep).any():
        print(f"  dropping {(~keep).sum()} subjects in batches with <10 members")
    df = vol_df[keep].reset_index(drop=True)

    cov = pd.DataFrame({"batch": df[batch_col].values})
    for c in covars:
        if c == "sex":
            cov["sex"] = (df["sex"] == "M").astype(int).values
        elif c in df:
            cov[c] = df[c].values
    keep_cols = [c for c in cov.columns if c != "batch"]
    if not preserve_label and "label" in keep_cols:
        keep_cols.remove("label")

    data = df[feature_cols].values.T          # neuroCombat wants features x samples
    res = neuroCombat(dat=data, covars=cov, batch_col="batch",
                      categorical_cols=[c for c in ("sex", "label") if c in keep_cols],
                      continuous_cols=[c for c in ("age",) if c in keep_cols])
    out = df.copy()
    out[feature_cols] = res["data"].T
    print(f"harmonised {len(feature_cols)} features across "
          f"{df[batch_col].nunique()} {batch_col} batches")
    return out
