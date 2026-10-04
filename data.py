"""Cohort loading, matching, DICOM caching and torch datasets."""
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pydicom
import torch
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .config import Config


# --------------------------------------------------------------- cohort
def load_cohort() -> pd.DataFrame:
    df = pd.read_csv(Config.MANIFEST)
    df = df[(df["complete"] == True) & (df["group"].isin(Config.LABELS))].copy()
    df["label"] = df["group"].map(Config.LABELS)
    df["age"] = df["age"].astype(float)
    # ADNI subject ids are SSS_S_NNNN, where SSS is the acquisition site
    df["site"] = df["subject"].str[:3]
    return df.reset_index(drop=True)


def match_pairs(df: pd.DataFrame, age_tol: float = 4.0) -> pd.DataFrame:
    """Greedy 1:1 AD<->CN match on sex + field strength, nearest age."""
    ad = df[df.group == "AD"].sort_values("age").to_dict("records")
    cn = df[df.group == "CN"].to_dict("records")
    used, keep = set(), []
    for a in ad:
        best, gap = None, None
        for c in cn:
            if (c["subject"] in used or c["sex"] != a["sex"]
                    or c["field_strength"] != a["field_strength"]):
                continue
            g = abs(c["age"] - a["age"])
            if g > age_tol:
                continue
            if gap is None or g < gap:
                gap, best = g, c
        if best is not None:
            used.add(best["subject"])
            keep += [a, best]
    return pd.DataFrame(keep).reset_index(drop=True)


def describe(df: pd.DataFrame):
    print(f"n = {len(df)}  {df.group.value_counts().to_dict()}")
    for g, s in df.groupby("group"):
        print(f"  {g}: n={len(s)}  age {s.age.mean():.1f} ± {s.age.std():.1f}  "
              f"F={(s.sex == 'F').sum()}  3T={(s.field_strength == 3.0).sum()}")


# ---------------------------------------------------------- preprocessing
class VolumePreprocessor:
    @staticmethod
    def load_dicom_series(folder: str) -> Optional[np.ndarray]:
        if not os.path.isdir(folder):
            return None
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".dcm"))
        if len(files) < Config.MIN_SLICES:
            return None
        try:
            head = pydicom.dcmread(os.path.join(folder, files[0]), force=True)
            h, w = head.pixel_array.shape
        except Exception:
            return None

        keyed = []
        for f in files:
            try:
                d = pydicom.dcmread(os.path.join(folder, f), force=True,
                                    stop_before_pixels=True)
                z = (float(d.ImagePositionPatient[2])
                     if hasattr(d, "ImagePositionPatient")
                     else float(getattr(d, "InstanceNumber", 0)))
            except Exception:
                z = 0.0
            keyed.append((z, f))
        keyed.sort()

        vol = np.zeros((h, w, len(keyed)), dtype=np.float32)
        bad = 0
        for i, (_, f) in enumerate(keyed):
            try:
                vol[:, :, i] = pydicom.dcmread(
                    os.path.join(folder, f), force=True).pixel_array.astype(np.float32)
            except Exception:
                bad += 1
                if bad > 0.1 * len(keyed):
                    return None
        return vol

    @staticmethod
    def preprocess(v: np.ndarray) -> np.ndarray:
        m, s = v.mean(), v.std()
        v = (v - m) / s if s > 0 else v - m
        return ndimage.zoom(v, [t / x for t, x in zip(Config.IMG_SIZE, v.shape)], order=1)


def cache_path(subject: str, mod: str) -> str:
    tag = "x".join(map(str, Config.IMG_SIZE))
    return os.path.join(Config.CACHE_DIR, f"{subject}_{mod}_{tag}.npy")


def build_cache(df: pd.DataFrame, verbose=True) -> pd.DataFrame:
    ok = []
    for r in tqdm(df.to_dict("records"), desc="cache", disable=not verbose):
        good = True
        for mod, col in (("mri", "t1_path"), ("pet", "pet_path")):
            p = cache_path(r["subject"], mod)
            if os.path.exists(p):
                continue
            v = VolumePreprocessor.load_dicom_series(r[col])
            if v is None:
                print("  FAILED", r["subject"], mod)
                good = False
                break
            np.save(p, VolumePreprocessor.preprocess(v).astype(np.float32))
        if good:
            ok.append(r)
    out = pd.DataFrame(ok).reset_index(drop=True)
    if verbose:
        print(f"usable: {len(out)} / {len(df)}  {out.group.value_counts().to_dict()}")
    return out


def prepare_cohort(verbose=True) -> pd.DataFrame:
    df = load_cohort()
    if verbose:
        print("complete in manifest:")
        describe(df)
    if Config.USE_MATCHED_ONLY:
        df = match_pairs(df)
        if verbose:
            print("\nafter 1:1 matching:")
            describe(df)
    return build_cache(df, verbose)


# --------------------------------------------------------------- dataset
class CMUPFDataset(Dataset):
    """Reads the cached .npy volumes. `modalities` selects which to return."""

    def __init__(self, df: pd.DataFrame, is_training: bool = True,
                 modalities=("mri", "pet")):
        self.records = df.to_dict("records")
        self.is_training = is_training
        self.modalities = tuple(modalities)

    def __len__(self):
        return len(self.records)

    def _aug(self, v: np.ndarray) -> np.ndarray:
        if np.random.rand() < 0.5:
            v = np.flip(v, axis=0)
        v = v * np.random.uniform(0.95, 1.05) + np.random.uniform(-0.05, 0.05)
        return np.ascontiguousarray(v)

    def __getitem__(self, i):
        r = self.records[i]
        out = {"label": torch.tensor(int(r["label"]), dtype=torch.long),
               "subject_id": r["subject"], "group": r["group"]}
        for m in self.modalities:
            v = np.load(cache_path(r["subject"], m))
            if self.is_training:
                v = self._aug(v)
            out[m] = torch.from_numpy(v).float().unsqueeze(0)
        return out


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    out = {}
    for k in batch[0]:
        out[k] = ([b[k] for b in batch] if k in ("subject_id", "group")
                  else torch.stack([b[k] for b in batch]))
    return out


def make_loader(df: pd.DataFrame, train: bool, modalities=("mri", "pet")) -> DataLoader:
    return DataLoader(
        CMUPFDataset(df, train, modalities),
        batch_size=Config.BATCH_SIZE, shuffle=train,
        num_workers=Config.NUM_WORKERS, collate_fn=collate,
        drop_last=train and len(df) > Config.BATCH_SIZE)


# ------------------------------------------------- tabular features (baseline)
def volume_summary(subject: str, mod: str) -> np.ndarray:
    """Cheap intensity descriptors from a cached volume."""
    v = np.load(cache_path(subject, mod)).ravel()
    qs = np.percentile(v, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    return np.concatenate([[v.mean(), v.std(), float(np.abs(v).mean())], qs])


def tabular_features(df: pd.DataFrame, modalities=("mri", "pet")) -> Tuple[np.ndarray, list]:
    rows, names = [], []
    for m in modalities:
        names += [f"{m}_{n}" for n in
                  ["mean", "std", "absmean", "p1", "p5", "p10", "p25",
                   "p50", "p75", "p90", "p95", "p99"]]
    names += ["age", "sex_M"]
    for r in df.to_dict("records"):
        f = [volume_summary(r["subject"], m) for m in modalities]
        f.append([r["age"], 1.0 if r["sex"] == "M" else 0.0])
        rows.append(np.concatenate(f))
    return np.asarray(rows, dtype=np.float64), names
