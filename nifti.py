"""DICOM -> NIfTI conversion and N4 bias-field correction.

First stage of the structural MRI pipeline. Everything downstream
(FastSurfer / SynthSeg, registration, volumetry) expects NIfTI with a
correct affine, which is what SimpleITK's series reader gives us —
slice order and spacing come from the DICOM headers rather than from
filename sorting.
"""
import os
from typing import Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm.auto import tqdm

from .config import Config

NIFTI_DIR = "/mnt/External/Seagate/dawood/datasets/ADNI_new/nifti"


def dicom_to_nifti(dicom_dir: str, out_path: str, overwrite=False) -> Optional[str]:
    """Read a DICOM series into a NIfTI volume with correct geometry."""
    if os.path.exists(out_path) and not overwrite:
        return out_path
    try:
        reader = sitk.ImageSeriesReader()
        ids = reader.GetGDCMSeriesIDs(dicom_dir)
        if not ids:
            return None
        # a folder occasionally holds more than one series; take the longest
        best, best_n = None, -1
        for sid in ids:
            files = reader.GetGDCMSeriesFileNames(dicom_dir, sid)
            if len(files) > best_n:
                best, best_n = files, len(files)
        reader.SetFileNames(best)
        img = reader.Execute()
        if img.GetDimension() != 3 or min(img.GetSize()) < 10:
            return None
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        sitk.WriteImage(sitk.Cast(img, sitk.sitkFloat32), out_path)
        return out_path
    except Exception as e:
        print(f"    convert failed {dicom_dir}: {e}")
        return None


def n4_bias_correct(in_path: str, out_path: str, shrink=4, overwrite=False) -> Optional[str]:
    """N4 inhomogeneity correction (SimpleITK — no ANTs install needed).

    Scanner bias fields differ by vendor and coil; leaving them in is one of
    the ways site leaks into intensity-based features.
    """
    if os.path.exists(out_path) and not overwrite:
        return out_path
    try:
        img = sitk.ReadImage(in_path, sitk.sitkFloat32)
        mask = sitk.OtsuThreshold(img, 0, 1, 200)

        small = sitk.Shrink(img, [shrink] * img.GetDimension())
        small_mask = sitk.Shrink(mask, [shrink] * img.GetDimension())

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([50] * 4)
        corrector.Execute(small, small_mask)

        log_bias = corrector.GetLogBiasFieldAsImage(img)
        out = img / sitk.Exp(log_bias)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        sitk.WriteImage(sitk.Cast(out, sitk.sitkFloat32), out_path)
        return out_path
    except Exception as e:
        print(f"    N4 failed {in_path}: {e}")
        return None


def convert_cohort(cohort: pd.DataFrame, modality="mri", n4=True,
                   nifti_dir=NIFTI_DIR, overwrite=False) -> pd.DataFrame:
    """Convert every subject's series; returns the cohort with paths attached."""
    col = "t1_path" if modality == "mri" else "pet_path"
    raw_dir = os.path.join(nifti_dir, f"{modality}_raw")
    n4_dir = os.path.join(nifti_dir, f"{modality}_n4")

    rows = []
    for r in tqdm(cohort.to_dict("records"), desc=f"{modality} -> nifti"):
        s = r["subject"]
        raw = os.path.join(raw_dir, f"{s}.nii.gz")
        got = dicom_to_nifti(r[col], raw, overwrite)
        rec = dict(r)
        rec[f"{modality}_nii"] = got or ""
        if got and n4:
            corrected = n4_bias_correct(got, os.path.join(n4_dir, f"{s}.nii.gz"),
                                        overwrite=overwrite)
            rec[f"{modality}_nii_n4"] = corrected or ""
        rows.append(rec)

    out = pd.DataFrame(rows)
    ok = (out[f"{modality}_nii"] != "").sum()
    print(f"converted {ok} / {len(out)}")
    if n4:
        print(f"N4 corrected {(out.get(f'{modality}_nii_n4', pd.Series(dtype=str)) != '').sum()}")
    return out


def check_volume(path: str) -> dict:
    img = sitk.ReadImage(path)
    a = sitk.GetArrayFromImage(img)
    return {"size": img.GetSize(), "spacing": tuple(round(x, 3) for x in img.GetSpacing()),
            "min": float(a.min()), "max": float(a.max()), "mean": float(a.mean())}
