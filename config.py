"""Configuration, seeding and device selection."""
import os
import random

import numpy as np
import torch


class Config:
    # ---- paths ------------------------------------------------------
    MANIFEST  = "/home/dawood/lab2_rotaion/CMUPF ISBI/cohort_manifest.csv"
    CACHE_DIR = "/mnt/External/Seagate/dawood/datasets/ADNI_new/cache"
    OUT_DIR   = "/mnt/External/Seagate/dawood/datasets/ADNI_new/runs/v3"

    # ---- cohort -----------------------------------------------------
    USE_MATCHED_ONLY = True
    IMG_SIZE   = (96, 96, 96)
    MIN_SLICES = 20
    LABELS      = {"CN": 0, "AD": 1}
    CLASS_NAMES = ["CN", "AD"]
    NUM_CLASSES = 2

    # ---- model ------------------------------------------------------
    FEATURE_DIM = 256
    SHARED_DIM  = 64
    DROPOUT     = 0.3

    LEGACY = False               # reproduce the original (broken) behaviour

    ATTENTION_MODE       = "per_sample"   # "original" = cross-sample leak
    IMPUTE_FROM_TRANSFER = True
    CONSISTENCY_SPACE    = "shared"       # "raw" forces unc_mri == unc_pet
    HETEROSCEDASTIC      = True
    HETERO_SAMPLES       = 8

    MODALITY_DROPOUT   = 0.25
    IMPUTE_UNC_PENALTY = 0.2

    MC_SAMPLES_TRAIN = 5
    MC_SAMPLES_EVAL  = 20

    # ---- loss weights ----------------------------------------------
    TASK_WEIGHT        = 1.0
    AUX_WEIGHT         = 1.0
    CONSISTENCY_WEIGHT = 0.1
    TRANSFER_WEIGHT    = 0.1

    # ---- training ---------------------------------------------------
    SEED         = 42
    N_FOLDS      = 5
    # Group-aware CV. Site predicts diagnosis at AUC 0.640 in this cohort, so
    # letting subjects from one site sit in both train and test lets a model be
    # rewarded for learning scanner signature. "site" keeps each site whole.
    GROUP_CV_BY  = "site"        # None = plain StratifiedKFold (leaky)
    BATCH_SIZE   = 4
    NUM_EPOCHS   = 80
    EARLY_STOP   = 20
    LR           = 1e-4
    WEIGHT_DECAY = 1e-5
    GRAD_CLIP    = 1.0
    INNER_VAL    = 0.176
    NUM_WORKERS  = 0
    EVAL_REPEATS = 10

    # ---- evaluation --------------------------------------------------
    TUNE_THRESHOLD = True     # Youden's J on validation instead of a fixed 0.5
    MIN_VAL_FOR_CALIB = 20    # below this, skip temperature/threshold fitting

    # ---- baselines ---------------------------------------------------
    BASELINE_EPOCHS = 60

    # ---- tabular CMUPF (derived features) ---------------------------
    # 44 structural volumes + 12 PET summaries is a far smaller input than a
    # 96^3 volume, so the encoders and heads shrink to match.
    TAB_FEATURE_DIM  = 64
    TAB_SHARED_DIM   = 32
    TAB_HIDDEN       = 128
    TAB_EPOCHS       = 400
    TAB_EARLY_STOP   = 60
    TAB_LR           = 1e-3
    TAB_WEIGHT_DECAY = 1e-4
    TAB_BATCH        = 32

    GPU_INDEX = None
    DEVICE    = None


def apply_legacy():
    """Restore the original behaviour, for ablation against the fixes."""
    if not Config.LEGACY:
        return
    Config.ATTENTION_MODE = "original"
    Config.IMPUTE_FROM_TRANSFER = False
    Config.CONSISTENCY_SPACE = "raw"
    Config.HETEROSCEDASTIC = False
    Config.MODALITY_DROPOUT = 0.0
    Config.TUNE_THRESHOLD = False


def pick_gpu(verbose=True) -> str:
    """Shared machine: take the emptiest GPU rather than always cuda:0."""
    if not torch.cuda.is_available():
        return "cpu"
    if Config.GPU_INDEX is not None:
        return f"cuda:{Config.GPU_INDEX}"
    free = []
    for i in range(torch.cuda.device_count()):
        try:
            f, _ = torch.cuda.mem_get_info(i)
        except Exception:
            f = 0
        free.append((f, i))
    f, i = max(free)
    if verbose:
        print(f"auto-selected cuda:{i} ({f / 1e9:.1f} GB free)")
    return f"cuda:{i}"


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def setup(verbose=True):
    apply_legacy()
    Config.DEVICE = pick_gpu(verbose)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.OUT_DIR, exist_ok=True)
    set_seed(Config.SEED)
    if verbose:
        print("device:", Config.DEVICE, "| legacy:", Config.LEGACY)
    return Config
