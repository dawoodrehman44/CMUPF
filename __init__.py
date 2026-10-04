"""CMUPF — MRI + PET uncertainty-propagation framework for CN vs AD."""
from .config import Config, setup, set_seed
from .data import (prepare_cohort, load_cohort, match_pairs, describe,
                   make_loader, CMUPFDataset, cache_path, tabular_features)
from .models import (CMUPF, CMUPFLoss, SingleModalityCNN, SingleModalityLoss,
                     BayesianEncoder, CMUPFTabular, TabularEncoder)
from .metrics import compute_metrics, fit_temperature, fit_threshold
from .engine import cross_validate, report, save_summary, make_folds, repeated_eval
from .baselines import logistic_cv, compare
from .plots import plot_curves, plot_comparison
from . import nifti, structural, qc, tabular
from .tabular import cross_validate_tabular, ablate

__all__ = [
    "Config", "setup", "set_seed",
    "prepare_cohort", "load_cohort", "match_pairs", "describe",
    "make_loader", "CMUPFDataset", "cache_path", "tabular_features",
    "CMUPF", "CMUPFLoss", "SingleModalityCNN", "SingleModalityLoss",
    "BayesianEncoder",
    "compute_metrics", "fit_temperature", "fit_threshold",
    "cross_validate", "report", "save_summary", "make_folds", "repeated_eval",
    "logistic_cv", "compare",
    "plot_curves", "plot_comparison",
    "nifti", "structural", "qc", "tabular",
    "CMUPFTabular", "TabularEncoder",
    "cross_validate_tabular", "ablate",
]
