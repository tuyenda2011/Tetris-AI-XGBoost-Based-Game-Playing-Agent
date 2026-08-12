"""Central configuration for reproducible experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed_1000_r3"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass(frozen=True)
class EnvConfig:
    """Configuration for the fallback Gymnasium-compatible Tetris environment."""

    rows: int = 20
    cols: int = 10
    max_pieces: int = 100000
    seed: int = 42


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for dataset generation and model training."""

    seed: int = 42
    test_size: float = 0.15
    validation_size: float = 0.15
    n_iter_tuning: int = 12
    cv_folds: int = 3


DEFAULT_XGB_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 350,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 2,
    "reg_alpha": 0.01,
    "reg_lambda": 1.0,
    "random_state": 42,
    "tree_method": "hist",
    "device": "cuda",
}

