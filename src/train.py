"""Training orchestration for baseline and XGBoost models."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.config import TrainingConfig
from src.features import FEATURE_NAMES
from src.model import (
    regression_metrics,
    save_model,
    train_baseline,
    train_xgboost,
    tune_xgboost,
)

LOGGER = logging.getLogger(__name__)


def train_from_splits(
    *,
    train_path: Path,
    validation_path: Path,
    test_path: Path,
    model_dir: Path,
    tune: bool,
    config: TrainingConfig,
) -> dict[str, Path]:
    """Train baseline and XGBoost models from saved CSV splits."""

    model_dir.mkdir(parents=True, exist_ok=True)
    missing_paths = [
        path for path in [train_path, validation_path, test_path] if not path.exists()
    ]
    if missing_paths:
        missing = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(
            "Dataset split files are missing. Generate the dataset first with:\n"
            "  python scripts/generate_dataset.py --episodes 20 --max-pieces 120 --seed 42\n"
            f"Missing files:\n{missing}"
        )
    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)
    test = pd.read_csv(test_path)
    if train.empty or validation.empty or test.empty:
        raise ValueError("Train, validation, and test splits must all be non-empty")
    required_columns = set(FEATURE_NAMES + ["target_quality", "target_mode"])
    missing_columns = required_columns.difference(train.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            "Dataset is missing required columns and may be from an older generator. "
            "Regenerate it with:\n"
            "  python scripts/generate_dataset.py --episodes 20 --max-pieces 120 --seed 42\n"
            f"Missing columns: {missing}"
        )

    baseline = train_baseline(train)
    baseline_metrics = {
        "validation": regression_metrics(baseline, validation),
        "test": regression_metrics(baseline, test),
    }
    baseline_path = model_dir / "baseline_dummy.joblib"
    save_model(baseline, baseline_path, baseline_metrics)

    if tune:
        xgb = tune_xgboost(
            train,
            n_iter=config.n_iter_tuning,
            cv=config.cv_folds,
            seed=config.seed,
        )
    else:
        xgb = train_xgboost(train, validation, params={"random_state": config.seed})
    xgb_metrics = {
        "validation": regression_metrics(xgb, validation),
        "test": regression_metrics(xgb, test),
    }
    xgb_path = model_dir / "xgboost_tetris.joblib"
    save_model(xgb, xgb_path, xgb_metrics)

    metrics_path = model_dir / "training_metrics.json"
    metadata_path = train_path.parent / "dataset_metadata.json"
    dataset_metadata = {}
    if metadata_path.exists():
        dataset_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metrics_path.write_text(
        json.dumps(
            {
                "dataset": dataset_metadata,
                "baseline": baseline_metrics,
                "xgboost": xgb_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    LOGGER.info("Saved models and metrics to %s", model_dir)
    return {
        "baseline": baseline_path,
        "xgboost": xgb_path,
        "metrics": metrics_path,
    }
