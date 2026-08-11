"""Model training, tuning, persistence, and prediction helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

from src.config import DEFAULT_XGB_PARAMS
from src.features import FEATURE_NAMES

LOGGER = logging.getLogger(__name__)


def _import_xgboost() -> Any:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:  # pragma: no cover - depends on local env.
        raise ImportError(
            "xgboost is required for training. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc
    return XGBRegressor


def split_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a dataset frame into model features and target quality."""

    return frame[FEATURE_NAMES], frame["target_quality"]


def train_baseline(train: pd.DataFrame) -> DummyRegressor:
    """Train a simple mean-regression baseline model."""

    x_train, y_train = split_xy(train)
    model = DummyRegressor(strategy="mean")
    model.fit(x_train, y_train)
    return model


def train_xgboost(
    train: pd.DataFrame,
    validation: pd.DataFrame | None = None,
    *,
    params: dict[str, Any] | None = None,
) -> Any:
    """Train an XGBRegressor using non-default, reproducible parameters."""

    XGBRegressor = _import_xgboost()
    model_params = dict(DEFAULT_XGB_PARAMS)
    if params:
        model_params.update(params)
    model = XGBRegressor(**model_params)
    x_train, y_train = split_xy(train)
    if validation is not None and len(validation) > 0:
        x_val, y_val = split_xy(validation)
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    else:
        model.fit(x_train, y_train)
    return model


def tune_xgboost(
    train: pd.DataFrame,
    *,
    n_iter: int,
    cv: int,
    seed: int,
) -> Any:
    """Run a bounded randomized hyperparameter search for XGBoost."""

    XGBRegressor = _import_xgboost()
    x_train, y_train = split_xy(train)
    search_space = {
        "n_estimators": [120, 180, 240, 320],
        "max_depth": [2, 3, 4, 5],
        "learning_rate": [0.03, 0.05, 0.08, 0.1],
        "subsample": [0.75, 0.85, 0.95, 1.0],
        "colsample_bytree": [0.75, 0.85, 0.95, 1.0],
        "min_child_weight": [1, 2, 4, 6],
        "reg_alpha": [0.0, 0.01, 0.1],
        "reg_lambda": [0.5, 1.0, 1.5, 2.0],
    }
    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=-1,
    )
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=search_space,
        n_iter=n_iter,
        cv=cv,
        random_state=seed,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1,
    )
    search.fit(x_train, y_train)
    LOGGER.info("Best XGBoost params: %s", search.best_params_)
    return search.best_estimator_


def regression_metrics(model: Any, frame: pd.DataFrame) -> dict[str, float]:
    """Evaluate target-quality prediction quality."""

    x_data, y_true = split_xy(frame)
    predictions = np.asarray(model.predict(x_data))
    mse = mean_squared_error(y_true, predictions)
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, predictions)),
    }


def save_model(model: Any, path: Path, metrics: dict[str, Any] | None = None) -> None:
    """Persist a model plus sidecar metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    metadata = {"feature_names": FEATURE_NAMES, "metrics": metrics or {}}
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_model(path: Path) -> Any:
    """Load a persisted model."""

    return joblib.load(path)
