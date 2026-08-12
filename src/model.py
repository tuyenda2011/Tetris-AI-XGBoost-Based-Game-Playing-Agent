"""Model training, tuning, persistence, and prediction helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import sys
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

warnings.filterwarnings("ignore", message=".*mismatched devices.*")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

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


def _print_train_progress(done: int, total: int, label: str, start_time: float, extra: str = "") -> None:
    """Print a compact progress bar to stdout."""
    done = max(1, min(done, total))
    pct = done / total
    bar_len = 30
    filled = int(bar_len * pct)
    bar = "#" * filled + "-" * (bar_len - filled)
    elapsed = time.time() - start_time
    eta = (elapsed / done) * (total - done) if done > 0 else 0.0
    eta_str = f"{int(eta // 60):02d}:{int(eta % 60):02d}"
    extra_str = f" {extra} " if extra else ""
    line = (
        f"\r  [{bar}] {pct*100:5.1f}%  "
        f"{label} {done:>3}/{total}  "
        f"{extra_str}"
        f"ETA {eta_str}"
    )
    sys.stdout.buffer.write(line.encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()
    if done == total:
        sys.stdout.buffer.write(b"\n")
        sys.stdout.buffer.flush()


def train_baseline(train: pd.DataFrame) -> DummyRegressor:
    """Train a simple mean-regression baseline model."""

    x_train, y_train = split_xy(train)
    model = DummyRegressor(strategy="mean")
    model.fit(x_train, y_train)
    return model


from tqdm import tqdm

try:
    from xgboost.callback import TrainingCallback

    class XGBProgressBarCallback(TrainingCallback):
        def __init__(self, progress_bar: tqdm):
            super().__init__()
            self.progress_bar = progress_bar

        def after_iteration(self, model, epoch, evals_log):
            self.progress_bar.update(1)
            return False
except ImportError:
    XGBProgressBarCallback = None


def train_xgboost(
    train: pd.DataFrame,
    validation: pd.DataFrame | None = None,
    *,
    params: dict[str, Any] | None = None,
) -> Any:
    """Train an XGBRegressor using non-default, reproducible parameters."""

    XGBRegressor = _import_xgboost()
    start_time = time.time()

    model_params = dict(DEFAULT_XGB_PARAMS)
    if params:
        model_params.update(params)

    n_estimators = model_params.get("n_estimators", 350)
    pbar = tqdm(total=n_estimators, desc=f"  Training XGBoost ({len(train):,} samples)", unit="tree", leave=True)

    if XGBProgressBarCallback is not None:
        model_params["callbacks"] = [XGBProgressBarCallback(pbar)]

    model = XGBRegressor(**model_params)
    x_train, y_train = split_xy(train)

    if validation is None or len(validation) == 0:
        val_size = min(5000, max(100, int(len(train) * 0.05)))
        x_val, y_val = x_train.iloc[-val_size:], y_train.iloc[-val_size:]
    else:
        x_val, y_val = split_xy(validation)

    try:
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    finally:
        pbar.close()

    # Clear callbacks to avoid pickling issues when dumping model with joblib
    model.set_params(callbacks=None)

    elapsed = time.time() - start_time
    print(f"  Done! Model trained in {elapsed:.1f}s")
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
    total_fits = n_iter * cv
    start_time = time.time()
    pbar = tqdm(total=total_fits, desc=f"  Tuning XGBoost ({n_iter} iters x {cv} folds)", unit="fit", leave=True)

    search_space = {
        "n_estimators": [200, 300, 400, 500],
        "max_depth": [4, 5, 6, 7, 8],
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
        tree_method="hist",
        device="cuda",
    )
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=search_space,
        n_iter=n_iter,
        cv=cv,
        random_state=seed,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
        verbose=0,
    )

    class ProgressParallel(joblib.parallel.Parallel):
        def __init__(self, progress_bar: tqdm, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._progress_bar = progress_bar

        def print_progress(self):
            self._progress_bar.update(1)
            try:
                super().print_progress()
            except Exception:
                pass

    old_parallel = joblib.parallel.Parallel
    try:
        joblib.parallel.Parallel = lambda *a, **kw: ProgressParallel(pbar, *a, **kw)
        search.fit(x_train, y_train)
    finally:
        joblib.parallel.Parallel = old_parallel
        pbar.close()

    elapsed = time.time() - start_time
    print(f"  Done! Tuning completed in {elapsed:.1f}s")
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
