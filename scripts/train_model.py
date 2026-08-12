"""Train baseline and XGBoost Tetris action-quality models."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, MODEL_DIR, TrainingConfig
from src.train import train_from_splits

DEFAULT_CONFIG = {
    "data_dir": DATA_DIR,
    "model_dir": MODEL_DIR,
    "seed": 42,
    "tune": False,
    "n_iter": 12,
    "cv_folds": 3,
}
CONFIG_KEYS = set(DEFAULT_CONFIG)


def _load_config(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("YAML config must be a mapping")
    config = raw.get("training", raw.get("train", raw))
    if not isinstance(config, dict):
        raise ValueError("YAML config field 'training' must be a mapping")
    unknown = set(config).difference(CONFIG_KEYS)
    if unknown:
        raise ValueError(f"Unknown training config keys: {sorted(unknown)}")
    return dict(config)


def _resolve_project_path(value: object) -> Path:
    path = value if isinstance(value, Path) else Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, remaining = config_parser.parse_known_args()
    config_path = _resolve_project_path(config_args.config) if config_args.config else None

    parser = argparse.ArgumentParser(description=__doc__, parents=[config_parser])
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--tune", action=argparse.BooleanOptionalAction)
    parser.add_argument("--n-iter", type=int)
    parser.add_argument("--cv-folds", type=int)
    args = parser.parse_args(remaining)

    merged = dict(DEFAULT_CONFIG)
    merged.update(_load_config(config_path))
    cli_overrides = {
        key: value
        for key, value in vars(args).items()
        if key != "config" and value is not None
    }
    merged.update(cli_overrides)
    merged["data_dir"] = _resolve_project_path(merged["data_dir"])
    merged["model_dir"] = _resolve_project_path(merged["model_dir"])
    merged["config"] = config_path
    return argparse.Namespace(**merged)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    config = TrainingConfig(
        seed=args.seed,
        n_iter_tuning=args.n_iter,
        cv_folds=args.cv_folds,
    )
    outputs = train_from_splits(
        train_path=args.data_dir / "train.csv",
        validation_path=args.data_dir / "validation.csv",
        test_path=args.data_dir / "test.csv",
        model_dir=args.model_dir,
        tune=args.tune,
        config=config,
    )
    logging.info("Training outputs: %s", outputs)


if __name__ == "__main__":
    main()
