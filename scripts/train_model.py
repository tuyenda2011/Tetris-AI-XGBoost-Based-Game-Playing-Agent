"""Train baseline and XGBoost Tetris action-quality models."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, MODEL_DIR, TrainingConfig
from src.train import train_from_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR / "processed")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--n-iter", type=int, default=12)
    parser.add_argument("--cv-folds", type=int, default=3)
    return parser.parse_args()


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
