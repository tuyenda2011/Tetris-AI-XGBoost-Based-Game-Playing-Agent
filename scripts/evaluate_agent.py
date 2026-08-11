"""Evaluate Random and XGBoost Tetris agents."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import RandomAgent, XGBoostAgent
from src.config import MODEL_DIR, RESULTS_DIR
from src.evaluate import evaluate_agent, save_evaluation, summarize_metrics
from src.model import load_model
from src.visualize import plot_evaluation, plot_feature_importance, save_shap_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-pieces", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", type=Path, default=MODEL_DIR / "xgboost_tetris.joblib")
    parser.add_argument("--test-data", type=Path, default=Path("data/processed/test.csv"))
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--skip-shap", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    if not args.model_path.exists():
        raise SystemExit(
            f"Model file not found: {args.model_path}\n"
            "Train a model first, for example:\n"
            "  python scripts/generate_dataset.py --episodes 20 --max-pieces 120 --seed 42\n"
            "  python scripts/train_model.py --seed 42"
        )
    random_agent = RandomAgent(seed=args.seed)
    xgb_agent = XGBoostAgent(model_path=args.model_path)

    metrics_dir = args.output_dir / "metrics"
    figures_dir = args.output_dir / "figures"
    shap_dir = args.output_dir / "shap"

    random_results = evaluate_agent(
        random_agent,
        episodes=args.episodes,
        seed=args.seed,
        max_pieces=args.max_pieces,
        render=args.render,
    )
    xgb_results = evaluate_agent(
        xgb_agent,
        episodes=args.episodes,
        seed=args.seed,
        max_pieces=args.max_pieces,
        render=args.render,
    )
    save_evaluation(random_results, metrics_dir, "random")
    save_evaluation(xgb_results, metrics_dir, "xgboost")
    comparison = pd.concat(
        [
            summarize_metrics(random_results).assign(agent="random"),
            summarize_metrics(xgb_results).assign(agent="xgboost"),
        ],
        ignore_index=True,
    )
    comparison.to_csv(metrics_dir / "comparison_summary.csv", index=False)
    plot_evaluation({"random": random_results, "xgboost": xgb_results}, figures_dir)

    model = load_model(args.model_path)
    plot_feature_importance(model, figures_dir)
    if not args.skip_shap and args.test_data.exists():
        test_frame = pd.read_csv(args.test_data)
        sample = test_frame.sample(n=min(200, len(test_frame)), random_state=args.seed)
        save_shap_summary(model, sample, shap_dir)
    logging.info("Evaluation complete. See %s", args.output_dir)


if __name__ == "__main__":
    main()
