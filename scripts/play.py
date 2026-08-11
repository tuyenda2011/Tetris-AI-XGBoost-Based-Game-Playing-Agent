"""Play one Tetris episode with a selected agent."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import HeuristicAgent, RandomAgent, XGBoostAgent
from src.config import MODEL_DIR
from src.evaluate import run_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=["random", "heuristic", "xgboost"], default="xgboost")
    parser.add_argument("--model-path", type=Path, default=MODEL_DIR / "xgboost_tetris.joblib")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pieces", type=int, default=120)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.agent == "random":
        agent = RandomAgent(seed=args.seed)
    elif args.agent == "heuristic":
        agent = HeuristicAgent()
    else:
        if not args.model_path.exists():
            raise SystemExit(
                f"Model file not found: {args.model_path}\n"
                "Train a model first, for example:\n"
                "  python scripts/generate_dataset.py --episodes 20 --max-pieces 120 --seed 42\n"
                "  python scripts/train_model.py --seed 42"
            )
        agent = XGBoostAgent(model_path=args.model_path)
    metrics = run_episode(agent, seed=args.seed, max_pieces=args.max_pieces, render=args.render)
    print(metrics)


if __name__ == "__main__":
    main()
