"""Episode evaluation for Tetris agents."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from src.agent import Agent
from src.environment import TetrisEnv


def run_episode(
    agent: Agent,
    *,
    seed: int,
    max_pieces: int,
    render: bool = False,
) -> dict[str, float | int]:
    """Run one episode and collect game-level metrics."""

    env = TetrisEnv(max_pieces=max_pieces, seed=seed, render_mode="human" if render else None)
    board, info = env.reset(seed=seed)
    done = False
    survival_steps = 0
    while not done:
        action = agent.select_action(board, info["current_piece"])
        if action is None:
            break
        board, _, done, _, info = env.step(action)
        survival_steps += 1

    return {
        "seed": seed,
        "score": float(info["score"]),
        "lines_cleared": float(info["lines_cleared"]),
        "pieces_placed": float(info["pieces_placed"]),
        "survival_time": float(survival_steps),
    }


def evaluate_agent(
    agent: Agent,
    *,
    episodes: int,
    seed: int,
    max_pieces: int,
    render: bool = False,
) -> pd.DataFrame:
    """Evaluate an agent across multiple random seeds."""

    rows = [
        run_episode(agent, seed=seed + idx, max_pieces=max_pieces, render=render)
        for idx in range(episodes)
    ]
    return pd.DataFrame(rows)


def summarize_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize evaluation metrics with mean/std/median/max and 95% CI."""

    metrics = ["score", "lines_cleared", "pieces_placed", "survival_time"]
    rows: list[dict[str, float | str]] = []
    n = max(len(frame), 1)
    for metric in metrics:
        values = frame[metric]
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        ci95 = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
        rows.append(
            {
                "metric": metric,
                "mean": float(values.mean()),
                "std": std,
                "median": float(values.median()),
                "max": float(values.max()),
                "ci95": float(ci95),
            }
        )
    return pd.DataFrame(rows)


def save_evaluation(frame: pd.DataFrame, output_dir: Path, name: str) -> dict[str, Path]:
    """Save per-episode and summary evaluation CSV files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = output_dir / f"{name}_episodes.csv"
    summary_path = output_dir / f"{name}_summary.csv"
    frame.to_csv(episodes_path, index=False)
    summarize_metrics(frame).to_csv(summary_path, index=False)
    return {"episodes": episodes_path, "summary": summary_path}
