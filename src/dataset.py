"""Dataset generation for supervised action-quality learning."""

from __future__ import annotations

import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.actions import CandidateAction, generate_candidate_actions
from src.config import EnvConfig, TrainingConfig
from src.environment import TETROMINOES, TetrisEnv
from src.features import FEATURE_NAMES, extract_heuristic_features

LOGGER = logging.getLogger(__name__)
DatasetRow = dict[str, float | int | str]


def heuristic_quality(candidate: CandidateAction) -> float:
    """Pseudo-label for supervised learning, inspired by classic Tetris heuristics."""

    f = candidate.features
    return (
        0.760666 * f["cleared_lines"]
        - 0.510066 * f["aggregate_height"]
        - 0.35663 * f["holes"]
        - 0.184483 * f["bumpiness"]
        - 0.08 * f["wells"]
        - 0.02 * f["landing_height"]
    )


def _score_lines(lines: int) -> float:
    return {0: 0.0, 1: 40.0, 2: 100.0, 3: 300.0, 4: 1200.0}.get(lines, 0.0)


def _static_board_value(board: np.ndarray) -> float:
    """Evaluate final rollout boards without using the current candidate row directly."""

    f = extract_heuristic_features(board)
    return (
        -0.510066 * f["aggregate_height"]
        - 0.35663 * f["holes"]
        - 0.184483 * f["bumpiness"]
        - 0.08 * f["wells"]
    )


def rollout_quality(
    candidate: CandidateAction,
    future_pieces: list[str],
    *,
    discount: float = 0.92,
    terminal_penalty: float = 100.0,
) -> float:
    """Label a candidate by simulating greedy play over a fixed future piece sequence.

    Unlike the immediate heuristic label, this target is not a direct algebraic
    function of the candidate's own feature row. Every candidate from the same
    decision point is evaluated against the same future pieces.
    """

    total = _score_lines(candidate.lines_cleared)
    board = candidate.board_after.copy()
    for depth, piece in enumerate(future_pieces, start=1):
        candidates = generate_candidate_actions(board, piece, full_features=False)
        if not candidates:
            return total - (discount**depth) * terminal_penalty
        best = max(candidates, key=heuristic_quality)
        total += (discount**depth) * _score_lines(best.lines_cleared)
        board = best.board_after
    total += (discount ** (len(future_pieces) + 1)) * 0.25 * _static_board_value(board)
    return float(total)


def _choose_behavior_action(
    candidates: list[CandidateAction],
    rng: np.random.Generator,
    epsilon: float,
) -> CandidateAction:
    if rng.random() < epsilon:
        return candidates[int(rng.integers(0, len(candidates)))]
    return max(candidates, key=heuristic_quality)


def _print_progress(episode: int, episodes: int, samples: int, start_time: float) -> None:
    """Print a compact progress bar to stdout."""
    done = episode + 1
    pct = done / episodes
    bar_len = 30
    filled = int(bar_len * pct)
    bar = "#" * filled + "-" * (bar_len - filled)
    elapsed = time.time() - start_time
    eta = (elapsed / done) * (episodes - done) if done > 0 else 0.0
    eta_str = f"{int(eta // 60):02d}:{int(eta % 60):02d}"
    line = (
        f"\r  [{bar}] {pct*100:5.1f}%  "
        f"ep {done:>3}/{episodes}  "
        f"samples {samples:>7,}  "
        f"ETA {eta_str}"
    )
    sys.stdout.buffer.write(line.encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()
    if done == episodes:
        sys.stdout.buffer.write(b"\n")
        sys.stdout.buffer.flush()


def _generate_episode_rows(
    *,
    episode: int,
    max_pieces: int,
    seed: int,
    epsilon: float,
    target_mode: str,
    rollout_steps: int,
    discount: float,
    board_rows: int,
    board_cols: int,
) -> list[DatasetRow]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, episode]))
    pieces = list(TETROMINOES.keys())
    rows: list[DatasetRow] = []
    env_seed = seed + episode
    env = TetrisEnv(
        rows=board_rows,
        cols=board_cols,
        max_pieces=max_pieces,
        seed=env_seed,
    )
    board, info = env.reset(seed=env_seed)
    done = False
    while not done:
        candidates = generate_candidate_actions(board, info["current_piece"])
        if not candidates:
            break
        future_pieces = [
            str(rng.choice(pieces)) for _ in range(max(0, rollout_steps))
        ]
        for idx, candidate in enumerate(candidates):
            if target_mode == "rollout":
                target_quality = rollout_quality(
                    candidate,
                    future_pieces,
                    discount=discount,
                )
            else:
                target_quality = heuristic_quality(candidate)
            row: DatasetRow = {
                name: candidate.features[name] for name in FEATURE_NAMES
            }
            row.update(
                {
                    "target_quality": target_quality,
                    "episode": episode,
                    "piece_index": info["pieces_placed"],
                    "piece": candidate.placement.piece,
                    "rotation": candidate.placement.rotation,
                    "x": candidate.placement.x,
                    "candidate_index": idx,
                    "target_mode": target_mode,
                    "rollout_steps": rollout_steps if target_mode == "rollout" else 0,
                }
            )
            rows.append(row)

        action = _choose_behavior_action(candidates, rng, epsilon).placement
        board, _, done, _, info = env.step(action)

    return rows


def generate_dataset(
    *,
    episodes: int,
    max_pieces: int,
    seed: int,
    epsilon: float = 0.15,
    target_mode: str = "rollout",
    rollout_steps: int = 1,
    discount: float = 0.92,
    env_config: EnvConfig | None = None,
    workers: int = 1,
) -> pd.DataFrame:
    """Generate candidate-action samples and heuristic quality targets."""

    if target_mode not in {"immediate", "rollout"}:
        raise ValueError("target_mode must be either 'immediate' or 'rollout'")
    if workers < 1:
        raise ValueError("workers must be at least 1")

    config = env_config or EnvConfig(seed=seed, max_pieces=max_pieces)
    rows: list[DatasetRow] = []
    start_time = time.time()
    actual_workers = min(workers, episodes)

    print(f"  Generating {episodes} episodes x {max_pieces} pieces  "
          f"[rollout={rollout_steps}, mode={target_mode}, workers={actual_workers}]")

    task_args = {
        "max_pieces": max_pieces,
        "seed": seed,
        "epsilon": epsilon,
        "target_mode": target_mode,
        "rollout_steps": rollout_steps,
        "discount": discount,
        "board_rows": config.rows,
        "board_cols": config.cols,
    }
    if actual_workers == 1:
        for episode in range(episodes):
            LOGGER.debug("Generating episode %s/%s", episode + 1, episodes)
            rows.extend(_generate_episode_rows(episode=episode, **task_args))
            _print_progress(episode, episodes, len(rows), start_time)
    else:
        results_by_episode: dict[int, list[DatasetRow]] = {}
        sample_count = 0
        with ProcessPoolExecutor(max_workers=actual_workers) as executor:
            futures = {
                executor.submit(_generate_episode_rows, episode=episode, **task_args): episode
                for episode in range(episodes)
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                episode = futures[future]
                episode_rows = future.result()
                results_by_episode[episode] = episode_rows
                sample_count += len(episode_rows)
                _print_progress(completed - 1, episodes, sample_count, start_time)
        for episode in range(episodes):
            rows.extend(results_by_episode[episode])

    dataset = pd.DataFrame(rows)
    elapsed = time.time() - start_time
    print(f"  Done! {len(dataset):,} samples in {elapsed:.1f}s  "
          f"({len(dataset)/elapsed:.0f} samples/s)")
    LOGGER.info("Generated %s samples from %s episodes", len(dataset), episodes)
    return dataset


def save_dataset_splits(
    dataset: pd.DataFrame,
    output_dir: Path,
    config: TrainingConfig,
) -> dict[str, Path]:
    """Save train/validation/test splits without leaking candidate groups."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if dataset.empty:
        raise ValueError("Cannot split an empty dataset")
    groups = dataset[["episode", "piece_index"]].drop_duplicates()
    if len(groups) < 4:
        raise ValueError(
            "Need at least 4 decision groups to create train/validation/test splits. "
            "Increase --episodes or --max-pieces."
        )
    train_groups, temp_groups = train_test_split(
        groups,
        test_size=config.validation_size + config.test_size,
        random_state=config.seed,
    )
    relative_test = config.test_size / (config.validation_size + config.test_size)
    val_groups, test_groups = train_test_split(
        temp_groups,
        test_size=relative_test,
        random_state=config.seed,
    )

    def select(groups_df: pd.DataFrame) -> pd.DataFrame:
        return dataset.merge(groups_df, on=["episode", "piece_index"], how="inner").reset_index(
            drop=True
        )

    paths = {
        "train": output_dir / "train.csv",
        "validation": output_dir / "validation.csv",
        "test": output_dir / "test.csv",
    }
    select(train_groups).to_csv(paths["train"], index=False)
    select(val_groups).to_csv(paths["validation"], index=False)
    select(test_groups).to_csv(paths["test"], index=False)
    return paths
