"""Agents that choose candidate Tetris placements."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from src.actions import CandidateAction, generate_candidate_actions
from src.dataset import heuristic_quality
from src.environment import Board, Placement
from src.features import FEATURE_NAMES
from src.model import load_model


class Agent(Protocol):
    """Common interface for placement-selecting agents."""

    def select_action(self, board: Board, piece: str) -> Placement | None:
        """Return a legal placement or None if no placement exists."""


@dataclass
class RandomAgent:
    """Uniform-random legal placement baseline."""

    seed: int = 42

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def select_action(self, board: Board, piece: str) -> Placement | None:
        candidates = generate_candidate_actions(board, piece)
        if not candidates:
            return None
        return candidates[int(self.rng.integers(0, len(candidates)))].placement


class HeuristicAgent:
    """Greedy oracle used for behavior generation and sanity checks."""

    def select_action(self, board: Board, piece: str) -> Placement | None:
        candidates = generate_candidate_actions(board, piece)
        if not candidates:
            return None
        return max(candidates, key=heuristic_quality).placement


class XGBoostAgent:
    """Select the candidate with the highest predicted action quality."""

    def __init__(self, model_path: Path | str | None = None, model: object | None = None) -> None:
        if model is None and model_path is None:
            raise ValueError("Provide either model_path or model")
        self.model = model if model is not None else load_model(Path(model_path))

    def select_action(self, board: Board, piece: str) -> Placement | None:
        candidates = generate_candidate_actions(board, piece)
        if not candidates:
            return None
        return self.select_candidate(candidates).placement

    def select_candidate(self, candidates: list[CandidateAction]) -> CandidateAction:
        matrix = pd.DataFrame(
            [[candidate.features[name] for name in FEATURE_NAMES] for candidate in candidates],
            columns=FEATURE_NAMES,
        )
        predictions = np.asarray(self.model.predict(matrix), dtype=float)
        return candidates[int(np.argmax(predictions))]
