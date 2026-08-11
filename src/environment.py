"""Tetris environment adapter and a small Gymnasium-compatible fallback engine.

The project is organized around a high-level placement action: choose a rotation
and horizontal position, then hard-drop the active tetromino. This keeps game
logic separate from the XGBoost model while still matching Gymnasium's reset /
step style for reproducible experiments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

try:  # pragma: no cover - optional dependency shape differs by installation.
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    gym = None
    spaces = None

LOGGER = logging.getLogger(__name__)

Board = np.ndarray

TETROMINOES: dict[str, np.ndarray] = {
    "I": np.array([[1, 1, 1, 1]], dtype=np.int8),
    "O": np.array([[1, 1], [1, 1]], dtype=np.int8),
    "T": np.array([[0, 1, 0], [1, 1, 1]], dtype=np.int8),
    "S": np.array([[0, 1, 1], [1, 1, 0]], dtype=np.int8),
    "Z": np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8),
    "J": np.array([[1, 0, 0], [1, 1, 1]], dtype=np.int8),
    "L": np.array([[0, 0, 1], [1, 1, 1]], dtype=np.int8),
}


@dataclass(frozen=True)
class Placement:
    """A complete Tetris move expressed as rotation index and x position."""

    piece: str
    rotation: int
    x: int
    y: int


@dataclass(frozen=True)
class StepResult:
    """Result of simulating a placement action."""

    board: Board
    lines_cleared: int
    landing_height: float
    y: int
    game_over: bool


@lru_cache(maxsize=len(TETROMINOES))
def unique_rotations(piece: str) -> tuple[np.ndarray, ...]:
    """Return unique 90-degree rotations for a tetromino."""

    shape = TETROMINOES[piece]
    rotations: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rotated = np.rot90(shape, k=k)
        key = (rotated.shape, rotated.tobytes())
        if key not in seen:
            rotations.append(rotated.astype(np.int8))
            seen.add(key)
    return tuple(rotations)


def collides(board: Board, shape: np.ndarray, x: int, y: int) -> bool:
    """Return True if a piece shape at x/y collides with walls, floor, or blocks."""

    rows, cols = board.shape
    height, width = shape.shape
    if x < 0 or x + width > cols or y + height > rows:
        return True
    if y < 0:
        return True
    window = board[y : y + height, x : x + width]
    return bool(np.any((window > 0) & (shape > 0)))


def drop_y(board: Board, shape: np.ndarray, x: int) -> int | None:
    """Find the final y position after a hard drop, or None if spawn is blocked."""

    rows, cols = board.shape
    height, width = shape.shape
    if x < 0 or x + width > cols:
        return None

    filled_shape = shape > 0
    occupied_shape_cols = np.any(filled_shape, axis=0)
    shape_row_indices = np.arange(height)[:, None]
    piece_bottoms = np.max(
        np.where(filled_shape, shape_row_indices, -1),
        axis=0,
    )

    board_window = board[:, x : x + width] > 0
    occupied_board_cols = np.any(board_window, axis=0)
    board_tops = np.where(occupied_board_cols, np.argmax(board_window, axis=0), rows)
    y = int(np.min(board_tops[occupied_shape_cols] - piece_bottoms[occupied_shape_cols] - 1))
    if y < 0:
        return None
    return y


def clear_completed_lines(board: Board) -> tuple[Board, int]:
    """Clear complete rows and return the new board plus line count."""

    completed = np.all(board > 0, axis=1)
    lines = int(np.sum(completed))
    if lines == 0:
        return board, 0
    remaining = board[~completed]
    empty = np.zeros((lines, board.shape[1]), dtype=np.int8)
    return np.vstack([empty, remaining]).astype(np.int8), lines


def simulate_placement(
    board: Board,
    piece: str,
    rotation: int,
    x: int,
    shape: np.ndarray | None = None,
) -> StepResult | None:
    """Simulate a hard-drop placement without mutating the input board."""

    if shape is None:
        rotations = unique_rotations(piece)
        shape = rotations[rotation % len(rotations)]
    y = drop_y(board, shape, x)
    if y is None:
        return None

    next_board = board.copy()
    height, width = shape.shape
    next_board[y : y + height, x : x + width] = np.where(
        shape > 0,
        1,
        next_board[y : y + height, x : x + width],
    )
    next_board, lines = clear_completed_lines(next_board)
    landing_height = float(board.shape[0] - (y + height / 2.0))
    return StepResult(
        board=next_board.astype(np.int8),
        lines_cleared=lines,
        landing_height=landing_height,
        y=y,
        game_over=False,
    )


class TetrisEnv(gym.Env if gym else object):
    """Small hard-drop Tetris environment with Gymnasium-style API."""

    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(
        self,
        rows: int = 20,
        cols: int = 10,
        max_pieces: int = 500,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.max_pieces = max_pieces
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)
        self.board = np.zeros((rows, cols), dtype=np.int8)
        self.current_piece = "I"
        self.pieces_placed = 0
        self.lines_cleared = 0
        self.score = 0
        if spaces is not None:
            self.observation_space = spaces.Box(0, 1, shape=(rows, cols), dtype=np.int8)
            self.action_space = spaces.Discrete(rows * cols * 4)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Board, dict[str, Any]]:
        """Reset the environment and return board observation plus info."""

        del options
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.current_piece = self._sample_piece()
        self.pieces_placed = 0
        self.lines_cleared = 0
        self.score = 0
        return self.board.copy(), self._info()

    def step(self, action: Placement | int) -> tuple[Board, float, bool, bool, dict[str, Any]]:
        """Apply a placement, update score, and spawn the next piece."""

        from src.actions import generate_candidate_actions

        if isinstance(action, int):
            candidates = generate_candidate_actions(self.board, self.current_piece)
            if not candidates:
                return self.board.copy(), -10.0, True, False, self._info()
            action = candidates[action % len(candidates)].placement

        if action.piece != self.current_piece:
            raise ValueError(f"Expected piece {self.current_piece}, got {action.piece}")

        result = simulate_placement(self.board, action.piece, action.rotation, action.x)
        if result is None:
            return self.board.copy(), -10.0, True, False, self._info()

        self.board = result.board
        self.pieces_placed += 1
        self.lines_cleared += result.lines_cleared
        reward = self._score_lines(result.lines_cleared)
        self.score += int(reward)
        self.current_piece = self._sample_piece()

        terminated = self._is_spawn_blocked() or self.pieces_placed >= self.max_pieces
        if self.render_mode == "human":
            print(self.render())
        return self.board.copy(), reward, terminated, False, self._info()

    def render(self) -> str:
        """Render the board as text."""

        chars = np.where(self.board > 0, "#", ".")
        return "\n".join("".join(row) for row in chars)

    def _sample_piece(self) -> str:
        return str(self.rng.choice(list(TETROMINOES.keys())))

    def _is_spawn_blocked(self) -> bool:
        from src.actions import generate_candidate_actions

        return len(generate_candidate_actions(self.board, self.current_piece)) == 0

    def _info(self) -> dict[str, Any]:
        return {
            "current_piece": self.current_piece,
            "score": self.score,
            "lines_cleared": self.lines_cleared,
            "pieces_placed": self.pieces_placed,
        }

    @staticmethod
    def _score_lines(lines: int) -> float:
        return {0: 0.0, 1: 40.0, 2: 100.0, 3: 300.0, 4: 1200.0}.get(lines, 0.0)
