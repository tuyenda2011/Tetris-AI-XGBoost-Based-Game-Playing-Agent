"""Feature engineering for Tetris board-state quality."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

Board = np.ndarray

BASE_FEATURE_NAMES = [
    "aggregate_height",
    "center_height",
    "max_height",
    "min_height",
    "holes",
    "bumpiness",
    "row_transitions",
    "col_transitions",
    "occupied_cells",
    "cleared_lines",
    "wells",
    "landing_height",
    "height_variance",
    "row_density",
    "column_density",
]

COLUMN_DIFF_NAMES = [f"column_diff_{idx}" for idx in range(9)]
FEATURE_NAMES = BASE_FEATURE_NAMES + COLUMN_DIFF_NAMES
HEURISTIC_FEATURE_NAMES = [
    "aggregate_height",
    "center_height",
    "holes",
    "bumpiness",
    "row_transitions",
    "col_transitions",
    "wells",
    "cleared_lines",
    "landing_height",
]


def column_heights(board: Board) -> np.ndarray:
    """Return the filled height of each column."""

    rows, cols = board.shape
    filled = board > 0
    has_block = np.any(filled, axis=0)
    first_block = np.argmax(filled, axis=0)
    return np.where(has_block, rows - first_block, 0).astype(float)


def count_holes(board: Board) -> int:
    """Count empty cells with at least one filled cell above in the same column."""

    filled = board > 0
    has_block_above_or_here = np.maximum.accumulate(filled, axis=0)
    return int(np.sum(has_block_above_or_here & ~filled))


def count_wells(board: Board) -> int:
    """Count well depth: empty cells bounded by filled cells or walls."""

    filled = board > 0
    left_blocked = np.ones_like(filled, dtype=bool)
    right_blocked = np.ones_like(filled, dtype=bool)
    left_blocked[:, 1:] = filled[:, :-1]
    right_blocked[:, :-1] = filled[:, 1:]
    return int(np.sum(~filled & left_blocked & right_blocked))


def count_row_transitions(board: Board) -> int:
    """Count state changes (empty<->filled) along rows, treating board walls as filled."""

    b = (board > 0).astype(np.int8)
    padded = np.pad(b, ((0, 0), (1, 1)), constant_values=1)
    return int(np.sum(padded[:, :-1] != padded[:, 1:]))


def count_col_transitions(board: Board) -> int:
    """Count state changes (empty<->filled) along columns, treating board floor as filled."""

    b = (board > 0).astype(np.int8)
    padded = np.pad(b, ((0, 1), (0, 0)), constant_values=1)
    return int(np.sum(padded[:-1, :] != padded[1:, :]))


def extract_features(
    board: Board,
    *,
    cleared_lines: int = 0,
    landing_height: float = 0.0,
) -> OrderedDict[str, float]:
    """Extract a stable feature vector from a board after a candidate action."""

    board = (board > 0).astype(np.int8)
    heights = column_heights(board)
    diffs = np.diff(heights)
    occupied = float(np.sum(board))
    row_occupancy = np.mean(np.sum(board, axis=1) / board.shape[1])
    column_density = np.mean(heights / board.shape[0])
    center_h = float(np.sum(heights[3:7])) if len(heights) >= 7 else 0.0

    features: OrderedDict[str, float] = OrderedDict()
    features["aggregate_height"] = float(np.sum(heights))
    features["center_height"] = center_h
    features["max_height"] = float(np.max(heights))
    features["min_height"] = float(np.min(heights))
    features["holes"] = float(count_holes(board))
    features["bumpiness"] = float(np.sum(np.abs(diffs)))
    features["row_transitions"] = float(count_row_transitions(board))
    features["col_transitions"] = float(count_col_transitions(board))
    features["occupied_cells"] = occupied
    features["cleared_lines"] = float(cleared_lines)
    features["wells"] = float(count_wells(board))
    features["landing_height"] = float(landing_height)
    features["height_variance"] = float(np.var(heights))
    features["row_density"] = float(row_occupancy)
    features["column_density"] = float(column_density)
    for idx, value in enumerate(diffs):
        features[f"column_diff_{idx}"] = float(value)
    return features


def extract_heuristic_features(
    board: Board,
    *,
    cleared_lines: int = 0,
    landing_height: float = 0.0,
) -> dict[str, float]:
    """Extract only the features needed by the heuristic label."""

    board = (board > 0).astype(np.int8)
    heights = column_heights(board)
    diffs = np.diff(heights)
    center_h = float(np.sum(heights[3:7])) if len(heights) >= 7 else 0.0
    return {
        "aggregate_height": float(np.sum(heights)),
        "center_height": center_h,
        "holes": float(count_holes(board)),
        "bumpiness": float(np.sum(np.abs(diffs))),
        "row_transitions": float(count_row_transitions(board)),
        "col_transitions": float(count_col_transitions(board)),
        "wells": float(count_wells(board)),
        "cleared_lines": float(cleared_lines),
        "landing_height": float(landing_height),
    }


def feature_vector(
    board: Board,
    *,
    cleared_lines: int = 0,
    landing_height: float = 0.0,
) -> np.ndarray:
    """Return features as a numpy vector in FEATURE_NAMES order."""

    features = extract_features(
        board,
        cleared_lines=cleared_lines,
        landing_height=landing_height,
    )
    return np.array([features[name] for name in FEATURE_NAMES], dtype=float)

