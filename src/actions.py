"""Candidate action generation and board simulation utilities."""

from __future__ import annotations

from dataclasses import dataclass

from src.environment import Board, Placement, simulate_placement, unique_rotations
from src.features import extract_features, extract_heuristic_features


@dataclass(frozen=True)
class CandidateAction:
    """A legal placement with the resulting board and engineered features."""

    placement: Placement
    board_after: Board
    features: dict[str, float]
    lines_cleared: int


def generate_candidate_actions(
    board: Board,
    piece: str,
    *,
    full_features: bool = True,
) -> list[CandidateAction]:
    """Generate every legal hard-drop placement for the current tetromino."""

    candidates: list[CandidateAction] = []
    cols = board.shape[1]
    feature_extractor = extract_features if full_features else extract_heuristic_features
    for rotation_idx, shape in enumerate(unique_rotations(piece)):
        max_x = cols - shape.shape[1]
        for x in range(max_x + 1):
            result = simulate_placement(board, piece, rotation_idx, x, shape)
            if result is None:
                continue
            placement = Placement(piece=piece, rotation=rotation_idx, x=x, y=result.y)
            features = feature_extractor(
                result.board,
                cleared_lines=result.lines_cleared,
                landing_height=result.landing_height,
            )
            candidates.append(
                CandidateAction(
                    placement=placement,
                    board_after=result.board,
                    features=dict(features),
                    lines_cleared=result.lines_cleared,
                )
            )
    return candidates
