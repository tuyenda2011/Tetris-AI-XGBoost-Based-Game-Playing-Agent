"""Pygame GUI demo - watch the AI play Tetris in a proper window."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pygame
import numpy as np

from src.agent import HeuristicAgent, RandomAgent, XGBoostAgent
from src.config import MODEL_DIR
from src.environment import TetrisEnv, TETROMINOES, unique_rotations, simulate_placement

# ── Layout constants ──────────────────────────────────────────────────────────
CELL          = 32          # pixels per cell
COLS, ROWS    = 10, 20
PANEL_W       = 220         # right info panel
MARGIN        = 20
WIN_W         = COLS * CELL + PANEL_W + MARGIN * 3
WIN_H         = ROWS * CELL + MARGIN * 2
FPS           = 60

# ── Colour palette ────────────────────────────────────────────────────────────
BG            = (15,  15,  25)
GRID_LINE     = (30,  30,  50)
PANEL_BG      = (20,  20,  35)
BORDER        = (60,  60,  100)
TEXT_COL      = (220, 220, 240)
DIM_COL       = (100, 100, 130)
GHOST_COL     = (60,  60,  80)

PIECE_COLORS: dict[str, tuple[int,int,int]] = {
    "I": (0,   200, 255),   # cyan
    "O": (255, 210,  0 ),   # yellow
    "T": (160,  0,  220),   # purple
    "S": (0,   200,  60),   # green
    "Z": (230,  40,  40),   # red
    "J": (0,    80, 255),   # blue
    "L": (255, 140,   0),   # orange
}

SCORE_TABLE = {0: 0, 1: 40, 2: 100, 3: 300, 4: 1200}


def draw_cell(surf: pygame.Surface, col: int, row: int,
              color: tuple, ox: int, oy: int, alpha: int = 255) -> None:
    x = ox + col * CELL
    y = oy + row * CELL
    rect = pygame.Rect(x + 1, y + 1, CELL - 2, CELL - 2)
    if alpha < 255:
        s = pygame.Surface((CELL - 2, CELL - 2), pygame.SRCALPHA)
        s.fill((*color, alpha))
        surf.blit(s, (x + 1, y + 1))
    else:
        pygame.draw.rect(surf, color, rect, border_radius=4)
        # highlight top-left edge
        pygame.draw.line(surf, tuple(min(255, c + 60) for c in color),
                         (x+2, y+2), (x + CELL - 3, y+2), 2)
        pygame.draw.line(surf, tuple(min(255, c + 40) for c in color),
                         (x+2, y+2), (x+2, y + CELL - 3), 2)
        # shadow bottom-right
        pygame.draw.line(surf, tuple(max(0, c - 60) for c in color),
                         (x+2, y + CELL - 3), (x + CELL - 3, y + CELL - 3), 2)


def draw_board(surf: pygame.Surface, board: np.ndarray,
               piece: str, ghost_row: int | None,
               rotation: int, placement_x: int,
               ox: int, oy: int) -> None:
    # Grid background
    board_rect = pygame.Rect(ox - 2, oy - 2, COLS * CELL + 4, ROWS * CELL + 4)
    pygame.draw.rect(surf, BORDER, board_rect, border_radius=6)
    pygame.draw.rect(surf, (10, 10, 20),
                     pygame.Rect(ox, oy, COLS * CELL, ROWS * CELL))

    # Grid lines
    for c in range(COLS + 1):
        pygame.draw.line(surf, GRID_LINE,
                         (ox + c * CELL, oy), (ox + c * CELL, oy + ROWS * CELL))
    for r in range(ROWS + 1):
        pygame.draw.line(surf, GRID_LINE,
                         (ox, oy + r * CELL), (ox + COLS * CELL, oy + r * CELL))

    # Ghost piece
    if ghost_row is not None:
        rotations = unique_rotations(piece)
        shape = rotations[rotation % len(rotations)]
        h, w = shape.shape
        for rr in range(h):
            for cc in range(w):
                if shape[rr, cc]:
                    draw_cell(surf, placement_x + cc, ghost_row + rr,
                              GHOST_COL, ox, oy, alpha=180)

    # Placed blocks
    piece_names = list(TETROMINOES.keys())
    for r in range(ROWS):
        for c in range(COLS):
            if board[r, c]:
                draw_cell(surf, c, r, (80, 80, 120), ox, oy)

    # Active piece preview (current piece at top)
    rotations = unique_rotations(piece)
    shape = rotations[rotation % len(rotations)]
    h, w = shape.shape
    color = PIECE_COLORS.get(piece, (180, 180, 180))
    for rr in range(h):
        for cc in range(w):
            if shape[rr, cc]:
                draw_cell(surf, placement_x + cc, rr, color, ox, oy)


def draw_panel(surf: pygame.Surface, font_big, font_med, font_sm,
               piece: str, next_pieces: list[str],
               score: int, lines: int, pieces: int,
               agent_name: str, speed: float,
               ox: int, oy: int) -> None:
    panel_rect = pygame.Rect(ox, oy, PANEL_W, ROWS * CELL)
    pygame.draw.rect(surf, PANEL_BG, panel_rect, border_radius=8)
    pygame.draw.rect(surf, BORDER, panel_rect, 1, border_radius=8)

    y = oy + 14

    def label(text, color=DIM_COL, fnt=font_sm):
        nonlocal y
        s = fnt.render(text, True, color)
        surf.blit(s, (ox + 12, y))
        y += s.get_height() + 4

    def value(text, color=TEXT_COL, fnt=font_med):
        nonlocal y
        s = fnt.render(text, True, color)
        surf.blit(s, (ox + 12, y))
        y += s.get_height() + 10

    def divider():
        nonlocal y
        pygame.draw.line(surf, BORDER, (ox + 10, y), (ox + PANEL_W - 10, y))
        y += 10

    # Agent title
    label("AGENT", DIM_COL, font_sm)
    value(agent_name.upper(), (100, 200, 255), font_med)
    divider()

    # Stats
    label("SCORE")
    value(f"{score:,}", (255, 220, 50), font_big)
    label("LINES CLEARED")
    value(str(lines), (0, 220, 100))
    label("PIECES PLACED")
    value(str(pieces))
    divider()

    # Speed
    label("SPEED")
    value(f"{speed:.0f} ms/move")
    divider()

    # Next piece mini-preview
    label("CURRENT PIECE")
    mini_cell = 18
    rotations = unique_rotations(piece)
    shape = rotations[0]
    color = PIECE_COLORS.get(piece, (180, 180, 180))
    h, w = shape.shape
    px = ox + 12
    for rr in range(h):
        for cc in range(w):
            if shape[rr, cc]:
                r = pygame.Rect(px + cc * mini_cell + 1,
                                y + rr * mini_cell + 1,
                                mini_cell - 2, mini_cell - 2)
                pygame.draw.rect(surf, color, r, border_radius=3)
    y += h * mini_cell + 14
    divider()

    # Controls hint
    label("CONTROLS", DIM_COL)
    label("↑/↓  Speed", DIM_COL)
    label("SPACE Pause", DIM_COL)
    label("R     Restart", DIM_COL)
    label("ESC   Quit", DIM_COL)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agent", choices=["random", "heuristic", "xgboost"],
                   default="xgboost")
    p.add_argument("--model-path", type=Path,
                   default=MODEL_DIR / "xgboost_tetris.joblib")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-pieces", type=int, default=500)
    p.add_argument("--delay-ms", type=int, default=120,
                   help="Milliseconds between moves (default 120)")
    return p.parse_args()


def make_agent(args: argparse.Namespace):
    if args.agent == "random":
        return RandomAgent(seed=args.seed)
    elif args.agent == "heuristic":
        return HeuristicAgent()
    else:
        if not args.model_path.exists():
            raise SystemExit(
                f"Model not found: {args.model_path}\n"
                "Train first: python scripts/train_model.py --seed 42"
            )
        return XGBoostAgent(model_path=args.model_path)


def main() -> None:
    args = parse_args()
    pygame.init()
    pygame.display.set_caption(f"Tetris AI — {args.agent.upper()} Agent")
    surf = pygame.display.set_mode((WIN_W, WIN_H))
    clock = pygame.time.Clock()

    font_big = pygame.font.SysFont("consolas", 28, bold=True)
    font_med = pygame.font.SysFont("consolas", 20, bold=True)
    font_sm  = pygame.font.SysFont("consolas", 14)

    ox = MARGIN          # board origin x
    oy = MARGIN          # board origin y
    px = MARGIN * 2 + COLS * CELL  # panel origin x

    agent = make_agent(args)
    env   = TetrisEnv(max_pieces=args.max_pieces, seed=args.seed)
    board, info = env.reset(seed=args.seed)

    delay_ms = args.delay_ms
    paused   = False
    done     = False
    last_move = time.time()

    ghost_row    = None
    best_rotation = 0
    best_x        = 0

    def compute_ghost(board, piece, rotation, x):
        from src.environment import unique_rotations, drop_y
        rots = unique_rotations(piece)
        shape = rots[rotation % len(rots)]
        y = drop_y(board, shape, x)
        return y

    def precompute_best(board, piece):
        """Find the best placement the agent would pick."""
        from src.actions import generate_candidate_actions
        candidates = generate_candidate_actions(board, piece)
        if not candidates:
            return None
        action = agent.select_action(board, piece)
        return action

    action_pending = precompute_best(board, info["current_piece"])
    if action_pending:
        best_rotation = action_pending.rotation
        best_x        = action_pending.x
        ghost_row     = compute_ghost(board, info["current_piece"],
                                      best_rotation, best_x)

    running = True
    while running:
        dt = clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_UP:
                    delay_ms = max(20, delay_ms - 20)
                elif event.key == pygame.K_DOWN:
                    delay_ms = min(2000, delay_ms + 20)
                elif event.key == pygame.K_r:
                    board, info = env.reset(seed=args.seed)
                    done = False
                    action_pending = precompute_best(board, info["current_piece"])
                    if action_pending:
                        best_rotation = action_pending.rotation
                        best_x        = action_pending.x
                        ghost_row     = compute_ghost(board, info["current_piece"],
                                                      best_rotation, best_x)

        # ── AI step ──────────────────────────────────────────────────────────
        now = time.time()
        if not paused and not done and (now - last_move) * 1000 >= delay_ms:
            if action_pending is not None:
                board, reward, terminated, truncated, info = env.step(action_pending)
                done = terminated or truncated
            else:
                done = True

            if not done:
                action_pending = precompute_best(board, info["current_piece"])
                if action_pending:
                    best_rotation = action_pending.rotation
                    best_x        = action_pending.x
                    ghost_row     = compute_ghost(board, info["current_piece"],
                                                  best_rotation, best_x)
                else:
                    ghost_row = None

            last_move = now

        # ── Draw ─────────────────────────────────────────────────────────────
        surf.fill(BG)

        draw_board(surf, board,
                   info["current_piece"], ghost_row,
                   best_rotation, best_x, ox, oy)

        draw_panel(surf, font_big, font_med, font_sm,
                   info["current_piece"], [],
                   info["score"], info["lines_cleared"], info["pieces_placed"],
                   args.agent, delay_ms, px, oy)

        # Game over overlay
        if done:
            overlay = pygame.Surface((COLS * CELL, ROWS * CELL), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160))
            surf.blit(overlay, (ox, oy))
            go1 = font_big.render("GAME OVER", True, (255, 80, 80))
            go2 = font_sm.render(f"Score: {info['score']:,}  Lines: {info['lines_cleared']}", True, TEXT_COL)
            go3 = font_sm.render("Press R to restart", True, DIM_COL)
            cx = ox + (COLS * CELL) // 2
            cy = oy + (ROWS * CELL) // 2
            surf.blit(go1, go1.get_rect(center=(cx, cy - 30)))
            surf.blit(go2, go2.get_rect(center=(cx, cy + 10)))
            surf.blit(go3, go3.get_rect(center=(cx, cy + 40)))

        # Paused overlay
        if paused and not done:
            overlay = pygame.Surface((COLS * CELL, ROWS * CELL), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 120))
            surf.blit(overlay, (ox, oy))
            ps = font_big.render("PAUSED", True, (200, 200, 255))
            surf.blit(ps, ps.get_rect(center=(ox + COLS * CELL // 2,
                                               oy + ROWS * CELL // 2)))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
