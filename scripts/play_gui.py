"""Pygame GUI demo - watch the AI play Tetris in a proper window."""

from __future__ import annotations

import argparse
import math
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
from src.environment import TetrisEnv, TETROMINOES, unique_rotations, drop_y

# ── Layout constants ───────────────────────────────────────────────────────────
CELL     = 32
COLS, ROWS = 10, 20
PANEL_W  = 220
MARGIN   = 20
WIN_W    = COLS * CELL + PANEL_W + MARGIN * 3
WIN_H    = ROWS * CELL + MARGIN * 2
FPS      = 60

# ── Colour palette ─────────────────────────────────────────────────────────────
BG         = (15,  15,  25)
GRID_LINE  = (30,  30,  50)
PANEL_BG   = (20,  20,  35)
BORDER     = (60,  60, 100)
TEXT_COL   = (220, 220, 240)
DIM_COL    = (100, 100, 130)
GHOST_COL  = (60,  60,  80)

PIECE_COLORS: dict[str, tuple[int, int, int]] = {
    "I": (0,   200, 255),
    "O": (255, 210,   0),
    "T": (160,   0, 220),
    "S": (0,   200,  60),
    "Z": (230,  40,  40),
    "J": (0,    80, 255),
    "L": (255, 140,   0),
}

# ── Agent definitions for menu ─────────────────────────────────────────────────
AGENTS = [
    {
        "key":   "xgboost",
        "name":  "XGBoost AI",
        "desc":  "Trained machine learning model",
        "color": (0, 200, 255),
        "icon":  "X",
    },
    {
        "key":   "heuristic",
        "name":  "Heuristic AI",
        "desc":  "Dellacherie expert formula",
        "color": (0, 220, 100),
        "icon":  "H",
    },
    {
        "key":   "random",
        "name":  "Random Agent",
        "desc":  "Uniform random placement",
        "color": (255, 140, 0),
        "icon":  "R",
    },
]


# ══════════════════════════════════════════════════════════════════════════════
#  MENU
# ══════════════════════════════════════════════════════════════════════════════

def _lerp_color(c1, c2, t: float) -> tuple[int, int, int]:
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))  # type: ignore[return-value]


def draw_menu(surf: pygame.Surface, fonts: dict, selected: int, t: float, model_path: Path) -> None:
    """Draw the main agent-selection menu."""
    surf.fill(BG)
    W, H = surf.get_size()

    # ── Animated falling tetromino particles in background ────────────────────
    rng = np.random.default_rng(0)
    for i in range(18):
        px = int(rng.integers(20, W - 20))
        py_base = int(rng.integers(0, H))
        py = (py_base + int(t * 30 * (0.3 + rng.random() * 0.7))) % H
        alpha = int(25 + 15 * math.sin(t * 1.2 + i))
        s = pygame.Surface((12, 12), pygame.SRCALPHA)
        col_idx = int(rng.integers(0, len(PIECE_COLORS)))
        col = list(PIECE_COLORS.values())[col_idx]
        s.fill((*col, alpha))
        surf.blit(s, (px, py))

    # ── Title ─────────────────────────────────────────────────────────────────
    title_glow = int(200 + 55 * math.sin(t * 2.0))
    title_col  = (title_glow, title_glow, 255)
    title_surf = fonts["title"].render("TETRIS  AI", True, title_col)
    title_rect = title_surf.get_rect(center=(W // 2, 70))
    # glow shadow
    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
        glow = fonts["title"].render("TETRIS  AI", True, (40, 40, 120))
        surf.blit(glow, title_rect.move(dx, dy))
    surf.blit(title_surf, title_rect)

    sub = fonts["sub"].render("Select an Agent to Watch", True, DIM_COL)
    surf.blit(sub, sub.get_rect(center=(W // 2, 115)))

    # ── Agent cards ───────────────────────────────────────────────────────────
    card_w, card_h = 320, 105
    spacing        = 18
    start_y        = 160
    cx             = (W - card_w) // 2

    for idx, agent in enumerate(AGENTS):
        cy = start_y + idx * (card_h + spacing)
        is_sel = idx == selected

        # Card background
        pulse = 0.5 + 0.5 * math.sin(t * 3.0) if is_sel else 0.0
        bg_col = _lerp_color((22, 22, 38), (30, 30, 55), pulse)
        border_col = _lerp_color((60, 60, 100), agent["color"], pulse if is_sel else 0.0)
        border_w   = 3 if is_sel else 1

        card_rect = pygame.Rect(cx, cy, card_w, card_h)
        pygame.draw.rect(surf, bg_col, card_rect, border_radius=14)
        pygame.draw.rect(surf, border_col, card_rect, border_w, border_radius=14)

        # Colored left accent bar
        accent_rect = pygame.Rect(cx + 2, cy + 2, 6, card_h - 4)
        pygame.draw.rect(surf, agent["color"], accent_rect, border_radius=4)

        # Big icon letter
        icon_col   = agent["color"] if is_sel else DIM_COL
        icon_alpha = int(220 + 35 * math.sin(t * 3.0)) if is_sel else 160
        icon_surf  = fonts["icon"].render(agent["icon"], True, icon_col)
        icon_rect  = icon_surf.get_rect(center=(cx + 50, cy + card_h // 2))
        surf.blit(icon_surf, icon_rect)

        # Agent name
        name_col  = TEXT_COL if is_sel else DIM_COL
        name_surf = fonts["name"].render(agent["name"], True, name_col)
        name_rect = name_surf.get_rect(topleft=(cx + 94, cy + 30))
        surf.blit(name_surf, name_rect)

        # Desc
        desc_surf = fonts["desc"].render(agent["desc"], True, (90, 90, 120) if not is_sel else (160, 160, 190))
        desc_rect = desc_surf.get_rect(topleft=(cx + 94, cy + 56))
        surf.blit(desc_surf, desc_rect)

        # Selected indicator: blinking arrow left of card
        if is_sel:
            arrow_x = cx - 30
            arrow_y = cy + card_h // 2 - 10
            arrow_alpha = int(180 + 75 * math.sin(t * 4.0))
            arrow_col   = (*agent["color"], arrow_alpha)
            arr_surf = pygame.Surface((20, 20), pygame.SRCALPHA)
            pygame.draw.polygon(arr_surf, arrow_col,
                                [(0, 0), (0, 20), (15, 10)])
            surf.blit(arr_surf, (arrow_x, arrow_y))

    # ── XGBoost model availability warning ───────────────────────────────────
    bottom_y = start_y + len(AGENTS) * (card_h + spacing)
    if selected == 0 and not model_path.exists():
        warn_surf = fonts["desc"].render("No trained model found -- train first!", True, (255, 100, 80))
        surf.blit(warn_surf, warn_surf.get_rect(center=(W // 2, bottom_y + 10)))
    else:
        hint_surf = fonts["desc"].render("ENTER/SPACE to start   *   UP/DOWN to switch   *   ESC to quit", True, (70, 70, 100))
        surf.blit(hint_surf, hint_surf.get_rect(center=(W // 2, bottom_y + 15)))

    # ── Mini decorative Tetris pieces in corners ──────────────────────────────
    _draw_deco_piece(surf, 20, H - 55, "L", (50, 50, 90))
    _draw_deco_piece(surf, W - 80, H - 55, "J", (50, 50, 90))

    pygame.display.flip()


def _draw_deco_piece(surf, x, y, piece, color):
    rots  = unique_rotations(piece)
    shape = rots[0]
    sz    = 14
    for rr in range(shape.shape[0]):
        for cc in range(shape.shape[1]):
            if shape[rr, cc]:
                pygame.draw.rect(surf, color,
                                 pygame.Rect(x + cc * sz, y + rr * sz, sz - 2, sz - 2),
                                 border_radius=3)


def run_menu(surf: pygame.Surface, fonts: dict, model_path: Path) -> str | None:
    """Show menu, return selected agent key or None if user quit."""
    clock   = pygame.time.Clock()
    selected = 0
    t_start  = time.time()

    while True:
        t = time.time() - t_start
        draw_menu(surf, fonts, selected, t, model_path)
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                elif event.key in (pygame.K_UP, pygame.K_w):
                    selected = (selected - 1) % len(AGENTS)
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    selected = (selected + 1) % len(AGENTS)
                elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    return AGENTS[selected]["key"]
            elif event.type == pygame.MOUSEMOTION:
                mx, my = event.pos
                W = surf.get_width()
                card_w, card_h = 320, 105
                spacing = 18
                start_y = 160
                cx = (W - card_w) // 2
                for idx in range(len(AGENTS)):
                    cy = start_y + idx * (card_h + spacing)
                    if pygame.Rect(cx, cy, card_w, card_h).collidepoint(mx, my):
                        selected = idx
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                W = surf.get_width()
                card_w, card_h = 320, 105
                spacing = 18
                start_y = 160
                cx = (W - card_w) // 2
                for idx, agent in enumerate(AGENTS):
                    cy = start_y + idx * (card_h + spacing)
                    if pygame.Rect(cx, cy, card_w, card_h).collidepoint(mx, my):
                        return agent["key"]



# ══════════════════════════════════════════════════════════════════════════════
#  GAME DRAWING
# ══════════════════════════════════════════════════════════════════════════════

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
        pygame.draw.line(surf, tuple(min(255, c + 60) for c in color),
                         (x+2, y+2), (x + CELL - 3, y+2), 2)
        pygame.draw.line(surf, tuple(min(255, c + 40) for c in color),
                         (x+2, y+2), (x+2, y + CELL - 3), 2)
        pygame.draw.line(surf, tuple(max(0, c - 60) for c in color),
                         (x+2, y + CELL - 3), (x + CELL - 3, y + CELL - 3), 2)


def draw_board(surf: pygame.Surface, board: np.ndarray,
               piece: str, ghost_row: int | None,
               rotation: int, placement_x: int,
               ox: int, oy: int) -> None:
    board_rect = pygame.Rect(ox - 2, oy - 2, COLS * CELL + 4, ROWS * CELL + 4)
    pygame.draw.rect(surf, BORDER, board_rect, border_radius=6)
    pygame.draw.rect(surf, (10, 10, 20),
                     pygame.Rect(ox, oy, COLS * CELL, ROWS * CELL))

    for c in range(COLS + 1):
        pygame.draw.line(surf, GRID_LINE,
                         (ox + c * CELL, oy), (ox + c * CELL, oy + ROWS * CELL))
    for r in range(ROWS + 1):
        pygame.draw.line(surf, GRID_LINE,
                         (ox, oy + r * CELL), (ox + COLS * CELL, oy + r * CELL))

    if ghost_row is not None:
        rotations = unique_rotations(piece)
        shape = rotations[rotation % len(rotations)]
        h, w = shape.shape
        for rr in range(h):
            for cc in range(w):
                if shape[rr, cc]:
                    draw_cell(surf, placement_x + cc, ghost_row + rr,
                              GHOST_COL, ox, oy, alpha=180)

    for r in range(ROWS):
        for c in range(COLS):
            if board[r, c]:
                draw_cell(surf, c, r, (80, 80, 120), ox, oy)

    rotations = unique_rotations(piece)
    shape = rotations[rotation % len(rotations)]
    h, w = shape.shape
    color = PIECE_COLORS.get(piece, (180, 180, 180))
    for rr in range(h):
        for cc in range(w):
            if shape[rr, cc]:
                draw_cell(surf, placement_x + cc, rr, color, ox, oy)


def draw_panel(surf: pygame.Surface, fonts: dict,
               piece: str, score: int, lines: int, pieces: int,
               agent_name: str, speed: float,
               ox: int, oy: int) -> None:
    panel_rect = pygame.Rect(ox, oy, PANEL_W, ROWS * CELL)
    pygame.draw.rect(surf, PANEL_BG, panel_rect, border_radius=8)
    pygame.draw.rect(surf, BORDER,   panel_rect, 1, border_radius=8)

    y = oy + 14

    def label(text, color=DIM_COL):
        nonlocal y
        s = fonts["sm"].render(text, True, color)
        surf.blit(s, (ox + 12, y))
        y += s.get_height() + 4

    def value(text, color=TEXT_COL, big=False):
        nonlocal y
        fnt = fonts["big"] if big else fonts["med"]
        s = fnt.render(text, True, color)
        surf.blit(s, (ox + 12, y))
        y += s.get_height() + 10

    def divider():
        nonlocal y
        pygame.draw.line(surf, BORDER, (ox + 10, y), (ox + PANEL_W - 10, y))
        y += 10

    # Agent name with its color
    agent_color = next((a["color"] for a in AGENTS if a["key"] == agent_name.lower()), TEXT_COL)
    label("AGENT")
    value(agent_name.upper(), agent_color)
    divider()

    label("SCORE")
    value(f"{score:,}", (255, 220, 50), big=True)
    label("LINES CLEARED")
    value(str(lines), (0, 220, 100))
    label("PIECES PLACED")
    value(str(pieces))
    divider()

    label("SPEED")
    value(f"{speed:.0f} ms/move")
    divider()

    label("CURRENT PIECE")
    mini_cell = 18
    rotations = unique_rotations(piece)
    shape  = rotations[0]
    color  = PIECE_COLORS.get(piece, (180, 180, 180))
    h, w   = shape.shape
    px_off = ox + 12
    for rr in range(h):
        for cc in range(w):
            if shape[rr, cc]:
                r = pygame.Rect(px_off + cc * mini_cell + 1,
                                y + rr * mini_cell + 1,
                                mini_cell - 2, mini_cell - 2)
                pygame.draw.rect(surf, color, r, border_radius=3)
    y += h * mini_cell + 14
    divider()

    label("CONTROLS")
    label("↑/↓  Speed")
    label("SPACE Pause")
    label("M     Menu")
    label("R     Restart")
    label("ESC   Quit")


# ══════════════════════════════════════════════════════════════════════════════
#  GAME LOOP
# ══════════════════════════════════════════════════════════════════════════════

def make_agent(agent_key: str, model_path: Path, seed: int):
    if agent_key == "random":
        return RandomAgent(seed=seed)
    elif agent_key == "heuristic":
        return HeuristicAgent()
    else:
        if not model_path.exists():
            raise SystemExit(
                f"Model not found: {model_path}\n"
                "Train first: python scripts/train_model.py --config configs/model_train.yaml"
            )
        return XGBoostAgent(model_path=model_path)


def run_game(surf: pygame.Surface, fonts: dict,
             agent_key: str, model_path: Path,
             seed: int, max_pieces: int, delay_ms: int) -> str:
    """Run one game session. Returns 'menu', 'restart' or 'quit'."""
    pygame.display.set_caption(f"Tetris AI — {agent_key.upper()} Agent")
    clock  = pygame.time.Clock()
    ox, oy = MARGIN, MARGIN
    px     = MARGIN * 2 + COLS * CELL

    agent = make_agent(agent_key, model_path, seed)
    env   = TetrisEnv(max_pieces=max_pieces, seed=seed)
    board, info = env.reset(seed=seed)

    cur_delay = delay_ms
    paused    = False
    done      = False
    last_move = time.time()

    ghost_row    = None
    best_rotation = 0
    best_x        = 0

    def compute_ghost(board, piece, rotation, x):
        rots  = unique_rotations(piece)
        shape = rots[rotation % len(rots)]
        y_pos = drop_y(board, shape, x)
        return y_pos

    def precompute_best(board, piece):
        action = agent.select_action(board, piece)
        return action

    action_pending = precompute_best(board, info["current_piece"])
    if action_pending:
        best_rotation = action_pending.rotation
        best_x        = action_pending.x
        ghost_row     = compute_ghost(board, info["current_piece"], best_rotation, best_x)

    while True:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "quit"
                elif event.key == pygame.K_m:
                    return "menu"
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_UP:
                    cur_delay = max(20, cur_delay - 20)
                elif event.key == pygame.K_DOWN:
                    cur_delay = min(2000, cur_delay + 20)
                elif event.key == pygame.K_r:
                    board, info = env.reset(seed=seed)
                    done = False
                    action_pending = precompute_best(board, info["current_piece"])
                    if action_pending:
                        best_rotation = action_pending.rotation
                        best_x        = action_pending.x
                        ghost_row     = compute_ghost(board, info["current_piece"],
                                                      best_rotation, best_x)

        # AI step
        now = time.time()
        if not paused and not done and (now - last_move) * 1000 >= cur_delay:
            if action_pending is not None:
                board, _, terminated, truncated, info = env.step(action_pending)
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

        # Draw
        surf.fill(BG)
        draw_board(surf, board, info["current_piece"], ghost_row,
                   best_rotation, best_x, ox, oy)
        draw_panel(surf, fonts, info["current_piece"],
                   info["score"], info["lines_cleared"], info["pieces_placed"],
                   agent_key, cur_delay, px, oy)

        # Game over overlay
        if done:
            overlay = pygame.Surface((COLS * CELL, ROWS * CELL), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160))
            surf.blit(overlay, (ox, oy))
            cx = ox + (COLS * CELL) // 2
            cy = oy + (ROWS * CELL) // 2
            go1 = fonts["big"].render("GAME OVER", True, (255, 80, 80))
            go2 = fonts["sm"].render(f"Score: {info['score']:,}   Lines: {info['lines_cleared']}", True, TEXT_COL)
            go3 = fonts["sm"].render("R - Restart   M - Menu   ESC - Quit", True, DIM_COL)
            surf.blit(go1, go1.get_rect(center=(cx, cy - 30)))
            surf.blit(go2, go2.get_rect(center=(cx, cy + 10)))
            surf.blit(go3, go3.get_rect(center=(cx, cy + 40)))

        # Paused overlay
        if paused and not done:
            overlay = pygame.Surface((COLS * CELL, ROWS * CELL), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 120))
            surf.blit(overlay, (ox, oy))
            ps = fonts["big"].render("PAUSED", True, (200, 200, 255))
            surf.blit(ps, ps.get_rect(center=(ox + COLS * CELL // 2,
                                              oy + ROWS * CELL // 2)))

        pygame.display.flip()


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agent",      choices=["random", "heuristic", "xgboost"],
                   default=None,   help="Skip menu and start directly with this agent")
    p.add_argument("--model-path", type=Path, default=MODEL_DIR / "xgboost_tetris.joblib")
    p.add_argument("--seed",       type=int,  default=42)
    p.add_argument("--max-pieces", type=int,  default=100_000)
    p.add_argument("--delay-ms",   type=int,  default=120)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pygame.init()
    pygame.display.set_caption("Tetris AI")
    surf = pygame.display.set_mode((WIN_W, WIN_H))

    fonts = {
        "title": pygame.font.SysFont("consolas", 42, bold=True),
        "sub":   pygame.font.SysFont("consolas", 16),
        "icon":  pygame.font.SysFont("consolas", 44, bold=True),
        "name":  pygame.font.SysFont("consolas", 17, bold=True),
        "desc":  pygame.font.SysFont("consolas", 13),
        "big":   pygame.font.SysFont("consolas", 28, bold=True),
        "med":   pygame.font.SysFont("consolas", 20, bold=True),
        "sm":    pygame.font.SysFont("consolas", 14),
    }

    # If agent specified via CLI, skip menu
    agent_key = args.agent

    while True:
        if agent_key is None:
            pygame.display.set_caption("Tetris AI — Select Agent")
            agent_key = run_menu(surf, fonts, args.model_path)
            if agent_key is None:
                break

        result = run_game(
            surf, fonts,
            agent_key=agent_key,
            model_path=args.model_path,
            seed=args.seed,
            max_pieces=args.max_pieces,
            delay_ms=args.delay_ms,
        )

        if result == "quit":
            break
        elif result == "menu":
            agent_key = None   # go back to menu
        # "restart" → loop again with same agent_key

    pygame.quit()


if __name__ == "__main__":
    main()
