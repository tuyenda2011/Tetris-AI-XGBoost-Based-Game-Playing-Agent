"""Script to record Pygame gameplay sessions into animated GIFs for all agents."""

import os
import sys
from pathlib import Path

# Force headless Pygame
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import XGBoostAgent, HeuristicAgent, RandomAgent
from src.config import MODEL_DIR
from src.environment import TetrisEnv, drop_y
from scripts.play_gui import (
    draw_board, draw_panel, _draw_deco_piece,
    WIN_W, WIN_H, MARGIN, BG
)

def record_agent(agent_key: str, agent, fonts, out_dir: Path, surf: pygame.Surface):
    print(f"Recording {agent_key}...")
    env = TetrisEnv(max_pieces=1000, seed=42)
    board, info = env.reset(seed=42)
    
    frames = []
    pieces_to_record = 100
    
    while env.pieces_placed < pieces_to_record:
        piece = info["current_piece"]
        action = agent.select_action(board, piece)
        if action is None:
            break
            
        surf.fill(BG)
        board_x = MARGIN
        board_y = MARGIN
        panel_x = board_x + 10 * 32 + MARGIN
        
        from src.environment import unique_rotations
        rots = unique_rotations(piece)
        shape = rots[action.rotation % len(rots)]
        ghost_r = drop_y(board, shape, action.x)
        
        draw_board(surf, board, piece, ghost_r, action.rotation, action.x, board_x, board_y)
        draw_panel(surf, fonts, piece, env.score, env.lines_cleared, env.pieces_placed, agent_key, 50, panel_x, MARGIN)
        
        _draw_deco_piece(surf, 20, WIN_H - 55, "L", (50, 50, 90))
        _draw_deco_piece(surf, WIN_W - 80, WIN_H - 55, "J", (50, 50, 90))
        
        img_array = pygame.surfarray.array3d(surf)
        img_array = img_array.transpose([1, 0, 2])
        frames.append(Image.fromarray(img_array))
        
        board, reward, done, truncated, info = env.step(action)
        if done:
            break

    gif_path = out_dir / f"{agent_key}_demo.gif"
    print(f"Saving GIF to {gif_path} (this might take a moment)...")
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=150,  # 150ms per frame
        loop=0
    )

def main():
    model_path = MODEL_DIR / "xgboost_tetris.joblib"
    out_path = PROJECT_ROOT / "assets"
    out_path.mkdir(exist_ok=True)
    
    print("Initializing headless Pygame...")
    pygame.init()
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

    agents = [
        ("random", RandomAgent(seed=42)),
        ("heuristic", HeuristicAgent())
    ]
    
    if model_path.exists():
        agents.append(("xgboost", XGBoostAgent(model_path=model_path)))
    else:
        print("XGBoost model not found, skipping xgboost recording.")
        
    for name, agent in agents:
        record_agent(name, agent, fonts, out_path, surf)
        
    pygame.quit()
    print("All GIFs generated successfully!")

if __name__ == "__main__":
    main()
