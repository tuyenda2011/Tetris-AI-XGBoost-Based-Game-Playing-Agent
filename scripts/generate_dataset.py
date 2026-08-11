"""Generate supervised Tetris candidate-action datasets."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, TrainingConfig
from src.dataset import generate_dataset, save_dataset_splits

DEFAULT_CONFIG = {
    "episodes": 20,
    "max_pieces": 120,
    "seed": 42,
    "epsilon": 0.15,
    "target_mode": "rollout",
    "rollout_steps": 1,
    "discount": 0.92,
    "workers": 1,
    "output_dir": DATA_DIR / "processed",
}
CONFIG_KEYS = set(DEFAULT_CONFIG)


def _load_config(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("YAML config must be a mapping")
    config = raw.get("dataset", raw)
    if not isinstance(config, dict):
        raise ValueError("YAML config field 'dataset' must be a mapping")
    unknown = set(config).difference(CONFIG_KEYS)
    if unknown:
        raise ValueError(f"Unknown dataset config keys: {sorted(unknown)}")
    return dict(config)


def _resolve_project_path(value: object) -> Path:
    path = value if isinstance(value, Path) else Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, remaining = config_parser.parse_known_args()
    config_path = _resolve_project_path(config_args.config) if config_args.config else None

    parser = argparse.ArgumentParser(description=__doc__, parents=[config_parser])
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--max-pieces", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--target-mode", choices=["rollout", "immediate"])
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--discount", type=float)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(remaining)

    merged = dict(DEFAULT_CONFIG)
    merged.update(_load_config(config_path))
    cli_overrides = {
        key: value
        for key, value in vars(args).items()
        if key != "config" and value is not None
    }
    merged.update(cli_overrides)
    merged["output_dir"] = _resolve_project_path(merged["output_dir"])
    merged["config"] = config_path
    return argparse.Namespace(**merged)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    dataset = generate_dataset(
        episodes=args.episodes,
        max_pieces=args.max_pieces,
        seed=args.seed,
        epsilon=args.epsilon,
        target_mode=args.target_mode,
        rollout_steps=args.rollout_steps,
        discount=args.discount,
        workers=args.workers,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    full_path = args.output_dir / "candidate_actions.csv"
    dataset.to_csv(full_path, index=False)
    paths = save_dataset_splits(dataset, args.output_dir, TrainingConfig(seed=args.seed))
    metadata = {
        "episodes": args.episodes,
        "max_pieces": args.max_pieces,
        "seed": args.seed,
        "epsilon": args.epsilon,
        "target_mode": args.target_mode,
        "rollout_steps": args.rollout_steps if args.target_mode == "rollout" else 0,
        "discount": args.discount,
        "workers": args.workers,
        "config": args.config.relative_to(PROJECT_ROOT).as_posix() if args.config else None,
        "samples": len(dataset),
        "split_paths": {name: path.relative_to(PROJECT_ROOT).as_posix() for name, path in paths.items()},
    }
    (args.output_dir / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    logging.info("Saved full dataset to %s", full_path)
    logging.info("Saved splits: %s", paths)


if __name__ == "__main__":
    main()
