"""Visualization and explainability artifacts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features import FEATURE_NAMES

LOGGER = logging.getLogger(__name__)


def plot_evaluation(results: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Create score, lines, and comparison plots for evaluated agents."""

    output_dir.mkdir(parents=True, exist_ok=True)
    combined = []
    for name, frame in results.items():
        agent_frame = frame.copy()
        agent_frame["agent"] = name
        combined.append(agent_frame)
    data = pd.concat(combined, ignore_index=True)

    for metric in ["score", "lines_cleared"]:
        plt.figure(figsize=(8, 5))
        agents = data["agent"].unique()
        plot_data = [data[data["agent"] == a][metric].values for a in agents]
        
        plt.boxplot(plot_data, tick_labels=agents, patch_artist=True)
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f"{metric.replace('_', ' ').title()} Distribution by Agent")
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}_distribution.png", dpi=160)
        plt.close()

    means = data.groupby("agent")["score"].mean().sort_values()
    plt.figure(figsize=(7, 5))
    means.plot(kind="barh")
    plt.xlabel("Average Score")
    plt.tight_layout()
    plt.savefig(output_dir / "performance_comparison.png", dpi=160)
    plt.close()


def plot_feature_importance(model: Any, output_dir: Path) -> None:
    """Save an XGBoost feature-importance plot when the model exposes importances."""

    output_dir.mkdir(parents=True, exist_ok=True)
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        LOGGER.warning("Model does not expose feature_importances_")
        return
    order = np.argsort(importances)
    plt.figure(figsize=(9, 6))
    plt.barh(np.array(FEATURE_NAMES)[order], np.asarray(importances)[order])
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=160)
    plt.close()


def save_shap_summary(model: Any, sample: pd.DataFrame, output_dir: Path) -> None:
    """Save SHAP values and summary plot for a sample of model inputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import shap
    except ImportError:
        LOGGER.warning("SHAP is not installed; skipping SHAP artifacts")
        return

    x_sample = sample[FEATURE_NAMES].copy()
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_sample)
        np.save(output_dir / "shap_values.npy", shap_values)
        plt.figure()
        shap.summary_plot(shap_values, x_sample, show=False)
        plt.tight_layout()
        plt.savefig(output_dir / "shap_summary.png", dpi=160, bbox_inches="tight")
        plt.close()
    except Exception as exc:  # pragma: no cover - depends on SHAP/XGBoost versions.
        message = (
            "SHAP artifact generation failed, but model evaluation completed.\n"
            f"Error type: {type(exc).__name__}\n"
            f"Error message: {exc}\n"
            "Try rerunning with --skip-shap, or install compatible SHAP/XGBoost versions."
        )
        (output_dir / "shap_error.txt").write_text(message, encoding="utf-8")
        LOGGER.warning(message)
