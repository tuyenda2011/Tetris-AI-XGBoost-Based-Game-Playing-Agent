# Tetris AI: XGBoost-Based Game Playing Agent

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/XGBoost-2.0%2B-EE6C4D?logo=xgboost&logoColor=white" alt="XGBoost" />
  <img src="https://img.shields.io/badge/License-MIT-00C853" alt="License" />
  <img src="https://img.shields.io/badge/Status-Complete-6C5CE7" alt="Status" />
</p>

<p align="center">
  🌐 <b>English</b> | <a href="README_VN.md">Vietnamese</a>
</p>

An Advanced Machine Learning project that trains an **XGBoost regressor** to select optimal Tetris piece placements using engineered board-state features.

---

## 📌 Project Overview

Unlike Reinforcement Learning (DQN/PPO) which learns from raw pixel frames via trial-and-error, this project uses **Supervised Candidate-Action Learning**:

1. **Generate Candidates**: For each incoming tetromino, compute all valid placement positions (rotations and column offsets).
2. **Simulate & Extract Features**: Simulate each placement and extract 12 surface topology features (holes, bumpiness, height, cleared lines, etc.).
3. **Score & Select**: Use the trained XGBoost model to evaluate candidate placements and execute the move with the highest predicted quality score.

```text
Tetris State (Board & Piece)
            │
  Generate Candidate Placements (Rotation & Column)
            │
  Simulate & Extract Board Features (12 Topology Metrics)
            │
  XGBoost Regressor (Score Placement Quality)
            │
  Select & Execute Best Move (Max Predicted Score)
```

---

## 🔬 Engineered Board Features

The model evaluates candidate placements based on 12 domain-engineered board features:

| Feature Key | Description | Game Intuition |
| :--- | :--- | :--- |
| `cleared_lines` | Lines cleared by this placement (0–4) | Primary reward metric |
| `aggregate_height` | Total sum of all column heights | Penalizes high board stacks |
| `holes` | Total covered empty cells | Heavy penalty (prevents blocked spaces) |
| `bumpiness` | Sum of height differences between adjacent columns | Keeps board surface flat |
| `wells` | Single-column cavity depths | Prepares slots for I-pieces |
| `landing_height` | Height where the piece lands | Prefers lower placements |
| `max_height` | Highest column on the board | Early warning for game over |
| `min_height` | Lowest column on the board | Measures board floor level |
| `height_variance` | Variance across column heights | Measures overall board balance |
| `occupied_cells` | Total non-zero blocks | Tracks total board fill |
| `row_density` | Average fill ratio of active rows | Measures row compactness |
| `col_density` | Average fill ratio of active columns | Measures column compactness |

---

## ⚙️ Quickstart & Execution Pipeline

Activate your Conda environment before running commands:
```bash
conda activate tetris
```

### Step 1: Run Unit Tests
Verify environment logic, feature extraction, and candidate generation:
```bash
python -m pytest tests/ -v
```

### Step 2: Generate Training Dataset
Generate supervised rollout targets using parallel processing (`ProcessPoolExecutor`) with real-time terminal progress bar:
```bash
python scripts/generate_dataset.py --config configs/dataset_quality.yaml
```

### Step 3: Train Model
Train and tune the XGBoost regressor using the training configuration:
```bash
python scripts/train_model.py --config configs/model_train.yaml
```

### Step 4: Evaluate Agent Performance
Compare the trained XGBoost Agent against a Random Agent baseline:
```bash
python scripts/evaluate_agent.py --episodes 10 --max-pieces 120 --seed 42 --skip-shap
```

### Step 5: Run Pygame GUI Demo
Launch the visual Pygame GUI to watch the AI play in real time:
```bash
# XGBoost Model Agent
python scripts/play_gui.py --agent xgboost --seed 42 --delay-ms 150

# Heuristic Oracle Agent
python scripts/play_gui.py --agent heuristic --seed 42 --delay-ms 150

# Random Baseline Agent
python scripts/play_gui.py --agent random --seed 42 --delay-ms 150
```

> 🕹️ **GUI Controls:** `UP` / `DOWN` adjust speed · `SPACE` pause/resume · `R` restart · `ESC` exit

---

## 🗂️ Project Structure

```text
Tetris-AI-XGBoost-Based-Game-Playing-Agent/
├── configs/                     # YAML Configuration Files
│   └── dataset_quality.yaml     # Dataset generation parameters
├── src/                         # Core Source Code
│   ├── environment.py           # Gymnasium-compatible Tetris engine
│   ├── features.py               # Feature extraction module (12 metrics)
│   ├── actions.py                # Candidate placement generator
│   ├── dataset.py                # Multiprocessing dataset generator
│   ├── model.py                  # XGBoost model interface
│   ├── agent.py                  # Agent wrappers (Random, Heuristic, XGBoost)
│   ├── train.py                  # Training pipeline orchestrator
│   └── evaluate.py               # Benchmarking suite
├── scripts/                     # Executable CLI Scripts
│   ├── generate_dataset.py       # Dataset generation script
│   ├── train_model.py            # Model training script
│   ├── evaluate_agent.py         # Evaluation script
│   └── play_gui.py               # Pygame GUI visualization script
├── tests/                       # Unit Test Suite
├── data/                        # Processed dataset CSV splits
├── models/                      # Trained model binaries (.joblib)
└── results/                     # Metric summaries and figures
```
