import os
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from experiments.configs import EXPERIMENTS
from experiments.run_experiment import run_experiment


EXPERIMENT_NAME = "cnn_baseline"


def sanitize_training_config(cfg: dict):
    clean_cfg = {}
    for k, v in cfg.items():
        if isinstance(v, nn.Module):
            clean_cfg[k] = v.__class__.__name__
        else:
            clean_cfg[k] = v
    return clean_cfg


OUTPUT_ROOT = Path("results")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = OUTPUT_ROOT / f"{EXPERIMENT_NAME}_{timestamp}"
run_dir.mkdir(parents=True, exist_ok=True)

cfg = EXPERIMENTS[EXPERIMENT_NAME]


history = run_experiment(
    experiment_name=EXPERIMENT_NAME,
    model_cls=cfg["model_cls"],
    model_config=cfg["model_config"],
    training_config=cfg["training_config"],
    game_dir="./data_base/game2_per_frame"
)


metadata = {
    "experiment_name": EXPERIMENT_NAME,
    "timestamp": timestamp,
    "model_class": cfg["model_cls"].__name__,
    "model_config": cfg["model_config"],
    "training_config": sanitize_training_config(cfg["training_config"]),
}

with open(run_dir / "config.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

with open(run_dir / "history.json", "w", encoding="utf-8") as f:
    json.dump(history, f, indent=2)

print("✅ Experiment finished successfully")
print(f"📁 Results saved to: {run_dir}")
