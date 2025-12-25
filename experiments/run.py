import os
import re
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

from experiments.configs import EXPERIMENTS
from experiments.run_experiment import run_experiment

# -------------------------
# Choose experiment
# -------------------------
EXPERIMENT_NAME = "dropout"

# -------------------------
# Build dataset
# -------------------------
DATASET_PATH = "./data_base/game2_per_frame"
samples = build_dataset_from_game(DATASET_PATH)

# -------------------------
# Resolve config + output paths
# -------------------------
cfg = EXPERIMENTS[EXPERIMENT_NAME]

OUTPUT_ROOT = Path("resut")  # לפי הבקשה שלך
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

def _next_cfg_number(root: Path, experiment_name: str) -> int:
    """
    Finds the next available config number by scanning existing folders in resut/.
    Pattern used: {experiment_name}_cfgNNN
    """
    pattern = re.compile(rf"^{re.escape(experiment_name)}_cfg(\d+)$")
    max_n = 0
    for p in root.iterdir():
        if p.is_dir():
            m = pattern.match(p.name)
            if m:
                max_n = max(max_n, int(m.group(1)))
    return max_n + 1

cfg_number = _next_cfg_number(OUTPUT_ROOT, EXPERIMENT_NAME)
run_dir = OUTPUT_ROOT / f"{EXPERIMENT_NAME}_cfg{cfg_number:03d}"
run_dir.mkdir(parents=True, exist_ok=False)

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

# -------------------------
# Run experiment
# -------------------------
run_out = run_experiment(
    model_cls=cfg["model_cls"],
    model_config=cfg["model_config"],
    dataset_samples=samples,
    training_config=cfg["training_config"],
)

# -------------------------
# Unpack outputs robustly
# -------------------------
model = None
history = None

# Common patterns:
# 1) history only
# 2) (model, history)
# 3) {"model": ..., "history": ...}
if isinstance(run_out, tuple) and len(run_out) == 2:
    model, history = run_out
elif isinstance(run_out, dict):
    model = run_out.get("model", None)
    history = run_out.get("history", None)
else:
    history = run_out

# -------------------------
# Helpers
# -------------------------
def _to_jsonable(obj):
    """Best-effort conversion to JSON-serializable structure."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    # Keras-like History: history.history
    if hasattr(obj, "history") and isinstance(getattr(obj, "history"), dict):
        return _to_jsonable(obj.history)

    # Torch tensors / numpy scalars
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:
        pass

    try:
        import numpy as np
        if isinstance(obj, (np.generic,)):
            return obj.item()
    except Exception:
        pass

    if hasattr(obj, "__dict__"):
        try:
            return dict(obj.__dict__)
        except Exception:
            pass

    return str(obj)

def _save_model_checkpoint(model_obj, out_path: Path):
    """
    Saves PyTorch model checkpoint if possible.
    If model is not a torch.nn.Module, falls back to pickling (not recommended, but better than nothing).
    """
    try:
        import torch
        import torch.nn as nn

        if isinstance(model_obj, nn.Module):
            ckpt = {
                "state_dict": model_obj.state_dict(),
                "experiment_name": EXPERIMENT_NAME,
                "cfg_number": cfg_number,
                "timestamp": timestamp,
                "model_cls": getattr(cfg["model_cls"], "__name__", str(cfg["model_cls"])),
                "model_config": _to_jsonable(cfg["model_config"]),
            }
            torch.save(ckpt, str(out_path))
            return True, "torch_state_dict"
    except Exception:
        pass

    # fallback: best-effort pickle
    try:
        import pickle
        with open(out_path, "wb") as f:
            pickle.dump(model_obj, f)
        return True, "pickle_fallback"
    except Exception as e:
        return False, f"failed: {e}"

def _plot_history(hist_dict: dict, out_dir: Path):
    """
    Plots each metric in history as a separate PNG.
    Expects dict: {metric_name: [values...]}
    """
    if not isinstance(hist_dict, dict) or len(hist_dict) == 0:
        return []

    saved = []
    for metric, values in hist_dict.items():
        # values must be list-like
        if not isinstance(values, (list, tuple)) or len(values) == 0:
            continue

        plt.figure()
        plt.plot(list(range(1, len(values) + 1)), values)
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.title(metric)
        plt.grid(True)

        fname = f"plot_{metric}_cfg{cfg_number:03d}_{timestamp}.png"
        fpath = out_dir / fname
        plt.savefig(fpath, dpi=160, bbox_inches="tight")
        plt.close()
        saved.append(str(fpath))
    return saved

# -------------------------
# Save artifacts (config + results + model + plots)
# -------------------------
run_metadata = {
    "experiment_name": EXPERIMENT_NAME,
    "cfg_number": cfg_number,
    "timestamp": timestamp,
    "dataset_path": DATASET_PATH,
    "model_cls": getattr(cfg["model_cls"], "__name__", str(cfg["model_cls"])),
    "model_config": _to_jsonable(cfg["model_config"]),
    "training_config": _to_jsonable(cfg["training_config"]),
}

config_path = run_dir / f"config_cfg{cfg_number:03d}_{timestamp}.json"
results_path = run_dir / f"results_cfg{cfg_number:03d}_{timestamp}.json"

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(run_metadata, f, ensure_ascii=False, indent=2)

history_json = _to_jsonable(history)
with open(results_path, "w", encoding="utf-8") as f:
    json.dump({"history": history_json}, f, ensure_ascii=False, indent=2)

# Save model checkpoint if available
ckpt_info = None
if model is not None:
    ckpt_path = run_dir / f"model_cfg{cfg_number:03d}_{timestamp}.pt"
    ok, mode = _save_model_checkpoint(model, ckpt_path)
    ckpt_info = {"saved": ok, "mode": mode, "path": str(ckpt_path) if ok else None}
else:
    ckpt_info = {"saved": False, "mode": "no_model_returned", "path": None}

# Save plots if history is dict-like (or Keras style)
plots_saved = []
hist_for_plot = None
if isinstance(history, dict):
    hist_for_plot = history
elif hasattr(history, "history") and isinstance(history.history, dict):
    hist_for_plot = history.history
elif isinstance(history_json, dict) and all(isinstance(v, list) for v in history_json.values()):
    # if _to_jsonable already produced metric->list
    hist_for_plot = history_json

if isinstance(hist_for_plot, dict):
    plots_saved = _plot_history(hist_for_plot, run_dir)

# Save summary
summary_path = run_dir / f"summary_cfg{cfg_number:03d}_{timestamp}.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "run_dir": str(run_dir),
            "config_file": config_path.name,
            "results_file": results_path.name,
            "checkpoint": ckpt_info,
            "plots": plots_saved,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

print(f"[OK] Saved run artifacts to: {run_dir}")
print(f" - Config  : {config_path.name}")
print(f" - Results : {results_path.name}")
print(f" - Model   : {ckpt_info}")
print(f" - Plots   : {len(plots_saved)} files")
print(f" - Summary : {summary_path.name}")

def main():
    experiment_name = "cnn_baseline"
    exp = EXPERIMENTS[experiment_name]

    history = run_experiment(
        model_cls=exp["model_cls"],
        model_config=exp["model_config"],
        training_config=exp["training_config"],
        game_dir="./data_base/game2_per_frame"
    )

    print("Training finished")


if __name__ == "__main__":
    main()
