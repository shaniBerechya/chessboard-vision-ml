from experiments.configs import EXPERIMENTS
from experiments.run_experiment import run_experiment
from data_pros.data_preprocessing import build_dataset_from_game

# -------------------------
# Choose experiment
# -------------------------
EXPERIMENT_NAME = "cnn_baseline"

# -------------------------
# Build dataset
# -------------------------
samples = build_dataset_from_game(
    "./data_base/game2_per_frame"
)

# -------------------------
# Run experiment
# -------------------------
cfg = EXPERIMENTS[EXPERIMENT_NAME]

history = run_experiment(
    model_cls=cfg["model_cls"],
    model_config=cfg["model_config"],
    dataset_samples=samples,
    training_config=cfg["training_config"]
)
