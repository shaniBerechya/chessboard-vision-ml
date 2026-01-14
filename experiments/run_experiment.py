import os
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from data_pros.data_preprocessing import build_dataset_from_game
from data_pros.chess_dataset import ChessSquareDataset, LABEL_TO_INDEX
from experiments.train_model import TrainModel


# for ml_ae
def output_to_logist(outputs):
    if isinstance(outputs, (tuple, list)):
        logits = outputs[1]
    elif isinstance(outputs, dict):
        logits = outputs["logits"]
    else:
        logits = outputs
    return logits


def collect_preds_targets(model, device, loader):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            logits = output_to_logist(outputs)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)

# --------------------------------------------------
# Accuracy functions
# --------------------------------------------------
def accuracy_all_fn(outputs, targets):
    logits = output_to_logist(outputs)
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def accuracy_no_empty_fn(outputs, targets):
    logits = output_to_logist(outputs)
    preds = logits.argmax(dim=1)
    empty_idx = LABEL_TO_INDEX["empty"]
    mask = targets != empty_idx
    if mask.sum() == 0:
        return 0.0
    return (preds[mask] == targets[mask]).float().mean().item()


def accuracy_only_pieces_fn(outputs, targets):
    logits = output_to_logist(outputs)
    preds = logits.argmax(dim=1)
    empty_idx = LABEL_TO_INDEX["empty"]
    mask = targets != empty_idx
    if mask.sum() == 0:
        return 0.0
    return (preds[mask] == targets[mask]).float().mean().item()


# --------------------------------------------------
# Run experiment
# --------------------------------------------------
def run_experiment(
    model_cls,
    model_config,
    training_config,
    game_dirs,
    experiment_name,
    output_dir,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------
    # GAME-LEVEL SPLIT: TRAIN vs (VAL + TEST)
    # --------------------------------------------------
    game_dirs = sorted([str(g) for g in game_dirs])

    seed = int(training_config.get("split_seed", 42))
    rng = np.random.default_rng(seed)
    rng.shuffle(game_dirs)

    train_ratio = float(training_config.get("train_split", 0.6))
    val_ratio   = float(training_config.get("val_split", 0.2))
    test_ratio  = float(training_config.get("test_split", 0.2))

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_split + val_split + test_split must sum to 1.0")

    n_games = len(game_dirs)
    n_train_games = max(1, int(train_ratio * n_games))

    train_games = game_dirs[:n_train_games]
    eval_games  = game_dirs[n_train_games:]  # used for val + test

    print("📁 Game-level split:")
    print("  Train games:", train_games)
    print("  Eval games (val + test):", eval_games)

    # --------------------------------------------------
    # BUILD TRAIN DATASET (ONLY TRAIN GAMES)
    # --------------------------------------------------
    train_samples = []
    for g in train_games:
        train_samples.extend(build_dataset_from_game(g))

    train_dataset = ChessSquareDataset(
        train_samples,
        image_size=training_config["image_size"]
    )

    # --------------------------------------------------
    # BUILD EVAL DATASET (VAL + TEST MIXED)
    # --------------------------------------------------
    eval_samples = []
    for g in eval_games:
        eval_samples.extend(build_dataset_from_game(g))

    rng.shuffle(eval_samples)

    n_eval = len(eval_samples)
    val_frac = val_ratio / (val_ratio + test_ratio)

    n_val = int(val_frac * n_eval)
    val_samples  = eval_samples[:n_val]
    test_samples = eval_samples[n_val:]

    val_dataset = ChessSquareDataset(
        val_samples,
        image_size=training_config["image_size"]
    )

    test_dataset = ChessSquareDataset(
        test_samples,
        image_size=training_config["image_size"]
    )

    # --------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False
    )

    # --------------------------------------------------
    # MODEL + TRAINER (UNCHANGED)
    # --------------------------------------------------
    model = model_cls(**model_config).to(device)

    trainer = TrainModel(
        model=model,
        device=device,
        lr=training_config["lr"],
        loss_fn=training_config["loss_fn"]
    )

    best_ckpt_path = str(Path(output_dir) / "model_best.pth")

    # --------------------------------------------------
    # TRAINING
    # --------------------------------------------------
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=training_config["epochs"],
        metric_fn=None,
        early_stopping=training_config.get("early_stopping", None),
        checkpoint_path=best_ckpt_path,
        metric_fns={
            "accuracy_all": accuracy_all_fn,
            "accuracy_no_empty": accuracy_no_empty_fn,
            "accuracy_only_pieces": accuracy_only_pieces_fn,
        }
    )

    # --------------------------------------------------
    # EVALUATION
    # --------------------------------------------------
    val_preds, val_targets = collect_preds_targets(model, device, val_loader)
    test_preds, test_targets = collect_preds_targets(model, device, test_loader)

    empty_idx = LABEL_TO_INDEX["empty"]

    def compute_metrics(preds, targets):
        acc_all = (preds == targets).mean()
        mask = targets != empty_idx
        acc_no_empty = (preds[mask] == targets[mask]).mean() if mask.any() else 0.0
        acc_only_pieces = acc_no_empty
        return {
            "accuracy_all": float(acc_all),
            "accuracy_no_empty": float(acc_no_empty),
            "accuracy_only_pieces": float(acc_only_pieces),
        }

    val_metrics = compute_metrics(val_preds, val_targets)
    test_metrics = compute_metrics(test_preds, test_targets)

    history["accuracy_all"] = [val_metrics["accuracy_all"]]
    history["accuracy_no_empty"] = [val_metrics["accuracy_no_empty"]]
    history["accuracy_only_pieces"] = [val_metrics["accuracy_only_pieces"]]

    history["test_accuracy_all"] = [test_metrics["accuracy_all"]]
    history["test_accuracy_no_empty"] = [test_metrics["accuracy_no_empty"]]
    history["test_accuracy_only_pieces"] = [test_metrics["accuracy_only_pieces"]]

    # --------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = output_dir / "model.pth"
    torch.save(model.state_dict(), ckpt_path)

    print(f"✅ Model saved to {ckpt_path}")
    print("✅ Experiment finished successfully")

    return history

