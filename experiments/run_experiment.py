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

def collect_preds_targets(model, device, loader):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)

def run_experiment(
    model_cls,
    model_config,
    training_config,
    game_dirs,
    experiment_name,
    output_dir,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Dataset
    # -------------------------
    samples = []
    for game_dir in game_dirs:
        game_samples = build_dataset_from_game(game_dir)
        samples.extend(game_samples)
    dataset = ChessSquareDataset(
        samples,
        image_size=training_config["image_size"]
    )

    n = len(dataset)

    train_ratio = float(training_config.get("train_split", 0.6))
    val_ratio   = float(training_config.get("val_split", 0.2))
    test_ratio  = float(training_config.get("test_split", 0.2))

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_split + val_split + test_split must sum to 1.0")

    train_size = int(train_ratio * n)
    val_size   = int(val_ratio * n)
    test_size  = n - train_size - val_size

    seed = int(training_config.get("split_seed", 42))
    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

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


    # -------------------------
    # Model + Trainer
    # -------------------------
    model = model_cls(**model_config).to(device)

    trainer = TrainModel(
        model=model,
        device=device,
        lr=training_config["lr"],
        loss_fn=training_config["loss_fn"]
    )

   
    best_ckpt_path = str(Path(output_dir) / "model_best.pth")

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=training_config["epochs"],
        metric_fn=None,
        early_stopping=training_config.get("early_stopping", None),
        checkpoint_path=best_ckpt_path
    )

    # -------------------------
    # VAL + TEST predictions (use existing helper)
    # -------------------------
    val_preds, val_targets = collect_preds_targets(model, device, val_loader)
    test_preds, test_targets = collect_preds_targets(model, device, test_loader)
    # -------------------------
    # Accuracy metrics (VAL + TEST)
    # -------------------------
    empty_idx = LABEL_TO_INDEX["empty"]

    def compute_metrics(preds, targets):
        acc_all = (preds == targets).mean()

        mask_no_empty = targets != empty_idx
        acc_no_empty = (preds[mask_no_empty] == targets[mask_no_empty]).mean() if mask_no_empty.any() else 0.0

        mask_only_pieces = targets != empty_idx
        acc_only_pieces = (preds[mask_only_pieces] == targets[mask_only_pieces]).mean() if mask_only_pieces.any() else 0.0

        return {
            "accuracy_all": float(acc_all),
            "accuracy_no_empty": float(acc_no_empty),
            "accuracy_only_pieces": float(acc_only_pieces),
        }

    val_metrics = compute_metrics(val_preds, val_targets)
    test_metrics = compute_metrics(test_preds, test_targets)

    # Keep existing key names for VAL (backward compatible)
    history["accuracy_all"] = [val_metrics["accuracy_all"]]
    history["accuracy_no_empty"] = [val_metrics["accuracy_no_empty"]]
    history["accuracy_only_pieces"] = [val_metrics["accuracy_only_pieces"]]

    # Add NEW keys for TEST
    history["test_accuracy_all"] = [test_metrics["accuracy_all"]]
    history["test_accuracy_no_empty"] = [test_metrics["accuracy_no_empty"]]
    history["test_accuracy_only_pieces"] = [test_metrics["accuracy_only_pieces"]]

    # -------------------------
    # Confusion matrices (VAL + TEST)
    # -------------------------
    labels = list(LABEL_TO_INDEX.keys())
    label_ids = list(range(len(labels)))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def save_cm(targets, preds, title, filename):
        cm = confusion_matrix(targets, preds, labels=label_ids)
        fig, ax = plt.subplots(figsize=(8, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, cmap="Blues", colorbar=True)
        ax.set_title(title)
        path = output_dir / filename
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.close()
        return path

    val_cm_path = save_cm(val_targets, val_preds, "Confusion Matrix (VAL)", "confusion_matrix_val.png")
    test_cm_path = save_cm(test_targets, test_preds, "Confusion Matrix (TEST)", "confusion_matrix_test.png")

    print(f"✅ VAL confusion matrix saved to {val_cm_path}")
    print(f"✅ TEST confusion matrix saved to {test_cm_path}")

    # -------------------------
    # Table (CSV) + simple bar plot for quick comparison
    # -------------------------
    import csv

    metrics_table_path = output_dir / "metrics_table.csv"
    with open(metrics_table_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "accuracy_all", "accuracy_no_empty", "accuracy_only_pieces"])
        w.writerow(["val",
                    val_metrics["accuracy_all"],
                    val_metrics["accuracy_no_empty"],
                    val_metrics["accuracy_only_pieces"]])
        w.writerow(["test",
                    test_metrics["accuracy_all"],
                    test_metrics["accuracy_no_empty"],
                    test_metrics["accuracy_only_pieces"]])

    plt.figure(figsize=(6, 4))
    plt.bar(["val", "test"], [val_metrics["accuracy_all"], test_metrics["accuracy_all"]])
    plt.ylabel("Accuracy (all)")
    plt.title("VAL vs TEST Accuracy (all)")
    plt.grid(True, axis="y")
    metrics_plot_path = output_dir / "metrics_bar.png"
    plt.savefig(metrics_plot_path, dpi=160, bbox_inches="tight")
    plt.close()

    # Put artifact paths into history so they are visible in history.json
    history["artifacts"] = {
        "val_confusion_matrix": str(val_cm_path),
        "test_confusion_matrix": str(test_cm_path),
        "metrics_table_csv": str(metrics_table_path),
        "metrics_bar_png": str(metrics_plot_path),
    }

    # -------------------------
    # Save model
    # -------------------------
    ckpt_path = output_dir / "model.pth"
    torch.save(model.state_dict(), ckpt_path)

    print(f"✅ Model saved to {ckpt_path}")
    print("✅ Experiment finished successfully")

    return history
