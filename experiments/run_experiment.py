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


def run_experiment(
    model_cls,
    model_config,
    training_config,
    game_dir,
    experiment_name,
    output_dir,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Dataset
    # -------------------------
    samples = build_dataset_from_game(game_dir)
    dataset = ChessSquareDataset(
        samples,
        image_size=training_config["image_size"]
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size]
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

    # -------------------------
    # Model
    # -------------------------
    model = model_cls(**model_config).to(device)

    trainer = TrainModel(
        model=model,
        device=device,
        lr=training_config["lr"],
        loss_fn=training_config["loss_fn"]
    )

    # -------------------------
    # Training
    # -------------------------
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=training_config["epochs"],
        metric_fn=None
    )

    # -------------------------
    # Validation predictions
    # -------------------------
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    labels = list(LABEL_TO_INDEX.keys())

    cm = confusion_matrix(
        all_targets,
        all_preds,
        labels=list(range(len(labels)))
    )

    # -------------------------
    # Save confusion matrix INSIDE run_dir
    # -------------------------
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title("Confusion Matrix (with empty)")

    cm_path = output_dir / "confusion_matrix_with_empty.png"
    plt.savefig(cm_path, dpi=160, bbox_inches="tight")
    plt.close()

    print(f"✅ Confusion matrix saved to {cm_path}")

    # -------------------------
    # Save model INSIDE run_dir
    # -------------------------
    ckpt_path = output_dir / "model.pth"
    torch.save(model.state_dict(), ckpt_path)

    print(f"✅ Model saved to {ckpt_path}")
    print("✅ Experiment finished successfully")

    return history
