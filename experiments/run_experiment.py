import os
import torch
from torch.utils.data import DataLoader, random_split

from data_pros.data_preprocessing import build_dataset_from_game
from data_pros.chess_dataset import ChessSquareDataset
from experiments.train_model import TrainModel


def accuracy_fn(outputs, targets):
    """
    Simple classification accuracy
    """
    preds = outputs.argmax(dim=1)
    return (preds == targets).float().mean().item()


def run_experiment(
    model_cls,
    model_config,
    training_config,
    game_dir,
    experiment_name
):
    # -------------------------
    # Device
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Build dataset (raw samples)
    # -------------------------
    samples = build_dataset_from_game(game_dir)

    dataset = ChessSquareDataset(
        samples,
        image_size=training_config["image_size"]
    )

    # -------------------------
    # Train / Validation split
    # -------------------------
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size]
    )

    # -------------------------
    # DataLoaders
    # -------------------------
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

    # -------------------------
    # Trainer
    # -------------------------
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
        metric_fn=accuracy_fn
    )

    # -------------------------
    # Save trained model
    # -------------------------
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = f"checkpoints/{experiment_name}.pth"
    torch.save(model.state_dict(), checkpoint_path)

    print(f"\n✅ Model saved to {checkpoint_path}")

    return history
