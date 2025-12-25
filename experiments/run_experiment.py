"""
Generic experiment runner.

This file:
- builds dataset and dataloaders
- builds a model (any architecture)
- trains it using TrainModel
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiments.train_model import TrainModel
from data_pros.chess_dataset import ChessSquareDataset


def run_experiment(
    model_cls,
    model_config: dict,
    dataset_samples,
    training_config: dict,
):
    """
    Runs a full training experiment.

    Args:
        model_cls: model class (e.g. CNNGeneric, MLPModel)
        model_config: kwargs for model initialization
        dataset_samples: list of samples (from data preprocessing)
        training_config: dict with training hyperparameters
    """

    # -------------------------
    # Dataset & DataLoader
    # -------------------------
    dataset = ChessSquareDataset(
        dataset_samples,
        image_size=training_config.get("image_size", 96)
    )

    train_loader = DataLoader(
        dataset,
        batch_size=training_config.get("batch_size", 32),
        shuffle=True,
        num_workers=training_config.get("num_workers", 0)
    )

    # -------------------------
    # Model
    # -------------------------
    model = model_cls(**model_config)

    # -------------------------
    # Trainer
    # -------------------------
    trainer = TrainModel(
        model=model,
        lr=training_config.get("lr", 1e-3),
        loss_fn=training_config.get("loss_fn", nn.CrossEntropyLoss())
    )

    # -------------------------
    # Training
    # -------------------------
    history = trainer.fit(
        train_loader,
        epochs=training_config.get("epochs", 10)
    )

    return history
