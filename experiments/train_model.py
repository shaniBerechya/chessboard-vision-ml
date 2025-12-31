import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class TrainModel:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        lr: float = 1e-3,
        optimizer_cls=torch.optim.Adam,
        loss_fn=None
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.optimizer = optimizer_cls(model.parameters(), lr=lr)
        self.loss_fn = loss_fn

    # --------------------------------------------------
    # Train one epoch
    # --------------------------------------------------
    def train_epoch(self, dataloader: DataLoader):
        self.model.train()
        running_loss = 0.0

        for x, y in tqdm(dataloader, desc="Training"):
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(x)

            if hasattr(self.model, "compute_loss"):
                losses = self.model.compute_loss(x, outputs, y)
                loss = losses["total"]
            else:
                loss = self.loss_fn(outputs, y)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * x.size(0)

        return running_loss / len(dataloader.dataset)

    # --------------------------------------------------
    # Evaluate (loss + metrics)
    # --------------------------------------------------
    def evaluate(self, dataloader: DataLoader, metric_fns=None):
        self.model.eval()
        running_loss = 0.0

        metric_fns = metric_fns or {}
        metric_sums = {name: 0.0 for name in metric_fns}

        with torch.no_grad():
            for x, y in tqdm(dataloader, desc="Evaluating"):
                x = x.to(self.device)
                y = y.to(self.device)

                outputs = self.model(x)

                if hasattr(self.model, "compute_loss"):
                    losses = self.model.compute_loss(x, outputs, y)
                    loss = losses["total"]
                else:
                    loss = self.loss_fn(outputs, y)

                running_loss += loss.item() * x.size(0)

                for name, fn in metric_fns.items():
                    metric_sums[name] += fn(outputs, y) * x.size(0)

        avg_loss = running_loss / len(dataloader.dataset)
        avg_metrics = {
            name: metric_sums[name] / len(dataloader.dataset)
            for name in metric_sums
        }

        return avg_loss, avg_metrics

    # --------------------------------------------------
    # Fit
    # --------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        epochs: int = 10,
        metric_fns: dict | None = None
    ):
        metric_fns = metric_fns or {}

        history = {
            "train_loss": [],
            "val_loss": [],
        }

        for name in metric_fns:
            history[name] = []

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            train_loss = self.train_epoch(train_loader)
            history["train_loss"].append(train_loss)
            print(f"Train Loss: {train_loss:.4f}")

            if val_loader is not None:
                val_loss, val_metrics = self.evaluate(
                    val_loader, metric_fns=metric_fns
                )
                history["val_loss"].append(val_loss)
                print(f"Val Loss: {val_loss:.4f}")

                for name, value in val_metrics.items():
                    history[name].append(value)
                    print(f"{name}: {value:.4f}")

        return history
