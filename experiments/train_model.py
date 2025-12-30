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
        """
        Generic trainer for any PyTorch model.

        Args:
            model: PyTorch model to train
            device: torch.device ('cuda' or 'cpu')
            lr: learning rate
            optimizer_cls: optimizer class (Adam, SGD, etc.)
            loss_fn: loss function, can be None if model returns dict of losses
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer_cls(model.parameters(), lr=lr)
        self.loss_fn = loss_fn

    def train_epoch(self, dataloader: DataLoader):
        self.model.train()
        running_loss = 0.0

        for batch in tqdm(dataloader, desc="Training"):
            # Unpack batch: support (x, y) or just x
            if len(batch) == 2:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
            else:
                x = batch[0].to(self.device)
                y = None

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(x)

            if hasattr(self.model, "compute_loss"):
                losses = self.model.compute_loss(x, outputs, y)
                loss = losses["total"]
            else:
                loss = self.loss_fn(outputs, y)


            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * x.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        return epoch_loss

    def evaluate(self, dataloader: DataLoader, metric_fn=None):
        self.model.eval()
        running_loss = 0.0
        metric_total = 0.0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                if len(batch) == 2:
                    x, y = batch
                    x, y = x.to(self.device), y.to(self.device)
                else:
                    x = batch[0].to(self.device)
                    y = None

                outputs = self.model(x)

                if hasattr(self.model, "compute_loss"):
                    losses = self.model.compute_loss(x, outputs, y)
                    loss = losses["total"]
                else:
                    loss = self.loss_fn(outputs, y)

                running_loss += loss.item() * x.size(0)

                if metric_fn is not None and y is not None:
                    metric_total += metric_fn(outputs, y) * x.size(0)

        avg_loss = running_loss / len(dataloader.dataset)
        avg_metric = None
        if metric_fn is not None:
            avg_metric = metric_total / len(dataloader.dataset)

        return avg_loss, avg_metric

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        epochs: int = 10,
        metric_fn=None
    ):
        history = {"train_loss": [], "val_loss": [], "val_metric": []}

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            train_loss = self.train_epoch(train_loader)
            print(f"Train Loss: {train_loss:.4f}")
            history["train_loss"].append(train_loss)

            if val_loader is not None:
                val_loss, val_metric = self.evaluate(val_loader, metric_fn=metric_fn)
                print(f"Val Loss: {val_loss:.4f}")
                if val_metric is not None:
                    print(f"Val Metric: {val_metric:.4f}")
                    history["val_metric"].append(val_metric)
                history["val_loss"].append(val_loss)

        return history
