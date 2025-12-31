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
        metric_fn=None,
        # NEW: early stopping config
        early_stopping: dict | None = None,
        # NEW: optional checkpoint path for best model
        checkpoint_path: str | None = None,
    ):
        """
        Trains the model for up to `epochs`, with optional Early Stopping.

        early_stopping dict options:
        - enabled: bool (default False)
        - monitor: str ("val_loss" or "val_metric"), default "val_loss"
        - mode: str ("min" or "max"), default inferred from monitor
        - patience: int, default 5
        - min_delta: float, default 0.0
        - restore_best: bool, default True
        """

        history = {"train_loss": [], "val_loss": [], "val_metric": []}

        # -------------------------
        # Early stopping setup
        # -------------------------
        es = early_stopping or {}
        es_enabled = bool(es.get("enabled", False)) and (val_loader is not None)

        monitor = es.get("monitor", "val_loss")
        patience = int(es.get("patience", 5))
        min_delta = float(es.get("min_delta", 0.0))
        restore_best = bool(es.get("restore_best", True))

        # infer default mode
        if "mode" in es:
            mode = es["mode"]
        else:
            mode = "min" if monitor == "val_loss" else "max"

        best_score = None
        best_state_dict = None
        best_epoch = None
        bad_epochs = 0

        def is_improvement(score, best):
            if mode == "min":
                return score < (best - min_delta)
            else:
                return score > (best + min_delta)

        # -------------------------
        # Training loop
        # -------------------------
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            train_loss = self.train_epoch(train_loader)
            print(f"Train Loss: {train_loss:.4f}")
            history["train_loss"].append(train_loss)

            if val_loader is not None:
                val_loss, val_metric = self.evaluate(val_loader, metric_fn=metric_fn)
                print(f"Val Loss: {val_loss:.4f}")
                history["val_loss"].append(val_loss)

                if val_metric is not None:
                    print(f"Val Metric: {val_metric:.4f}")
                    history["val_metric"].append(val_metric)

                # -------------------------
                # Early stopping step
                # -------------------------
                if es_enabled:
                    if monitor == "val_loss":
                        current = val_loss
                    elif monitor == "val_metric":
                        # אם אין metric_fn, val_metric יהיה None -> אי אפשר לנטר אותו
                        if val_metric is None:
                            raise ValueError("EarlyStopping monitor='val_metric' requires metric_fn to be provided.")
                        current = val_metric
                    else:
                        raise ValueError(f"Unknown early_stopping.monitor='{monitor}'. Use 'val_loss' or 'val_metric'.")

                    if best_score is None:
                        best_score = current
                        best_epoch = epoch
                        if restore_best:
                            best_state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                        if checkpoint_path is not None:
                            torch.save(self.model.state_dict(), checkpoint_path)
                    else:
                        if is_improvement(current, best_score):
                            best_score = current
                            best_epoch = epoch
                            bad_epochs = 0
                            if restore_best:
                                best_state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                            if checkpoint_path is not None:
                                torch.save(self.model.state_dict(), checkpoint_path)
                            print(f"EarlyStopping: improvement. best_{monitor}={best_score:.6f} at epoch {best_epoch+1}")
                        else:
                            bad_epochs += 1
                            print(f"EarlyStopping: no improvement. bad_epochs={bad_epochs}/{patience}")

                            if bad_epochs >= patience:
                                print(
                                    f"EarlyStopping TRIGGERED. "
                                    f"Best epoch: {best_epoch+1}, best_{monitor}: {best_score:.6f}"
                                )
                                # restore best weights if requested
                                if restore_best and best_state_dict is not None:
                                    self.model.load_state_dict(best_state_dict)
                                    print("EarlyStopping: restored best model weights in-memory.")
                                break

        return history
