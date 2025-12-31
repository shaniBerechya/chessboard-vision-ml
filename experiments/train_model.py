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
        metric_fn=None,
        # NEW: early stopping config
        early_stopping: dict | None = None,
        # NEW: optional checkpoint path for best model
        checkpoint_path: str | None = None,
        metric_fns: dict | None = None
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

        metric_fns = metric_fns or {}

        history = {
            "train_loss": [],
            "val_loss": [],
        }

        for name in metric_fns:
            history[name] = []


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
