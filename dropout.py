import torch
import torch.nn as nn
from torchvision import models


class ChessSquareClassifier(nn.Module):
    """
    Generic classifier for chess square patches.

    Uses a torchvision backbone (e.g., 'resnet18') pretrained on ImageNet,
    replaces its classifier head with an MLP that includes Dropout.

    Works best when the input is normalized to ImageNet mean/std.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        num_classes: int = 13,
        pretrained: bool = True,
        dropout_p: float = 0.3,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.backbone_name = backbone_name

        backbone_name = backbone_name.lower().strip()
        if not hasattr(models, backbone_name):
            raise ValueError(
                f"Unknown backbone '{backbone_name}'. "
                f"Expected a torchvision.models entry like 'resnet18', 'resnet34', etc."
            )

        backbone_fn = getattr(models, backbone_name)
        self.backbone = backbone_fn(pretrained=pretrained)

        # --- Replace final layer depending on architecture family ---
        if backbone_name.startswith("resnet"):
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            feat_dim = in_features

        elif backbone_name.startswith("mobilenet_v3"):
            # MobileNetV3: classifier is a Sequential, last Linear usually at [-1]
            if not hasattr(self.backbone, "classifier"):
                raise ValueError(f"Backbone {backbone_name} has no .classifier attribute")
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier = nn.Identity()
            feat_dim = in_features

        elif backbone_name.startswith("efficientnet"):
            # EfficientNet: classifier is Sequential([Dropout, Linear])
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier = nn.Identity()
            feat_dim = in_features

        else:
            raise ValueError(
                f"Backbone '{backbone_name}' is not wired in this generic wrapper yet. "
                f"Use a ResNet family first (resnet18/34/50) or extend the adapter."
            )

        # --- Head with dropout (MC Dropout friendly) ---
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits


def enable_dropout_in_eval(model: nn.Module) -> None:
    """
    Enable dropout layers during evaluation for MC Dropout,
    while keeping BatchNorm layers in eval mode.
    Usage:
        model.eval()
        enable_dropout_in_eval(model)
    """
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
