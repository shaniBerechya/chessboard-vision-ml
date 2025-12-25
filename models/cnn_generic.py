"""
Generic CNN model for chessboard square classification.

This model is fully parameterized:
- Number of convolutional layers
- Number of channels per layer
- Input image size
- Number of output classes

A specific configuration is defined as the baseline model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNGeneric(nn.Module):
    """
    Generic CNN for image classification.
    """

    def __init__(
        self,
        in_channels: int,
        conv_channels: list,
        input_size: int,
        num_classes: int
    ):
        """
        Args:
            in_channels (int): number of input channels (RGB = 3)
            conv_channels (list[int]): output channels for each conv layer
            input_size (int): height/width of input image (assumed square)
            num_classes (int): number of output classes
        """
        super().__init__()

        layers = []
        current_channels = in_channels
        current_size = input_size

        # -------------------------
        # Convolutional feature extractor
        # -------------------------
        for out_channels in conv_channels:
            layers.append(
                nn.Conv2d(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2))

            current_channels = out_channels
            current_size //= 2

        self.feature_extractor = nn.Sequential(*layers)

        # -------------------------
        # Classifier
        # -------------------------
        self.classifier = nn.Sequential(
            nn.Linear(current_channels * current_size * current_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): shape [B, C, H, W]

        Returns:
            logits (Tensor): shape [B, num_classes]
        """
        x = self.feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


# -------------------------------------------------
# Baseline configuration
# -------------------------------------------------

BASELINE_CONFIG = {
    "in_channels": 3,
    "conv_channels": [32, 64, 128],
    "input_size": 96,
    "num_classes": 13
}
