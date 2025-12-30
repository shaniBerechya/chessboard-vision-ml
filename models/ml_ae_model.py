import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import resnet18, ResNet18_Weights


from data_pros.chess_dataset import ChessSquareDataset
from module_utils import build_model_conv, build_model_fully_conected

# ------------------------------------------------------------------------ #
# -------------------------- Model Struct -------------------------------- #

# Input Image
#    ↓
# Encoder
#    ↓
# Backbone model
#    ↓
# Latent vector z
#    ↓            ↓
# Decoder        Classifier

# L = α·MSE + β·NLL


# ------------------------------------------------------------------------ #

# =========================
# Hyperparameters
# =========================

# Input
IN_CHANNELS = 3
IMG_SIZE = 64          # assuming square crops (64x64)

# Latent space
LATENT_DIM = 256

# Classes
NUM_CLASSES = 13       # 12 pieces + empty

# Loss weights
ALPHA = 1.0      # MSE weight
BETA = 1.0         # Cross Entropy weight

# Backbone Module:

BACKBONE = resnet18(weights=ResNet18_Weights.DEFAULT)


# =========================
# Encoder - Decoder
# =========================

class Encoder(nn.Module):
    def __init__(self, latent_dim: int,backbone_module):
        super().__init__()

        self.backbone = backbone_module

        # Remove the classification head
        self.feature_extractor = nn.Sequential(
            *list(backbone_module.children())[:-1]  # removes final FC
        )

        self.fc = nn.Linear(512, latent_dim)

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: z (B, latent_dim)
        """
        features = self.feature_extractor(x)     # (B, 512, 1, 1)
        features = features.view(features.size(0), -1)  # (B, 512)
        z = self.fc(features)                     # (B, latent_dim)
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, out_channels: int):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)

        self.decoder = build_model_conv(
            in_channels=512,
            out_channels=3,
            hidden_channels=[512, 256, 128, 64],
            kernel_size=4,
            stride=2,
            padding=1,
            hidden_activation=nn.ReLU,
            output_activation=nn.Sigmoid
        )


    def forward(self, z):
        """
        z: (B, latent_dim)
        returns: reconstructed image (B, C, H, W)
        """
        x = self.fc(z)
        x = x.view(x.size(0), 512, 4, 4)
        x_hat = self.decoder(x)
        return x_hat

# =========================
# Classification Head
# =========================

class ClassificationHead(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int):
        super().__init__()
        self.classifier = build_model_fully_conected(
            input_dim=latent_dim,
            output_dim=num_classes,
            hidden_dim= 10,
            num_layers= 2
        )

    def forward(self, z):
        """
        z: (B, latent_dim)
        returns: logits (B, num_classes)
        """
        return self.classifier(z)


# =========================
# ML-AE Model
# =========================

class MLAutoEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        in_channels: int,
        backbone,
        alpha=1.0,
        beta=1.0
    ):
        super().__init__()

        self.encoder = Encoder(latent_dim, backbone)
        self.decoder = Decoder(latent_dim, in_channels)
        self.classifier = ClassificationHead(latent_dim, num_classes)

        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        logits = self.classifier(z)
        return x_hat, logits, z

    def compute_loss(self, x, outputs, labels):
        x_hat, logits, _ = outputs
        x_hat = F.interpolate(
                        x_hat,
                        size=x.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )

        recon_loss = F.mse_loss(x_hat, x)
        cls_loss = F.cross_entropy(logits, labels)

        total = self.alpha * recon_loss + self.beta * cls_loss

        return {
            "total": total,
            "recon": recon_loss,
            "cls": cls_loss
        }


BASELINE_CONFIG = {
    "latent_dim": 256,
    "num_classes": 13,
    "in_channels": 3,
    "backbone": resnet18(weights=ResNet18_Weights.DEFAULT),
    "alpha": 1.0,
    "beta": 1.0
}

def main():
    model = MLAutoEncoder(
        latent_dim=LATENT_DIM,
        num_classes=NUM_CLASSES,
        in_channels=IN_CHANNELS
    )

    x = torch.randn(4, IN_CHANNELS, IMG_SIZE, IMG_SIZE)
    labels = torch.randint(0, NUM_CLASSES, (4,))

    x_hat, logits, z = model(x)
    x_hat = F.interpolate(x_hat, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False) # Ajeset x_hat size to x
    losses = compute_loss(x, x_hat, logits, labels)

    print("Reconstruction shape:", x_hat.shape)
    print("Logits shape:", logits.shape)
    print("Latent shape:", z.shape)
    print("Losses:", losses)

if __name__ == "__main__":
    main()