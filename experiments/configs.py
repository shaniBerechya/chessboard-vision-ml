import torch
import torch.nn as nn

from torchvision.models import resnet18, ResNet18_Weights

from models.cnn_generic import CNNGeneric
from models.ml_ae_model import MLAutoEncoder
from models.dropout import Dropout


CNN_BASELINE_MODEL_CONFIG = {
    "in_channels": 3,
    "conv_channels": [32, 64, 128],
    "input_size": 96,
    "num_classes": 13
}


EXPERIMENTS = {
    "cnn_baseline": {
        "model_cls": CNNGeneric,
        "model_config": CNN_BASELINE_MODEL_CONFIG,
        "training_config": {
            "lr": 1e-3,
            "epochs": 10,
            "batch_size": 32,
            "image_size": 96,
            "loss_fn": nn.CrossEntropyLoss()
        }
    },

    "cnn_weighted_loss": {
        "model_cls": CNNGeneric,
        "model_config": CNN_BASELINE_MODEL_CONFIG,
        "training_config": {
            "lr": 1e-3,
            "epochs": 10,
            "batch_size": 32,
            "image_size": 96,
            "loss_fn": nn.CrossEntropyLoss(
                weight=torch.tensor(
                    [0.2] + [1.0] * 12,
                    dtype=torch.float
                )
            )
        }
    },

    "ml_ae": {
        "model_cls": MLAutoEncoder,
        "model_config": {
            "latent_dim": 256,
            "num_classes": 13,
            "in_channels": 3,
            "backbone": resnet18(weights=ResNet18_Weights.DEFAULT),
            "alpha": 1.0,
            "beta": 1.0
        },
        "training_config": {
            "lr": 1e-3,
            "epochs": 10,
            "batch_size": 32,
            "image_size": 96,
            "loss_fn": None  # loss is handled inside the model
        }
    },

    "dropout": {
        "model_cls": Dropout,
        "model_config": {
            "backbone_name": "resnet18",
            "num_classes": 13,
            "pretrained": True,
            "dropout_p": 0.3,
            "hidden_dim": 256
        },
        "training_config": {
            "lr": 1e-3,
            "epochs": 10,
            "batch_size": 32,
            "image_size": 96,
            "loss_fn": nn.CrossEntropyLoss()
        }
    }
}
