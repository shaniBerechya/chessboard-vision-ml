import torch.nn as nn
from models.cnn_generic import CNNGeneric, BASELINE_CONFIG

EXPERIMENTS = {
    "cnn_baseline": {
        "model_cls": CNNGeneric,
        "model_config": BASELINE_CONFIG,
        "training_config": {
            "lr": 1e-3,
            "epochs": 10,
            "batch_size": 32,
            "image_size": 96,
            "loss_fn": nn.CrossEntropyLoss()
        }
    }
}
