import torch.nn as nn

from models.cnn_generic import CNNGeneric, CNN_BASELINE_CONFIG
from models.ml_ae_model import MLAutoEncoder, ML_AE_BASELINE_CONFIG
from models.dropout import Dropout, BASELINE_CONFIG as DROPOUT_BASELINE_CONFIG


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
    },

    "ml_ae": {
        "model_cls": MLAutoEncoder,
        "model_config": BASELINE_CONFIG,
        "training_config": {
            "lr": 1e-3,
            "epochs": 10,
            "batch_size": 32,
            "image_size": 96,
            "loss_fn": None # the loos func is a class method
        }
    },

    "dropout": {
        "model_cls": Dropout,
        "model_config": DROPOUT_BASELINE_CONFIG,
        "training_config": {
            "lr": 1e-3,
            "epochs": 10,
            "batch_size": 32,
            "image_size": 96,
            "loss_fn": nn.CrossEntropyLoss()
        }
    }
}
