import torch.nn as nn
from models.cnn_generic import CNNGeneric, BASELINE_CONFIG
# from models.ml_ae_model import MLAutoEncoder, BASELINE_CONFIG
from models.dropout import Dropout, BASELINE_CONFIG


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
    # "ml_ae": {
    #     "model_cls": MLAutoEncoder,
    #     "model_config": BASELINE_CONFIG,
    #     "training_config": {
    #         "lr": 1e-3,
    #         "epochs": 10,
    #         "batch_size": 32,
    #         "image_size": 96,
    #         "loss_fn": compute_loss()
    #     }
    # },
    "dropout": {
        "model_cls": Dropout,
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
