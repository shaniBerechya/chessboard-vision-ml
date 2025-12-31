import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


from models.cnn_generic import CNNGeneric
from models.ml_ae_model import MLAutoEncoder
from models.dropout import Dropout


EXPERIMENTS = {
    "cnn_baseline": {
        "model_cls": CNNGeneric,
        "model_config":{
            "in_channels": 3,
            "conv_channels": [32, 64, 128],
            "input_size": 96,
            "num_classes": 13
        },
        "training_config": {
            "lr": 1e-3,
            "epochs": 10,
            "batch_size": 32,
            "image_size": 96,
            "loss_fn": nn.CrossEntropyLoss(),

        #     "train_split": 0.6,
        #     "val_split": 0.2,
        #     "test_split": 0.2,
        #     "split_seed": 42,
         }
    },

    # "ml_ae": {
    #     "model_cls": MLAutoEncoder,
    #     "model_config": {
    #         "latent_dim": 256,
    #         "num_classes": 13,
    #         "in_channels": 3,
    #         "backbone": resnet18(weights=ResNet18_Weights.DEFAULT),
    #         "alpha": 1.0,
    #         "beta": 1.0
    #     },
    #     "training_config": {
    #         "lr": 1e-3,
    #         "epochs": 10,
    #         "batch_size": 32,
    #         "image_size": 96,
    #         "loss_fn": None # the loos func is a class method
    #     }
    # },

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
            "loss_fn": nn.CrossEntropyLoss(),

            "train_split": 0.6,
            "val_split": 0.2,
            "test_split": 0.2,
            "split_seed": 42,
            "early_stopping": {
                "enabled": True,
                "monitor": "val_loss",
                "patience": 5,
                "min_delta": 1e-4,
                "restore_best": True
            }
        }

    }
}
