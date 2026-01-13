# Chessboard Square Classification and Board-State Reconstruction

This repository implements a system for **classifying each square of a chessboard** in real images and reconstructing the board state in **FEN notation**. The project handles occlusions using an Out-of-Distribution approach and is robust to unseen games.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Environment Setup and Requirements](#environment-setup)
- [Training](#training)
- [Running Inference / Evaluation](#running-inference--evaluation)


---

## Project Overview

The system performs:

1. **Per-square classification**: Each of the 64 squares is classified into one of the possible piece classes (white pawn, black knight, etc.) or as `unknown` if occluded.
2. **Board reconstruction**: Reconstructs the chessboard state into FEN notation and generates board images.
3. **Robustness**: The model is designed to handle unseen games and occluded squares.

The project uses **CNN-based models** and can optionally leverage additional datasets or temporal frame information for data augmentation.

---

## Environment Setup and Requirements

Install the Python dependencies:
**Windows:**

```bash

git clone (https://github.com/shaniBerechya/chessboard-vision-ml.git
cd <repo-folder>
python -m venv venv
.\venv\Scripts\activate 
pip install -r requirements.txt

```

**Linux / macOS:**

```bash

git clone (https://github.com/shaniBerechya/chessboard-vision-ml.git
cd <repo-folder>
python -m venv venv
source venv/bin/activate   
pip install -r requirements.txt

```

## Training

1. Configure the model:
Open the experiments/configs.py file and select the model and training configuration:
```json
EXPERIMENTS = {
    "<model_name>": {
        "model_cls": <model_class>,
        "model_config": <dict_of_model_args>,
        "training_config": {
            "lr": 1e-3,
            "epochs": 10,
            "batch_size": 32,
            "image_size": 96,
            "loss_fn": nn.CrossEntropyLoss(
                weight=torch.tensor(
                    [1.0] * 12 + [0.2],
                    dtype=torch.float
                )
            ),
            "ood_threshold": <ood_threshold>,
            "train_split": 0.6,
            "val_split": 0.2,
            "test_split": 0.2
        }
    }
}
```

You can modify the model_config or training_config to experiment with different architectures, learning rates, or epochs.

**Run training:**

```bash
python -m experiments.run -e cnn_baseline -o ./results --num_frames 10
```

**Parameters:**

*-e:* experiment name from EXPERIMENTS

*-o:* output folder for model checkpoints and logs

*--num_frames:* number of frames to use per game (optional)

**Training outputs:**

Model checkpoints (.pt files) saved in results/

Training logs and accuracy plots

## Running-inference--evaluation

After training, you can use the predict_board.py script to classify a chessboard image and reconstruct the board state.

```bash
python predict_board.py --model_path <path_to_model_pat> --image_path <path_to_image_path>
```

**Options:**

*--model_path:* path to your trained model checkpoint

*--image_path:* path to the input chessboard image

*--output_fen:* optional path to save FEN notation output

*--show:* display the reconstructed board image

The script outputs:

Per-square predictions

Reconstructed FEN string

Optional board image with predicted pieces


