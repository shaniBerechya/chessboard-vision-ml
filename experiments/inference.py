import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from models.cnn_generic import CNNGeneric, BASELINE_CONFIG
from data_pros.data_preprocessing import split_board_with_context
from data_pros.chess_dataset import LABEL_TO_INDEX


INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}


def run_inference(
    image_path,
    model_checkpoint_path,
    image_size=96
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNGeneric(**BASELINE_CONFIG)
    model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    patches = split_board_with_context(image_path)
    predictions = []

    with torch.no_grad():
        for patch in patches:
            patch = cv2.resize(patch, (image_size, image_size))
            tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).to(device)

            logits = model(tensor)
            pred_idx = logits.argmax(dim=1).item()
            predictions.append(INDEX_TO_LABEL[pred_idx])

    return predictions


def print_board(preds):
    print("\nPredicted board:")
    for i in range(8):
        row = preds[i * 8:(i + 1) * 8]
        print(" ".join(f"{p:5s}" for p in row))


if __name__ == "__main__":
    image_path = "./data_base/game4_per_frame/tagged_images/frame_000924.jpg"
    checkpoint_path = "./checkpoints/cnn_baseline.pth"

    preds = run_inference(
        image_path=image_path,
        model_checkpoint_path=checkpoint_path
    )

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Input chessboard image")
    plt.show()

    print_board(preds)
