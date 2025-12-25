import torch
import cv2
import numpy as np

from models.cnn_generic import CNNGeneric, BASELINE_CONFIG
from data_pros.data_preprocessing import split_board_with_context
from data_pros.chess_dataset import LABEL_TO_INDEX

# Reverse mapping: index -> label
INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}


def run_inference(
    image_path,
    model_checkpoint_path,
    image_size=96
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Load model
    # -------------------------
    model = CNNGeneric(**BASELINE_CONFIG)
    model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # -------------------------
    # Split image into 64 patches
    # -------------------------
    patches = split_board_with_context(image_path)

    predictions = []

    with torch.no_grad():
        for patch in patches:
            patch = cv2.resize(patch, (image_size, image_size))
            tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).to(device)

            logits = model(tensor)
            pred_idx = logits.argmax(dim=1).item()
            pred_label = INDEX_TO_LABEL[pred_idx]

            predictions.append(pred_label)

    return predictions


if __name__ == "__main__":
    preds = run_inference(
        image_path="./data_base/game2_per_frame/tagged_images/frame_000200.jpg",
        model_checkpoint_path="./checkpoints/cnn_baseline.pth"
    )

    print("Predicted labels:")
    print(preds)
