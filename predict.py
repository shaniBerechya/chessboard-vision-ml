import argparse

import torch
import numpy as np
import cv2
import os


from models.resnet_classifier import ResNetClassifier
from data_pros.data_preprocessing import extract_square_with_context, CONTEXT_RATIO
from data_pros.data_preprocessing import split_board_with_context

from data_pros.chess_dataset import LABEL_TO_INDEX



# ==================================================
# Constants
# ==================================================
IMAGE_SIZE = 96
BOARD_SIZE = 8
_DEVICE = torch.device("cpu")
_MODEL = None


# ==================================================
# Utils
# ==================================================
def output_to_logist(outputs):
    if isinstance(outputs, (tuple, list)):
        return outputs[1]
    elif isinstance(outputs, dict):
        return outputs["logits"]
    return outputs

INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}



# ==================================================
# Model loading (once)
# ==================================================
def _load_model():
    global _MODEL
    if _MODEL is None:
        model = ResNetClassifier(
            num_classes=13,
            pretrained=False,
            freeze_backbone=False
        )
        ckpt = torch.load(
            _MODEL_PATH,
            map_location="cpu"
        )
        model.load_state_dict(ckpt)
        model.eval()
        _MODEL = model
    return _MODEL


# ==================================================
# predict_board API
# ==================================================
def predict_board(image: np.ndarray) -> torch.Tensor:
    """
    Predict the chessboard state from a single RGB image.
    """

    # --- required input validation ---
    assert isinstance(image, np.ndarray)
    assert image.ndim == 3 and image.shape[2] == 3
    assert image.dtype == np.uint8

    model = _load_model()

    h, w, _ = image.shape
    square_h = h // BOARD_SIZE
    square_w = w // BOARD_SIZE

    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int64)

    with torch.no_grad():
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                # patch extraction
                patch = extract_square_with_context(
                    image,
                    r,
                    c,
                    square_h,
                    square_w,
                    CONTEXT_RATIO
                )

                # preprocessing
                patch = cv2.resize(patch, (IMAGE_SIZE, IMAGE_SIZE))
                tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
                tensor = tensor.unsqueeze(0)

                outputs = model(tensor)
                logits = output_to_logist(outputs)

                probs = torch.softmax(logits, dim=1)
                conf, pred_idx = probs.max(dim=1)

                pred_idx = int(pred_idx.item())
                conf = float(conf.item())

                # OOD logic
                if pred_idx == LABEL_TO_INDEX["empty"]:
                    ood_threshold = 0.75
                else:
                    ood_threshold = 0.5

                if conf < ood_threshold:
                    board[r, c] = 13  # OOD
                else:
                    board[r, c] = pred_idx

    return torch.tensor(board, dtype=torch.int64, device="cpu")


# ==================================================
# DEBUG VISUALIZATION (local use only)
# ==================================================
def save_debug_image(
    image_rgb: np.ndarray,
    board: torch.Tensor,
    out_path: str = "debug_result.png"
):
    CLASS_NAMES = {
        0: "WP", 1: "WR", 2: "WN", 3: "WB", 4: "WQ", 5: "WK",
        6: "BP", 7: "BR", 8: "BN", 9: "BB", 10: "BQ", 11: "BK",
        12: "Empty",
        13: "OOD"
    }

    img = image_rgb.copy()
    h, w = img.shape[:2]
    cell_h, cell_w = h // BOARD_SIZE, w // BOARD_SIZE

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            y0, y1 = r * cell_h, (r + 1) * cell_h
            x0, x1 = c * cell_w, (c + 1) * cell_w

            # draw grid
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 1)

            # draw label
            label = CLASS_NAMES[int(board[r, c].item())]
            cv2.putText(
                img,
                label,
                (x0 + 5, y0 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

    # always overwrite
    cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# ==================================================
# Local test only (safe to keep for development)
# ==================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run chessboard prediction on a single image.")
    parser.add_argument(
        "--image",
        "-i",
        required=True,
        help="Path to an input image file (jpg/png).",
    )
    parser.add_argument(
    "--model_path","-m",
    default="trained_model/model.pth",
    help="Path to trained model (.pth). Default: trained_model/model.pth"
    )

    args = parser.parse_args()
    global _MODEL_PATH
    _MODEL_PATH = args.model_path

    if not os.path.exists(_MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {_MODEL_PATH}")


    # Load image (BGR) -> RGB
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise ValueError(f"cv2.imread failed to load image: {args.image}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    board = predict_board(img_rgb)
    print(board)

    save_debug_image(img_rgb, board, "debug_result.png")
