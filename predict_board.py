import torch
import numpy as np
from torchvision import transforms
import cv2
import argparse

from models.resnet_classifier import ResNetClassifier

_MODEL = None
_DEVICE = torch.device("cpu")

_preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # change if your model expects a different size
    transforms.ToTensor(),          # converts to [0,1]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def _load_model():
    global _MODEL
    if _MODEL is None:
        model = ResNetClassifier()
        ckpt = torch.load("model.pth", map_location="cpu")
        model.load_state_dict(ckpt)
        model.eval()
        _MODEL = model
    return _MODEL



# --------------------------------------------------
# Evaluation API function
# --------------------------------------------------

def predict_board(image: np.ndarray) -> torch.Tensor:
    """
    Predict the chessboard state from a single RGB image.

    Input:
        image: numpy.ndarray of shape (H, W, 3), dtype uint8, RGB

    Output:
        torch.Tensor of shape (8, 8), dtype torch.int64, on CPU
        Each value is an integer in [0, 12] according to the class encoding.
    """

    # Basic input validation
    assert isinstance(image, np.ndarray), "Input must be a numpy array"
    assert image.ndim == 3 and image.shape[2] == 3, "Input must be HxWx3 RGB image"
    assert image.dtype == np.uint8, "Input image must be uint8"

    model = _load_model()

    # --------------------------------------------------
    # Preprocess image
    # --------------------------------------------------
    x = _preprocess(image)          # (3, H, W)
    x = x.unsqueeze(0)              # (1, 3, H, W)

    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------
    with torch.no_grad():
        outputs = model(x)

    # outputs: (1, 64, C)
    probs = torch.softmax(outputs, dim=-1)
    conf, preds = probs.max(dim=-1)

    print("outputs shape:", outputs.shape)
    print("preds shape:", preds.shape)

    board = preds.view(8, 8)
    conf = conf.view(8, 8)


    # -----------------------------
    # OOD handling
    # -----------------------------
    OOD_THRESHOLD = 0.6  # tune on validation set if possible
    board[conf < OOD_THRESHOLD] = 12

    # --------------------------------------------------
    # Enforce API requirements
    # --------------------------------------------------
    board = board.to(device="cpu", dtype=torch.int64)

    return board


# --------------------------------------------------
# Converts board prediction to a PNG image for debugging
# --------------------------------------------------

def board_to_png(board: torch.Tensor, out_path: str = "board.png"):
    """
    Convert a predicted board tensor to a PNG image using python-chess.
    This function MUST NOT be called inside predict_board.
    """

    import chess
    import chess.svg
    import cairosvg

    board_np = board.cpu().numpy()

    # Integer class to FEN piece mapping
    mapping = {
        0: "P", 1: "R", 2: "N", 3: "B", 4: "Q", 5: "K",
        6: "p", 7: "r", 8: "n", 9: "b", 10: "q", 11: "k"
    }

    fen_rows = []
    for row in board_np:
        fen_row = ""
        empty_count = 0
        for v in row:
            if v == 12:
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += mapping.get(int(v), "1")
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)

    fen = "/".join(fen_rows) + " w - - 0 1"

    svg = chess.svg.board(chess.Board(fen))
    cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=out_path)


def main():
    """
    Main entry point for running predict_board on a single image.
    """

    parser = argparse.ArgumentParser(description="Run chessboard prediction on an image")
    parser.add_argument(
        "--image","-i",
        type=str,
        required=True,
        help="Path to input RGB image"
    )

    args = parser.parse_args()

    # ----------------------------------
    # Load image (OpenCV loads BGR!)
    # ----------------------------------
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    # Convert BGR -> RGB
    image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Ensure correct dtype
    image = image.astype(np.uint8)

    # ----------------------------------
    # Run prediction
    # ----------------------------------
    board = predict_board(image)

    # ----------------------------------
    # Display result
    # ----------------------------------
    print("Predicted board (8x8):")
    print(board)
    print(f"dtype: {board.dtype}, device: {board.device}")


if __name__ == "__main__":
    main()
