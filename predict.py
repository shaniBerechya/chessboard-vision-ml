import torch
import numpy as np
import cv2
from torchvision import transforms

from models.resnet_classifier import ResNetClassifier


# ==================================================
# Constants
# ==================================================
IMAGE_SIZE = 96
OOD_THRESHOLD = 0.077
_DEVICE = torch.device("cpu")

_MODEL = None


# ==================================================
# Preprocessing
# ==================================================
_preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


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
        ckpt = torch.load("model.pth", map_location="cpu")
        model.load_state_dict(ckpt)
        model.eval()
        _MODEL = model
    return _MODEL


# ==================================================
# Board splitting (array-based, safe for evaluation)
# ==================================================
def split_board_from_array(image_rgb: np.ndarray):
    h, w, _ = image_rgb.shape
    cell_h, cell_w = h // 8, w // 8

    patches = []
    for r in range(8):
        for c in range(8):
            y0, y1 = r * cell_h, (r + 1) * cell_h
            x0, x1 = c * cell_w, (c + 1) * cell_w
            patch = image_rgb[y0:y1, x0:x1]
            patches.append(patch)

    return patches


# ==================================================
# REQUIRED EVALUATION FUNCTION
# ==================================================
def predict_board(image: np.ndarray) -> torch.Tensor:
    """
    Predict the chessboard state from a single RGB image.
    """

    assert isinstance(image, np.ndarray)
    assert image.ndim == 3 and image.shape[2] == 3
    assert image.dtype == np.uint8

    model = _load_model()

    patches = split_board_from_array(image)
    preds = []
    confs = []

    with torch.no_grad():
        for patch in patches:
            x = _preprocess(patch).unsqueeze(0)  # (1,3,H,W)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)

            preds.append(int(pred.item()))
            confs.append(float(conf.item()))

    preds = np.array(preds).reshape(8, 8)
    confs = np.array(confs).reshape(8, 8)

    board = np.where(confs < OOD_THRESHOLD, 13, preds)

    return torch.tensor(board, dtype=torch.int64, device="cpu")


# ==================================================
# OPTIONAL DEBUG VISUALIZATION
# ==================================================
def save_debug_image(
    image_rgb: np.ndarray,
    board: torch.Tensor,
    out_path: str = "debug_result.png"
):
    img = image_rgb.copy()
    h, w = img.shape[:2]
    cell_h, cell_w = h // 8, w // 8

    for r in range(8):
        for c in range(8):
            y0, y1 = r * cell_h, (r + 1) * cell_h
            x0, x1 = c * cell_w, (c + 1) * cell_w

            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 1)

            if board[r, c].item() == 13:
                cv2.line(img, (x0, y0), (x1, y1), (255, 0, 0), 2)
                cv2.line(img, (x1, y0), (x0, y1), (255, 0, 0), 2)

    cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# ==================================================
# Local test only
# ==================================================
if __name__ == "__main__":
    img_bgr = cv2.imread("example.jpg")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    board = predict_board(img_rgb)
    print(board)

    save_debug_image(img_rgb, board, "debug_result.png")
