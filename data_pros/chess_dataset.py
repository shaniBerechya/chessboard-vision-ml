import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

LABEL_TO_INDEX = {
    "wp": 0,   # White Pawn
    "wr": 1,   # White Rook
    "wn": 2,   # White Knight
    "wb": 3,   # White Bishop
    "wq": 4,   # White Queen
    "wk": 5,   # White King

    "bp": 6,   # Black Pawn
    "br": 7,   # Black Rook
    "bn": 8,   # Black Knight
    "bb": 9,   # Black Bishop
    "bq": 10,  # Black Queen
    "bk": 11,  # Black King

    "empty": 12,  # Empty / OOD / Unknown
    "unknown": 13
}


class ChessSquareDataset(Dataset):
    """
    Dataset of individual chessboard squares with context.
    Each sample = (image_tensor, label_index)
    """

    def __init__(self, samples, image_size=96):
        """
        Args:
            samples: list of dicts with keys ["image", "label"]
            image_size: int, target size for square patches (H = W)
        """
        self.samples = samples
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img = sample["image"]          # numpy array (H, W, 3)
        label_str = sample["label"]    # string label

        # Resize to fixed size (important for batching!)
        img = cv2.resize(img, (self.image_size, self.image_size))

        # Convert to tensor (C, H, W)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        label = LABEL_TO_INDEX[label_str]
        label = torch.tensor(label, dtype=torch.long)

        return img, label
