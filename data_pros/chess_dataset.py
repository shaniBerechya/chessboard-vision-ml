import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

# Label mapping (shared, deterministic)
LABEL_TO_INDEX = {
    "empty": 0,
    "wp": 1, "wn": 2, "wb": 3, "wr": 4, "wq": 5, "wk": 6,
    "bp": 7, "bn": 8, "bb": 9, "br": 10, "bq": 11, "bk": 12
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
