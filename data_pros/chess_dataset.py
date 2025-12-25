"""
PyTorch Dataset for Chessboard Square Classification
Each sample is a single square with context.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

# -------------------------------------------------
# Label mapping
# -------------------------------------------------

LABEL_TO_INDEX = {
    "empty": 0,
    "wp": 1, "wn": 2, "wb": 3, "wr": 4, "wq": 5, "wk": 6,
    "bp": 7, "bn": 8, "bb": 9, "br": 10, "bq": 11, "bk": 12
}

INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}
NUM_CLASSES = len(LABEL_TO_INDEX)

# -------------------------------------------------
# Dataset
# -------------------------------------------------

class ChessSquareDataset(Dataset):
    """
    Dataset of context-aware chessboard square patches.
    """

    def __init__(self, samples, target_size=96):
        """
        Args:
            samples (list): list of dicts with keys:
                - image (H x W x 3 numpy array)
                - label (string)
            target_size (int): final square image size (target_size x target_size)
        """
        self.samples = samples
        self.target_size = target_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = sample["image"]
        label_str = sample["label"]

        # --- resize to fixed size ---
        image = cv2.resize(
            image,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_LINEAR
        )

        # --- normalize ---
        image = image.astype(np.float32) / 255.0

        # HWC -> CHW
        image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.tensor(image, dtype=torch.float32)

        # --- label encoding ---
        label_idx = LABEL_TO_INDEX[label_str]
        label_tensor = torch.tensor(label_idx, dtype=torch.long)

        return image_tensor, label_tensor
