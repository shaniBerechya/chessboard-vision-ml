"""
Chessboard Dataset Preparation
Stages 1–2: Data loading and preprocessing

This file contains utility functions to:
- Load chess game data from ZIP files
- Parse FEN strings into per-square labels
- Extract context-aware image patches for each board square
- Build training samples of (image patch, label)
"""

import os
import zipfile
import cv2
import pandas as pd

# -------------------------------------------------
# Configuration
# -------------------------------------------------

BOARD_SIZE = 8
CONTEXT_RATIO = 0.25  # Extra context around each square (relative to square size)

FEN_TO_LABEL = {
    'p': 'bp', 'r': 'br', 'n': 'bn', 'b': 'bb', 'q': 'bq', 'k': 'bk',
    'P': 'wp', 'R': 'wr', 'N': 'wn', 'B': 'wb', 'Q': 'wq', 'K': 'wk'
}

# -------------------------------------------------
# FEN parsing
# -------------------------------------------------

def fen_to_board(fen):
    """
    Convert a FEN board string into an 8x8 matrix of square labels.
    """
    board = []
    for row in fen.split('/'):
        board_row = []
        for char in row:
            if char.isdigit():
                board_row.extend(['empty'] * int(char))
            else:
                board_row.append(FEN_TO_LABEL[char])
        board.append(board_row)
    return board

# -------------------------------------------------
# Image patch extraction
# -------------------------------------------------

def extract_square_with_context(img, row, col, square_h, square_w, context_ratio):
    """
    Extract a context-aware image patch centered on a given board square.
    """
    ctx_h = int(square_h * context_ratio)
    ctx_w = int(square_w * context_ratio)

    y1 = max(0, row * square_h - ctx_h)
    y2 = min(img.shape[0], (row + 1) * square_h + ctx_h)

    x1 = max(0, col * square_w - ctx_w)
    x2 = min(img.shape[1], (col + 1) * square_w + ctx_w)

    return img[y1:y2, x1:x2]

def split_board_with_context(image_path, context_ratio=CONTEXT_RATIO):
    """
    Split a full chessboard image into 64 context-aware patches.
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, _ = img.shape
    square_h = h // BOARD_SIZE
    square_w = w // BOARD_SIZE

    patches = []

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            patch = extract_square_with_context(
                img,
                row,
                col,
                square_h,
                square_w,
                context_ratio
            )
            patches.append(patch)

    return patches

# -------------------------------------------------
# Sample construction
# -------------------------------------------------

def create_samples_from_frame(image_path, fen, context_ratio=CONTEXT_RATIO):
    """
    Create per-square training samples from a single labeled frame.
    Each sample contains a context-aware image patch and its label.
    """
    patches = split_board_with_context(image_path, context_ratio)
    labels_2d = fen_to_board(fen)

    samples = []
    idx = 0

    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            samples.append({
                "image": patches[idx],
                "label": labels_2d[i][j],
                "row": i,
                "col": j
            })
            idx += 1

    return samples

# -------------------------------------------------
# ZIP and metadata loading
# -------------------------------------------------

def extract_game_zip(zip_path, extract_to):
    """
    Extract a game ZIP file containing images and labels.
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def load_game_metadata(game_dir):
    """
    Load the game.csv file containing frame-to-FEN mappings.
    """
    csv_path = os.path.join(game_dir, "game.csv")
    return pd.read_csv(csv_path)

# -------------------------------------------------
# End-to-end dataset construction (single game)
# -------------------------------------------------

def build_dataset_from_game(game_dir, context_ratio=CONTEXT_RATIO):
    """
    Build a dataset from a single extracted game directory.
    Returns a list of per-square training samples.
    """
    df = load_game_metadata(game_dir)
    images_dir = os.path.join(game_dir, "images")

    dataset = []

    for _, row in df.iterrows():
        frame_id = int(row["from_frame"])
        fen = row["fen"]

        image_path = os.path.join(
            images_dir,
            f"frame_{frame_id:06d}.jpg"
        )

        samples = create_samples_from_frame(
            image_path,
            fen,
            context_ratio
        )

        dataset.extend(samples)

    return dataset
