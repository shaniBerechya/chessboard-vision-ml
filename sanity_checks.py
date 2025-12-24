import pandas as pd
import cv2
import matplotlib.pyplot as plt

from dataset_preparation import (
    fen_to_board,
    split_board_with_context,
    create_samples_from_frame
)

# -----------------------------
# Paths
# -----------------------------
GAME_DIR = "data_base/game2_per_frame"
CSV_PATH = f"{GAME_DIR}/game2.csv"
IMAGES_DIR = f"{GAME_DIR}/tagged_images"

# -----------------------------
# Load CSV
# -----------------------------
df = pd.read_csv(CSV_PATH)
print(df.head())

# -----------------------------
# Test FEN parsing
# -----------------------------
fen = df.iloc[0]["fen"]
board = fen_to_board(fen)

for row in board:
    print(row)

# -----------------------------
# Load image
# -----------------------------
frame_id = int(df.iloc[0]["from_frame"])
image_path = f"{IMAGES_DIR}/frame_{frame_id:06d}.jpg"

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.axis("off")
plt.title("Full board image")
plt.show()

# -----------------------------
# Split board into patches
# -----------------------------
patches = split_board_with_context(image_path)
print("Number of patches:", len(patches))

# -----------------------------
# Visualize patches
# -----------------------------
plt.figure(figsize=(8, 8))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(patches[i])
    plt.axis("off")
plt.show()

# -----------------------------
# Full sample creation
# -----------------------------
samples = create_samples_from_frame(image_path, fen)
sample = samples[0]

plt.imshow(sample["image"])
plt.axis("off")
plt.title(f"Label: {sample['label']}")
plt.show()
