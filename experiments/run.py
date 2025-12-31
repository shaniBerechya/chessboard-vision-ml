import argparse
import json
import os
import random
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import torch.nn as nn

from experiments.configs import EXPERIMENTS
from experiments.run_experiment import run_experiment
from data_pros.data_preprocessing import split_board_with_context
from data_pros.chess_dataset import LABEL_TO_INDEX


INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}


def sanitize_training_config(cfg: dict):
    clean_cfg = {}
    for k, v in cfg.items():
        if isinstance(v, nn.Module):
            clean_cfg[k] = v.__class__.__name__
        else:
            clean_cfg[k] = v
    return clean_cfg

def sanitize_model_config(cfg: dict):
    clean_cfg = {}
    for k, v in cfg.items():
        if isinstance(v, nn.Module):
            clean_cfg[k] = v.__class__.__name__
        else:
            clean_cfg[k] = v
    return clean_cfg


def plot_error_curves(history: dict, save_path: Path | None = None, show: bool = False):
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])

    if not train_loss and not val_loss:
        print("⚠️ No loss data found in history. Skipping plot.")
        return

    plt.figure(figsize=(8, 5))
    if train_loss:
        plt.plot(range(1, len(train_loss) + 1), train_loss, label="train_loss")
    if val_loss:
        plt.plot(range(1, len(val_loss) + 1), val_loss, label="val_loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Error Curve (Loss)")
    plt.grid(True)
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=160, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()


def find_frame_images(game_dir: str | Path) -> list[Path]:
    """
    Finds original frame images under game_dir.
    Checks common folders first, then falls back to recursive search.
    """
    game_dir = Path(game_dir)
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    candidates = [
        game_dir / "tagged_images",
        game_dir / "frames",
        game_dir / "images",
        game_dir / "raw_frames",
    ]
    for d in candidates:
        if d.exists() and d.is_dir():
            imgs = sorted([p for p in d.iterdir() if p.suffix.lower() in exts])
            if imgs:
                return imgs

    return sorted([p for p in game_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def load_trained_model(model_cls, model_config: dict, checkpoint_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_cls(**model_config).to(device)

    state = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.eval()
    return model, device


def predict_board_labels(model, device, image_path: Path, image_size: int) -> list[str]:
    patches = split_board_with_context(str(image_path))
    preds = []

    with torch.no_grad():
        for patch in patches:
            patch = cv2.resize(patch, (image_size, image_size))
            tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).to(device)
            logits = model(tensor)
            pred_idx = int(logits.argmax(dim=1).item())
            preds.append(INDEX_TO_LABEL.get(pred_idx, str(pred_idx)))

    return preds


def annotate_full_frame_with_grid_and_labels(image_bgr: np.ndarray, preds: list[str]) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    cell_w = w / 8.0
    cell_h = h / 8.0

    out = image_bgr.copy()

    for i in range(9):
        x = int(round(i * cell_w))
        y = int(round(i * cell_h))
        cv2.line(out, (x, 0), (x, h), (0, 255, 0), 1)
        cv2.line(out, (0, y), (w, y), (0, 255, 0), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for r in range(8):
        for c in range(8):
            idx = r * 8 + c
            if idx >= len(preds):
                continue
            label = preds[idx]
            x0 = int(c * cell_w)
            y0 = int(r * cell_h)
            org = (x0 + 4, y0 + 14)

            cv2.putText(out, label, org, font, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(out, label, org, font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return out


def save_qualitative_full_frames(
    game_dir: str | Path,
    out_dir: Path,
    model_cls,
    model_config: dict,
    checkpoint_path: Path,
    image_size: int,
    n_frames: int = 10,
    seed: int = 42,
):
    random.seed(seed)

    images = find_frame_images(game_dir)
    if not images:
        print(f"⚠️ No frame images found under: {game_dir}. Skipping qualitative frames.")
        return

    n_frames = min(n_frames, len(images))
    chosen = random.sample(images, k=n_frames)

    out_dir.mkdir(parents=True, exist_ok=True)
    orig_dir = out_dir / "original"
    pred_dir = out_dir / "predicted"
    orig_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    model, device = load_trained_model(model_cls, model_config, checkpoint_path)

    all_preds = {}
    for p in chosen:
        frame_name = p.name
        preds = predict_board_labels(model, device, p, image_size=image_size)
        all_preds[frame_name] = preds

        shutil.copy2(p, orig_dir / frame_name)

        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            print(f"⚠️ Failed reading image: {p}")
            continue

        annotated = annotate_full_frame_with_grid_and_labels(img_bgr, preds)
        cv2.imwrite(str(pred_dir / frame_name), annotated)

    with open(out_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(all_preds, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(all_preds)} qualitative full-frame examples to: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", "-e", default="cnn_baseline",
                        help=f"Experiment key from EXPERIMENTS. Options: {list(EXPERIMENTS.keys())}")
    parser.add_argument( "--game_dirs","-g",nargs="+",
        default=[
            "./data_base/game2_per_frame",
            "./data_base/game4_per_frame",
            "./data_base/game5_per_frame",
            "./data_base/game6_per_frame",
            "./data_base/game7_per_frame",
        ],
        help="List of game directories under data_base (default: games 2,4,5,6,7)"
    )
    parser.add_argument("--output_root", "-o", default="results",
                        help="Root folder to store run artifacts.")
    parser.add_argument("--num_frames", type=int, default=10,
                        help="How many random full frames to export for qualitative review.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for random frame sampling.")
    parser.add_argument("--show_plot", action="store_true",
                        help="Show the loss plot window at the end (plt.show).")

    args = parser.parse_args()

    if args.experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment '{args.experiment}'. Available: {list(EXPERIMENTS.keys())}")

    exp_name = args.experiment
    cfg = EXPERIMENTS[exp_name]

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{exp_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = run_dir / "model.pth"

    history = run_experiment(
        experiment_name=exp_name,
        model_cls=cfg["model_cls"],
        model_config=cfg["model_config"],
        training_config=cfg["training_config"],
        game_dirs=args.game_dirs,
        output_dir=run_dir,
    )

    metadata = {
        "experiment_name": exp_name,
        "timestamp": timestamp,
        "model_class": cfg["model_cls"].__name__,
        "model_config": sanitize_model_config(cfg["model_config"]),
        "training_config": sanitize_training_config(cfg["training_config"]),
        "game_dirs": args.game_dirs,
        "checkpoint_path": str(checkpoint_path),
        "seed": args.seed,
    }
    
    metadata["splits"] = {
        "train_split": cfg["training_config"].get("train_split", 0.6),
        "val_split": cfg["training_config"].get("val_split", 0.2),
        "test_split": cfg["training_config"].get("test_split", 0.2),
        "split_seed": cfg["training_config"].get("split_seed", 42),
    }

    metadata["test_results"] = {
        "accuracy_all": history.get("test_accuracy_all", [None])[-1],
        "accuracy_no_empty": history.get("test_accuracy_no_empty", [None])[-1],
        "accuracy_only_pieces": history.get("test_accuracy_only_pieces", [None])[-1],
        "artifacts": history.get("artifacts", {}),
    }


    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(run_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plot_error_curves(history, save_path=run_dir / "loss_curve.png", show=args.show_plot)

    save_qualitative_full_frames(
        game_dir=args.game_dirs[0],
        out_dir=run_dir / "qualitative_full_frames",
        model_cls=cfg["model_cls"],
        model_config=cfg["model_config"],
        checkpoint_path=checkpoint_path,
        image_size=cfg["training_config"]["image_size"],
        n_frames=args.num_frames,
        seed=args.seed,
    )

    print("✅ Experiment finished successfully")
    print(f"📁 Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
