import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from oxford_pet import make_datasets
from train import build_model
from utils import load_checkpoint


def save_mask_png(mask: np.ndarray, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    png = (mask.astype(np.uint8) * 255)
    cv2.imwrite(path, png)


@torch.no_grad()
def run_inference(model, loader, device, output_dir: str):
    model.eval()
    pred_dir = os.path.join(output_dir, "pred_masks")
    os.makedirs(pred_dir, exist_ok=True)
    submission_rows = []

    for images, _, names in tqdm(loader, desc="Inference"):
        images = images.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)

        for pred, name in zip(preds, names):
            mask_path = os.path.join(pred_dir, f"{name}.png")
            save_mask_png(pred, mask_path)
            submission_rows.append({"image_id": name, "mask_path": f"pred_masks/{name}.png"})

    csv_path = os.path.join(output_dir, "submission.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "mask_path"])
        writer.writeheader()
        writer.writerows(submission_rows)
    print(f"Saved predictions to {pred_dir}")
    print(f"Saved CSV to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="dataset/oxford-iiit-pet")
    parser.add_argument("--model", type=str, choices=["unet", "resnet34_unet"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = make_datasets(args.dataset_root, image_size=args.image_size)
    loader = DataLoader(bundle.test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.model).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    run_inference(model, loader, device, args.output_dir)
