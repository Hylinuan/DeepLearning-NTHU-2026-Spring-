import argparse
import os
from typing import Dict, List

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from oxford_pet import make_datasets
from models.unet import UNet
from models.resnet34_unet import ResNet34UNet
from utils import build_loss, count_parameters, dice_score_from_logits, save_checkpoint, seed_everything, write_metrics_csv


def build_model(model_name: str, num_classes: int = 2):
    if model_name == "unet":
        return UNet(in_channels=3, num_classes=num_classes)
    if model_name == "resnet34_unet":
        return ResNet34UNet(in_channels=3, num_classes=num_classes)
    raise ValueError(f"Unknown model: {model_name}")


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    n = 0
    for images, masks in tqdm(loader, desc="Valid", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        loss = criterion(logits, masks)
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_dice += dice_score_from_logits(logits, masks) * batch_size
        n += batch_size
    return total_loss / max(1, n), total_dice / max(1, n)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    n = 0

    for images, masks in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_dice += dice_score_from_logits(logits.detach(), masks) * batch_size
        n += batch_size

    return total_loss / max(1, n), total_dice / max(1, n)


def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle = make_datasets(args.dataset_root, image_size=args.image_size)
    train_loader = DataLoader(bundle.train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(bundle.valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.model).to(device)
    criterion = build_loss(args.loss)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    print(f"Model: {args.model}")
    print(f"Trainable params: {count_parameters(model):,}")

    best_dice = -1.0
    history: List[Dict] = []
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = train_one_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_dice = validate(model, valid_loader, criterion, device)
        scheduler.step(valid_dice)

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_dice": round(train_dice, 6),
            "valid_loss": round(valid_loss, 6),
            "valid_dice": round(valid_dice, 6),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(row)

        ckpt_path = os.path.join(args.save_dir, f"{args.model}_last.pth")
        save_checkpoint(
            ckpt_path,
            {
                "epoch": epoch,
                "model_name": args.model,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_valid_dice": best_dice,
                "args": vars(args),
            },
        )

        if valid_dice > best_dice:
            best_dice = valid_dice
            patience_counter = 0
            save_checkpoint(
                os.path.join(args.save_dir, f"{args.model}_best.pth"),
                {
                    "epoch": epoch,
                    "model_name": args.model,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_valid_dice": best_dice,
                    "args": vars(args),
                },
            )
            print(f"Saved best checkpoint with valid dice = {best_dice:.6f}")
        else:
            patience_counter += 1

        write_metrics_csv(os.path.join(args.save_dir, f"{args.model}_history.csv"), history)

        if patience_counter >= args.early_stop:
            print("Early stopping triggered.")
            break

    print(f"Best valid dice: {best_dice:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="dataset/oxford-iiit-pet")
    parser.add_argument("--model", type=str, choices=["unet", "resnet34_unet"], required=True)
    parser.add_argument("--save_dir", type=str, default="saved_models")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--loss", type=str, default="ce_dice", choices=["ce", "dice", "ce_dice"])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stop", type=int, default=8)
    args = parser.parse_args()
    main(args)
