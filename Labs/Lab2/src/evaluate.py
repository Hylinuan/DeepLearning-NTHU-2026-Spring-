import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from oxford_pet import make_datasets
from train import build_model
from utils import build_loss, dice_score_from_logits, load_checkpoint


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    n = 0
    for images, masks in tqdm(loader, desc="Evaluate"):
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        loss = criterion(logits, masks)
        batch = images.size(0)
        total_loss += loss.item() * batch
        total_dice += dice_score_from_logits(logits, masks) * batch
        n += batch
    print({"loss": total_loss / max(1, n), "dice": total_dice / max(1, n)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="dataset/oxford-iiit-pet")
    parser.add_argument("--model", type=str, choices=["unet", "resnet34_unet"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = make_datasets(args.dataset_root, image_size=args.image_size)
    loader = DataLoader(bundle.valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.model).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    criterion = build_loss("ce_dice")
    evaluate(model, loader, criterion, device)
