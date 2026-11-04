import csv
import os
import random
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)[:, 1]
        targets = targets.float()
        dims = (1, 2)
        intersection = (probs * targets).sum(dim=dims)
        union = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


def dice_score_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> float:
    preds = torch.argmax(logits, dim=1)
    return dice_score_from_masks(preds, targets, eps)


def dice_score_from_masks(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> float:
    preds = preds.float()
    targets = targets.float()
    dims = (1, 2)
    intersection = (preds * targets).sum(dim=dims)
    union = preds.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean().item()


def build_loss(loss_name: str = "ce_dice"):
    ce = nn.CrossEntropyLoss()
    dice = DiceLoss()

    def criterion(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if loss_name == "ce":
            return ce(logits, targets)
        if loss_name == "dice":
            return dice(logits, targets)
        return ce(logits, targets) + dice(logits, targets)

    return criterion


def save_checkpoint(path: str, state: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model: nn.Module, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def write_metrics_csv(path: str, rows: Iterable[Dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
