import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def read_split_file(split_file: str) -> List[str]:
    """Read Oxford-IIIT Pet split file.

    Each valid line is formatted like:
        Abyssinian_100 1 1 0
    We only need the image id in the first column.
    """
    names: List[str] = []
    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 0:
                continue
            names.append(parts[0])
    return names


def build_binary_mask(trimap: np.ndarray) -> np.ndarray:
    """Convert trimap to binary mask required by the assignment.

    Assignment rule:
      1 -> foreground (1)
      2 or 3 -> background (0)
    """
    mask = (trimap == 1).astype(np.uint8)
    return mask


class SegmentationTransform:
    def __init__(
        self,
        image_size: int = 256,
        train: bool = False,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.image_size = image_size
        self.train = train
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def _resize_pair(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        return image, mask

    def _random_flip(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
        return image, mask

    def _random_rotate_scale(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        angle = np.random.uniform(-15.0, 15.0)
        scale = np.random.uniform(0.9, 1.1)
        center = (w / 2.0, h / 2.0)
        mat = cv2.getRotationMatrix2D(center, angle, scale)
        image = cv2.warpAffine(
            image,
            mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        mask = cv2.warpAffine(
            mask,
            mat,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return image, mask

    def _color_jitter(self, image: np.ndarray) -> np.ndarray:
        alpha = np.random.uniform(0.9, 1.1)
        beta = np.random.uniform(-10, 10)
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] *= np.random.uniform(0.9, 1.1)
        hsv[..., 2] *= np.random.uniform(0.9, 1.1)
        hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return image

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = self._resize_pair(image, mask)
        if self.train:
            image, mask = self._random_flip(image, mask)
            image, mask = self._random_rotate_scale(image, mask)
            if np.random.rand() < 0.8:
                image = self._color_jitter(image)

        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = np.transpose(image, (2, 0, 1))

        image_tensor = torch.from_numpy(image).float()
        mask_tensor = torch.from_numpy(mask).long()
        return image_tensor, mask_tensor


class OxfordPetDataset(Dataset):
    def __init__(
        self,
        root: str,
        split_file: str,
        transform: Optional[Callable] = None,
        return_name: bool = False,
    ) -> None:
        self.root = root
        self.images_dir = os.path.join(root, "images")
        self.masks_dir = os.path.join(root, "annotations", "trimaps")
        self.names = read_split_file(split_file)
        self.transform = transform
        self.return_name = return_name

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        name = self.names[idx]
        image_path = os.path.join(self.images_dir, f"{name}.jpg")
        mask_path = os.path.join(self.masks_dir, f"{name}.png")

        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        trimap = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if trimap is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        mask = build_binary_mask(trimap)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        if self.return_name:
            return image, mask, name
        return image, mask


@dataclass
class DatasetBundle:
    train_dataset: OxfordPetDataset
    valid_dataset: OxfordPetDataset
    test_dataset: OxfordPetDataset


def make_datasets(dataset_root: str, image_size: int = 256) -> DatasetBundle:
    train_transform = SegmentationTransform(image_size=image_size, train=True)
    eval_transform = SegmentationTransform(image_size=image_size, train=False)

    train_split = os.path.join(dataset_root, "annotations", "trainval.txt")
    valid_split = os.path.join(dataset_root, "annotations", "list.txt")
    test_split = os.path.join(dataset_root, "annotations", "test.txt")

    train_dataset = OxfordPetDataset(dataset_root, train_split, train_transform)
    valid_dataset = OxfordPetDataset(dataset_root, valid_split, eval_transform)
    test_dataset = OxfordPetDataset(dataset_root, test_split, eval_transform, return_name=True)

    return DatasetBundle(train_dataset, valid_dataset, test_dataset)
