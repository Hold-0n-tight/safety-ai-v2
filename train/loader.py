"""
loader.py

Utility helpers for building DataLoader objects from pre-computed split JSON.

Supported datasets:
* custom9   — 9-class ImageFolder under `data/train/raw`, sibling `data/test/`
* pathmnist — MedMNIST PathMNIST (downloads to cfg.dataset.root)
* cifar*    — CIFAR-10 fallback

`_infer_img_size` picks the model-input resolution (380 for EfficientNet-B4,
else 224). PathMNIST also has an orthogonal `size` parameter — the resolution
of the *downloaded* arrays (28/64/128/224); the transform then resizes to
`img_size`. Defaults to 28 for fast iteration.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder, CIFAR10

from medmnist import PathMNIST


# ──────────────────────────────────────────────────────────────
# Helper: infer proper image size from model / dataset name
# ──────────────────────────────────────────────────────────────

def _infer_img_size(name: str) -> int:
    """Return input resolution based on model/dataset identifier."""
    name = name.lower()
    if "efficientnet_b4" in name or "efficientnet-b4" in name:
        return 380
    return 224


# ──────────────────────────────────────────────────────────────
# Dataset-specific transform builders
# ──────────────────────────────────────────────────────────────

def _get_transform(train: bool = True, img_size: int = 224):
    """Return torchvision transforms for train / test phases."""
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


def _flatten_label(t):
    """MedMNIST returns labels shaped (1,); CrossEntropyLoss wants a scalar long."""
    return int(t[0])


def _load_dataset(
    name: str,
    root: str | Path,
    *,
    train: bool = True,
    img_size: int = 224,
    size: int = 28,
) -> Dataset:
    """Instantiate a dataset based on name."""
    root = Path(root)
    name_lc = name.lower()

    if name_lc == "custom9":
        if not root.exists():
            raise FileNotFoundError(f"Dataset root path does not exist: {root}")
        return ImageFolder(root=root, transform=_get_transform(train, img_size))

    if name_lc == "pathmnist":
        root.mkdir(parents=True, exist_ok=True)
        split = "train" if train else "test"
        return PathMNIST(
            split=split,
            root=str(root),
            download=True,
            size=size,
            transform=_get_transform(train, img_size),
            target_transform=_flatten_label,
        )

    if name_lc.startswith("cifar"):
        if not root.exists():
            root.mkdir(parents=True, exist_ok=True)
        return CIFAR10(root=root,
                       train=train,
                       download=True,
                       transform=_get_transform(train, img_size))

    raise ValueError(
        f"Unsupported dataset identifier '{name}'. "
        "Supported: ['custom9', 'pathmnist', 'cifar10']"
    )


# ──────────────────────────────────────────────────────────────
# Public: shared test set resolver (used by FL clients & central eval)
# ──────────────────────────────────────────────────────────────

def get_test_dataset(
    dataset_name: str,
    data_root: str | Path,
    img_size: int,
    *,
    size: int = 28,
) -> Optional[Dataset]:
    """Return the held-out test set for the configured dataset.

    For datasets with a built-in test split (PathMNIST, CIFAR-10), uses it
    directly. For ImageFolder-style datasets ("custom9"), resolves a sibling
    `test/` (preferred) or `val/` directory next to `data_root.parent`. If
    neither exists, returns None so the caller can decide on a fallback.
    """
    name = dataset_name.lower()

    if name == "pathmnist":
        return _load_dataset(name, data_root, train=False, img_size=img_size, size=size)

    if name.startswith("cifar"):
        return _load_dataset(name, data_root, train=False, img_size=img_size)

    if name == "custom9":
        data_root = Path(data_root)
        for candidate in (data_root.parent.parent / "test",
                          data_root.parent.parent / "val"):
            if candidate.exists():
                return ImageFolder(
                    root=candidate,
                    transform=_get_transform(False, img_size),
                )
        return None

    raise ValueError(f"Unsupported dataset identifier '{dataset_name}'")


# ──────────────────────────────────────────────────────────────
# Public: DataLoader builder from client split indices
# ──────────────────────────────────────────────────────────────

def get_dataloaders_from_split(
    *,
    client_id: int,
    split_indices: List[int],
    data_root: str | Path,
    batch_size: int,
    dataset_name: str = "custom9",
    size: int = 28,
) -> Tuple[DataLoader, DataLoader]:
    """
    Return train & test DataLoader for a given client.

    Parameters
    ----------
    client_id : int
        Numeric client ID (for logging only).
    split_indices : list[int]
        Index list belonging to this client (train split).
    data_root : str | Path
        Root directory of train images (custom9) or download cache (pathmnist).
    batch_size : int
        Batch size for loaders.
    dataset_name : str, optional
        Dataset identifier; default "custom9".
    size : int, optional
        PathMNIST download resolution (28 / 64 / 128 / 224). Ignored for other
        datasets. Default 28 for fast iteration; override via cfg.dataset.size.
    """
    img_size = _infer_img_size(dataset_name)

    # 1) Train subset specific to this client
    full_train_ds = _load_dataset(
        dataset_name, data_root, train=True, img_size=img_size, size=size,
    )
    train_subset = Subset(full_train_ds, split_indices)
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    # 2) Shared test set; fall back to 10 % of this client's split for sanity
    #    if the dataset has no test split available on disk.
    test_ds = get_test_dataset(dataset_name, data_root, img_size, size=size)
    if test_ds is None:
        val_len = max(1, int(0.1 * len(split_indices)))
        test_ds = Subset(full_train_ds, split_indices[:val_len])

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader
