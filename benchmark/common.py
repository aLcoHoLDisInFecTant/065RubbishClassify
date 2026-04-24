"""Shared utilities for the benchmark scripts.

Keeps preprocessing identical to ``backend.py`` (Resize(256) ->
CenterCrop(224) -> ToTensor -> Normalize(ImageNet stats)) so the
offline numbers match the online serving path.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "best_model.pth"
DEFAULT_CLASS_NAMES = PROJECT_ROOT / "class_names.json"
BENCH_DIR = Path(__file__).resolve().parent
REPORT_DIR = BENCH_DIR / "report"
SPLITS_DIR = BENCH_DIR / "splits"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_eval_transform() -> transforms.Compose:
    """Preprocessing used at serving time (see backend.py)."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_tensor_normalize() -> transforms.Normalize:
    """Normalize-only transform for tensors already in [0,1] and 224x224."""
    return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)


def load_class_names(path: Path = DEFAULT_CLASS_NAMES) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pick_device(prefer: Optional[str] = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_model(num_classes: int) -> nn.Module:
    # weights=None avoids the deprecated `pretrained=False` warning and
    # skips downloading ImageNet weights (we load our own state dict).
    try:
        model = models.resnet50(weights=None)
    except TypeError:
        # Older torchvision
        model = models.resnet50(pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_model(
    model_path: Path = DEFAULT_MODEL_PATH,
    class_names: Optional[Sequence[str]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, List[str], torch.device]:
    names = list(class_names) if class_names is not None else load_class_names()
    dev = device if device is not None else pick_device()
    model = build_model(num_classes=len(names))
    state = torch.load(str(model_path), map_location=dev)
    model.load_state_dict(state)
    model.to(dev)
    model.eval()
    return model, names, dev


@dataclass
class ManifestRow:
    path: Path
    label_name: str
    label_idx: int


def read_manifest(csv_path: Path) -> List[ManifestRow]:
    rows: List[ManifestRow] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(ManifestRow(
                path=Path(r["path"]),
                label_name=r["label_name"],
                label_idx=int(r["label_idx"]),
            ))
    if not rows:
        raise RuntimeError(f"Manifest is empty: {csv_path}")
    return rows


class ManifestDataset(Dataset):
    """Image dataset backed by a manifest CSV.

    Parameters
    ----------
    rows:
        Parsed manifest rows.
    transform:
        Callable applied to the PIL image. Defaults to the serving
        transform (Resize/CenterCrop/Normalize).
    return_raw:
        If True, additionally returns the un-normalised 224x224 float
        tensor in [0,1] (useful for robustness perturbations).
    """

    def __init__(
        self,
        rows: Sequence[ManifestRow],
        transform=None,
        return_raw: bool = False,
    ) -> None:
        self.rows = list(rows)
        self.transform = transform or get_eval_transform()
        self.return_raw = return_raw
        if return_raw:
            self._raw_tf = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        img = Image.open(row.path).convert("RGB")
        tensor = self.transform(img)
        if self.return_raw:
            raw = self._raw_tf(img)
            return tensor, raw, row.label_idx, str(row.path)
        return tensor, row.label_idx, str(row.path)


def ensure_report_dir() -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    return REPORT_DIR
