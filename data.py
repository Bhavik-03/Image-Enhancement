"""Paired underwater dataset: raw image -> reference image.

Expected layout (symlink or copy EUVP / UIEB into it):
    <root>/raw/<name>.png    degraded underwater photo
    <root>/ref/<name>.png    reference (same stem, same size)
"""
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def list_pairs(root):
    """Return {stem: (raw_path, ref_path)}; fail early if the folders are not paired."""
    folders = []
    for name in ("raw", "ref"):
        folder = Path(root) / name
        if not folder.is_dir():
            raise FileNotFoundError(f"missing folder: {folder}")
        folders.append({p.stem: p for p in sorted(folder.iterdir()) if p.suffix.lower() in EXTS})
    raw, ref = folders
    if raw.keys() != ref.keys():
        odd = sorted(raw.keys() ^ ref.keys())[:5]
        raise ValueError(f"{root}: raw/ and ref/ are not paired, e.g. {odd}")
    if not raw:
        raise ValueError(f"{root}: no images found")
    return {stem: (raw[stem], ref[stem]) for stem in sorted(raw)}


def get_split(root, seed=0, val=0.1, test=0.1):
    """Seeded train/val/test split of the stems, saved to <root>/split.json and reused after."""
    path = Path(root) / "split.json"
    if path.exists():
        return json.loads(path.read_text())
    stems = list(list_pairs(root))
    random.Random(seed).shuffle(stems)
    n_val, n_test = round(len(stems) * val), round(len(stems) * test)
    split = {
        "val": sorted(stems[:n_val]),
        "test": sorted(stems[n_val : n_val + n_test]),
        "train": sorted(stems[n_val + n_test :]),
    }
    path.write_text(json.dumps(split, indent=1))
    return split


def read_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_tensor(img):
    """HxWx3 uint8 -> float tensor [3,H,W] in [0,1]."""
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255


class PairedDataset(Dataset):
    """Returns (raw, ref). train: random crop + flips/rot90; val/test: full images (use batch size 1)."""

    def __init__(self, root, split="train", crop=256):
        pairs = list_pairs(root)
        self.items = [pairs[stem] for stem in get_split(root)[split]]
        self.train, self.crop = split == "train", crop

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        raw, ref = (read_rgb(p) for p in self.items[i])
        if raw.shape != ref.shape:
            raise ValueError(f"size mismatch {raw.shape} vs {ref.shape}: {self.items[i][0]}")
        if self.train:
            raw, ref = self.augment(raw, ref)
        return to_tensor(raw), to_tensor(ref)

    def augment(self, raw, ref):
        h, w = raw.shape[:2]
        if min(h, w) < self.crop:  # too small to crop: scale up so the short side fits
            s = self.crop / min(h, w)
            size = (max(self.crop, round(w * s)), max(self.crop, round(h * s)))
            raw, ref = cv2.resize(raw, size), cv2.resize(ref, size)
            h, w = raw.shape[:2]
        y, x = random.randint(0, h - self.crop), random.randint(0, w - self.crop)
        raw, ref = (a[y : y + self.crop, x : x + self.crop] for a in (raw, ref))
        if random.random() < 0.5:
            raw, ref = raw[:, ::-1], ref[:, ::-1]
        if random.random() < 0.5:
            raw, ref = raw[::-1], ref[::-1]
        k = random.randint(0, 3)
        return np.rot90(raw, k), np.rot90(ref, k)
