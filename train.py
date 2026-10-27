"""Fine-tune Shallow-UWnet on paired underwater data.

    python train.py --data data/uieb --run ft1
Saves the best model by validation PSNR to checkpoints/<run>_best.pth (a plain state_dict) and the
config and per-epoch log to results/<run>/.
"""

import argparse
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from data import PairedDataset
from evaluate import score
from losses import Loss
from model import AUTHORS_SHA256, UWnet, load_pretrained

PRETRAINED = "data/downloads/shallow_uwnet.ckpt"


def git(*args):
    result = subprocess.run(["git", *args], check=False, capture_output=True, text=True)
    return result.stdout.strip()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA


def fit(model, criterion, train_loader, val_loader, epochs, lr, run, device):
    """Train, validate each epoch, keep the best checkpoint. Returns [(loss, psnr, ssim), ...]."""
    model, criterion = model.to(device), criterion.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    cuda = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=cuda)
    ckpt = Path("checkpoints") / f"{run}_best.pth"
    log = Path("results") / run / "log.txt"
    ckpt.parent.mkdir(exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    history, best = [], -float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for raw, ref in train_loader:
            raw, ref = raw.to(device), ref.to(device)
            with torch.autocast(device, dtype=torch.float16, enabled=cuda):
                loss = criterion(model(raw), ref)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item()
        sched.step()
        psnr, ssim = score(model, val_loader, device)
        history.append((total / len(train_loader), psnr, ssim))
        line = f"epoch {epoch}/{epochs} loss {history[-1][0]:.4f} val_psnr {psnr:.2f} val_ssim {ssim:.4f}"
        if psnr > best:
            best = psnr
            torch.save(model.state_dict(), ckpt)
            line += "  (best)"
        print(line, flush=True)
        with log.open("a") as f:
            f.write(line + "\n")
    return history


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data", nargs="+", required=True, help="folders with raw/ and ref/"
    )
    p.add_argument("--run", required=True, help="name for checkpoints/ and results/")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument(
        "--batch-size", type=int, default=16, help="drop to 8 on CUDA out-of-memory"
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--init", default=PRETRAINED, help="authors' checkpoint, or 'scratch'"
    )
    args = p.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_set = ConcatDataset([PairedDataset(d, "train", args.crop) for d in args.data])
    val_set = ConcatDataset([PairedDataset(d, "val") for d in args.data])
    if len(train_set) < args.batch_size or len(val_set) == 0:
        raise ValueError(
            f"need >= {args.batch_size} train and >= 1 val images, got {len(train_set)}/{len(val_set)}"
        )
    train_loader = DataLoader(
        train_set, args.batch_size, shuffle=True, drop_last=True, num_workers=2,
        generator=torch.Generator().manual_seed(args.seed),
    )  # fmt: skip
    val_loader = DataLoader(val_set, batch_size=1, num_workers=2)
    model = (
        UWnet()
        if args.init == "scratch"
        else load_pretrained(args.init, AUTHORS_SHA256)
    )

    out = Path("results") / args.run
    if out.exists() or (Path("checkpoints") / f"{args.run}_best.pth").exists():
        raise FileExistsError(f"run '{args.run}' already exists; pick a new --run name")
    out.mkdir(parents=True)
    config = {
        **vars(args),
        "device": device,
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(
            git("status", "--porcelain")
        ),  # trained from uncommitted code?
    }
    (out / "config.json").write_text(json.dumps(config, indent=1))
    print(f"{len(train_set)} train / {len(val_set)} val images on {device}", flush=True)
    fit(model, Loss(), train_loader, val_loader, args.epochs, args.lr, args.run, device)


if __name__ == "__main__":
    main()
