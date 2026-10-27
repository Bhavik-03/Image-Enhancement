"""Score models on a held-out split with PSNR and SSIM (higher is better).

Always compare against baselines: the raw input itself (nn.Identity) and the pretrained model.
    python evaluate.py --data data/uieb --split test --run baseline
    python evaluate.py --data data/uieb --checkpoint checkpoints/ft_best.pth --run ft
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch import nn
from torch.utils.data import DataLoader

from data import PairedDataset
from model import AUTHORS_SHA256, load_finetuned, load_pretrained


def to_img(t):
    """[1,3,H,W] tensor in [0,1] -> HxWx3 float array."""
    return t.squeeze(0).clamp(0, 1).permute(1, 2, 0).float().cpu().numpy()


@torch.no_grad()
def score(model, loader, device="cpu", save_dir=None, n_save=8):
    """Mean (PSNR, SSIM) of model(raw) against ref; batch size must be 1 (full-size images).

    Output is clamped to [0,1] first, as at inference. The first n_save results go to save_dir as
    raw | enhanced | reference strips.
    """
    model = model.to(device).eval()
    psnrs, ssims = [], []
    for i, (raw, ref) in enumerate(loader):
        pred, ref_img = to_img(model(raw.to(device))), to_img(ref)
        psnrs.append(peak_signal_noise_ratio(ref_img, pred, data_range=1.0))
        ssims.append(
            structural_similarity(ref_img, pred, data_range=1.0, channel_axis=2)
        )
        if save_dir is not None and i < n_save:
            strip = np.concatenate([to_img(raw), pred, ref_img], axis=1)
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(Path(save_dir) / f"{i:03d}.png"),
                cv2.cvtColor((strip * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR),
            )
    return float(np.mean(psnrs)), float(np.mean(ssims))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="folder with raw/ and ref/")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--checkpoint", help="state_dict from train.py (fine-tuned model)")
    p.add_argument("--pretrained", default="data/downloads/shallow_uwnet.ckpt")
    p.add_argument("--run", default="baseline")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = PairedDataset(args.data, args.split)
    if len(dataset) == 0:
        raise ValueError(f"{args.data}: the {args.split} split is empty")
    loader = DataLoader(dataset, batch_size=1, num_workers=2)
    models = {
        "raw input": nn.Identity(),
        "pretrained": load_pretrained(args.pretrained, AUTHORS_SHA256),
    }
    if args.checkpoint:
        models["fine-tuned"] = load_finetuned(args.checkpoint)

    out = Path("results") / args.run
    results = {}
    for name, model in models.items():
        results[name] = dict(
            zip(
                ("psnr", "ssim"),
                score(model, loader, device, out / name.replace(" ", "_")),
            )
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(
            {"data": args.data, "split": args.split, "n": len(loader), **results},
            indent=1,
        )
    )

    print(
        f"{args.split} split, {len(loader)} images\n{'model':<12} {'PSNR':>7} {'SSIM':>7}"
    )
    for name, m in results.items():
        print(f"{name:<12} {m['psnr']:>7.2f} {m['ssim']:>7.4f}")


if __name__ == "__main__":
    main()
