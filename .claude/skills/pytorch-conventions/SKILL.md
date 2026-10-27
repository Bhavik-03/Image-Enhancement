---
name: pytorch-conventions
description: Code conventions for the PyTorch underwater enhancement code (data, model, losses, training, inference).
user-invocable: false
paths:
  - "**/*.py"
---

**Interfaces**
- Models are `nn.Module` with `forward(x) -> y`; images are float tensors `[B,3,H,W]` in `[0,1]`. Swapping a model must not change other files.
- `data.py`: `PairedDataset(root, split, crop=256)` returns `(raw, ref)`; train takes random crops, val/test use full images.
- `losses.py`: one function `loss(pred, ref)` = L1 + 0.1 x VGG perceptual (VGG frozen, in eval mode).

**Training**
- Seed everything (`torch`, `numpy`, `random`); splits come from a saved, seeded split file.
- Augment with flips and 90 degree rotations only. No hue or colour jitter: colour is the signal.
- `torch.autocast` fp16 on CUDA; save the best checkpoint by validation PSNR to `checkpoints/`; log config per run.
- Validate each epoch with PSNR and SSIM (`skimage.metrics`) on the held-out split.

**Inference**
- Fully convolutional, run at full resolution (pad to a multiple of 8 if the model needs it). Clamp output to `[0,1]`.
- Video: process frame by frame; smooth lightly across frames to reduce flicker.

**Errors and tests**
- Fail early with clear messages for missing paths and unpaired files; do not catch broad exceptions.
- Tests are small and CPU-only: shape checks, loss is finite, one batch overfit, no network access.
