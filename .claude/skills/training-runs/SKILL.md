---
name: training-runs
description: How to launch, watch and record training and evaluation runs (train.py, evaluate.py, enhance.py). Use whenever running anything longer than a minute or touching checkpoints and results.
user-invocable: false
---

**Before a run**
- `.venv/bin/python -c "import torch; print(torch.cuda.is_available())"` must print `True`; check free VRAM with `nvidia-smi`.
- Smoke test first: one batch or one epoch on a few images, to catch shape, path and OOM errors in seconds, not after an hour.
- Start at batch size 16; on CUDA out-of-memory drop to 8 and note it in the run config.

**Launching**
- Run long jobs with Bash `run_in_background`, output to `results/<run>/log.txt`, then watch with Monitor on the loss and val PSNR lines. Do not poll in a loop and do not run training in the foreground.
- One folder per run: `results/<run>/` (config, log, metrics) and `checkpoints/<run>_best.pth`. Never overwrite a previous run.
- The config (seed, lr, batch size, epochs, data roots, git commit) is written to the run folder at start.

**Reading results**
- Always report against the baseline: raw input vs pretrained vs fine-tuned, PSNR and SSIM on the held-out test split. A number without a baseline means nothing.
- Val PSNR falling while train loss falls = overfitting: stop and keep the best checkpoint.
- Failure cases go in the report too: look at side-by-side images, not only the metrics.

**Git**
- `data/`, `checkpoints/`, `results/` are gitignored. Commit code and the results table (as text in the README), never weights or images.
