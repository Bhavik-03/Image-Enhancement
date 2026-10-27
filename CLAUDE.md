# Underwater Image Enhancement

College project, ~1 day. Enhance real underwater photos/video by fine-tuning a small pretrained model (Shallow-UWnet; fallback: small residual U-Net) on EUVP + UIEB. The user knows DL theory but is new to this task: keep code minimal and explain each step.

Hardware: RTX 4060 Laptop (8 GB VRAM), 15 GB RAM, Python 3.10.

## Commands
- Env: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- CUDA check: `python -c "import torch; print(torch.cuda.is_available())"`

## Layout (planned)
`data.py` `model.py` `losses.py` `train.py` `evaluate.py` `enhance.py` `app.py`; `data/`, `checkpoints/`, `results/` are gitignored.

## Decisions
- Train on 256x256 random crops, infer at full resolution; L1 + 0.1 x VGG perceptual loss; Adam lr 1e-4; fp16 autocast; flips/rot90 only (no colour jitter).
- Seeded train/test split saved to file for reproducibility.

## Git workflow
Repo: github.com/Bhavik-03/Image-Enhancement. Feature branch per step (`feat/data`, `feat/model-loss`, `feat/train`, ...), merged via PR (`gh pr create`). Conventional Commits, small atomic commits. Never commit data, checkpoints or secrets. Show the commit list before pushing.

Full plan: `~/.claude/plans/my-plan-is-to-zazzy-liskov.md`
