# Underwater Image Enhancement

Enhances real underwater photos and video by fine-tuning a small pretrained network, [Shallow-UWnet](https://arxiv.org/abs/2101.02073) (219K parameters), on the UIEB benchmark. Water absorbs red light first, so raw footage looks blue-green and hazy; the model maps a degraded image to a clearer one.

## Results

Held-out UIEB test split: 89 image pairs the model never saw, scored at full resolution against UIEB's reference images (higher is better).

| Model | PSNR (dB) | SSIM |
|---|---|---|
| Raw input (no enhancement) | 18.07 | 0.7753 |
| Shallow-UWnet, authors' pretrained weights | 17.97 | 0.7700 |
| **Shallow-UWnet, fine-tuned here on UIEB** | **19.20** | **0.8296** |

- The authors' weights (trained on EUVP) do not transfer to UIEB: they score slightly *below* the raw input and tint the images grey-purple. Measuring this baseline first is what justified fine-tuning.
- Fine-tuning gives +1.13 dB PSNR and +0.054 SSIM over the raw input.
- Visually the gain is modest: the purple haze is gone and images are a little clearer, but they stay much bluer and duller than the bright reference images.

Run `ft1`: 712 train / 89 val / 89 test images (seed 0), 30 epochs, batch 16, Adam lr 1e-4 with cosine decay, fp16, 256x256 random crops with flips and 90 degree rotations only, loss = L1 + 0.1 x VGG16 perceptual. About 15 minutes on an RTX 4060 Laptop (8 GB). Validation PSNR stopped improving around epoch 15 (18.00 at epoch 1, 18.51 at the end); the best checkpoint by validation PSNR is the one scored above.

## Limitations

- **Small model, small data.** 219K parameters and 712 training images limit what it can restore; it does not reach the references' colour.
- **References are not ground truth.** UIEB references are the best of several enhancement algorithms picked by people, so PSNR/SSIM measure similarity to that choice, not true colour. No-reference metrics (UIQM, UCIQE) are not computed.
- **One run.** A single seed and a single split; the 89-image test set gives a rough, not a tight, estimate. EUVP was not used for fine-tuning (the authors' weights were trained on it).
- **Video.** Frames are enhanced independently, then lightly blended with the previous output (exponential smoothing, `--smooth 0.3`) to reduce flicker. This is not evaluated quantitatively.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
```

`source .venv/bin/activate` must be run in every new shell. On this setup it also puts the venv's NVIDIA libraries on `LD_LIBRARY_PATH` (see `CLAUDE.md`); without it every GPU convolution fails with a cuDNN error.

**Data** (not in the repo: UIEB is for non-commercial use and may not be redistributed). Download the UIEB raw and reference archives from the [authors' page](https://li-chongyi.github.io/proj_benchmark.html), extract them (the page lists the archive passwords) and arrange them as:

```
data/uieb/raw/<name>.png     data/uieb/ref/<name>.png     (same file names)
```

Download the Shallow-UWnet weights (`model.ckpt`) from the [authors' repo](https://github.com/mkartik/Shallow-UWnet) to `data/downloads/shallow_uwnet.ckpt`. The file is a pickled model, so it is loaded only after a SHA-256 check and with an allowlist of classes (`model.load_pretrained`); never load it with `weights_only=False`.

## Usage

```bash
python evaluate.py --data data/uieb --split test --run baseline      # raw vs pretrained
python train.py --data data/uieb --run ft1 --epochs 30               # fine-tune; best weights -> checkpoints/ft1_best.pth
python evaluate.py --data data/uieb --split test --checkpoint checkpoints/ft1_best.pth --run ft1_test
python enhance.py photo.jpg --checkpoint checkpoints/ft1_best.pth    # also videos: clip.mp4 --smooth 0.3
python app.py                                                        # Gradio demo at http://127.0.0.1:7860
```

- `train.py` refuses to reuse a run name, saves a config (including the git commit) and per-epoch log to `results/<run>/`. If the GPU runs out of memory, pass `--batch-size 8`.
- `evaluate.py` writes `results/<run>/metrics.json` and raw | enhanced | reference strips for the first images.
- `enhance.py` works at full resolution; frames over 4 megapixels are processed in overlapping bands (identical result, about 3.6 GB VRAM up to 8K).

## Project layout

`data.py` paired dataset and seeded split | `model.py` Shallow-UWnet and safe loaders | `losses.py` L1 + perceptual loss | `train.py` | `evaluate.py` | `enhance.py` | `app.py` | `tests/` CPU-only tests (`python -m pytest -q`)

`data/`, `checkpoints/` and `results/` are gitignored.
