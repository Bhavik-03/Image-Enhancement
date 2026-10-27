"""Shallow-UWnet (~220K params): a residual-style CNN, image [B,3,H,W] in [0,1] -> enhanced image.

Layer names match the authors' checkpoint (github.com/mkartik/Shallow-UWnet) so its weights load as is.
"""

import hashlib
from pathlib import Path

import torch
from torch import nn

# SHA-256 of the authors' model.ckpt (Google Drive link in their README), the exact file we inspected.
AUTHORS_SHA256 = "16c090a847e990bfd56a6466fd36f5595cc3a4c514a5ce941caa1a1806441835"


class ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv is applied twice in forward (shared weights)
        self.conv = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.conv1 = nn.Conv2d(64, 61, 3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=0.2)

    def forward(self, x):
        x, input_x = x
        h = self.relu(self.drop(self.conv(x)))
        h = self.relu(self.drop(self.conv(h)))
        h = self.relu(self.conv1(h))
        # 61 learned channels + the 3 raw input channels = 64 again
        return torch.cat((h, input_x), 1), input_x


class UWnet(nn.Module):
    def __init__(self, num_layers=3):
        super().__init__()
        self.input = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.output = nn.Conv2d(64, 3, 3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.blocks = nn.Sequential(*(ConvBlock() for _ in range(num_layers)))

    def forward(self, x):
        out, _ = self.blocks((self.relu(self.input(x)), x))
        return self.output(out)


def load_pretrained(path, sha256=None):
    """Load a checkpoint that is a whole pickled model (the authors' format), not a state_dict.

    Pass `sha256=AUTHORS_SHA256` for the authors' file: a different file is refused before it is
    unpickled. The pickle is then read with weights_only=True, which only lets the classes below be
    rebuilt. That limits the damage a bad file can do (needs torch >= 2.6) but does not make an
    untrusted file safe, so only load files whose hash you have checked.
    """
    if (
        sha256 is not None
        and hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha256
    ):
        raise ValueError(f"{path}: SHA-256 does not match the expected checkpoint")
    allowed = [UWnet, ConvBlock, nn.Conv2d, nn.ReLU, nn.Dropout2d, nn.Sequential]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)
