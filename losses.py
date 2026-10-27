"""Training loss: L1 + 0.1 x perceptual (frozen VGG16 features), both on [0,1] images."""

import torch
import torch.nn.functional as F
from torch import nn


def vgg_features():
    """VGG16 up to relu3_3 with ImageNet weights (downloads ~530 MB once)."""
    from torchvision.models import VGG16_Weights, vgg16

    return vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16]


class Loss(nn.Module):
    """loss = Loss(); loss(pred, ref). `features` can be swapped for a tiny net in tests."""

    def __init__(self, features=None, weight=0.1):
        super().__init__()
        self.features = (features if features is not None else vgg_features()).eval()
        self.features.requires_grad_(False)
        self.weight = weight
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def train(self, mode=True):
        super().train(mode)
        self.features.eval()  # VGG stays in eval mode even when the surrounding code is training
        return self

    def forward(self, pred, ref):
        f_pred, f_ref = (self.features((x - self.mean) / self.std) for x in (pred, ref))
        return F.l1_loss(pred, ref) + self.weight * F.l1_loss(f_pred, f_ref)
