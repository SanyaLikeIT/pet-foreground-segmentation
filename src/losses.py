"""Losses for raw logits and binary foreground masks of shape (N, 1, H, W)."""

import math

import torch
from torch import nn


class SoftDiceLoss(nn.Module):
    """Mean per-image soft Dice loss; smooth=1e-6 is added to numerator/denominator.

    Dice = (2 * sum(sigmoid(logits) * target) + smooth) /
           (sum(sigmoid(logits)) + sum(target) + smooth).
    All non-batch dimensions are reduced per image. No thresholding is used.
    Smoothing makes an exactly empty prediction and target have zero loss.
    """

    def __init__(self, smooth=1e-6):
        super().__init__()
        if not math.isfinite(smooth) or smooth <= 0:
            raise ValueError("smooth must be finite and positive")
        self.smooth = smooth

    def forward(self, logits, targets):
        if logits.shape != targets.shape or logits.ndim < 2 or logits.shape[0] == 0:
            raise ValueError("logits and targets must have matching, nonempty batched shapes")
        probabilities = torch.sigmoid(logits).flatten(1)
        targets = targets.flatten(1)
        intersection = (probabilities * targets).sum(dim=1)
        denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
        return (1 - dice).mean()


class BCEDiceLoss(nn.Module):
    """BCEWithLogitsLoss + dice_weight * mean per-image SoftDiceLoss."""

    def __init__(self, dice_weight=1.0, smooth=1e-6):
        super().__init__()
        if not math.isfinite(dice_weight) or dice_weight < 0:
            raise ValueError("dice_weight must be finite and nonnegative")
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss(smooth)

    def forward(self, logits, targets):
        return self.bce(logits, targets) + self.dice_weight * self.dice(logits, targets)


def make_loss(name="bce", dice_weight=1.0):
    if not math.isfinite(dice_weight) or dice_weight < 0:
        raise ValueError("dice_weight must be finite and nonnegative")
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "bce_dice":
        return BCEDiceLoss(dice_weight)
    raise ValueError("loss must be bce or bce_dice")
