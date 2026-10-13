"""Hard foreground metrics; image means for IoU/Dice, pixel totals for accuracy.

No smoothing is used. Both masks empty => IoU=Dice=1; only one empty => 0.
Inputs must be binary, matching batched tensors. Threshold logits externally.
"""

import torch


METRIC_DEFINITIONS = {
    "foreground_iou": "Per-image TP / (TP + FP + FN), arithmetic mean over images",
    "foreground_dice": "Per-image 2*TP / (2*TP + FP + FN), arithmetic mean over images",
    "pixel_accuracy": "Total correct foreground/background pixels / total evaluated pixels",
    "empty_mask_convention": "Both prediction and target empty: IoU=Dice=1; only one empty: 0",
    "prediction": "sigmoid(raw logits) >= threshold; trimap == 1 is foreground",
}


def binary_metrics(predictions, targets):
    if predictions.shape != targets.shape or predictions.ndim < 2 or predictions.shape[0] == 0:
        raise ValueError("predictions and targets must have matching, nonempty batched shapes")
    for tensor in (predictions, targets):
        if not torch.all((tensor == 0) | (tensor == 1)):
            raise ValueError("metrics require binary masks with values 0 and 1")
    predictions = predictions.bool().flatten(1)
    targets = targets.bool().flatten(1)
    tp = (predictions & targets).sum(dim=1)
    fp = (predictions & ~targets).sum(dim=1)
    fn = (~predictions & targets).sum(dim=1)
    union = tp + fp + fn
    dice_denominator = 2 * tp + fp + fn
    iou = torch.where(union > 0, tp.double() / union.clamp_min(1), 1.0)
    dice = torch.where(dice_denominator > 0, 2 * tp.double() / dice_denominator.clamp_min(1), 1.0)
    return {
        "foreground_iou": iou,
        "foreground_dice": dice,
        "correct_pixels": (predictions == targets).sum(dim=1),
        "total_pixels": torch.full_like(tp, predictions.shape[1]),
        "true_positive_pixels": tp,
        "false_positive_pixels": fp,
        "false_negative_pixels": fn,
    }


def aggregate_metrics(rows):
    """Aggregate per-image records; never average unweighted batch means."""
    if not rows:
        raise ValueError("cannot evaluate an empty dataset")
    return {
        "foreground_iou": sum(row["foreground_iou"] for row in rows) / len(rows),
        "foreground_dice": sum(row["foreground_dice"] for row in rows) / len(rows),
        "pixel_accuracy": sum(row["correct_pixels"] for row in rows)
        / sum(row["total_pixels"] for row in rows),
    }
