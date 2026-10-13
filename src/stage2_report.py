"""Plot original losses and optionally render metric-selected validation examples."""

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch

from src.check_dataset import denormalize
from src.dataset import PetSegDataset, get_val_transform
from src.evaluate import file_sha256, load_checkpoint_model
from src.metrics import binary_metrics
from src.train import select_device


def plot_history(history_path, output_dir):
    history = json.loads(Path(history_path).read_text(encoding="utf-8"))
    epochs = history["epochs"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([row["epoch"] for row in epochs], [row["train_loss"] for row in epochs], label="Train BCE")
    ax.plot([row["epoch"] for row in epochs], [row["val_loss"] for row in epochs], label="Validation BCE")
    ax.scatter([history["best_epoch"]], [history["best_val_loss"]], color="black", zorder=5,
               label=f"Selected epoch {history['best_epoch']} (BCE={history['best_val_loss']:.6f})")
    ax.set(xlabel="Epoch", ylabel="BCEWithLogitsLoss", title="Existing baseline: original training history")
    ax.set_xticks([row["epoch"] for row in epochs])
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close(fig)


def render_examples(model, dataset, rows, device, threshold, destination):
    model.eval()
    fig, axes = plt.subplots(len(rows), 4, figsize=(16, 4 * len(rows)), squeeze=False)
    with torch.inference_mode():
        for axes_row, row in zip(axes, rows):
            index = int(row["dataset_index"])
            if dataset.image_id(index) != row["image_id"]:
                raise ValueError("CSV identifiers do not match the validation dataset")
            image, target = dataset[index]
            prediction = (model(image.unsqueeze(0).to(device)).sigmoid() >= threshold).cpu()
            scores = binary_metrics(prediction, target.unsqueeze(0))
            for key in ("foreground_iou", "foreground_dice"):
                if abs(scores[key].item() - float(row[key])) > 1e-3:
                    raise ValueError("Rendered prediction differs materially from recorded metrics")
            image = denormalize(image)
            target = target[0].numpy().astype(bool)
            prediction = prediction[0, 0].numpy()
            errors = np.zeros((*prediction.shape, 4), dtype=np.float32)
            errors[prediction & target] = (0.2, 0.9, 0.2, 0.45)
            errors[prediction & ~target] = (1, 0, 0, 0.8)
            errors[~prediction & target] = (0, 0.8, 1, 0.8)
            axes_row[0].imshow(image)
            axes_row[0].set_title(row["image_id"])
            axes_row[0].set_ylabel(f"IoU={float(row['foreground_iou']):.4f}\nDice={float(row['foreground_dice']):.4f}")
            axes_row[1].imshow(target, cmap="gray", vmin=0, vmax=1)
            axes_row[1].set_title("Ground truth (trimap == 1)")
            axes_row[2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
            axes_row[2].set_title(f"Prediction (threshold {threshold})")
            axes_row[3].imshow(image)
            axes_row[3].imshow(errors)
            axes_row[3].set_title("Errors: red FP / cyan FN")
            for ax in axes_row:
                ax.set_xticks([])
                ax.set_yticks([])
    fig.legend(handles=[Patch(color="red", label="False positive (extra foreground)"),
                        Patch(color="cyan", label="False negative (missed foreground)"),
                        Patch(color="limegreen", label="True positive")],
               loc="lower center", ncol=3)
    fig.suptitle("Validation predictions", fontsize=16)
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", default="runs/baseline/history.json")
    parser.add_argument("--metrics-dir", help="Validation evaluation directory; omit for loss plot only")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--checkpoint", default="runs/baseline/best.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--poor-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="report/stage2")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    plot_history(args.history, output)
    print(f"Saved {output / 'loss_curve.png'}")
    if not args.metrics_dir:
        return
    metrics_dir = Path(args.metrics_dir)
    summary = json.loads((metrics_dir / "metrics.json").read_text(encoding="utf-8"))
    if summary["split"] != "val":
        raise ValueError("Stage 2 failure analysis requires validation metrics")
    if "precision" in summary:
        torch.backends.cuda.matmul.allow_tf32 = summary["precision"]["matmul_allow_tf32"]
        torch.backends.cudnn.allow_tf32 = summary["precision"]["cudnn_allow_tf32"]
    if summary["checkpoint_sha256"] != file_sha256(args.checkpoint):
        raise ValueError("Metrics were produced with a different checkpoint")
    split_path = Path(__file__).resolve().parents[1] / "data" / "splits" / "val.txt"
    if summary["split_sha256"] != file_sha256(split_path):
        raise ValueError("Metrics were produced with a different validation split")
    with (metrics_dir / "per_image.csv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not 1 <= args.count <= len(rows) or not 3 <= args.poor_count <= len(rows):
        raise ValueError("count must be positive; poor-count must be at least 3, within dataset size")
    dataset = PetSegDataset(args.data_root, "val", get_val_transform())
    if len(rows) != len(dataset) or summary["sample_count"] != len(dataset):
        raise ValueError("Metrics must cover the entire validation split")
    device = select_device(args.device)
    model, _ = load_checkpoint_model(args.checkpoint, device)
    selected = random.Random(args.seed).sample(rows, args.count)
    poor = sorted(rows, key=lambda row: (float(row["foreground_iou"]), row["image_id"]))[:args.poor_count]
    nonempty_rows = [row for row in rows
                     if int(row["true_positive_pixels"]) + int(row["false_negative_pixels"]) > 0]
    poor_nonempty = sorted(nonempty_rows, key=lambda row: (float(row["foreground_iou"]), row["image_id"]))[:args.poor_count]
    render_examples(model, dataset, selected, device, summary["threshold"], output / "validation_examples.png")
    render_examples(model, dataset, poor, device, summary["threshold"], output / "validation_poor.png")
    if poor_nonempty:
        render_examples(model, dataset, poor_nonempty, device, summary["threshold"],
                        output / "validation_poor_nonempty.png")
    (output / "example_selection.json").write_text(json.dumps({
        "split": "val", "threshold": summary["threshold"], "random_seed": args.seed,
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "poor_selection": "Lowest per-image foreground IoU, ties broken by image_id",
        "poor_nonempty_selection": "Lowest per-image foreground IoU among nonempty targets; supplementary analysis only",
        "empty_target_count": len(rows) - len(nonempty_rows),
        "examples": selected, "poor_predictions": poor, "poor_nonempty_predictions": poor_nonempty,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved validation examples, {len(poor)} poor predictions, and identifiers/scores to {output}")


if __name__ == "__main__":
    main()
