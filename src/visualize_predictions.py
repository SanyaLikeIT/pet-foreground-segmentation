import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.check_dataset import denormalize
from src.dataset import PetSegDataset, get_val_transform
from src.evaluate import load_checkpoint_model
from src.train import select_device


def parse_args():
    parser = argparse.ArgumentParser(description="Compare predicted and reference pet masks")
    parser.add_argument("--checkpoint", default="runs/baseline/best.pt")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output")
    return parser.parse_args()


def select_indices(dataset_size, count, indices, seed):
    if indices is not None:
        if any(index < 0 or index >= dataset_size for index in indices):
            raise ValueError(f"indices must be between 0 and {dataset_size - 1}")
        return indices
    if count < 1 or count > dataset_size:
        raise ValueError(f"count must be between 1 and {dataset_size}")
    return random.Random(seed).sample(range(dataset_size), count)


def load_model(checkpoint_path, device):
    model, _ = load_checkpoint_model(checkpoint_path, device)
    return model


def show_prediction(axes, image, mask, prediction, sample_index):
    image = denormalize(image)
    mask = mask[0].numpy()
    prediction = prediction.numpy()

    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[0].set_ylabel(f"Sample {sample_index}")
    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Reference mask")
    axes[2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Predicted mask")
    overlay = np.zeros((*prediction.shape, 4), dtype=np.float32)
    overlay[prediction] = (1.0, 0.0, 0.0, 0.45)
    axes[3].imshow(image)
    axes[3].imshow(overlay)
    axes[3].set_title("Prediction overlay")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])


def main():
    args = parse_args()
    if not 0 <= args.threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    device = select_device(args.device)
    dataset = PetSegDataset("data/raw", split=args.split, transform=get_val_transform())
    indices = select_indices(len(dataset), args.count, args.indices, args.seed)
    model = load_model(args.checkpoint, device)
    figure, axes = plt.subplots(len(indices), 4, figsize=(16, 4 * len(indices)), squeeze=False)

    with torch.inference_mode():
        for row, index in enumerate(indices):
            image, mask = dataset[index]
            logits = model(image.unsqueeze(0).to(device))
            prediction = (torch.sigmoid(logits)[0, 0] >= args.threshold).cpu()
            show_prediction(axes[row], image, mask, prediction, index)

    output_path = Path(args.output or f"report/predictions_{args.split}.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
