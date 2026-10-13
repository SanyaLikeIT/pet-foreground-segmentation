"""Evaluate a saved checkpoint on validation by default, without retraining."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset import PetSegDataset, get_val_transform
from src.metrics import METRIC_DEFINITIONS, aggregate_metrics, binary_metrics
from src.model import UNet
from src.train import select_device


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint_model(checkpoint_path, device):
    """Restore a complete checkpoint safely, strictly, and without a weight download."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError("checkpoint must contain a model_state dictionary")
    model = UNet(pretrained_encoder=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    # Do not keep a potentially large optimizer state alive during inference.
    metadata = {key: value for key, value in checkpoint.items()
                if key not in ("model_state", "optimizer_state")}
    return model, metadata


def evaluate_loader(model, loader, device, threshold=0.5):
    """Return metrics and per-image rows in loader order (shuffle must be False).

    Counts and scores are stored per image before aggregating, including the
    final incomplete batch. dataset.image_id(index) supplies stable identifiers.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    model.eval()
    rows = []
    with torch.inference_mode():
        for images, targets in loader:
            # Fingerprint the actual normalized inputs and resized masks, before
            # device transfer, to verify preprocessing/order across batch sizes.
            input_hashes = [hashlib.sha256(image.contiguous().numpy().tobytes()).hexdigest()
                            for image in images]
            target_hashes = [hashlib.sha256(target.contiguous().numpy().tobytes()).hexdigest()
                             for target in targets]
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            predictions = torch.sigmoid(model(images)) >= threshold
            values = {key: value.cpu().tolist()
                      for key, value in binary_metrics(predictions, targets).items()}
            for batch_index in range(images.shape[0]):
                dataset_index = len(rows)
                row = {"image_id": loader.dataset.image_id(dataset_index),
                       "dataset_index": dataset_index,
                       "input_sha256": input_hashes[batch_index],
                       "target_sha256": target_hashes[batch_index]}
                row.update({key: value[batch_index] for key, value in values.items()})
                rows.append(row)
    return aggregate_metrics(rows), rows


def save_results(output_dir, summary, rows):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (output_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--checkpoint", default="runs/baseline/best.pt")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", help="Directory for metrics.json and per_image.csv")
    parser.add_argument("--check-batch-size", type=int,
                        help="Repeat the full split with another batch size and compare per-image scores")
    parser.add_argument("--disable-tf32", action="store_true",
                        help="Disable TF32 in CUDA matrix multiplication and cuDNN convolutions")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers nonnegative")
    if not 0 <= args.threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if args.check_batch_size is not None and args.check_batch_size < 1:
        raise ValueError("check-batch-size must be positive")
    device = select_device(args.device)
    if args.disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    data_dir = Path(args.data_root) / "oxford-iiit-pet"
    if not (data_dir / "images").is_dir() or not (data_dir / "annotations" / "trimaps").is_dir():
        raise FileNotFoundError(
            f"Dataset missing at {data_dir}. Supply --data-root pointing to the directory "
            "containing oxford-iiit-pet, or download it as documented in README."
        )
    dataset = PetSegDataset(args.data_root, args.split, get_val_transform())
    model, checkpoint = load_checkpoint_model(args.checkpoint, device)

    def loader(batch_size):
        return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=device.type == "cuda")

    metrics, rows = evaluate_loader(model, loader(args.batch_size), device, args.threshold)
    split_path = Path(__file__).resolve().parents[1] / "data" / "splits" / f"{args.split}.txt"
    summary = {
        "split": args.split,
        "sample_count": len(rows),
        "threshold": args.threshold,
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_loss": checkpoint.get("val_loss"),
        "checkpoint_config": checkpoint.get("config"),
        "data_root": str(Path(args.data_root).resolve()),
        "split_sha256": file_sha256(split_path),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "precision": {
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "inference_dtype": "float32 (no autocast)",
        },
        "evaluation_mode": "model.eval() and torch.inference_mode()",
        "preprocessing": {
            "image_size": [256, 256],
            "image_resize": "bilinear", "mask_resize": "nearest neighbor",
            "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
            "augmentation": "none", "foreground": "trimap == 1",
        },
        "batch_size": args.batch_size,
        "metric_definitions": METRIC_DEFINITIONS,
        "metrics": metrics,
    }
    if args.check_batch_size is not None:
        other_metrics, other_rows = evaluate_loader(
            model, loader(args.check_batch_size), device, args.threshold
        )
        if [row["image_id"] for row in rows] != [row["image_id"] for row in other_rows]:
            raise RuntimeError("image ordering changed between batch-size checks")
        same_inputs = all(row["input_sha256"] == other["input_sha256"]
                          and row["target_sha256"] == other["target_sha256"]
                          for row, other in zip(rows, other_rows))
        if not same_inputs:
            raise RuntimeError("preprocessed images or masks changed between batch-size checks")
        differences = {key: abs(metrics[key] - other_metrics[key]) for key in metrics}
        per_image_difference = max(
            abs(row[key] - other[key])
            for row, other in zip(rows, other_rows)
            for key in ("foreground_iou", "foreground_dice")
        )
        # Floating-point kernel differences can flip pixels very close to 0.5.
        summary["batch_size_check"] = {
            "other_batch_size": args.check_batch_size,
            "other_metrics": other_metrics,
            "identical_image_order": True,
            "identical_preprocessed_images_and_masks": same_inputs,
            "aggregate_absolute_differences": differences,
            "max_per_image_score_difference": per_image_difference,
            "aggregate_tolerance": 1e-5,
            "per_image_tolerance": 1e-3,
            "passed": max(differences.values()) <= 1e-5 and per_image_difference <= 1e-3,
        }
        # Preserve the second pass so differences can be inspected per image.
        comparison_output = Path(args.output or f"report/stage2/{args.split}") / f"batch_size_{args.check_batch_size}"
        save_results(comparison_output, {
            **summary, "batch_size": args.check_batch_size, "metrics": other_metrics,
        }, other_rows)
    output = args.output or f"report/stage2/{args.split}"
    save_results(output, summary, rows)
    print(json.dumps(summary, indent=2))
    print(f"Saved metrics.json and per_image.csv to {output}")
    if summary.get("batch_size_check", {}).get("passed") is False:
        raise RuntimeError("Batch-size check failed; inspect saved differences before reporting results")


if __name__ == "__main__":
    main()
