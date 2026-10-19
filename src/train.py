import argparse
import json
import random
import hashlib
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import PetSegDataset, get_train_transform, get_val_transform
from src.model import UNet
from src.losses import SoftDiceLoss, make_loss
from src.metrics import binary_metrics


@dataclass
class TrainingConfig:
    data_root: str = "data/raw"
    output_dir: str = "runs/baseline"
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 0.0003
    weight_decay: float = 0.0001
    num_workers: int = 0
    seed: int = 42
    freeze_encoder_epochs: int = 0
    pretrained_encoder: bool = True
    device: str = "auto"
    patience: int = 5
    loss: str = "bce"
    dice_weight: float = 1.0
    selection_metric: str = "val_loss"
    disable_tf32: bool = False


def validate_config(config):
    make_loss(config.loss, config.dice_weight)
    if config.selection_metric not in ("val_loss", "val_bce"):
        raise ValueError("selection_metric must be val_loss or val_bce")
    if config.epochs < 1 or config.batch_size < 1 or config.learning_rate <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive")
    if config.weight_decay < 0 or config.num_workers < 0:
        raise ValueError("weight_decay and num_workers cannot be negative")
    if config.freeze_encoder_epochs < 0 or config.patience < 0:
        raise ValueError("freeze_encoder_epochs and patience cannot be negative")

    data_dir = Path(config.data_root) / "oxford-iiit-pet"
    if not (data_dir / "images").is_dir() or not (data_dir / "annotations" / "trimaps").is_dir():
        raise FileNotFoundError("Dataset missing. Run: python src/prepare_data.py")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested_device):
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def create_loaders(config, device):
    train_dataset = PetSegDataset(config.data_root, "train", get_train_transform(config.seed))
    val_dataset = PetSegDataset(config.data_root, "val", get_val_transform())
    loader_options = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
    }
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    return train_loader, val_loader


def run_epoch(model, loader, criterion, device, optimizer=None, encoder_trainable=True):
    training = optimizer is not None
    model.train(training)
    if training:
        model.set_encoder_trainable(encoder_trainable)

    total_loss = 0.0
    total_samples = 0
    with torch.set_grad_enabled(training):
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), masks)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

    return total_loss / total_samples


def validate_epoch(model, loader, criterion, device):
    """Log a common BCE and hard per-image metrics independently of the objective."""
    model.eval()
    bce = nn.BCEWithLogitsLoss()
    dice = SoftDiceLoss()
    totals = {key: 0.0 for key in ("val_loss", "val_bce", "val_soft_dice_loss",
                                    "val_iou", "val_dice")}
    samples = 0
    with torch.inference_mode():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            count = images.size(0)
            totals["val_loss"] += criterion(logits, masks).item() * count
            totals["val_bce"] += bce(logits, masks).item() * count
            totals["val_soft_dice_loss"] += dice(logits, masks).item() * count
            scores = binary_metrics(logits.sigmoid() >= 0.5, masks)
            totals["val_iou"] += scores["foreground_iou"].sum().item()
            totals["val_dice"] += scores["foreground_dice"].sum().item()
            samples += count
    return {key: value / samples for key, value in totals.items()}


def save_history(output_dir, config, loss_name, history, best_epoch, best_val_loss,
                 dice_weight=None, best_selection_value=None, training_seconds=None):
    summary = {
        "config": asdict(config),
        "loss": loss_name,
        "dice_weight": config.dice_weight if dice_weight is None else dice_weight,
        "selection_metric": config.selection_metric,
        "best_selection_value": best_val_loss if best_selection_value is None else best_selection_value,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "epochs": history,
        "training_seconds": training_seconds,
    }
    (output_dir / "history.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def save_checkpoint(output_dir, model, optimizer, config, epoch, val_loss, loss_name=None,
                    dice_weight=None, selection_value=None):
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(config),
            "epoch": epoch,
            "val_loss": val_loss,
            "loss": loss_name or config.loss,
            "dice_weight": config.dice_weight if dice_weight is None else dice_weight,
            "selection_metric": config.selection_metric,
            "selection_value": val_loss if selection_value is None else selection_value,
        },
        output_dir / "best.pt",
    )


def train_model(config, criterion=None):
    validate_config(config)
    output_dir = Path(config.output_dir)
    if any((output_dir / name).exists() for name in ("best.pt", "history.json", "run_info.json")):
        raise FileExistsError(f"Training artifacts already exist in {output_dir}; use a new output directory")
    started = time.perf_counter()
    set_seed(config.seed)
    device = select_device(config.device)
    if config.disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    train_loader, val_loader = create_loaders(config, device)
    model = UNet(pretrained_encoder=config.pretrained_encoder).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    if criterion is None:
        criterion = make_loss(config.loss, config.dice_weight)
    if isinstance(criterion, nn.Module):
        criterion = criterion.to(device)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_info(output_dir, config, model, device)
    best_val_loss = float("inf")
    best_selection_value = float("inf")
    best_epoch = 0
    history = []
    loss_name = type(criterion).__name__
    effective_dice_weight = getattr(criterion, "dice_weight", 0.0)

    for epoch in range(1, config.epochs + 1):
        epoch_started = time.perf_counter()
        train_loss = run_epoch(
            model, train_loader, criterion, device, optimizer,
            encoder_trainable=epoch > config.freeze_encoder_epochs,
        )
        validation = validate_epoch(model, val_loader, criterion, device)
        val_loss = validation["val_loss"]
        selection_value = validation[config.selection_metric]
        history.append({"epoch": epoch, "train_loss": train_loss, **validation,
                        "epoch_seconds": time.perf_counter() - epoch_started})
        print(f"epoch {epoch}/{config.epochs}: train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_bce={validation['val_bce']:.6f} "
              f"val_iou={validation['val_iou']:.6f} val_dice={validation['val_dice']:.6f}", flush=True)

        if selection_value < best_selection_value:
            best_selection_value = selection_value
            best_val_loss = val_loss
            best_epoch = epoch
            save_checkpoint(output_dir, model, optimizer, config, epoch, val_loss,
                            loss_name, effective_dice_weight, selection_value)
        save_history(output_dir, config, loss_name, history, best_epoch, best_val_loss,
                     effective_dice_weight, best_selection_value, time.perf_counter() - started)
        if config.patience and epoch - best_epoch >= config.patience:
            print(f"early stopping at epoch {epoch}")
            break

    return save_history(output_dir, config, loss_name, history, best_epoch, best_val_loss,
                        effective_dice_weight, best_selection_value, time.perf_counter() - started)


def write_run_info(output_dir, config, model, device):
    """Keep source, split, initialization and environment provenance with the run."""
    import albumentations
    import torchvision

    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    paths = [root / "src" / name for name in
             ("train.py", "model.py", "dataset.py", "losses.py", "metrics.py")]
    paths += [root / "data" / "splits" / f"{split}.txt" for split in ("train", "val", "test")]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                              capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=root).returncode != 0
    info = {
        "config": asdict(config), "git_commit": revision, "tracked_files_modified": dirty,
        "initial_model_state_sha256": digest.hexdigest(),
        "file_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in paths},
        "python": platform.python_version(), "torch": str(torch.__version__),
        "torchvision": torchvision.__version__, "albumentations": albumentations.__version__,
        "numpy": np.__version__, "cuda_build": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "precision": {"dtype": "float32; no autocast", "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                      "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32},
    }
    (output_dir / "run_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a U-Net on Oxford-IIIT Pet masks")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--output-dir", default="runs/baseline")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=0)
    parser.add_argument("--random-encoder", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--loss", choices=("bce", "bce_dice"), default="bce")
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--selection-metric", choices=("val_loss", "val_bce"), default="val_loss")
    parser.add_argument("--disable-tf32", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainingConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
        pretrained_encoder=not args.random_encoder,
        device=args.device,
        patience=args.patience,
        loss=args.loss,
        dice_weight=args.dice_weight,
        selection_metric=args.selection_metric,
        disable_tf32=args.disable_tf32,
    )
    train_model(config)


if __name__ == "__main__":
    main()
