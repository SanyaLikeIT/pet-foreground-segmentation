import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import PetSegDataset, get_train_transform, get_val_transform
from src.model import UNet


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


def validate_config(config):
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
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested_device)


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


def save_history(output_dir, config, loss_name, history, best_epoch, best_val_loss):
    summary = {
        "config": asdict(config),
        "loss": loss_name,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "epochs": history,
    }
    (output_dir / "history.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def save_checkpoint(output_dir, model, optimizer, config, epoch, val_loss):
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(config),
            "epoch": epoch,
            "val_loss": val_loss,
        },
        output_dir / "best.pt",
    )


def train_model(config, criterion=None):
    validate_config(config)
    set_seed(config.seed)
    device = select_device(config.device)
    train_loader, val_loader = create_loaders(config, device)
    model = UNet(pretrained_encoder=config.pretrained_encoder).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()
    if isinstance(criterion, nn.Module):
        criterion = criterion.to(device)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_epoch = 0
    history = []
    loss_name = type(criterion).__name__

    for epoch in range(1, config.epochs + 1):
        train_loss = run_epoch(
            model, train_loader, criterion, device, optimizer,
            encoder_trainable=epoch > config.freeze_encoder_epochs,
        )
        val_loss = run_epoch(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch}/{config.epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_checkpoint(output_dir, model, optimizer, config, epoch, val_loss)
        save_history(output_dir, config, loss_name, history, best_epoch, best_val_loss)
        if config.patience and epoch - best_epoch >= config.patience:
            print(f"early stopping at epoch {epoch}")
            break

    return save_history(output_dir, config, loss_name, history, best_epoch, best_val_loss)


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
    )
    train_model(config)


if __name__ == "__main__":
    main()
