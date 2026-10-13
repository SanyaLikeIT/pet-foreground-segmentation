"""Audit the existing baseline without changing its checkpoint or history."""

import argparse
import json
import platform
import subprocess
from pathlib import Path

import albumentations
import numpy
import torch
import torchvision

from src.evaluate import file_sha256, load_checkpoint_model
from src.train import select_device
from src.validate_data import check_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/baseline/best.pt")
    parser.add_argument("--history", default="runs/baseline/history.json")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="report/stage2/baseline_audit.json")
    args = parser.parse_args()
    before = {str(path): file_sha256(path) for path in (args.checkpoint, args.history)}
    history = json.loads(Path(args.history).read_text(encoding="utf-8"))
    best = min(history["epochs"], key=lambda row: row["val_loss"])
    device = select_device(args.device)
    model, checkpoint = load_checkpoint_model(args.checkpoint, device)
    with torch.inference_mode():
        logits = model(torch.zeros(1, 3, 256, 256, device=device))
    if logits.shape != (1, 1, 256, 256) or not torch.isfinite(logits).all():
        raise RuntimeError("checkpoint forward check failed")
    checks = {
        "strict_state_dict_load": True,
        "checkpoint_epoch_matches_history": checkpoint["epoch"] == history["best_epoch"] == best["epoch"],
        "checkpoint_loss_matches_history": abs(checkpoint["val_loss"] - history["best_val_loss"]) < 1e-12,
        "best_history_loss_is_minimum": abs(best["val_loss"] - history["best_val_loss"]) < 1e-12,
        "checkpoint_config_matches_history": checkpoint.get("config") == history.get("config"),
        "baseline_artifacts_unchanged": before == {
            str(path): file_sha256(path) for path in (args.checkpoint, args.history)
        },
    }
    if not all(checks.values()):
        raise RuntimeError(f"Baseline consistency checks failed: {checks}")
    try:
        nvidia = subprocess.run([
            "nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"
        ], capture_output=True, text=True, check=False)
        nvidia_status = {"return_code": nvidia.returncode,
                         "output": (nvidia.stdout + nvidia.stderr).strip()}
    except FileNotFoundError:
        nvidia_status = {"output": "nvidia-smi is not installed"}
    data_dir = Path(args.data_root) / "oxford-iiit-pet"
    data_available = all(path.is_dir() for path in (
        data_dir / "images", data_dir / "annotations" / "trimaps"
    ))
    last_epoch = history["epochs"][-1]["epoch"]
    patience = history["config"].get("patience", 0)
    cpu_model = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu_model = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                          if line.startswith("model name")), cpu_model)
    summary = {
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "history_path": str(Path(args.history).resolve()),
        "sha256": before,
        "checkpoint_metadata": checkpoint,
        "history_loss": history["loss"],
        "best_epoch": history["best_epoch"],
        "best_validation_bce": history["best_val_loss"],
        "last_recorded_epoch": last_epoch,
        "history_consistent_with_early_stopping": bool(
            patience and last_epoch < history["config"]["epochs"]
            and last_epoch - history["best_epoch"] >= patience
        ),
        "checks": checks,
        "dummy_forward_shape": list(logits.shape),
        "dummy_forward_finite": True,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "albumentations": albumentations.__version__,
            "numpy": numpy.__version__, "cpu_model": cpu_model,
            "cuda_available": torch.cuda.is_available(), "torch_cuda_build": torch.version.cuda,
            "actual_device": str(device), "nvidia_smi": nvidia_status,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        },
        "splits": check_splits(),
        "data_available": data_available,
        "validation_data_status": "available" if data_available else "local dataset missing",
        "training_hardware": "Not recorded in original artifacts; cannot infer from this audit device",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
