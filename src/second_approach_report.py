"""Compare two validation evaluations and render matched predictions; never use test."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.check_dataset import denormalize
from src.dataset import PetSegDataset, get_val_transform
from src.evaluate import file_sha256, load_checkpoint_model, split_sha256
from src.metrics import aggregate_metrics, binary_metrics
from src.train import select_device


def load_evidence(directory):
    directory = Path(directory)
    summary = json.loads((directory / "metrics.json").read_text())
    if summary["split"] != "val":
        raise ValueError("Second approach comparison accepts validation evidence only")
    with (directory / "per_image.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != summary["sample_count"] or len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("Incomplete or duplicate per-image evidence")
    for row in rows:
        for key in row:
            if key in ("image_id", "input_sha256", "target_sha256"):
                continue
            row[key] = float(row[key]) if key in ("foreground_iou", "foreground_dice") else int(row[key])
    reaggregated = aggregate_metrics(rows)
    if any(abs(reaggregated[key] - summary["metrics"][key]) > 1e-12 for key in reaggregated):
        raise ValueError("Per-image evidence does not reproduce aggregate metrics")
    return summary, rows


def compare_evidence(baseline, approach):
    base_summary, base_rows = baseline
    new_summary, new_rows = approach
    for key in ("split", "sample_count", "threshold", "split_sha256", "preprocessing",
                "metric_definitions", "precision"):
        if base_summary[key] != new_summary[key]:
            raise ValueError(f"Evaluation protocols differ: {key}")
    paired = []
    if [r["image_id"] for r in base_rows] != [r["image_id"] for r in new_rows]:
        raise ValueError("Evaluation image order differs")
    for base, new in zip(base_rows, new_rows):
        for key in ("dataset_index", "input_sha256", "target_sha256", "total_pixels"):
            if base[key] != new[key]:
                raise ValueError(f"Inputs differ for {base['image_id']}: {key}")
        if base["true_positive_pixels"] + base["false_negative_pixels"] != new["true_positive_pixels"] + new["false_negative_pixels"]:
            raise ValueError("Target foreground area differs")
        row = {"image_id": base["image_id"], "dataset_index": base["dataset_index"],
               "empty_target": base["true_positive_pixels"] + base["false_negative_pixels"] == 0}
        for suffix, key in (("iou", "foreground_iou"), ("dice", "foreground_dice"),
                            ("fp", "false_positive_pixels"), ("fn", "false_negative_pixels")):
            row.update({f"baseline_{suffix}": base[key], f"approach_{suffix}": new[key],
                        f"delta_{suffix}": new[key] - base[key]})
        paired.append(row)
    delta = {key: new_summary["metrics"][key] - base_summary["metrics"][key]
             for key in base_summary["metrics"]}
    return {
        "split": "val", "sample_count": len(paired), "threshold": base_summary["threshold"],
        "selection_metric": "minimum validation BCE for both checkpoints and early stopping",
        "baseline": base_summary["metrics"], "bce_dice_lambda1": new_summary["metrics"],
        "delta": delta,
        "images_with_improved_dice": sum(r["delta_dice"] > 1e-12 for r in paired),
        "images_with_worse_dice": sum(r["delta_dice"] < -1e-12 for r in paired),
        "images_with_unchanged_dice": sum(abs(r["delta_dice"]) <= 1e-12 for r in paired),
        "empty_target_ids": [r["image_id"] for r in paired if r["empty_target"]],
        "identical_image_order_and_preprocessed_inputs": True,
        "baseline_checkpoint_sha256": base_summary["checkpoint_sha256"],
        "approach_checkpoint_sha256": new_summary["checkpoint_sha256"],
        "validation_split_sha256": base_summary["split_sha256"],
    }, paired


def select_examples(paired):
    nonempty = [r for r in paired if not r["empty_target"]]
    ordered = sorted(nonempty, key=lambda r: (r["delta_dice"], r["image_id"]))
    examples = [{**ordered[-1], "selection": "largest Dice increase among nonempty targets"},
                {**ordered[0], "selection": "largest Dice decrease among nonempty targets"}]
    remaining = [r for r in nonempty if r["image_id"] not in {e["image_id"] for e in examples}]
    typical = min(remaining, key=lambda r: (abs(r["delta_dice"]), r["image_id"]))
    examples.append({**typical, "selection": "smallest absolute Dice change among remaining nonempty targets"})
    known_ids = ("Abyssinian_164", "beagle_147", "pomeranian_18")
    known = [{**r, "selection": "previously documented baseline failure"}
             for name in known_ids for r in paired if r["image_id"] == name]
    return examples, known


def render_comparison(models, dataset, rows, device, threshold, destination):
    fig, axes = plt.subplots(len(rows), 4, figsize=(14, 3.5 * len(rows)), squeeze=False)
    with torch.inference_mode():
        for axes_row, row in zip(axes, rows):
            index = row["dataset_index"]
            if dataset.image_id(index) != row["image_id"]:
                raise ValueError("Figure image ID differs from metric evidence")
            image, target = dataset[index]
            axes_row[0].imshow(denormalize(image))
            axes_row[0].set_title(row["image_id"])
            axes_row[1].imshow(target[0], cmap="gray", vmin=0, vmax=1)
            axes_row[1].set_title("Ground truth")
            for ax, model, prefix, label in zip(axes_row[2:], models,
                                                ("baseline", "approach"), ("BCE", "BCE + Dice, lambda=1")):
                prediction = (model(image.unsqueeze(0).to(device)).sigmoid() >= threshold).cpu()
                scores = binary_metrics(prediction, target.unsqueeze(0))
                for suffix, key in (("iou", "foreground_iou"), ("dice", "foreground_dice")):
                    if abs(scores[key].item() - row[f"{prefix}_{suffix}"]) > 1e-3:
                        raise ValueError("Rendered prediction differs from saved evaluation")
                ax.imshow(prediction[0, 0], cmap="gray", vmin=0, vmax=1)
                ax.set_title(f"{label}\nIoU {row[f'{prefix}_iou']:.4f}, Dice {row[f'{prefix}_dice']:.4f}")
            for ax in axes_row:
                ax.set_xticks([])
                ax.set_yticks([])
    fig.suptitle("Matched validation predictions (foreground probability >= 0.5)")
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def plot_histories(base_history, new_history, destination):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for history, key, label in ((base_history, "val_loss", "Archived BCE baseline"),
                                (new_history, "val_bce", "BCE + Dice, lambda=1")):
        axes[0].plot([r["epoch"] for r in history["epochs"]],
                     [r[key] for r in history["epochs"]], marker=".", label=label)
        selected = next(r for r in history["epochs"] if r["epoch"] == history["best_epoch"])
        axes[0].scatter(selected["epoch"], selected[key], marker="x", s=70)
    axes[0].set(xlabel="Epoch", ylabel="Validation BCE", title="Common checkpoint selection criterion")
    axes[0].legend()
    for key, label in (("train_loss", "Training BCE + Dice"), ("val_loss", "Validation BCE + Dice")):
        axes[1].plot([r["epoch"] for r in new_history["epochs"]],
                     [r[key] for r in new_history["epochs"]], label=label)
    axes[1].set(xlabel="Epoch", ylabel="Combined objective", title="Second approach objective")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-metrics", default="report/second_approach/baseline/val")
    parser.add_argument("--approach-metrics", default="report/second_approach/bce_dice_lambda1/val")
    parser.add_argument("--baseline-history", default="runs/baseline/history.json")
    parser.add_argument("--approach-history", default="runs/second_approach/bce_dice_lambda1/history.json")
    parser.add_argument("--baseline-checkpoint", default="runs/baseline/best.pt")
    parser.add_argument("--approach-checkpoint", default="runs/second_approach/bce_dice_lambda1/best.pt")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="report/second_approach")
    args = parser.parse_args()
    baseline = load_evidence(args.baseline_metrics)
    approach = load_evidence(args.approach_metrics)
    summary, paired = compare_evidence(baseline, approach)
    split_path = Path(__file__).resolve().parents[1] / "data/splits/val.txt"
    if summary["validation_split_sha256"] != split_sha256(split_path):
        raise ValueError("Saved validation split changed")
    base_history = json.loads(Path(args.baseline_history).read_text())
    new_history = json.loads(Path(args.approach_history).read_text())
    if base_history.get("selection_metric", "val_loss") not in ("val_loss", "val_bce") or base_history["loss"] != "BCEWithLogitsLoss":
        raise ValueError("Baseline must be selected by validation BCE")
    if new_history["selection_metric"] != "val_bce" or new_history["config"]["loss"] != "bce_dice" or new_history["dice_weight"] != 1:
        raise ValueError("Second approach must use BCE+Dice lambda=1 with BCE selection")
    for history, (evaluation, _) in zip((base_history, new_history), (baseline, approach)):
        if history["best_epoch"] != evaluation["checkpoint_epoch"]:
            raise ValueError("History and evaluated checkpoint epochs differ")
        key = "val_bce" if history.get("selection_metric") == "val_bce" else "val_loss"
        if min(history["epochs"], key=lambda r: r[key])["epoch"] != history["best_epoch"]:
            raise ValueError("Selected checkpoint is not the minimum validation BCE")
        last_epoch = history["epochs"][-1]["epoch"]
        config = history["config"]
        if last_epoch < config["epochs"] and not (config["patience"] > 0 and last_epoch - history["best_epoch"] >= config["patience"]):
            raise ValueError("Training run is incomplete")
    for key in ("epochs", "batch_size", "learning_rate", "weight_decay", "num_workers",
                "seed", "freeze_encoder_epochs", "pretrained_encoder", "patience"):
        if base_history["config"][key] != new_history["config"][key]:
            raise ValueError(f"Training configurations differ: {key}")
    summary.update(baseline_best_epoch=base_history["best_epoch"],
                   approach_best_epoch=new_history["best_epoch"],
                   approach_training_seconds=new_history["training_seconds"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "per_image_comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    with (output / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "foreground_iou", "foreground_dice", "pixel_accuracy"])
        writer.writeheader()
        for condition in ("baseline", "bce_dice_lambda1", "delta"):
            writer.writerow({"condition": condition, **summary[condition]})
    examples, known = select_examples(paired)
    (output / "example_selection.json").write_text(json.dumps({"split": "val", "examples": examples,
          "previous_baseline_failures": known}, indent=2) + "\n")
    plot_histories(base_history, new_history, output / "learning_curves.png")
    torch.backends.cuda.matmul.allow_tf32 = baseline[0]["precision"]["matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = baseline[0]["precision"]["cudnn_allow_tf32"]
    device = select_device(args.device)
    checkpoints = (args.baseline_checkpoint, args.approach_checkpoint)
    for checkpoint, key in zip(checkpoints, ("baseline_checkpoint_sha256", "approach_checkpoint_sha256")):
        if file_sha256(checkpoint) != summary[key]:
            raise ValueError("Metrics were produced with a different checkpoint")
    models = [load_checkpoint_model(p, device)[0] for p in checkpoints]
    dataset = PetSegDataset(args.data_root, "val", get_val_transform())
    if len(dataset) != len(paired):
        raise ValueError("Evaluation must cover the complete validation dataset")
    render_comparison(models, dataset, examples, device, summary["threshold"], output / "validation_comparison.png")
    if known:
        render_comparison(models, dataset, known, device, summary["threshold"], output / "previous_failures.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
