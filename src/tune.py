import argparse
import itertools
import json
from pathlib import Path

from src.train import TrainingConfig, train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Compare training hyperparameters on validation loss")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--output-dir", default="runs/tuning")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.0003, 0.0001])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8])
    parser.add_argument("--weight-decays", type=float, nargs="+", default=[0.0, 0.0001])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=0)
    parser.add_argument("--random-encoder", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--patience", type=int, default=5)
    return parser.parse_args()


def make_trial_config(args, trial_dir, learning_rate, batch_size, weight_decay):
    return TrainingConfig(
        data_root=args.data_root,
        output_dir=str(trial_dir),
        epochs=args.epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
        pretrained_encoder=not args.random_encoder,
        device=args.device,
        patience=args.patience,
    )


def save_tuning_summary(output_dir, trials):
    best_trial = min(trials, key=lambda trial: trial["best_val_loss"])
    summary = {"selection_metric": "val_loss", "best_trial": best_trial, "trials": trials}
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return best_trial


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trials = []
    combinations = itertools.product(
        args.learning_rates, args.batch_sizes, args.weight_decays
    )

    for trial_number, (learning_rate, batch_size, weight_decay) in enumerate(combinations, start=1):
        trial_dir = output_dir / f"trial_{trial_number:02d}"
        config = make_trial_config(
            args, trial_dir, learning_rate, batch_size, weight_decay
        )
        print(f"trial {trial_number}: lr={learning_rate} batch={batch_size} wd={weight_decay}")
        result = train_model(config)
        trials.append(
            {
                "trial": trial_number,
                "output_dir": str(trial_dir),
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
                "best_epoch": result["best_epoch"],
                "best_val_loss": result["best_val_loss"],
            }
        )
        best_trial = save_tuning_summary(output_dir, trials)

    print(f"best trial: {best_trial['trial']} val_loss={best_trial['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()
