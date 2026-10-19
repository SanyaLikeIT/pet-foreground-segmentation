"""Check common checkpoint selection without dataset downloads or GPU training."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.losses import BCEDiceLoss, SoftDiceLoss
from src.metrics import binary_metrics
from src.train import TrainingConfig, train_model, validate_config, validate_epoch
from src.second_approach_report import compare_evidence, load_evidence, select_examples
from src.metrics import aggregate_metrics
from src.evaluate import save_results


class ValidationChecks(unittest.TestCase):
    def test_common_metrics_and_uneven_batches(self):
        logits = torch.tensor([[[[2., -2.]]], [[[0., 1.]]], [[[-1., 3.]]]])
        targets = torch.tensor([[[[1., 0.]]], [[[0., 0.]]], [[[1., 1.]]]])
        criterion = BCEDiceLoss(2)
        expected = {
            "val_loss": criterion(logits, targets).item(),
            "val_bce": nn.BCEWithLogitsLoss()(logits, targets).item(),
            "val_soft_dice_loss": SoftDiceLoss()(logits, targets).item(),
        }
        scores = binary_metrics(logits.sigmoid() >= 0.5, targets)
        expected.update(val_iou=scores["foreground_iou"].mean().item(),
                        val_dice=scores["foreground_dice"].mean().item())
        for size in (1, 2, 3):
            actual = validate_epoch(nn.Identity(), DataLoader(TensorDataset(logits, targets),
                                    batch_size=size), criterion, torch.device("cpu"))
            for key, value in expected.items():
                self.assertAlmostEqual(actual[key], value, places=6)

    def test_bce_selects_checkpoint_and_controls_early_stopping(self):
        # The combined objective improves every epoch, but common BCE is best at epoch 1.
        records = [{"val_loss": loss, "val_bce": bce, "val_soft_dice_loss": loss - bce,
                    "val_iou": .7, "val_dice": .8}
                   for loss, bce in ((.8, .2), (.6, .3), (.4, .25))]
        with tempfile.TemporaryDirectory() as temporary:
            config = TrainingConfig(output_dir=temporary, epochs=10, patience=2,
                                    device="cpu", loss="bce_dice", selection_metric="val_bce")
            with patch("src.train.validate_config"), \
                 patch("src.train.create_loaders", return_value=(None, None)), \
                 patch("src.train.UNet", side_effect=lambda **kwargs: nn.Conv2d(1, 1, 1)), \
                 patch("src.train.run_epoch", return_value=.9), \
                 patch("src.train.validate_epoch", side_effect=records), \
                 patch("src.train.write_run_info"):
                result = train_model(config)
            checkpoint = torch.load(Path(temporary) / "best.pt", weights_only=True)
            saved = json.loads((Path(temporary) / "history.json").read_text())
            self.assertEqual(len(result["epochs"]), 3)
            self.assertEqual(result["best_epoch"], 1)
            self.assertEqual(result["best_selection_value"], .2)
            self.assertEqual(result["best_val_loss"], .8)
            self.assertEqual(saved, result)
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(checkpoint["selection_value"], .2)
            self.assertEqual(checkpoint["selection_metric"], "val_bce")
            self.assertGreater(result["training_seconds"], 0)

    def test_existing_run_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "history.json"
            path.write_text("preserve")
            with patch("src.train.validate_config"), self.assertRaises(FileExistsError):
                train_model(TrainingConfig(output_dir=temporary))
            self.assertEqual(path.read_text(), "preserve")

    def test_invalid_selection_metric(self):
        with self.assertRaisesRegex(ValueError, "selection_metric"):
            validate_config(TrainingConfig(selection_metric="test_dice"))


class EvidenceChecks(unittest.TestCase):
    def fixture(self):
        rows = []
        for index, prediction in enumerate(([1, 0], [0, 1], [1, 1], [0, 0])):
            target = [1, 0] if index < 3 else [0, 0]
            values = binary_metrics(torch.tensor([prediction]), torch.tensor([target]))
            row = {key: value.item() for key, value in values.items()}
            rows.append({"image_id": f"image_{index}", "dataset_index": index,
                         "input_sha256": str(index), "target_sha256": str(index), **row})
        summary = {"split": "val", "sample_count": 4, "threshold": .5,
                   "split_sha256": "split", "preprocessing": {}, "metric_definitions": {},
                   "precision": {}, "metrics": aggregate_metrics(rows), "checkpoint_sha256": "checkpoint"}
        return summary, rows

    def test_roundtrip_and_matched_comparison(self):
        evidence = self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            save_results(temporary, *evidence)
            loaded = load_evidence(temporary)
        self.assertEqual(loaded, evidence)
        summary, paired = compare_evidence(evidence, loaded)
        self.assertTrue(all(value == 0 for value in summary["delta"].values()))
        self.assertEqual(summary["images_with_unchanged_dice"], 4)
        self.assertEqual(summary["empty_target_ids"], ["image_3"])
        examples, _ = select_examples(paired)
        self.assertEqual(len({r["image_id"] for r in examples}), 3)
        self.assertTrue(all(not r["empty_target"] for r in examples))

    def test_reject_changed_inputs_and_protocol(self):
        for changed in ("input_sha256", "target_sha256", "threshold", "precision"):
            baseline = self.fixture()
            approach = self.fixture()
            if changed in ("threshold", "precision"):
                approach[0][changed] = "different"
            else:
                approach[1][0][changed] = "different"
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                compare_evidence(baseline, approach)

    def test_reject_test_split_and_inconsistent_metrics(self):
        for change in ("test", "metrics"):
            summary, rows = self.fixture()
            if change == "test":
                summary["split"] = "test"
            else:
                summary["metrics"]["foreground_iou"] = -1
            with tempfile.TemporaryDirectory() as temporary:
                save_results(temporary, summary, rows)
                with self.assertRaises(ValueError):
                    load_evidence(temporary)


if __name__ == "__main__":
    unittest.main()
