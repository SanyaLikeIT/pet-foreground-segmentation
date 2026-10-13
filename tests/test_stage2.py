"""Small numerical checks; no training runs or downloads are performed."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.dataset import PetSegDataset, get_train_transform, get_val_transform
from src.evaluate import evaluate_loader, save_results
from src.losses import BCEDiceLoss, SoftDiceLoss, make_loss
from src.metrics import aggregate_metrics, binary_metrics
from src.train import TrainingConfig, run_epoch, save_checkpoint, save_history
from src.validate_data import check_splits


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tensor(values):
    return torch.tensor(values, dtype=torch.float32, device=DEVICE)


class MetricChecks(unittest.TestCase):
    def scores(self, prediction, target):
        values = binary_metrics(tensor([prediction]), tensor([target]))
        return {key: value.item() for key, value in values.items()}

    def test_perfect_overlap(self):
        scores = self.scores([1, 0, 1, 0], [1, 0, 1, 0])
        self.assertEqual(scores["foreground_iou"], 1)
        self.assertEqual(scores["foreground_dice"], 1)
        self.assertEqual(scores["correct_pixels"], 4)

    def test_no_overlap(self):
        scores = self.scores([1, 1, 0, 0], [0, 0, 1, 1])
        self.assertEqual(scores["foreground_iou"], 0)
        self.assertEqual(scores["foreground_dice"], 0)
        self.assertEqual(scores["correct_pixels"], 0)

    def test_partial_overlap_by_hand(self):
        # TP=1, FP=1, FN=1, TN=1 => IoU=1/3, Dice=1/2, accuracy=1/2.
        scores = self.scores([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertAlmostEqual(scores["foreground_iou"], 1 / 3)
        self.assertEqual(scores["foreground_dice"], 0.5)
        self.assertEqual(scores["correct_pixels"] / scores["total_pixels"], 0.5)

    def test_empty_masks(self):
        for prediction, target, expected in (
            ([0, 0], [0, 0], 1), ([1, 0], [0, 0], 0), ([0, 0], [1, 0], 0)
        ):
            with self.subTest(prediction=prediction, target=target):
                scores = self.scores(prediction, target)
                self.assertEqual(scores["foreground_iou"], expected)
                self.assertEqual(scores["foreground_dice"], expected)

    def test_image_mean_and_pixel_totals(self):
        # Different image sizes make the required accuracy weighting explicit.
        rows = [self.scores([1, 0], [1, 0]),
                self.scores([1, 1, 0, 0], [0, 0, 1, 1])]
        metrics = aggregate_metrics(rows)
        self.assertEqual(metrics["foreground_iou"], 0.5)
        self.assertEqual(metrics["foreground_dice"], 0.5)
        self.assertAlmostEqual(metrics["pixel_accuracy"], 2 / 6)

    def test_reject_nonbinary_masks(self):
        with self.assertRaises(ValueError):
            binary_metrics(tensor([[0.2, 1]]), tensor([[0, 1]]))


class LossChecks(unittest.TestCase):
    def test_bce_default_and_zero_dice_weight(self):
        self.assertIsInstance(make_loss(), nn.BCEWithLogitsLoss)
        logits = tensor([[0.7, -1.2, 2.1]])
        target = tensor([[1, 0, 0]])
        torch.testing.assert_close(BCEDiceLoss(0)(logits, target), nn.BCEWithLogitsLoss()(logits, target))

    def test_soft_dice_per_image_by_hand(self):
        probabilities = tensor([[0.8, 0.2, 0.1, 0.9], [0.1, 0.2, 0.3, 0.4]])
        target = tensor([[1, 0, 0, 1], [0, 0, 0, 0]])
        smooth = 1e-6
        expected = ((1 - (3.4 + smooth) / (4 + smooth))
                    + (1 - smooth / (1 + smooth))) / 2
        actual = SoftDiceLoss(smooth)(torch.logit(probabilities), target)
        self.assertAlmostEqual(actual.item(), expected, places=6)

    def test_confident_overlap_and_empty_behavior(self):
        target = tensor([[1, 0, 1, 0]])
        perfect = SoftDiceLoss()(tensor([[100, -100, 100, -100]]), target)
        wrong = SoftDiceLoss()(tensor([[-100, 100, -100, 100]]), target)
        empty = SoftDiceLoss()(tensor([[-100, -100]]), tensor([[0, 0]]))
        self.assertLess(perfect.item(), 1e-6)
        self.assertGreater(wrong.item(), 0.999)
        self.assertLess(empty.item(), 1e-6)

    def test_bce_dice_finite_gradients_including_empty_target(self):
        logits = tensor([[0.0, -2.0, 3.0, -100.0], [100.0, -100.0, 1.0, -1.0]])
        logits.requires_grad_()
        target = tensor([[1, 0, 1, 0], [0, 0, 0, 0]])
        loss = make_loss("bce_dice", 2.0)(logits, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_invalid_loss_settings(self):
        for weight in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                make_loss("bce_dice", weight)
        with self.assertRaises(ValueError):
            SoftDiceLoss(0)


class TinyDataset(Dataset):
    def __init__(self):
        self.logits = torch.tensor([
            [[[2., -2.], [2., -2.]]], [[[2., 2.], [-2., -2.]]],
            [[[-2., -2.], [-2., -2.]]], [[[0., -2.], [-2., -2.]]],
            [[[2., -2.], [-2., -2.]]],
        ])
        self.targets = torch.tensor([
            [[[1., 0.], [1., 0.]]], [[[0., 0.], [1., 1.]]],
            [[[0., 0.], [0., 0.]]], [[[0., 0.], [0., 0.]]],
            [[[1., 0.], [0., 0.]]],
        ])

    def __len__(self):
        return len(self.logits)

    def __getitem__(self, index):
        return self.logits[index], self.targets[index]

    def image_id(self, index):
        return f"fixture_{index}"


class InferenceModel(nn.Module):
    def forward(self, images):
        if self.training or not torch.is_inference_mode_enabled():
            raise AssertionError("evaluation must use eval() and inference_mode()")
        return images


class PipelineChecks(unittest.TestCase):
    def test_uneven_batches_and_inclusive_threshold(self):
        dataset = TinyDataset()
        results = [evaluate_loader(InferenceModel(), DataLoader(dataset, batch_size=size), DEVICE)
                   for size in (1, 2, 3)]
        for result in results[1:]:
            self.assertEqual(result, results[0])
        metrics, rows = results[0]
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[3]["false_positive_pixels"], 1)  # sigmoid(0) == 0.5
        self.assertEqual(metrics["foreground_iou"], 0.6)
        self.assertEqual(metrics["foreground_dice"], 0.6)
        self.assertEqual(metrics["pixel_accuracy"], 15 / 20)
        with tempfile.TemporaryDirectory() as temporary:
            save_results(temporary, {"sample_count": 5, "metrics": metrics}, rows)
            saved = json.loads((Path(temporary) / "metrics.json").read_text())
            self.assertEqual(saved["metrics"], metrics)
            self.assertIn("fixture_4", (Path(temporary) / "per_image.csv").read_text())

    def test_validation_loss_weights_last_batch(self):
        dataset = TinyDataset()
        criterion = nn.BCEWithLogitsLoss()
        expected = criterion(dataset.logits.to(DEVICE), dataset.targets.to(DEVICE)).item()
        actual = run_epoch(nn.Identity(), DataLoader(dataset, batch_size=2), criterion, DEVICE)
        self.assertAlmostEqual(actual, expected, places=6)

    def test_new_metadata_and_preserved_defaults(self):
        config = TrainingConfig()
        self.assertEqual(config.loss, "bce")
        self.assertEqual(config.freeze_encoder_epochs, 0)
        self.assertEqual(config.patience, 5)
        config.loss = "bce_dice"
        config.dice_weight = 2.0
        model = nn.Conv2d(1, 1, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            summary = save_history(destination, config, "BCEDiceLoss", [], 1, 0.2, 2.0)
            save_checkpoint(destination, model, optimizer, config, 1, 0.2, "BCEDiceLoss", 2.0)
            checkpoint = torch.load(destination / "best.pt", weights_only=True)
            for record in (summary, checkpoint):
                self.assertEqual(record["loss"], "BCEDiceLoss")
                self.assertEqual(record["config"]["loss"], "bce_dice")
                self.assertEqual(record["dice_weight"], 2.0)
                self.assertEqual(record["selection_metric"], "val_loss")

    def test_saved_split_integrity(self):
        result = check_splits()
        self.assertEqual(result["sample_counts"], {"train": 2944, "val": 736, "test": 3669})
        self.assertTrue(result["train_val_disjoint"])

    def test_trimap_mapping_and_transformed_binary_values(self):
        class FixtureOxford:
            def __init__(self, **kwargs):
                self._images = [Path(f"fixture_{index}.jpg") for index in range(3680)]

            def __getitem__(self, index):
                image = Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8))
                trimap = Image.fromarray(np.array([[1, 2, 3], [3, 1, 2]], dtype=np.uint8))
                return image, trimap

        with patch("src.dataset.OxfordIIITPet", FixtureOxford):
            dataset = PetSegDataset("unused", "val")
            _, target = dataset[0]
            torch.testing.assert_close(target, torch.tensor([[[1., 0., 0.], [0., 1., 0.]]]))
            self.assertEqual(dataset.image_id(0), f"fixture_{dataset.indices[0]}")
            for transform in (get_val_transform(), get_train_transform()):
                dataset.transform = transform
                image, mask = dataset[0]
                self.assertEqual(image.shape, (3, 256, 256))
                self.assertEqual(mask.shape, (1, 256, 256))
                self.assertEqual(mask.dtype, torch.float32)
                self.assertEqual(set(mask.unique().tolist()), {0., 1.})


if __name__ == "__main__":
    print(f"Numerical checks device: {DEVICE}")
    unittest.main()
