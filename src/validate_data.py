"""Check saved split integrity; optionally inspect every transformed train/val mask."""

import argparse
import json
from pathlib import Path


SPLITS_DIR = Path(__file__).resolve().parents[1] / "data" / "splits"


def check_splits(splits_dir=SPLITS_DIR):
    expected = {"train": 2944, "val": 736, "test": 3669}
    splits = {}
    for split, count in expected.items():
        indices = [int(line) for line in (Path(splits_dir) / f"{split}.txt").read_text().splitlines()]
        if len(indices) != count or len(set(indices)) != count:
            raise ValueError(f"{split}: incorrect count or duplicate indices")
        upper = 3669 if split == "test" else 3680
        if any(index < 0 or index >= upper for index in indices):
            raise ValueError(f"{split}: indices outside official split range")
        splits[split] = set(indices)
    if splits["train"] & splits["val"]:
        raise ValueError("train and validation overlap")
    if splits["train"] | splits["val"] != set(range(3680)):
        raise ValueError("train and validation do not cover the official trainval split")
    if splits["test"] != set(range(3669)):
        raise ValueError("test indices do not cover the official test split")
    return {"sample_counts": expected, "train_val_disjoint": True,
            "trainval_complete": True, "test_complete": True,
            "index_namespaces": "Train/val indices address trainval; test indices address test"}


def check_masks(data_root, splits=("train", "val")):
    import numpy as np
    import torch
    from src.dataset import PetSegDataset, get_train_transform, get_val_transform

    checked = {}
    annotations = Path(data_root) / "oxford-iiit-pet" / "annotations"
    official_ids = {
        split: [line.split()[0] for line in (annotations / f"{split}.txt").read_text().splitlines()]
        for split in ("trainval", "test")
    }
    if len(set(official_ids["trainval"])) != 3680 or len(set(official_ids["test"])) != 3669:
        raise ValueError("Official annotations have unexpected counts or duplicate image IDs")
    if set(official_ids["trainval"]) & set(official_ids["test"]):
        raise ValueError("Official trainval and test image identifiers overlap")
    image_ids = set()
    for split in splits:
        transform = get_train_transform() if split == "train" else get_val_transform()
        dataset = PetSegDataset(data_root, split, transform)
        untransformed_dataset = PetSegDataset(data_root, split)
        for index in range(len(dataset)):
            identifier = dataset.image_id(index)
            if identifier in image_ids:
                raise ValueError(f"Duplicate image identifier across checked splits: {identifier}")
            image_ids.add(identifier)
            # Also verify the original trimap values and exact binary mapping.
            _, raw_mask = dataset.dataset[dataset.indices[index]]
            raw_mask = np.asarray(raw_mask)
            if not np.isin(raw_mask, (1, 2, 3)).all():
                raise ValueError(f"Unexpected trimap value: {identifier}")
            _, untransformed = untransformed_dataset[index]
            if not np.array_equal(untransformed[0].numpy(), raw_mask == 1):
                raise ValueError(f"Incorrect trimap-to-binary mapping: {identifier}")
            image, mask = dataset[index]
            if image.shape != (3, 256, 256) or mask.shape != (1, 256, 256):
                raise ValueError(f"Unexpected transformed shape: {identifier}")
            if mask.dtype != torch.float32 or not torch.all((mask == 0) | (mask == 1)):
                raise ValueError(f"Mask is not binary float32: {identifier}")
        checked[split] = len(dataset)
    return {"checked_masks": checked, "binary_values": [0, 1],
            "original_trimap_values": [1, 2, 3], "shape": [1, 256, 256],
            "official_trainval_test_image_ids_disjoint": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--splits-only", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = check_splits()
    if not args.splits_only:
        result.update(check_masks(args.data_root))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
