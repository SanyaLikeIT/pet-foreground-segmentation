# Pet Foreground Segmentation with U-Net

Binary segmentation of pets (cats and dogs) from the background on the
Oxford-IIIT Pet dataset. We train a U-Net with a pretrained encoder and
compare two loss functions: BCE vs. BCE + Dice.

## Task

Each pixel is classified as **pet** (foreground) or **non-pet** (background).
The original trimap has three classes (1 = pet, 2 = background, 3 = boundary);
we merge background and boundary into a single background class.

## Repository structure

```
pet_seg/
├── data/
│   ├── raw/                 # downloaded dataset (not in git)
│   └── splits/              # train.txt, val.txt, test.txt
├── src/
│   ├── prepare_data.py      # download dataset and create splits
│   ├── dataset.py           # PetSegDataset + transforms
│   ├── check_dataset.py     # sanity check and visualizations
│   ├── model.py             # U-Net with a ResNet-34 encoder
│   ├── train.py             # training and validation
│   ├── tune.py              # hyperparameter search
│   └── visualize_predictions.py  # checkpoint visualization
├── report/
│   ├── example_train.png
│   └── example_val.png
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/SanyaLikeIT/pet-foreground-segmentation.git
cd pet-foreground-segmentation
pip install -r requirements.txt
```
## Data preparation

Download the dataset and create the splits:

```bash
python src/prepare_data.py
```

This produces:

```
data/raw/oxford-iiit-pet/
├── images/                  # .jpg images
└── annotations/trimaps/     # .png masks with values 1, 2, 3

data/splits/
├── train.txt                # 2944 indices from official trainval
├── val.txt                  # 736 indices from official trainval
└── test.txt                 # 3669 indices from official test
```

The dataset itself is **not** stored in the repository. It is downloaded
automatically by the script.

### Splits

- **train** — 80% of the official `trainval` split, seed 42.
- **val** — 20% of the official `trainval` split, seed 42.
- **test** — official `test` split, used only for final evaluation.

Train and val are disjoint subsets of `trainval`. The test set is a separate
official split and is never mixed with train/val.

## Dataset check

Run the sanity check from the project root:

```bash
python -m src.check_dataset
```

It saves sample visualizations to `report/example_train.png` and
`report/example_val.png`, and prints the size of each split.

Expected output:

```
train: 2944 samples
val:   736 samples
test:  3669 samples
```

## Preprocessing and augmentations

- All images and masks are resized to **256 x 256**.
- Pixel values are normalized with ImageNet statistics:
  `mean = (0.485, 0.456, 0.406)`, `std = (0.229, 0.224, 0.225)`.
- **Train:** random horizontal flip (p = 0.5) + resize + normalization.
- **Val / test:** resize + normalization only.

Masks are binary `float32` tensors of shape `(1, 256, 256)` with values `{0, 1}`.

## Training

Run the commands from the project root after preparing the data:

```bash
python -m src.train
```

The model uses an ImageNet-pretrained ResNet-34 encoder and produces one logit
per pixel.
# !!! (Поменяйте потом кто этим занимается) Loss:
`BCEWithLogitsLoss` is used as a loss function
The `train_model` function accepts a `criterion`
argument for experiments with other loss functions. The criterion receives
raw logits and binary masks of the same shape.

## Checkpoints

The best checkpoint is saved to `runs/baseline/best.pt`. The per-epoch losses,
configuration, and best epoch are saved to `runs/baseline/history.json`.

## Train examples

```bash
python -m src.train --epochs 20 --batch-size 8 --learning-rate 0.0003 --weight-decay 0.0001 --freeze-encoder-epochs 2
```

### Additional flags
- `--random-encoder`. Starts training without pretrained weights.
- `--patience 0` disables early stopping.

## Hyperparameter tuning

Currently, we imlemented the tuning of LR, batch size and weight decay.

Other parameters from the train function could be used here as well, but they won't support tuning.

This is an example of how to start a hyperparameter tuning.
```bash
python -m src.tune --learning-rates 0.0003 0.0001 --batch-sizes 8 16 --weight-decays 0 0.0001
```

Each configuration trains on `train` and is selected by the lowest `val` loss.

Tuning checkpoints and histories are stored under `runs/tuning/trial_XX/` \
`runs/tuning/summary.json` identifies the best trial.
The full command above runs eight training trials. Reduce the value of
`--epochs` for a quick check.

## Visual inspection

After the model was trained, this script can be used to compare images, reference masks, predicted masks, and overlays.

```bash
python -m src.visualize_predictions --checkpoint runs/baseline/best.pt --split val
```
This saves four randomly selected validation examples to `report/predictions_val.png`. \

The argument `--indices 0 1 2` can be used to inspect specific examples,
`--count 8` , `--seed 7` to change the selection.
The default foreground probability threshold is `0.5` and can be changed with `--threshold`.
`--split test` and `--split val` can be used to pick val or test set.

## Reproducibility

- Fixed random seed (42) for the train/val split.
- Split indices are stored as plain text files and committed to the repository.
- Dataset download and preprocessing are fully scripted.
