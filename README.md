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
│   └── model.py             # U-Net (to be added)
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

## Reproducibility

- Fixed random seed (42) for the train/val split.
- Split indices are stored as plain text files and committed to the repository.
- Dataset download and preprocessing are fully scripted.

