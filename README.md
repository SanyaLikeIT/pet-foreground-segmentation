# Pet Foreground Segmentation with U-Net

Binary segmentation of pets (cats and dogs) from the background on the
Oxford-IIIT Pet dataset. Stage 2 evaluates the existing BCE baseline. BCE
versus BCE + Dice comparisons and sensitivity experiments are later work.

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
│   ├── losses.py            # BCE, per-image soft Dice, and BCE + Dice
│   ├── metrics.py           # foreground IoU/Dice and pixel accuracy
│   ├── evaluate.py          # checkpoint evaluation to JSON and CSV
│   ├── inspect_baseline.py  # safe checkpoint/history/device audit
│   ├── validate_data.py     # split integrity and binary mask checks
│   ├── stage2_report.py     # original loss curve and validation errors
│   ├── tune.py              # hyperparameter search
│   └── visualize_predictions.py  # checkpoint visualization
├── report/
│   ├── example_train.png
│   ├── example_val.png
│   └── stage2/              # new evidence; previous artifacts are preserved
├── tests/test_stage2.py     # numerical and aggregation checks
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/SanyaLikeIT/pet-foreground-segmentation.git
cd pet-foreground-segmentation
```

Use the project virtual environment for all commands:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The verified Stage 2 environment uses Python 3.14.7, PyTorch 2.14.1+cu130,
torchvision 0.29.1+cu130, and albumentations 2.0.8. Its installed packages are
recorded in `report/stage2/environment.txt`. The GPU build was installed with:

```bash
python -m pip install --force-reinstall torch==2.14.1+cu130 torchvision==0.29.1+cu130 --index-url https://download.pytorch.org/whl/cu130
```

Evaluation ran on an NVIDIA GeForce RTX 4060 Laptop GPU (8 GB). In the Codex
sandbox, NVIDIA devices are hidden; CUDA commands must run outside that sandbox.
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

These commands are for future training, **not needed for Stage 2 evaluation**.
Use a new output directory to preserve the archived baseline:

```bash
python -m src.train --output-dir runs/new_baseline
```

The architecture is the custom torchvision-based U-Net in `src/model.py`,
with an ImageNet-pretrained ResNet-34 encoder, five decoder blocks, skip
connections, and a one-channel convolutional head. It is **not** a
segmentation-models-pytorch model. It produces raw logits at 256 x 256.

The default `--loss bce` reuses `torch.nn.BCEWithLogitsLoss`. The optional
`--loss bce_dice --dice-weight 1.0` computes BCE + lambda * soft Dice loss.
For each image, soft Dice is `(2*sum(sigmoid(logits)*mask) + 1e-6) /
(sum(sigmoid(logits)) + sum(mask) + 1e-6)`; loss is one minus this score,
averaged over images. The smoothing constant is `1e-6`; predictions are never
thresholded in the loss. `dice_weight` must be finite and nonnegative.

The `train_model` function still accepts a `criterion`
argument for experiments with other loss functions. The criterion receives
raw logits and binary masks of the same shape.

## Checkpoints

The best checkpoint is saved to `runs/baseline/best.pt`. The per-epoch losses,
configuration, and best epoch are saved to `runs/baseline/history.json`.

The archived baseline was selected by **minimum validation BCE**, not Dice.
Its verified settings are: AdamW, learning rate 0.0003, weight decay 0.0001,
batch size 8, seed 42, zero frozen-encoder epochs, pretrained encoder, zero
data-loader workers, 20 requested epochs, and early-stopping patience 5.
The best epoch is **11**, validation BCE **0.10701909580308458**; history ends
at epoch **16**, consistent with early stopping. `device=auto` was recorded;
the original training hardware and runtime were not recorded.

Defaults and checkpoint selection are preserved: each new run minimizes its
chosen validation loss. New metadata records the configured loss, effective
criterion class, Dice weight (zero for plain BCE), and `selection_metric=val_loss`.
Existing checkpoint and history files are not migrated or rewritten.

## Train examples

```bash
python -m src.train --output-dir runs/new_baseline --epochs 20 --batch-size 8 --learning-rate 0.0003 --weight-decay 0.0001 --freeze-encoder-epochs 0 --seed 42 --patience 5 --loss bce
```

### Additional flags
- `--random-encoder`. Starts training without pretrained weights.
- `--patience 0` disables early stopping.

## Hyperparameter tuning

The existing tuning script supports LR, batch size and weight decay. It was
not run for Stage 2; its commands below are for later milestones.

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

## Stage 2: baseline validation evidence

Run from the repository root using the activated `.venv`. No training is involved:

```bash
python -m src.inspect_baseline --checkpoint runs/baseline/best.pt --history runs/baseline/history.json --device cuda --output report/stage2/baseline_audit.json
python -m src.validate_data --data-root data/raw --output report/stage2/data_check.json
python -m unittest discover -s tests -v
python -m src.evaluate --data-root data/raw --checkpoint runs/baseline/best.pt --split val --device cuda --batch-size 8 --check-batch-size 3 --disable-tf32 --output report/stage2/val_fp32
python -m src.stage2_report --history runs/baseline/history.json --metrics-dir report/stage2/val_fp32 --data-root data/raw --checkpoint runs/baseline/best.pt --device cuda --count 4 --poor-count 3 --output report/stage2
```

The audit uses `torch.load(..., weights_only=True)` and strict model-state
loading. Restoration instantiates `UNet(pretrained_encoder=False)`, so it
does not download ImageNet weights. `--data-root` is the directory containing
`oxford-iiit-pet`; saved split files remain those in this repository.
For a CPU-only environment use `--device cpu`; explicit `cuda` never silently
falls back to CPU. Evaluation also accepts `auto` and `cuda:N`.

Verified **validation results**, 736 images, checkpoint epoch 11, threshold 0.5:

| Metric | Value | Aggregation |
| --- | ---: | --- |
| Foreground IoU | 0.852994681 | Arithmetic mean of per-image IoU |
| Foreground Dice | 0.915092323 | Arithmetic mean of per-image Dice |
| Pixel accuracy | 0.957708048 | Total correct pixels / total pixels |

Evaluation uses `model.eval()` and `torch.inference_mode()`, with
`sigmoid(logits) >= 0.5`. IoU is TP/(TP+FP+FN); Dice is
2TP/(2TP+FP+FN). Both masks empty gives IoU=Dice=1; only one empty gives 0.
No metric smoothing is used. Image scores are aggregated over all images,
including incomplete final batches; batch means are never averaged equally.

`metrics.json` records the checkpoint path, SHA-256 and epoch, split, sample
count, threshold, preprocessing, device, precision and metric definitions.
`per_image.csv` includes stable Oxford image identifiers, image scores,
pixel counts and hashes of the actual preprocessed inputs/masks.
The second pass is saved under `val_fp32/batch_size_3/`.

The first evaluation is preserved in `report/stage2/val/`. Its batch-size
check exceeded the per-image tolerance (maximum score difference 0.001048308;
aggregate differences below 0.000004). Exactly one full comparison was
repeated with TF32 disabled for CUDA matmul and cuDNN convolutions. Batch
sizes 8 and 3 then produced identical input hashes, per-image metrics and
aggregate results. The original tolerances remain 1e-5 for aggregate metrics
and 1e-3 for per-image IoU/Dice. This verifies the two tested batch sizes on
this GPU; it does not guarantee identical floating-point results everywhere.

Evidence and the English results note are in [report/stage2/results.md](report/stage2/results.md).
Figures show image, ground truth, prediction, and errors: red = false
positive, cyan = false negative, green = true positive. Four validation
examples use seed 42. Three worst predictions are selected by foreground
IoU, with ties broken by image identifier. Their original trimaps contain
only background; they are retained in all aggregate results. A separate
figure shows the three worst cases with nonempty targets for interpretation.
Previous `report/*.png`, `best.pt`, `history.json`, and saved splits are intact.
No test inference, retraining, tuning or sensitivity experiments were run.

If data are missing, `python -m src.validate_data --splits-only` checks the
saved indices, and `python -m src.stage2_report` can still plot the original
history. Full evaluation requires the images, trimaps and official
`annotations/trainval.txt` (or `annotations/test.txt` for explicit final test
evaluation), plus a compatible complete checkpoint. Use a new output path
when rerunning evidence to retain earlier results.

## Later loss comparison and sensitivity experiments

Plan BCE versus BCE+Dice and lambda in **{0.5, 1.0, 2.0}**. Use the same
saved splits, initialization seed (42), pretrained weights, transformations,
optimizer, training settings, evaluation threshold and checkpoint-selection
policy. Before these runs, implement a common selection metric, for example
validation BCE for every objective, matching the archived baseline. Currently
`--loss bce_dice` selects by combined validation loss; Stage 2 deliberately
preserves this existing loss-based selection behavior.

The future objective can be configured with, for example:

```bash
python -m src.train --output-dir runs/later_bce_dice_lambda1 --loss bce_dice --dice-weight 1.0 --epochs 20 --batch-size 8 --learning-rate 0.0003 --weight-decay 0.0001 --freeze-encoder-epochs 0 --seed 42 --patience 5 --device cuda
```

This command has **not** been executed and should wait for the common
selection policy. Compare approaches using the same validation IoU/Dice
metrics; do not select a winner by comparing raw BCE and raw BCE+Dice loss
values, which have different scales. Reserve test evaluation for the final
fixed experiment design. No loss-comparison or tuning results are claimed.
