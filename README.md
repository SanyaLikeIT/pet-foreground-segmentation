# Pet Foreground Segmentation with U-Net

Binary segmentation of pets (cats and dogs) from the background on the
Oxford-IIIT Pet dataset. Stage 2 evaluates the BCE baseline. The
second approach compares it with BCE + Dice (lambda=1) on validation;
see [Second approach](#second-approach-bce--dice).

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
│   ├── inspect_baseline.py  # checkpoint, history and device inspection
│   ├── validate_data.py     # split integrity and binary mask checks
│   ├── stage2_report.py     # original loss curve and validation errors
│   ├── second_approach_report.py  # matched BCE / BCE+Dice validation comparison
│   ├── tune.py              # hyperparameter search
│   └── visualize_predictions.py  # checkpoint visualization
├── report/
│   ├── example_train.png
│   ├── example_val.png
│   ├── stage2/              # first working model evidence
│   └── second_approach/     # BCE / BCE+Dice validation evidence
├── tests/test_stage2.py     # numerical and aggregation checks
├── tests/test_second_approach.py  # common selection and paired evidence checks
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

On Windows, activate the environment with `.venv\Scripts\activate` in
Command Prompt or `.venv\Scripts\Activate.ps1` in PowerShell.

The verified Stage 2 environment uses Python 3.14.7, PyTorch 2.14.1+cu130,
torchvision 0.29.1+cu130, and albumentations 2.0.8. Its installed packages are
recorded in `report/stage2/environment.txt`. The GPU build was installed with:

```bash
python -m pip install --force-reinstall torch==2.14.1+cu130 torchvision==0.29.1+cu130 --index-url https://download.pytorch.org/whl/cu130
```

Evaluation ran on an NVIDIA GeForce RTX 4060 Laptop GPU (8 GB). CUDA commands
require a GPU-enabled execution environment.

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

Train a model with:

```bash
python -m src.train --output-dir runs/new_baseline
```

The architecture is the custom torchvision-based U-Net in `src/model.py`,
with an ImageNet-pretrained ResNet-34 encoder, five decoder blocks, skip
connections, and a one-channel convolutional head producing raw logits at
256 x 256.

The default `--loss bce` uses `torch.nn.BCEWithLogitsLoss`. The optional
`--loss bce_dice --dice-weight 1.0` computes BCE + lambda * soft Dice loss.
For each image, soft Dice is `(2*sum(sigmoid(logits)*mask) + 1e-6) /
(sum(sigmoid(logits)) + sum(mask) + 1e-6)`; loss is one minus this score,
averaged over images. The smoothing constant is `1e-6`; predictions are never
thresholded in the loss. `dice_weight` must be finite and nonnegative.

The `train_model` function accepts a `criterion`
argument for experiments with other loss functions. The criterion receives
raw logits and binary masks of the same shape.

## Checkpoints

By default, the best checkpoint is saved to `runs/baseline/best.pt`. The
per-epoch losses, configuration, and best epoch are saved to
`runs/baseline/history.json`. `--output-dir` changes this directory. Choose a
new directory for each run: training refuses to overwrite existing artifacts
and does not support resuming interrupted runs.

The baseline was selected by **minimum validation BCE**.
Its verified settings are: AdamW, learning rate 0.0003, weight decay 0.0001,
batch size 8, seed 42, zero frozen-encoder epochs, pretrained encoder, zero
data-loader workers, 20 requested epochs, and early-stopping patience 5.
The best epoch is **11**, validation BCE **0.10701909580308458**; history ends
at epoch **16**, consistent with early stopping. `device=auto` was recorded;
the original training hardware and runtime were not recorded.

The default selection policy minimizes the chosen validation loss.
`--selection-metric val_bce` instead uses a common validation BCE for checkpoint
selection and early stopping, independently of the training objective. Each
epoch also records validation BCE, soft Dice loss, hard IoU/Dice at threshold
0.5 and elapsed time. `--disable-tf32` disables TF32 for both CUDA matmul and
cuDNN convolutions. Training metadata records the configured loss, effective
criterion class, Dice weight (zero for plain BCE), selection criterion and
its value. A separate `run_info.json` records initialization, source/split
hashes, Git revision, device, library versions and precision flags.

## Train examples

```bash
python -m src.train --output-dir runs/new_baseline --epochs 20 --batch-size 8 --learning-rate 0.0003 --weight-decay 0.0001 --freeze-encoder-epochs 0 --seed 42 --patience 5 --loss bce
```

### Additional flags

- `--random-encoder`. Starts training without pretrained weights.
- `--patience 0` disables early stopping.

## Hyperparameter tuning

The tuning script searches over learning rate, batch size and weight decay:

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

Evaluate the saved baseline from the repository root using the activated `.venv`:

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

The initial evaluation in `report/stage2/val/` exceeded the per-image
batch-size tolerance (maximum score difference 0.001048308; aggregate
differences below 0.000004). With TF32 disabled for CUDA matmul and cuDNN, batch
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

If data are missing, `python -m src.validate_data --splits-only` checks the
saved indices, and `python -m src.stage2_report` can still plot the original
history. Full evaluation requires the images, trimaps and official
`annotations/trainval.txt` (or `annotations/test.txt` for explicit final test
evaluation), plus a compatible complete checkpoint.

## Second approach: BCE + Dice

The second approach changes the training objective to
`BCEWithLogitsLoss + 1.0 * SoftDiceLoss`. The custom U-Net, ImageNet-pretrained
ResNet-34 encoder, saved splits, transforms, AdamW settings, batch size 8 and
seed 42 match the archived baseline. Training starts from pretrained encoder
weights and a newly initialized decoder, not from the trained BCE checkpoint.
Both approaches use minimum validation BCE for checkpoint selection and early
stopping (20 maximum epochs, patience 5). Lambda=1 was fixed before training.
The comparison uses the validation split.

Activate `.venv` and run from the repository root on a CUDA-enabled machine:

```bash
python -m unittest discover -s tests -v
python -m src.validate_data --data-root data/raw --output report/second_approach/data_check.json
python -u -m src.train --output-dir runs/second_approach/bce_dice_lambda1 --loss bce_dice --dice-weight 1.0 --selection-metric val_bce --disable-tf32 --epochs 20 --batch-size 8 --learning-rate 0.0003 --weight-decay 0.0001 --freeze-encoder-epochs 0 --num-workers 0 --seed 42 --patience 5 --device cuda
python -m src.evaluate --checkpoint runs/baseline/best.pt --split val --device cuda --batch-size 8 --threshold 0.5 --disable-tf32 --output report/second_approach/baseline/val
python -m src.evaluate --checkpoint runs/second_approach/bce_dice_lambda1/best.pt --split val --device cuda --batch-size 8 --check-batch-size 3 --threshold 0.5 --disable-tf32 --output report/second_approach/bce_dice_lambda1/val
python -m src.second_approach_report --device cuda
```

To generate the comparison report from saved evaluations and checkpoints on CPU:

```bash
python -m src.second_approach_report --device cpu
```

For a rerun, choose a fresh training output directory and pass the corresponding
checkpoint, history and evaluation paths to the evaluation/report commands.
The report command checks validation-only evidence, matching protocols,
identical input/target hashes, aggregate metrics against CSV rows, completed
runs and common BCE selection. It saves comparison tables, paired per-image
changes, learning curves, and predictions for the largest Dice increase,
largest decrease and smallest absolute change among nonempty targets. A
second figure revisits three previously documented baseline failures.

Results and limitations are recorded in
[report/second_approach/results.md](report/second_approach/results.md).
Histories and run metadata accompany the evaluation results. Generating the
figures requires both checkpoints: `runs/baseline/best.pt` and
`runs/second_approach/bce_dice_lambda1/best.pt`. Checkpoints are stored locally
and are not included in Git.

The comparison uses one seed. The baseline training environment and runtime
were not fully recorded, which limits reproducibility. Global IoU/Dice measure
foreground overlap; boundary quality requires a dedicated metric. Lambda
sensitivity and final test evaluation have not been measured.
