# Evaluation

The evaluation layer is framework independent. It reads two CSV files — the ground truth published with each dataset and the predictions produced by your model — and writes the performance metrics. Nothing about how you trained the model matters, so you can benchmark a model written in any framework.

## Running an evaluation

```bash
python -m ultrabench.evaluate task=<classification|segmentation> \
  ground_truth=<path to ground truth CSV> \
  predictions=<path to predictions CSV>
```

Results are written to the Hydra run directory:

| File | Task | Contents |
| ---- | ---- | -------- |
| `summary.csv` | Both | One row per metric, with `metric` and `value` columns. |
| `per_image_metrics.csv` | Segmentation | The Dice score of each (image, class) pair. |
| `per_sequence_predictions.csv` | Classification | The pooled prediction of each sequence, written only when `aggregate_by` is set. |

## Ground-truth CSVs

Each dataset publishes its ground truth alongside the images on the Hugging Face Hub. Evaluation is test-only, so the files cover the two test splits: `test-ground-truth-<task>.csv` describes the original-resolution `test` split, and `test.256-ground-truth-<task>.csv` describes the `test.256` split. A dataset with subsets prefixes the subset name, as in `segmentation-test.256-ground-truth-segmentation.csv`.

Score your predictions against the split you predicted on. A segmentation CSV encodes its masks at the resolution of its own split, so scoring `test.256` predictions against the full-resolution ground truth gives a near-zero Dice score rather than an error.

| Column | Task | Description |
| ------ | ---- | ----------- |
| `image_id` | Both | The image's `filename` value in the dataset. |
| `label` | Classification | The integer class index. |
| `sequence_id` | Classification | The sequence the image belongs to. Present only for tasks with sequence-level labels. |
| `class_id` | Segmentation | The foreground class index. Background (0) is omitted. |
| `encoded_pixels` | Segmentation | The run-length encoded binary mask for that class, or an empty string. |

### Revisions before 0.4

Do not use the ground truth published with revisions `0.1` to `0.3`. Two defects make those files unusable: the segmentation CSVs encode full-resolution masks despite their `test.256` name, and CAMUS publishes an un-prefixed classification CSV that predates the `sequence_id` column. Revision `0.4` fixes both.

For an experiment pinned to an older revision, rebuild the ground truth from the dataset's `test.256` split instead:

```bash
python -m ultrabench.evaluate.ground_truth output_dir=<dir>
```

This writes one `<task>.csv` per task. Pass `tasks='[tn3k_segmentation]'` to build a subset, and `overwrite=true` to rebuild files that already exist. The ground truth is shared by every model, so build it once and point each evaluation at it. This command is retired once every task is pinned to `0.4` or later.

## Prediction CSVs

Write one row per test image, using the same `image_id` values as the ground truth. Images missing from your predictions are logged as warnings and excluded from the metrics.

Classification:

| Column | Description |
| ------ | ----------- |
| `image_id` | The image identifier. |
| `label` | The predicted class index. |
| `prob_{k}` | The predicted probability of class `k`, one column per class. Required for sequence-level evaluation. |

```csv
image_id,label,prob_0,prob_1
41_0,0,0.82,0.18
41_1,1,0.34,0.66
```

Segmentation:

| Column | Description |
| ------ | ----------- |
| `image_id` | The image identifier. |
| `class_id` | The foreground class index. |
| `encoded_pixels` | The run-length encoded binary mask for that class. |

Run-length encoding is Kaggle style: 1-indexed, column-major, and space separated pairs of start position and run length. An all-zero mask encodes to an empty string. Use `ultrabench.core.rle.mask_to_rle` and `rle_to_mask` to convert.

## Sequence-level tasks

Some datasets label a whole sequence rather than an individual image. CAMUS grades the quality of a `(patient, view)` acquisition, COVID-BLUES records a PCR result per patient, and MASLD records a diagnosis per patient. Scoring these per image would weight each sequence by its frame count and overstate the size of the test set, so they are scored on one prediction per sequence.

| Task | Aggregation unit | Test sequences |
| ---- | ---------------- | -------------- |
| CAMUS image quality | `(patient, view)` | 100 |
| COVID-BLUES COVID-19 | `patient` | 20 |
| MASLD | `patient` | 11 |

Pool the predictions by averaging the predicted probabilities over every image in the sequence, then take the argmax. Pass `aggregate_by=sequence_id` to do this:

```bash
python -m ultrabench.evaluate task=classification aggregate_by=sequence_id \
  ground_truth=test-ground-truth-classification.csv \
  predictions=predictions.csv
```

Your predictions CSV stays per image. The evaluator reads the grouping from the ground truth's `sequence_id` column, so you do not need to know how the sequences are laid out. You do need the `prob_{k}` columns, since probabilities are averaged rather than votes counted.

The training layer reports the same metric inline. Set `sequence_by` in the task configuration (`src/ultrabench/training/config/task/*.yaml`) to the dataset columns that identify a sequence, and `test_metrics.csv` gains `seq_`-prefixed metrics next to the per-image ones.

## Metrics

| Task | Metrics |
| ---- | ------- |
| Classification | Balanced accuracy. |
| Segmentation | Positive Dice over the foreground classes, and the mean Dice of each class. |

### The positive Dice convention

The positive Dice score excludes the background class, and follows two further conventions. Both match `torchmetrics.segmentation.DiceScore(include_background=False)`, which the training layer uses for the `positive_dice` column of `test_metrics.csv`, so the evaluator reproduces that column exactly.

- A class that is empty in both the ground truth and the prediction is undefined, and drops out of the score. Scoring it 1.0 instead rewards a model for predicting nothing on an image the class is absent from, which inflates the score on tasks where many images hold no instance of a class. An image whose every foreground class is undefined drops out of the score entirely.
- The score averages the defined foreground classes of an image first, and the images second. Averaging per class first gives a different number, since the undefined pairs leave the per-class groups unequal in size.

The undefined pairs are still written to `per_image_metrics.csv`, with an empty `dice` field.
