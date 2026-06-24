# Pipeline

## Overview

The `pipeline/` package sits between the Hydra entry point (`main2.py`) and the lower-level training, validation, and testing modules. It is responsible for:

- Wiring together plans, splits, trainer construction, and validation into a single `ExperimentRunner`
- Managing 5-fold cross-validation splits (`splits.py`)
- Optional VRAM sanity-check before training starts (`vram_estimation.py`)

---

## File Map

| File | Responsibility |
|---|---|
| `experiment.py` | `ExperimentRunner` — top-level command dispatcher |
| `splits.py` | k-fold split generation and loading |
| `vram_estimation.py` | Dummy forward/backward VRAM check |

---

## `ExperimentRunner` (`experiment.py`)

The single orchestrator object created by `main2.py` for `command=train`, `command=validate`, and `command=test`.

### Construction

```python
ExperimentRunner(cfg)
```

On construction the runner:

1. Loads `experiment_plans.json` and `raw/dataset.json` from `cfg.data.dataset_dir`.
2. Validates that `cfg.configuration` exists in the plans.
3. Enforces that models with `requires_isotropic_preprocessing=true` (e.g. `mednext`, `medmoenext`) are paired with an `_iso` configuration.
4. Derives `self.preprocessed_folder` from the `data_identifier` stored in the selected plan configuration.
5. Calls `generate_or_load_splits` so `self.num_folds` is available for all commands.
6. Computes `self.experiments_base`:

```
experiments/{data.name}/{model.name}_{configuration}/{experiment_name}/
```

### `run_train()`

Resolves which folds to train from `cfg.fold` and `cfg.run_all_folds`, then calls `_train_one_fold` for each. When multiple folds complete, logs cross-validation mean ± std of best EMA Dice.

#### `_train_one_fold(fold)`

For a single fold:

1. Resolves train/val keys from `self.splits`.
2. Constructs `BioAITrainer` and calls `trainer.initialize(...)`.
3. Optionally runs `check_vram_and_adjust` (when `training.vram_check=true`) — raises if the model does not fit even at `min_batch_size=2`.
4. Optionally initialises Weights & Biases (`logging.wandb_logging.enabled=true`).
5. Calls `trainer.run_training()`.
6. Immediately after training completes, runs `perform_full_validation` on the validation split using the trained network — results saved to `fold_{N}/validation/summary.json`.
7. Finishes the W&B run.

Output folder: `experiments_base/fold_{N}/`.

### `run_validate()`

Standalone validation for one or all folds from saved checkpoints.

- If all `num_folds` checkpoints exist and `cfg.fold != "all"`, calls `_run_cross_validation()` to aggregate across all folds.
- Otherwise calls `_validate_one_fold(fold)`.

#### `_validate_one_fold(fold)`

Rebuilds the network (deep supervision disabled), loads `checkpoint_best.pth` (fallback: `checkpoint_final.pth`), and runs `perform_full_validation`. Results written to `fold_{N}/validation/summary.json`.

#### `_run_cross_validation()`

Collects `validation/summary.json` from every fold (running validation on-demand if a summary is missing). Aggregates all metrics with mean ± std across folds:

- Foreground: Dice, IoU, Sensitivity, Precision, HD95 (optional), Surface Dice / NSD (optional)
- Per-class: same metrics

Output: `experiments_base/cross_validation_summary.json`.

```json
{
  "folds": 5,
  "foreground": {
    "foreground_mean_dice": {"mean": 0.91, "std": 0.02, "per_fold": [...]},
    ...
  },
  "per_class": {
    "1": {"Dice": {"mean": 0.91, "std": 0.02, ...}, ...}
  }
}
```

### `run_test()`

Delegates to `testing.test_runner.run_testing`, then optionally calls `testing.postprocessing.run_postprocessing` when `postprocessing.enabled=true`. The raw data directory is forwarded as the CRF bilateral image source.

---

## Cross-Validation Splits (`splits.py`)

### `generate_or_load_splits`

Looks for `{dataset_dir}/splits_final.json`. If found, loads and returns it. Otherwise:

1. Discovers case identifiers by scanning `{preprocessed_folder}/*.npz`.
2. Calls `generate_crossval_split` — sklearn `KFold(n_splits=5, shuffle=True, random_state=12345)` — matching the nnUNet default seed for reproducibility.
3. Saves to `splits_final.json`.

The file is stored at the dataset root (not the preprocessed subfolder) so it is shared across configurations.

### `get_fold_keys(splits, fold)`

Returns `(train_keys, val_keys)` for a given fold index.

| `fold` value | Behaviour |
|---|---|
| Integer (0–4) | Returns the corresponding split entry |
| `"all"` | All cases in both train and val (no held-out set) |
| Out-of-range integer | Falls back to a random 80/20 split seeded at `12345 + fold` |

### `splits_final.json` format

```json
[
  {"train": ["case_001", "case_002", ...], "val": ["case_005", ...]},
  ...
]
```

---

## VRAM Estimation (`vram_estimation.py`)

`check_vram_and_adjust(model, patch_size, batch_size, num_input_channels, device)` runs a dummy forward + backward pass under `autocast` to verify that the configured batch size and patch size fit in GPU memory.

### Behaviour

1. **Analytical pre-check** (nnUNet models only): uses `model.compute_conv_feature_map_size(patch_size)` to estimate activation memory before attempting the actual pass.
2. **Empirical check**: creates a random `(batch_size, C, *patch_size)` tensor, runs the full forward + backward, and reads `torch.cuda.max_memory_allocated`.
3. **Batch-size halving**: if OOM at `batch_size`, retries at `batch_size // 2` down to `min_batch_size=2`.
4. **Patch-size suggestion**: if even `min_batch_size` fails, calls `_next_smaller_patch` to suggest the next valid smaller patch (preserving aspect ratio and divisibility by `div_by`).

Returns `(fits: bool, recommended_batch_size: int, recommended_patch_size: list)`.

Only executed when `training.vram_check=true` (opt-in, disabled by default since it adds startup latency).

---

## Experiment Output Layout

```
experiments/
  {data.name}/
    {model.name}_{configuration}/
      {experiment_name}/
        fold_0/
          checkpoint_best.pth
          checkpoint_latest.pth      ← deleted on completion
          checkpoint_final.pth
          plans.json                 ← copy for reproducibility
          dataset.json
          config.json
          run_log.txt                ← pointer to Hydra log
          validation/
            summary.json
            {case_id}.nii.gz         ← optional exported predictions
        fold_1/ ... fold_4/
        test_ensemble/
          {case_id}.nii.gz
          summary.json
        test_ensemble_postprocessed/
          {case_id}.nii.gz
        cross_validation_summary.json
```
