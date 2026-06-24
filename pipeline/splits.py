"""Dataset splitting for k-fold cross-validation.

Creates and manages ``splits_final.json`` following the nnUNet convention.
"""
from __future__ import annotations

import json
import logging
import os
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


def generate_crossval_split(
    train_identifiers: List[str],
    seed: int = 12345, # Default seed used in nnUNet for reproducibility
    n_splits: int = 5,
) -> List[dict]:
    """Generate reproducible k-fold cross-validation splits.

    Ported from ``nnunetv2/utilities/crossval_split.py``.
    """
    splits = []
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, val_idx in kfold.split(train_identifiers):
        train_keys = np.array(train_identifiers)[train_idx]
        val_keys = np.array(train_identifiers)[val_idx]
        splits.append({"train": train_keys.tolist(), "val": val_keys.tolist()})
    return splits


def generate_or_load_splits(
    dataset_dir: str,
    preprocessed_folder: str,
    n_splits: int = 5,
    seed: int = 12345,
) -> List[dict]:
    """Load existing splits or create new ones.

    Parameters
    ----------
    dataset_dir : str
        Root dataset directory (e.g. ``data/spleen``).
    preprocessed_folder : str
        Path to the preprocessed data folder containing ``.npz`` files.
    n_splits : int
        Number of folds.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list of dict
        Each dict has ``'train'`` and ``'val'`` keys with lists of case
        identifiers.
    """
    splits_file = os.path.join(dataset_dir, "splits_final.json")

    if os.path.isfile(splits_file):
        logger.info("Loading existing splits from %s", splits_file)
        with open(splits_file) as fh:
            splits = json.load(fh)
        logger.info("Loaded %d folds from splits file.", len(splits))
        return splits

    # Discover case identifiers from preprocessed .npz files
    identifiers = sorted(
        f[:-4] # Remove .npz extension
        for f in os.listdir(preprocessed_folder)
        if f.endswith(".npz")
    )
    if len(identifiers) == 0:
        raise FileNotFoundError(
            f"No .npz files found in {preprocessed_folder}. "
            "Run preprocessing first."
        )

    logger.info(
        "Creating new %d-fold cross-validation split from %d cases...",
        n_splits,
        len(identifiers),
    )
    splits = generate_crossval_split(identifiers, seed=seed, n_splits=n_splits)

    with open(splits_file, "w") as fh:
        json.dump(splits, fh, indent=2)
    logger.info("Saved splits to %s", splits_file)

    return splits


def get_fold_keys(
    splits: List[dict],
    fold,
) -> Tuple[List[str], List[str]]:
    """Return (train_keys, val_keys) for the requested fold.

    Parameters
    ----------
    splits : list of dict
        As returned by :func:`generate_or_load_splits`.
    fold : int or str
        Fold index (0-based) or ``"all"`` to use every case for both
        training and validation.

    Returns
    -------
    (train_keys, val_keys)
    """
    if fold == "all":
        all_keys = sorted(
            set(
                k
                for s in splits
                for k in s["train"] + s["val"]
            )
        )
        return all_keys, all_keys

    fold = int(fold)
    if fold < len(splits):
        tr = splits[fold]["train"]
        val = splits[fold]["val"]
        logger.info(
            "Fold %d: %d training cases, %d validation cases.",
            fold, len(tr), len(val),
        )
        return tr, val

    # Fallback: requested fold not in file → random 80/20
    logger.warning(
        "Fold %d not in splits (only %d folds). "
        "Creating random 80:20 split (seed=%d).",
        fold, len(splits), 12345 + fold,
    )
    all_keys = sorted(
        set(k for s in splits for k in s["train"] + s["val"])
    )
    rng = np.random.RandomState(seed=12345 + fold)
    idx_tr = rng.choice(len(all_keys), int(len(all_keys) * 0.8), replace=False)
    idx_val = [i for i in range(len(all_keys)) if i not in idx_tr]
    return [all_keys[i] for i in idx_tr], [all_keys[i] for i in idx_val]
