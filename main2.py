"""Entry point for BioAI medical image segmentation pipeline.

Commands:
  plan        — Extract dataset fingerprint + generate experiment plans
  preprocess  — Preprocess raw data according to experiment plans
  train       — Train a model (single fold, all folds, or fold="all")
  validate    — Run full-volume validation on a trained model
  test        — Run inference on test data
"""
import os
import logging
import warnings
from pathlib import Path

# Install process guard BEFORE any fork / subprocess spawns so that child
# processes inherit PR_SET_PDEATHSIG and cannot outlive the parent — even
# when the parent is killed with SIGKILL.
from pipeline.process_guard import install as _install_process_guard
_install_process_guard()

warnings.filterwarnings("ignore", message="Using a non-tuple sequence for multidimensional indexing")
warnings.filterwarnings("ignore", message=".*always_return_as_numpy.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*cuda.cudart module is deprecated.*", category=FutureWarning)

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")

logging.basicConfig(level=logging.INFO)

# Patch the console StreamHandler so log messages don't break tqdm progress bars.
# tqdm.write() clears the bar, prints the message, and redraws the bar cleanly.
for _handler in logging.getLogger().handlers:
    if isinstance(_handler, logging.StreamHandler) and not isinstance(_handler, logging.FileHandler):
        _orig_emit = _handler.emit
        def _tqdm_emit(record, _orig=_orig_emit):
            try:
                from tqdm import tqdm
                msg = _handler.format(record)
                tqdm.write(msg)
            except Exception:
                _orig(record)
        _handler.emit = _tqdm_emit
        break

logger = logging.getLogger(__name__)


def _seed_everything(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    _seed_everything()
    os.environ["HYDRA_FULL_ERROR"] = "1"

    # Attach FileHandler to root logger so all logger.xxx() calls in every module
    # write to both console (existing StreamHandler) and this log file.
    _hydra_dir = HydraConfig.get().runtime.output_dir
    _fh = logging.FileHandler(os.path.join(_hydra_dir, "run.log"))
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(_fh)

    logger.info("Command: %s", cfg.command)
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    try:
        if cfg.command == "plan":
            _run_plan(cfg)

        elif cfg.command == "preprocess":
            _run_preprocess(cfg)

        elif cfg.command == "train":
            _run_train(cfg)

        elif cfg.command == "validate":
            _run_validate(cfg)

        elif cfg.command == "test":
            _run_test(cfg)

        else:
            raise ValueError(
                f"Unknown command: {cfg.command}. "
                "Use: plan, preprocess, train, validate, test"
            )
    except Exception:
        logger.exception("Fatal error — run aborted")
        raise


# ======================================================================
# Plan
# ======================================================================

def _run_plan(cfg: DictConfig):
    logger.info("Starting planning...")
    from preprocessing.plan import plan_dataset, plan_experiment

    dataset_dir = Path(cfg.data.dataset_dir)
    raw_data_dir = dataset_dir / "raw"
    fp_file = dataset_dir / "dataset_fingerprint.json"

    plan_dataset(
        raw_data_dir=raw_data_dir,
        output_dir=dataset_dir,
        num_workers=int(cfg.preprocessing.num_workers),
    )

    plan_experiment(
        fingerprint_file=fp_file,
        output_dir=dataset_dir,
        patch_size_override=list(cfg.data.roi_size) if "roi_size" in cfg.data else None,
        batch_size=int(cfg.planning.batch_size),
        min_feature_map_size=cfg.planning.get("min_feature_map_size", None),
        max_numpool=cfg.planning.get("max_numpool", None),
        initial_patch_ref_3d=int(cfg.planning.initial_patch_ref_3d),
        initial_patch_ref_2d=int(cfg.planning.initial_patch_ref_2d),
        iso_spacing_strategy=cfg.planning.get("iso_spacing_strategy", "voxel_budget"),
    )

    # Store network topology in plans (for training to use later)
    _store_topology_in_plans(
        dataset_dir,
        min_feature_map_size=cfg.planning.get("min_feature_map_size", None),
        max_numpool=cfg.planning.get("max_numpool", None),
    )

    logger.info("Planning completed. Run 'command=preprocess' next.")


def _store_topology_in_plans(
    dataset_dir: Path,
    min_feature_map_size=None,
    max_numpool=None,
):
    """Extend experiment_plans.json with network topology info."""
    import json
    from preprocessing.plan import _get_pool_and_conv_props, _MIN_FEATURE_MAP_SIZE

    plans_file = dataset_dir / "experiment_plans.json"
    with open(plans_file) as f:
        plans = json.load(f)

    _min_fmap = int(min_feature_map_size) if min_feature_map_size is not None else _MIN_FEATURE_MAP_SIZE
    _max_pool = int(max_numpool) if max_numpool is not None else 999999

    for config_name, config_data in plans["configurations"].items():
        if "pool_op_kernel_sizes" in config_data:
            continue  # already computed

        spacing = config_data["spacing"]
        patch_size = config_data["patch_size"]

        num_pool, pool_ops, conv_kernels, _, _ = _get_pool_and_conv_props(
            spacing, patch_size, _min_fmap, _max_pool,
        )

        config_data["pool_op_kernel_sizes"] = [list(p) for p in pool_ops]
        config_data["conv_kernel_sizes"] = [list(c) for c in conv_kernels]
        config_data["num_pool_per_axis"] = list(num_pool)

    with open(plans_file, "w") as f:
        json.dump(plans, f, indent=2)

    logger.info("Stored network topology in experiment_plans.json.")


# ======================================================================
# Preprocess
# ======================================================================

def _run_preprocess(cfg: DictConfig):
    logger.info("Starting preprocessing...")
    from preprocessing.preprocessing4 import preprocess

    dataset_dir = Path(cfg.data.dataset_dir)
    raw_data_dir = dataset_dir / "raw"

    plans_file = dataset_dir / "experiment_plans.json"
    if not plans_file.exists():
        raise FileNotFoundError(
            f"Experiment plans not found at '{plans_file}'. "
            "Run 'command=plan' first."
        )

    preprocess(
        raw_data_dir=raw_data_dir,
        output_dir=dataset_dir,
        configuration=cfg.configuration,
        num_workers=int(cfg.preprocessing.num_workers),
        convert_to_binary_mask=bool(cfg.preprocessing.convert_to_binary_mask),
        unpack=bool(cfg.preprocessing.get("unpack", False)),
    )
    logger.info("Preprocessing completed.")


# ======================================================================
# Train
# ======================================================================

def _run_train(cfg: DictConfig):
    logger.info("Starting training...")
    from pipeline.experiment import ExperimentRunner

    runner = ExperimentRunner(cfg)
    runner.run_train()

    logger.info("Training completed.")


# ======================================================================
# Validate
# ======================================================================

def _run_validate(cfg: DictConfig):
    logger.info("Starting validation...")
    from pipeline.experiment import ExperimentRunner

    runner = ExperimentRunner(cfg)
    runner.run_validate()

    logger.info("Validation completed.")


# ======================================================================
# Test
# ======================================================================

def _run_test(cfg: DictConfig):
    logger.info("Starting test inference...")
    from pipeline.experiment import ExperimentRunner

    runner = ExperimentRunner(cfg)
    runner.run_test()

    logger.info("Test inference completed.")


if __name__ == "__main__":
    main()
