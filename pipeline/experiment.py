"""High-level experiment orchestration for training and validation workflows.

This module wires together plan loading, split management, trainer
construction, checkpoint-based validation, and cross-validation aggregation
for BioAI experiments.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from omegaconf import DictConfig, OmegaConf

import torch

from pipeline.splits import generate_or_load_splits, get_fold_keys
from pipeline.vram_estimation import check_vram_and_adjust
from models.topology import extract_topology_from_plans
from models.model_factory import get_model
from training.trainer import BioAITrainer
from training.full_validation import perform_full_validation
from training.checkpoint import load_checkpoint
from data_loading.dataset import BioAIDataset

logger = logging.getLogger(__name__)

SPLIT_NUMER : int = 5


class ExperimentRunner:
    """Coordinate end-to-end experiment execution for one configuration.

    The runner loads dataset metadata and experiment plans, resolves the
    preprocessed data folder for the selected configuration, and exposes
    command-style entry points for training and validation. It is responsible
    for fold selection, experiment output paths, optional W&B setup, launching
    full-volume validation after training, and aggregating fold summaries.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra configuration describing the dataset location, selected
        plan configuration, model, training, validation, and logging options.
    """

    def __init__(self, cfg: DictConfig):
        """Load plans, dataset metadata, and experiment paths from config.

        Parameters
        ----------
        cfg : DictConfig
            Hydra configuration for the current experiment run.

        Raises
        ------
        FileNotFoundError
            If the experiment plans, dataset metadata, or preprocessed data
            folder required by the selected configuration are missing.
        ValueError
            If ``cfg.configuration`` is not present in the loaded plans.
        """
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load plans and dataset.json
        dataset_dir = Path(cfg.data.dataset_dir)
        plans_file = dataset_dir / "experiment_plans.json"
        dataset_json_file = dataset_dir / "raw" / "dataset.json"

        if not plans_file.exists():
            raise FileNotFoundError(
                f"Experiment plans not found at '{plans_file}'. "
                "Run 'command=plan' first."
            )
        if not dataset_json_file.exists():
            raise FileNotFoundError(
                f"dataset.json not found at '{dataset_json_file}'."
            )

        with open(plans_file) as f:
            self.plans = json.load(f)
        with open(dataset_json_file) as f:
            self.dataset_json = json.load(f)

        self.configuration = cfg.configuration
        if self.configuration not in self.plans["configurations"]:
            available = list(self.plans["configurations"].keys())
            raise ValueError(
                f"Configuration '{self.configuration}' not in plans. "
                f"Available: {available}"
            )

        if cfg.model.get("requires_isotropic_preprocessing", False):
            if not self.configuration.endswith("_iso"):
                raise ValueError(
                    f"Model '{cfg.model.name}' requires isotropic preprocessing "
                    f"(requires_isotropic_preprocessing=true), but the selected "
                    f"configuration '{self.configuration}' does not end with '_iso'. "
                    f"Use a configuration ending in '_iso' (e.g. '3d_fullres_iso')."
                )

        # Derive preprocessed data folder
        config_data = self.plans["configurations"][self.configuration]
        data_identifier = config_data["data_identifier"]
        self.preprocessed_folder = str(dataset_dir / data_identifier)

        if not os.path.isdir(self.preprocessed_folder):
            raise FileNotFoundError(
                f"Preprocessed data folder '{self.preprocessed_folder}' not found. "
                "Run 'command=preprocess' first."
            )

        # Experiment path: experiments/{data}/{model}_{config}/{experiment_name}/fold_N
        self.experiment_name = cfg.get("experiment_name", None) or "default"
        self.experiments_base = os.path.join(
            "experiments",
            cfg.data.name,
            f"{cfg.model.name}_{self.configuration}",
            self.experiment_name,
        )

        # Load (or generate) splits now so num_folds is known for all commands.
        self.splits = generate_or_load_splits(
            dataset_dir=str(dataset_dir),
            preprocessed_folder=self.preprocessed_folder,
            n_splits=SPLIT_NUMER
        )
        self.num_folds = len(self.splits)

    # ==================================================================
    # Command: train
    # ==================================================================

    def run_train(self):
        """Train the configured fold or folds.

        Depending on ``cfg.fold`` and ``cfg.run_all_folds``, this method runs
        one fold, a user-provided list of folds, or the full k-fold sequence.
        Each fold is trained independently through :meth:`_train_one_fold`, and
        when multiple folds are executed a mean/std summary of the best EMA
        Dice scores is logged at the end.
        """
        fold = self.cfg.fold
        run_all = self.cfg.get("run_all_folds", False)

        if run_all:
            folds = list(range(self.num_folds))
            logger.info("Running all %d folds sequentially.", self.num_folds)
        elif isinstance(fold, (list, tuple)):
            folds = [int(f) for f in fold]
        else:
            folds = [fold]

        scores = []
        for f in folds:
            logger.info("=" * 60)
            logger.info("Starting fold %s", f)
            logger.info("=" * 60)
            score = self._train_one_fold(f)
            scores.append(score)
            logger.info("Fold %s completed with best EMA Dice: %.4f", f, score)

        if len(scores) > 1:
            arr = np.array(scores)
            logger.info(
                "Cross-validation results: mean=%.4f, std=%.4f",
                arr.mean(), arr.std(ddof=1),
            )

    def _train_one_fold(self, fold) -> float:
        """Train one fold and run full-volume validation afterward.

        The method resolves the train/validation split for ``fold``,
        initializes :class:`training.trainer.BioAITrainer`, optionally checks
        VRAM fit, sets up Weights & Biases logging when enabled, runs patch
        training, and then evaluates the best trained model on full validation
        volumes using sliding-window inference.

        Parameters
        ----------
        fold : int or str
            Fold identifier passed through the configured split logic.

        Returns
        -------
        float
            Best EMA foreground Dice reported by the trainer for this fold.
        """
        # Splits
        tr_keys, val_keys = get_fold_keys(self.splits, fold)

        # Output folder
        fold_str = f"fold_{fold}" if fold != "all" else "fold_all"
        output_folder = os.path.join(self.experiments_base, fold_str)

        # Trainer
        trainer = BioAITrainer(
            plans=self.plans,
            dataset_json=self.dataset_json,
            configuration=self.configuration,
            fold=fold,
            cfg=self.cfg,
            device=self.device,
            output_folder=output_folder,
        )

        trainer.initialize(tr_keys, val_keys, self.preprocessed_folder)

        assert trainer.network is not None, "Network not initiated, please initialize the trainer object."
        # Optional VRAM check
        if self.cfg.training.get("vram_check", False):
            fits, recommended_bs, recommended_patch = check_vram_and_adjust(
                model=trainer.network,
                patch_size=trainer.topology.patch_size,
                batch_size=int(self.cfg.training.batch_size) if self.cfg.training.get("batch_size") is not None
                    else self.plans["configurations"][self.configuration].get("batch_size", 2),
                num_input_channels=trainer.num_input_channels,
                device=self.device,
                div_by=trainer.topology.shape_must_be_divisible_by,
            )
            if not fits:
                raise RuntimeError(
                    f"Model does not fit in VRAM with patch_size={trainer.topology.patch_size}. "
                    f"Suggested smaller patch: {recommended_patch}. "
                    "Re-run planning with a smaller configuration or set the patch manually."
                )

        # W&B
        if hasattr(self.cfg, "logging") and self.cfg.logging.get("wandb_logging", {}).get("enabled", False):
            try:
                import wandb
                trainer.wandb_run = wandb.init(
                    project=self.cfg.logging.wandb_logging.project,
                    entity=self.cfg.logging.wandb_logging.get("entity", None),
                    config=OmegaConf.to_container(self.cfg, resolve=True), # type: ignore
                    name=f"{self.experiment_name}_{fold_str}",
                    group=self.experiment_name,
                    tags=self.cfg.logging.wandb_logging.get("tags", []),
                    reinit=True,
                )
            except Exception as e:
                logger.warning("W&B init failed: %s", e)

        # Train
        best_score = trainer.run_training()

        # Persist best EMA dice and model parameter count so run_experiment.py
        # can read them without loading the full checkpoint.
        num_params = sum(p.numel() for p in trainer.network.parameters())
        training_info = {"best_ema_dice": best_score, "num_params": num_params}
        with open(os.path.join(output_folder, "training_info.json"), "w") as _f:
            json.dump(training_info, _f)

        # Load best checkpoint for full validation (fall back to final if best
        # was never produced — e.g. EMA Dice never improved past NaN).
        ckpt_for_val = os.path.join(output_folder, "checkpoint_best.pth")
        if not os.path.isfile(ckpt_for_val):
            ckpt_for_val = os.path.join(output_folder, "checkpoint_final.pth")

        # When in-run UpKern completes phase 2, trainer.network has the larger
        # kernel. If checkpoint_best was saved during phase 1 (smaller kernel),
        # loading it into the phase-2 network causes a size mismatch. Fall back
        # to checkpoint_final which is always written at the end of training
        # (i.e. from phase 2).
        if (
            os.path.isfile(ckpt_for_val)
            and getattr(trainer, "_inrun_upkern_done", False)
        ):
            meta = torch.load(ckpt_for_val, map_location="cpu", weights_only=False)
            if meta.get("kernel_upsize_phase", 2) == 1:
                final_ckpt = os.path.join(output_folder, "checkpoint_final.pth")
                if os.path.isfile(final_ckpt):
                    logger.warning(
                        "checkpoint_best was saved in phase-1 (small kernel); "
                        "falling back to checkpoint_final (phase-2) for validation."
                    )
                    ckpt_for_val = final_ckpt

        if os.path.isfile(ckpt_for_val):
            logger.info("Loading %s for full validation", os.path.basename(ckpt_for_val))
            load_checkpoint(ckpt_for_val, trainer.network, device=self.device)
        else:
            logger.warning(
                "No checkpoint found in %s; validating with in-memory weights.",
                output_folder,
            )

        # Full validation after training
        logger.info("Running full validation for fold %s...", fold)
        val_output = os.path.join(output_folder, "validation")

        max_val_cases = self.cfg.validation.get("max_cases", None)
        val_keys_for_validation = val_keys[:int(max_val_cases)] if max_val_cases is not None else val_keys
        dataset_val = BioAIDataset(self.preprocessed_folder, val_keys_for_validation)

        summary = perform_full_validation(
            network=trainer.network,
            dataset=dataset_val,
            val_keys=val_keys_for_validation,
            patch_size=trainer.topology.patch_size,
            num_classes=trainer.num_classes,
            device=self.device,
            output_folder=val_output,
            tile_step_size=self.cfg.validation.get("tile_step_size", 0.5),
            use_gaussian=self.cfg.validation.get("use_gaussian", True),
            use_mirroring=self.cfg.validation.get("use_mirroring", True),
            mirror_axes=trainer.inference_allowed_mirroring_axes,

            compute_hd95_flag=self.cfg.validation.get("compute_hd95", True),
            compute_surface_dice_flag=self.cfg.validation.get("compute_surface_dice", False),
            surface_dice_tolerance=self.cfg.validation.get("surface_dice_tolerance", 2.0),
            enable_deep_supervision=trainer.enable_deep_supervision,
            sw_batch_size=self.cfg.validation.get("sw_batch_size", 2),
            compute_flops=self.cfg.validation.get("compute_flops", True),
            num_input_channels=trainer.num_input_channels,
            save_per_case_metrics=self.cfg.validation.get("save_per_case_metrics", True),
        )

        # Close W&B
        if trainer.wandb_run is not None:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass

        return best_score

    # ==================================================================
    # Command: test
    # ==================================================================

    def run_test(self):
        """Run test inference with optional ensemble and post-processing.

        Delegates to :func:`testing.test_runner.run_testing` for inference,
        then optionally runs post-processing if configured.
        """
        from testing.test_runner import run_testing
        from omegaconf import OmegaConf

        summary = run_testing(
            cfg=self.cfg,
            plans=self.plans,
            dataset_json=self.dataset_json,
            configuration=self.configuration,
            experiments_base=self.experiments_base,
            preprocessed_folder=self.preprocessed_folder,
            device=self.device,
        )

        # Post-processing
        if hasattr(self.cfg, "postprocessing") and self.cfg.postprocessing.get("enabled", False):
            from testing.postprocessing import run_postprocessing

            # Determine prediction folder
            testing_cfg = self.cfg.testing
            ensemble_folds = testing_cfg.get("ensemble_folds", True)
            if ensemble_folds:
                predictions_folder = os.path.join(self.experiments_base, "test_ensemble")
                output_folder = os.path.join(self.experiments_base, "test_ensemble_postprocessed")
            else:
                fold = self.cfg.fold
                fold_str = f"fold_{fold}" if fold != "all" else "fold_all"
                predictions_folder = os.path.join(self.experiments_base, fold_str, "test_predictions")
                output_folder = os.path.join(self.experiments_base, fold_str, "test_predictions_postprocessed")

            # Raw data dir for CRF bilateral term
            raw_data_dir = str(Path(self.cfg.data.dataset_dir) / "raw")

            pp_cfg = OmegaConf.to_container(self.cfg.postprocessing, resolve=True)

            run_postprocessing(
                predictions_folder=predictions_folder,
                output_folder=output_folder,
                cfg_postprocessing=pp_cfg, # type: ignore
                raw_data_dir=raw_data_dir,
                softmax_folder=predictions_folder,
                properties_folder=self.preprocessed_folder,
            )
            logger.info("Post-processed predictions saved to %s", output_folder)

        if summary is not None:
            logger.info("Test summary: %s", json.dumps(summary, indent=2))

    # ==================================================================
    # Command: validate
    # ==================================================================

    def run_validate(self):
        """Run checkpoint-based validation for one fold or all folds.

        If all fold checkpoints are available and ``cfg.fold`` is not
        ``"all"``, the method aggregates validation summaries across folds via
        :meth:`_run_cross_validation`. Otherwise it performs validation only
        for the requested fold.
        """
        fold = self.cfg.fold

        # Check if all folds are trained
        all_folds_trained = all(
            os.path.isfile(
                os.path.join(self.experiments_base, f"fold_{f}", "checkpoint_best.pth")
            )
            for f in range(self.num_folds)
        )

        if all_folds_trained and fold != "all":
            logger.info("All %d folds trained. Running cross-validation aggregation.", self.num_folds)
            self._run_cross_validation()
        else:
            self._validate_one_fold(fold)

    def _validate_one_fold(self, fold):
        """Validate one trained fold from its saved checkpoint.

        This method rebuilds the network for inference, loads the checkpoint
        selected by ``validation.checkpoint`` (``best``, ``final``, or
        ``latest``; falling back to ``checkpoint_best.pth`` if the requested
        one is unavailable), restores the validation case list from the saved
        split definition, and runs full-volume sliding-window validation into
        the fold's ``validation/`` directory.

        Parameters
        ----------
        fold : int or str
            Fold identifier to validate.

        Raises
        ------
        FileNotFoundError
            If no suitable checkpoint exists for the requested fold.
        """
        fold_str = f"fold_{fold}" if fold != "all" else "fold_all"
        fold_dir = os.path.join(self.experiments_base, fold_str)

        ckpt_choice = str(self.cfg.validation.get("checkpoint", "best")).lower()
        ckpt_filenames = {
            "best": "checkpoint_best.pth",
            "final": "checkpoint_final.pth",
            "latest": "checkpoint_latest.pth",
        }
        if ckpt_choice not in ckpt_filenames:
            raise ValueError(
                f"Invalid validation.checkpoint='{ckpt_choice}'. "
                f"Expected one of {list(ckpt_filenames)}."
            )

        ckpt_path = os.path.join(fold_dir, ckpt_filenames[ckpt_choice])
        if not os.path.isfile(ckpt_path) and ckpt_choice != "best":
            fallback = os.path.join(fold_dir, ckpt_filenames["best"])
            if os.path.isfile(fallback):
                logger.warning(
                    "Requested %s not found in %s; falling back to checkpoint_best.pth",
                    ckpt_filenames[ckpt_choice], fold_dir,
                )
                ckpt_path = fallback
        if not os.path.isfile(ckpt_path):
            ckpt_path = os.path.join(fold_dir, "checkpoint_final.pth")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"No checkpoint found in {fold_dir}. Train first."
            )
        logger.info("Loading %s for validation", os.path.basename(ckpt_path))

        # Load model
        topology = extract_topology_from_plans(
            self.plans,
            self.configuration,
            min_feature_map_size=self.cfg.planning.get("min_feature_map_size", 4),
            max_numpool=self.cfg.planning.get("max_numpool", None),
        )
        labels = self.plans.get("labels", self.dataset_json.get("labels", {}))
        num_classes = len(labels)
        ch_names = self.plans.get("channel_names", self.dataset_json.get("channel_names", {}))
        num_input_channels = len(ch_names)

        network = get_model(
            model_cfg=self.cfg.model,
            topology=topology,
            num_input_channels=num_input_channels,
            num_output_channels=num_classes,
            enable_deep_supervision=False,
        ).to(self.device)

        # strict=False: the network is built with deep supervision disabled,
        # so MedNeXt-family checkpoints trained with deep supervision carry
        # extra (unused-at-inference) head weights that the network lacks.
        load_checkpoint(ckpt_path, network, device=self.device, strict=False)

        # Get val keys
        _, val_keys = get_fold_keys(self.splits, fold)

        max_val_cases = self.cfg.validation.get("max_cases", None)
        val_keys_for_validation = val_keys[:int(max_val_cases)] if max_val_cases is not None else val_keys
        dataset_val = BioAIDataset(self.preprocessed_folder, val_keys_for_validation)

        # Determine mirror axes from training geometry
        from augmentation.geometry import determine_training_geometry
        geometry = determine_training_geometry(topology.patch_size)

        val_output = os.path.join(fold_dir, "validation")
        summary = perform_full_validation(
            network=network,
            dataset=dataset_val,
            val_keys=val_keys_for_validation,
            patch_size=topology.patch_size,
            num_classes=num_classes,
            device=self.device,
            output_folder=val_output,
            tile_step_size=self.cfg.validation.get("tile_step_size", 0.5),
            use_gaussian=self.cfg.validation.get("use_gaussian", True),
            use_mirroring=self.cfg.validation.get("use_mirroring", True),
            mirror_axes=geometry.mirror_axes,
            compute_hd95_flag=self.cfg.validation.get("compute_hd95", True),
            compute_surface_dice_flag=self.cfg.validation.get("compute_surface_dice", False),
            surface_dice_tolerance=self.cfg.validation.get("surface_dice_tolerance", 2.0),
            sw_batch_size=self.cfg.validation.get("sw_batch_size", 2),
            compute_flops=self.cfg.validation.get("compute_flops", True),
            num_input_channels=num_input_channels,
            save_per_case_metrics=self.cfg.validation.get("save_per_case_metrics", True),
        )

        logger.info("Validation summary: %s", json.dumps(summary, indent=2))

    def _run_cross_validation(self):
        """Aggregate saved validation summaries across all folds.

        For each fold, the method looks for ``validation/summary.json`` and
        triggers validation on-demand if the summary is missing but a checkpoint
        exists. It then computes the mean and sample standard deviation across
        folds for every metric returned by ``perform_full_validation``:
        foreground-level means (Dice, IoU, Sensitivity, Precision, HD95) and
        per-class breakdowns. Results are written to
        ``cross_validation_summary.json`` in the experiment root.
        """
        all_summaries = []
        for f in range(self.num_folds):
            summary_path = os.path.join(
                self.experiments_base, f"fold_{f}", "validation", "summary.json"
            )
            if os.path.isfile(summary_path):
                logger.info("Found existing summary for fold %d. Loading...", f)
                with open(summary_path) as fh:
                    all_summaries.append(json.load(fh))
            else:
                logger.info(
                    "No summary found for fold %d. Running validation for this fold.", f
                )
                self._validate_one_fold(f)
                if os.path.isfile(summary_path):
                    with open(summary_path) as fh:
                        all_summaries.append(json.load(fh))

        if not all_summaries:
            logger.error("No validation summaries found.")
            return

        n_folds = len(all_summaries)

        def _agg(values: List[float]) -> dict:
            """Return mean/std dict, handling inf and nan gracefully."""
            finite = [v for v in values if v is not None and not np.isinf(v) and not np.isnan(v)]
            return {
                "mean": float(np.mean(finite)) if finite else float("nan"),
                "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
                "per_fold": values,
            }

        # ── Foreground-level aggregation ──────────────────────────────────
        fg_metrics = ["foreground_mean_dice", "foreground_mean_iou",
                      "foreground_mean_sensitivity", "foreground_mean_precision"]
        cv_summary: dict = {"folds": n_folds, "foreground": {}}
        for key in fg_metrics:
            values = [s.get(key, float("nan")) for s in all_summaries]
            cv_summary["foreground"][key] = _agg(values)

        # HD95 is optional
        if any("foreground_mean_hd95" in s for s in all_summaries):
            hd_values = [s.get("foreground_mean_hd95", float("inf")) for s in all_summaries]
            cv_summary["foreground"]["foreground_mean_hd95"] = _agg(hd_values)

        # Surface Dice is optional
        if any("foreground_mean_surface_dice" in s for s in all_summaries):
            nsd_values = [s.get("foreground_mean_surface_dice", float("nan")) for s in all_summaries]
            cv_summary["foreground"]["foreground_mean_surface_dice"] = _agg(nsd_values)

        # ── Per-class aggregation ─────────────────────────────────────────
        # Collect the union of class keys present across all fold summaries.
        class_keys: List[str] = sorted(
            {ck for s in all_summaries for ck in s.get("per_class", {})},
            key=lambda x: int(x),
        )
        cv_summary["per_class"] = {}
        per_class_metrics = ["Dice", "IoU", "Sensitivity", "Precision"]
        for ck in class_keys:
            cv_summary["per_class"][ck] = {}
            for metric in per_class_metrics:
                values = [
                    s["per_class"][ck][metric]
                    for s in all_summaries
                    if ck in s.get("per_class", {}) and metric in s["per_class"][ck]
                ]
                cv_summary["per_class"][ck][metric] = _agg(values)

            # HD95 per class (optional)
            if any(
                ck in s.get("per_class", {}) and "HD95" in s["per_class"][ck]
                for s in all_summaries
            ):
                hd_values = [
                    s["per_class"][ck].get("HD95", float("inf"))
                    for s in all_summaries
                    if ck in s.get("per_class", {})
                ]
                cv_summary["per_class"][ck]["HD95"] = _agg(hd_values)

            # Surface Dice per class (optional)
            if any(
                ck in s.get("per_class", {}) and "SurfaceDice" in s["per_class"][ck]
                for s in all_summaries
            ):
                nsd_values = [
                    s["per_class"][ck].get("SurfaceDice", float("nan"))
                    for s in all_summaries
                    if ck in s.get("per_class", {})
                ]
                cv_summary["per_class"][ck]["SurfaceDice"] = _agg(nsd_values)

        # ── Save ──────────────────────────────────────────────────────────
        cv_path = os.path.join(self.experiments_base, "cross_validation_summary.json")
        with open(cv_path, "w") as f:
            json.dump(cv_summary, f, indent=2)

        # ── Logging ───────────────────────────────────────────────────────
        fg = cv_summary["foreground"]
        logger.info("Cross-validation complete (%d folds):", n_folds)
        logger.info(
            "  Foreground Dice      : %.4f ± %.4f",
            fg["foreground_mean_dice"]["mean"], fg["foreground_mean_dice"]["std"],
        )
        logger.info(
            "  Foreground IoU       : %.4f ± %.4f",
            fg["foreground_mean_iou"]["mean"], fg["foreground_mean_iou"]["std"],
        )
        logger.info(
            "  Foreground Sensitivity: %.4f ± %.4f",
            fg["foreground_mean_sensitivity"]["mean"], fg["foreground_mean_sensitivity"]["std"],
        )
        logger.info(
            "  Foreground Precision : %.4f ± %.4f",
            fg["foreground_mean_precision"]["mean"], fg["foreground_mean_precision"]["std"],
        )
        if "foreground_mean_hd95" in fg:
            logger.info(
                "  Foreground HD95      : %.4f ± %.4f",
                fg["foreground_mean_hd95"]["mean"], fg["foreground_mean_hd95"]["std"],
            )
        if "foreground_mean_surface_dice" in fg:
            logger.info(
                "  Foreground NSD       : %.4f ± %.4f",
                fg["foreground_mean_surface_dice"]["mean"], fg["foreground_mean_surface_dice"]["std"],
            )
        for ck in class_keys:
            cls_data = cv_summary["per_class"][ck]
            hd_str = ""
            if "HD95" in cls_data:
                hd_str = f"  HD95={cls_data['HD95']['mean']:.4f}±{cls_data['HD95']['std']:.4f}"
            nsd_str = ""
            if "SurfaceDice" in cls_data:
                nsd_str = f"  NSD={cls_data['SurfaceDice']['mean']:.4f}±{cls_data['SurfaceDice']['std']:.4f}"
            logger.info(
                "  Class %s — Dice=%.4f±%.4f  IoU=%.4f±%.4f  "
                "Sens=%.4f±%.4f  Prec=%.4f±%.4f%s%s",
                ck,
                cls_data["Dice"]["mean"], cls_data["Dice"]["std"],
                cls_data["IoU"]["mean"], cls_data["IoU"]["std"],
                cls_data["Sensitivity"]["mean"], cls_data["Sensitivity"]["std"],
                cls_data["Precision"]["mean"], cls_data["Precision"]["std"],
                hd_str,
                nsd_str,
            )
        logger.info("Cross-validation summary saved to %s", cv_path)
