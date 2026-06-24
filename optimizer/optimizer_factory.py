"""Optimizer and LR scheduler factory.

Builds optimizer + scheduler from Hydra config.  Supports both plain
parameter iterators (backwards compatible) and explicit parameter group
dicts (for per-group learning rates used by the pretraining/freeze system).
"""
from __future__ import annotations

import logging
from typing import List, Tuple, Union

import torch
from omegaconf import DictConfig

from optimizer.poly_lr import (
    PolyLRScheduler,
    ExponentialLRScheduler,
    CosineAnnealingLRScheduler,
    WarmupWrapper,
)
from torch.optim.lr_scheduler import LRScheduler

logger = logging.getLogger(__name__)


def build_optimizer_and_scheduler(
    opt_cfg: DictConfig,
    model_parameters_or_groups: Union[List[dict], any],
    num_epochs: int,
    warmup_steps: int = 0,
    warmup_start_step: int = 0,
    warmup_start_factor: float = 0.1,
) -> Tuple[torch.optim.Optimizer, LRScheduler]:
    """Build optimizer and learning rate scheduler from config.

    Parameters
    ----------
    opt_cfg : DictConfig
        Optimizer Hydra config group (e.g. ``configs/optimizer/sgd.yaml``).
    model_parameters_or_groups
        Either ``model.parameters()`` (backwards compatible) or a list of
        parameter group dicts produced by
        :func:`models.pretrained.build_param_groups`.  Each dict must have a
        ``params`` key and may have ``initial_lr``, ``lr_scale``, and
        ``freeze_group_idx`` metadata.
    num_epochs : int
        Total number of training epochs.
    warmup_steps : int
        When ``> 0``, the returned scheduler is wrapped in a
        :class:`~optimizer.poly_lr.WarmupWrapper` that linearly ramps the LR
        over this many epochs.  Used by the UpKern fine-tuning paths.  ``0``
        (the default) leaves the scheduler untouched.
    warmup_start_step : int
        Absolute epoch at which the warmup window begins (``0`` for regular
        UpKern, the handoff epoch for in-run UpKern phase 2).
    warmup_start_factor : float
        LR multiplier at the first warmup epoch.

    Returns
    -------
    (optimizer, lr_scheduler)
    """
    name = opt_cfg.name.lower()

    # Detect parameter groups vs plain parameter iterator
    _is_param_groups = (
        isinstance(model_parameters_or_groups, list)
        and len(model_parameters_or_groups) > 0
        and isinstance(model_parameters_or_groups[0], dict)
    )

    if _is_param_groups:
        param_groups = model_parameters_or_groups
        # Set per-group optimizer defaults from opt_cfg
        default_wd = opt_cfg.get(
            "weight_decay", 3e-5 if name == "sgd" else 1e-4
        )
        for pg in param_groups:
            # Use initial_lr as the starting lr for the optimizer
            pg.setdefault("lr", pg.get("initial_lr", opt_cfg.lr))
            pg.setdefault("weight_decay", default_wd)
        params = param_groups
    else:
        params = model_parameters_or_groups

    # Build optimizer
    if name == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.get("weight_decay", 3e-5),
            momentum=opt_cfg.get("momentum", 0.99),
            nesterov=opt_cfg.get("nesterov", True),
        )
    elif name == "adamw":
        betas = opt_cfg.get("betas", (0.9, 0.999))
        optimizer = torch.optim.AdamW(
            params,
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.get("weight_decay", 1e-4),
            betas=tuple(betas),
            eps=opt_cfg.get("eps", 1e-8),
            amsgrad=opt_cfg.get("amsgrad", False),
        )
    elif name == "adam":
        betas = opt_cfg.get("betas", (0.9, 0.999))
        optimizer = torch.optim.Adam(
            params,
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.get("weight_decay", 0),
            betas=tuple(betas),
            eps=opt_cfg.get("eps", 1e-8),
            amsgrad=opt_cfg.get("amsgrad", False),
        )
    else:
        raise ValueError(f"Unknown optimizer '{name}'. Available: sgd, adamw, adam")

    # Scheduler
    sched_cfg = opt_cfg.get("scheduler", {})
    sched_name = sched_cfg.get("name", "poly_lr") if sched_cfg else "poly_lr"

    if sched_name == "poly_lr":
        scheduler = PolyLRScheduler(
            optimizer,
            initial_lr=opt_cfg.lr,
            max_steps=num_epochs,
            exponent=sched_cfg.get("exponent", 0.9),
        )
    elif sched_name == "cosine":
        eta_min = float(sched_cfg.get("eta_min", 0.0))
        eta_min_ratio = eta_min / opt_cfg.lr if opt_cfg.lr > 0 else 0.0
        scheduler = CosineAnnealingLRScheduler(
            optimizer,
            initial_lr=opt_cfg.lr,
            max_steps=num_epochs,
            eta_min_ratio=eta_min_ratio,
        )
    elif sched_name == "exponential":
        # lr = initial_lr * gamma^epoch. Either provide `gamma` directly, or
        # `final_lr_ratio` (final/initial ratio at the end of training) which we
        # convert to gamma = ratio^(1/num_epochs) for convenience.
        if "gamma" in sched_cfg:
            gamma = float(sched_cfg["gamma"])
        else:
            final_ratio = float(sched_cfg.get("final_lr_ratio", 0.01))
            if final_ratio <= 0.0:
                raise ValueError(
                    "scheduler.final_lr_ratio must be > 0 for exponential LR "
                    f"(got {final_ratio})."
                )
            gamma = final_ratio ** (1.0 / max(num_epochs, 1))
        scheduler = ExponentialLRScheduler(
            optimizer,
            initial_lr=opt_cfg.lr,
            max_steps=num_epochs,
            gamma=gamma,
        )
    else:
        raise ValueError(
            f"Unknown scheduler '{sched_name}'. Available: poly_lr, cosine, exponential"
        )

    # Optional LR warmup (UpKern fine-tuning). Wraps the base scheduler so the
    # warmup composes with any scheduler type and with per-group LRs.
    if warmup_steps and warmup_steps > 0:
        scheduler = WarmupWrapper(
            scheduler,
            optimizer,
            warmup_steps=warmup_steps,
            warmup_start_step=warmup_start_step,
            start_factor=warmup_start_factor,
        )
        logger.info(
            "LR warmup enabled: %d epochs from start_factor=%.3f (start_epoch=%d).",
            warmup_steps, warmup_start_factor, warmup_start_step,
        )

    logger.info("Optimizer: %s (lr=%.6f), Scheduler: %s", name, opt_cfg.lr, sched_name)
    return optimizer, scheduler
