"""Learning rate schedulers.

Defines idempotent, epoch-indexed schedulers used by the trainer. The
trainer's ``on_train_epoch_start`` calls ``scheduler.step(current_epoch)``
once per epoch; each scheduler computes the LR from ``current_epoch`` in
closed form. This is the pattern from ``nnunetv2/training/lr_scheduler``;
keeping it for the cosine/exponential variants too avoids PyTorch's
``UserWarning: Detected call of lr_scheduler.step() before optimizer.step()``
and the deprecation warning on the ``epoch`` arg of the built-in schedulers.

All variants support per-parameter-group ``initial_lr`` for gradual unfreezing.
"""
from __future__ import annotations
import math
from typing import Dict, List, Optional
import torch

from torch.optim.lr_scheduler import _LRScheduler


class PolyLRScheduler(_LRScheduler):
    """Polynomial decay: lr = initial_lr * (1 - epoch / max_steps) ^ exponent.

    Supports per-parameter-group initial learning rates.  Each optimizer
    param-group may carry an ``initial_lr`` key; if present, that value is
    used instead of the global ``initial_lr``.  This enables freeze-group
    workflows where unfrozen layers start with a scaled learning rate.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    initial_lr : float
        Global starting learning rate (fallback for groups without
        ``initial_lr``).
    max_steps : int
        Total number of epochs.
    exponent : float
        Polynomial exponent (default 0.9, the nnUNet default).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_lr: float,
        max_steps: int,
        exponent: float = 0.9,
        current_step: Optional[int] = None,
    ):
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0

        # Per-group initial LRs.  If a param-group has 'initial_lr' set,
        # use that; otherwise fall back to the global initial_lr.
        self.initial_lrs: List[float] = [
            pg.get("initial_lr", initial_lr) for pg in optimizer.param_groups
        ]

        super().__init__(optimizer, last_epoch=current_step if current_step is not None else -1)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        decay = max(0, 1 - current_step / self.max_steps) ** self.exponent
        for param_group, init_lr in zip(self.optimizer.param_groups, self.initial_lrs):
            param_group["lr"] = init_lr * decay

    # ------------------------------------------------------------------
    # Per-group LR updates (used by gradual unfreezing)
    # ------------------------------------------------------------------

    def update_group_initial_lr(self, group_idx: int, new_initial_lr: float):
        """Update the initial LR for one parameter group.

        Called when a frozen layer group is unfrozen.  The next
        :meth:`step` will apply polynomial decay from the new initial LR.
        """
        if group_idx < len(self.initial_lrs):
            self.initial_lrs[group_idx] = new_initial_lr

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict:
        state = super().state_dict()
        state["initial_lrs"] = self.initial_lrs
        state["ctr"] = self.ctr
        return state

    def load_state_dict(self, state_dict: Dict):
        self.initial_lrs = state_dict.pop("initial_lrs", self.initial_lrs)
        self.ctr = state_dict.pop("ctr", self.ctr)
        super().load_state_dict(state_dict)


class _IdempotentLRScheduler(_LRScheduler):
    """Common machinery for closed-form, epoch-indexed schedulers.

    Subclasses implement :meth:`_decay`, which maps a step to a multiplier in
    [0, 1] applied to each group's ``initial_lr``.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_lr: float,
        max_steps: int,
        current_step: Optional[int] = None,
    ):
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.ctr = 0
        self.initial_lrs: List[float] = [
            pg.get("initial_lr", initial_lr) for pg in optimizer.param_groups
        ]
        super().__init__(optimizer, last_epoch=current_step if current_step is not None else -1)

    def _decay(self, current_step: int) -> float:
        raise NotImplementedError

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1
        decay = self._decay(current_step)
        for param_group, init_lr in zip(self.optimizer.param_groups, self.initial_lrs):
            param_group["lr"] = init_lr * decay

    def update_group_initial_lr(self, group_idx: int, new_initial_lr: float):
        if group_idx < len(self.initial_lrs):
            self.initial_lrs[group_idx] = new_initial_lr

    def state_dict(self) -> Dict:
        state = super().state_dict()
        state["initial_lrs"] = self.initial_lrs
        state["ctr"] = self.ctr
        return state

    def load_state_dict(self, state_dict: Dict):
        self.initial_lrs = state_dict.pop("initial_lrs", self.initial_lrs)
        self.ctr = state_dict.pop("ctr", self.ctr)
        super().load_state_dict(state_dict)


class ExponentialLRScheduler(_IdempotentLRScheduler):
    """Exponential decay: lr = initial_lr * gamma^epoch."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_lr: float,
        max_steps: int,
        gamma: float,
        current_step: Optional[int] = None,
    ):
        self.gamma = gamma
        super().__init__(optimizer, initial_lr, max_steps, current_step)

    def _decay(self, current_step: int) -> float:
        return self.gamma ** max(0, current_step)


class WarmupWrapper:
    """Linear LR warmup layered on top of any epoch-indexed base scheduler.

    The base scheduler (``PolyLRScheduler`` / ``CosineAnnealingLRScheduler`` /
    ``ExponentialLRScheduler``) is stepped first to set each param-group's
    decayed LR.  While ``current_step`` lies inside the warmup window
    ``[warmup_start_step, warmup_start_step + warmup_steps)``, every group's LR
    is then multiplied by a factor that ramps linearly from ``start_factor`` to
    ``1.0`` (reaching exactly ``1.0`` at the last warmup epoch).

    Because the warmup is a *multiplicative* factor applied on top of the base
    scheduler's per-group output, it composes with every scheduler type and
    preserves per-group ``initial_lr`` values (used by the freeze/unfreeze
    workflow).  Each ``step`` is a pure function of ``current_step``, so the
    warmup survives checkpoint resume without any extra bookkeeping.

    This is used to stabilise the early epochs of UpKern fine-tuning, where the
    spatially-interpolated convolution kernels have not yet adapted and the
    full (decayed) LR can be destabilising.

    Parameters
    ----------
    base_scheduler
        The scheduler whose per-group LRs are scaled.
    optimizer : torch.optim.Optimizer
    warmup_steps : int
        Number of epochs spanned by the warmup ramp.  ``<= 0`` disables it.
    warmup_start_step : int
        Absolute epoch at which the warmup window begins.  ``0`` for regular
        UpKern (warmup covers the start of the run); the handoff epoch for
        in-run UpKern (warmup covers the start of phase 2).
    start_factor : float
        LR multiplier at the first warmup epoch.
    """

    def __init__(
        self,
        base_scheduler,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        warmup_start_step: int = 0,
        start_factor: float = 0.1,
    ):
        # Assign base first so __getattr__ never recurses before it exists.
        self.base = base_scheduler
        self.optimizer = optimizer
        self.warmup_steps = int(warmup_steps)
        self.warmup_start_step = int(warmup_start_step)
        self.start_factor = float(start_factor)

    def _warmup_factor(self, current_step: int) -> float:
        if self.warmup_steps <= 0:
            return 1.0
        t = current_step - self.warmup_start_step
        if t < 0 or t >= self.warmup_steps:
            return 1.0
        frac = (t + 1) / self.warmup_steps
        return self.start_factor + (1.0 - self.start_factor) * frac

    def step(self, current_step=None):
        # Let the base scheduler resolve the step (and its own ctr) and set the
        # decayed per-group LRs first.
        self.base.step(current_step)
        if current_step is None or current_step == -1:
            # Base used / advanced its internal counter; mirror it.
            current_step = self.base.ctr - 1

        factor = self._warmup_factor(current_step)
        if factor != 1.0:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = param_group["lr"] * factor

    # ------------------------------------------------------------------
    # Delegation to the base scheduler
    # ------------------------------------------------------------------

    def update_group_initial_lr(self, group_idx: int, new_initial_lr: float):
        """Delegate to the base scheduler (used by gradual unfreezing)."""
        self.base.update_group_initial_lr(group_idx, new_initial_lr)

    def __getattr__(self, name):
        # Only reached for attributes not found on the wrapper itself; forward
        # them (initial_lrs, ctr, max_steps, ...) to the base scheduler.
        base = self.__dict__.get("base")
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict:
        return {
            "base": self.base.state_dict(),
            "warmup_steps": self.warmup_steps,
            "warmup_start_step": self.warmup_start_step,
            "start_factor": self.start_factor,
        }

    def load_state_dict(self, state_dict: Dict):
        if isinstance(state_dict, dict) and "base" in state_dict:
            self.base.load_state_dict(state_dict["base"])
            self.warmup_steps = state_dict.get("warmup_steps", self.warmup_steps)
            self.warmup_start_step = state_dict.get("warmup_start_step", self.warmup_start_step)
            self.start_factor = state_dict.get("start_factor", self.start_factor)
        else:
            # Backward-compat: checkpoint written by a bare base scheduler.
            self.base.load_state_dict(state_dict)


class CosineAnnealingLRScheduler(_IdempotentLRScheduler):
    """Cosine annealing: lr = eta_min + 0.5 * (initial_lr - eta_min) * (1 + cos(pi * epoch / T_max)).

    ``eta_min`` is expressed as a ratio of ``initial_lr`` so per-group initial
    LRs (used by gradual unfreezing) decay to a consistent fraction of their
    own initial value.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_lr: float,
        max_steps: int,
        eta_min_ratio: float = 0.0,
        current_step: Optional[int] = None,
    ):
        self.eta_min_ratio = eta_min_ratio
        super().__init__(optimizer, initial_lr, max_steps, current_step)

    def _decay(self, current_step: int) -> float:
        t = min(max(current_step, 0), self.max_steps)
        cos = 0.5 * (1.0 + math.cos(math.pi * t / max(self.max_steps, 1)))
        return self.eta_min_ratio + (1.0 - self.eta_min_ratio) * cos
