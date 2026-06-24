"""Multi-process prefetching wrapper for BioAIDataLoader.

Wraps a BioAIDataLoader with ``NonDetMultiThreadedAugmenter`` from
``batchgenerators`` to prepare batches in background worker processes,
eliminating GPU idle time between training steps.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class PrefetchedDataLoader:
    """Wrap a BioAIDataLoader with multi-process background prefetching.

    Background worker processes independently call ``next(dataloader)`` and
    place the resulting batches into a shared queue.  When the training loop
    calls ``next()`` on this wrapper it receives a pre-computed batch from the
    queue instead of blocking on data loading + augmentation.

    Parameters
    ----------
    dataloader
        The base ``BioAIDataLoader`` to wrap.
    num_processes : int
        Number of background worker processes.
    num_cached : int
        Maximum number of batches to keep in the prefetch queue.
    pin_memory : bool
        Pin batch tensors in page-locked memory for faster CPU→GPU transfer.
    wait_time : float
        Polling interval in seconds for the background workers.
    seeds : list of int or None
        Optional per-worker random seeds.
    """

    def __init__(
        self,
        dataloader,
        num_processes: int = 12,
        num_cached: int = 6,
        pin_memory: bool = True,
        wait_time: float = 0.002,
        seeds: Optional[list] = None,
    ):
        self._dataloader = dataloader
        self._num_processes = num_processes
        self._num_cached = num_cached
        self._pin_memory = pin_memory
        self._wait_time = wait_time
        self._seeds = seeds
        self._augmenter = None

        logger.info(
            "PrefetchedDataLoader: %d workers, queue_size=%d, pin_memory=%s",
            num_processes, num_cached, pin_memory,
        )

    def _start(self):
        import io
        import contextlib
        from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
            NonDetMultiThreadedAugmenter,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            self._augmenter = NonDetMultiThreadedAugmenter(
                data_loader=self._dataloader,
                transform=None,  # transforms already applied inside generate_train_batch()
                num_processes=self._num_processes,
                num_cached=self._num_cached,
                seeds=self._seeds,
                pin_memory=self._pin_memory,
                wait_time=self._wait_time,
            )

    def __iter__(self):
        return self

    def __next__(self):
        if self._augmenter is None:
            self._start()
        return next(self._augmenter)

    def shutdown(self):
        """Gracefully stop all background workers."""
        if self._augmenter is not None:
            self._augmenter._finish()
            self._augmenter = None

    def __del__(self):
        self.shutdown()
