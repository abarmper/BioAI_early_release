"""BioAI dataset class for loading preprocessed .npz + .pkl pairs.

Analogous to ``nnUNetDatasetNumpy`` from nnUNet v2.
"""
from __future__ import annotations

import atexit
import os
import pickle
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Map .npz keys → corresponding .npy sidecar suffix.
# Kept in sync with cleanup_npz.py and BioAIDataset._load_from_disk.
_NPZ_KEY_TO_NPY_SUFFIX = {
    "data": ".npy",
    "seg": "_seg.npy",
    "distance_map": "_dist.npy",
}


def _unpack_one(folder_path: str, ident: str, delete_npz: bool = False) -> None:
    """Unpack a single .npz file into separate .npy files.

    If *delete_npz* is True, removes the source .npz after a successful unpack.
    """
    npz_path = os.path.join(folder_path, ident + ".npz")
    with np.load(npz_path) as npz:
        np.save(os.path.join(folder_path, ident + ".npy"), npz["data"])
        if "seg" in npz:
            np.save(os.path.join(folder_path, ident + "_seg.npy"), npz["seg"])
        if "distance_map" in npz:
            np.save(os.path.join(folder_path, ident + "_dist.npy"), npz["distance_map"])
    if delete_npz:
        os.remove(npz_path)


def _try_delete_redundant_npz(folder_path: str, ident: str) -> bool:
    """Delete ``{ident}.npz`` iff every key it holds has a .npy sidecar on disk.

    Returns True if the .npz was removed.
    """
    npz_path = os.path.join(folder_path, ident + ".npz")
    if not os.path.isfile(npz_path):
        return False
    try:
        with np.load(npz_path) as npz:
            keys = list(npz.keys())
    except Exception:
        return False
    for key in keys:
        suffix = _NPZ_KEY_TO_NPY_SUFFIX.get(key)
        if suffix is None:
            return False
        if not os.path.isfile(os.path.join(folder_path, ident + suffix)):
            return False
    os.remove(npz_path)
    return True


class BioAIDataset:
    """Lazy-loading dataset over preprocessed ``.npz`` + ``.pkl`` files.

    Each case is expected to provide array data in ``{id}.npz`` and metadata
    in ``{id}.pkl``. Segmentation targets are optional; when present they are
    read from the ``seg`` entry in ``{id}.npz`` or, if unpacked files are
    available, from ``{id}_seg.npy``. The loader similarly prefers ``{id}.npy``
    for image data to avoid repeated ``.npz`` reads.

    If distance maps are requested for boundary-based training, the dataset
    loads ``{id}_dist.npy`` when present and otherwise falls back to the
    ``distance_map`` entry stored inside ``{id}.npz``.

    Parameters
    ----------
    folder : str
        Directory containing the preprocessed files.
    identifiers : list of str or None
        Case identifiers to include.  If *None*, all ``.npz`` files in
        *folder* are used.
    load_distance_maps : bool
        If True, also load the ``distance_map`` key from ``.npz`` (for
        boundary loss).
    cache_cases : bool
        Enable in-memory caching of loaded cases.
    cache_max_bytes : int
        Maximum bytes to use for the cache (0 = unlimited).
    cache_mode : str
        Cache strategy when *cache_cases* is True:

        * ``"lazy"`` — cache on first access (default).  Each forked worker
          builds its own copy, so total RAM scales with ``num_workers``.
        * ``"eager"`` — pre-load all cases before workers fork.  On Linux
          the bulk data pages stay copy-on-write shared across workers, so
          memory usage is close to 1x the dataset size.
        * ``"shared"`` — pre-load into POSIX shared memory so forked
          workers share a single physical copy with zero COW overhead.
    """

    def __init__(
        self,
        folder: str,
        identifiers: Optional[List[str]] = None,
        load_distance_maps: bool = False,
        cache_cases: bool = False,
        cache_max_bytes: int = 0,
        cache_mode: str = "lazy",
    ):
        self.folder = folder
        self.load_distance_maps = load_distance_maps

        if identifiers is not None:
            self.identifiers = sorted(identifiers)
        else:
            self.identifiers = self.get_identifiers(folder)

        # Pre-load all properties into memory (small pkl files)
        self._properties_cache: Dict[str, dict] = {}

        # Caching state
        self._cache_mode = cache_mode if cache_cases else "off"
        self._cache_max_bytes = cache_max_bytes
        self._array_cache: Dict[str, dict] = {}
        self._cache_bytes: int = 0

        # Shared memory management (only for cache_mode == "shared")
        self._shm_objects: list = []
        self._creator_pid: int = os.getpid()

        if self._cache_mode == "eager":
            self._preload_eager()
        elif self._cache_mode == "shared":
            self._preload_shared()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_case(self, identifier: str) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
        """Load a single case.

        Returns
        -------
        data : np.ndarray, shape ``(C, *spatial)``
        seg : np.ndarray or None, shape ``(1, *spatial)``
        properties : dict
        """
        # Check cache (covers lazy, eager, and shared modes)
        if identifier in self._array_cache:
            cached = self._array_cache[identifier]
            props = self._load_properties(identifier)
            if cached.get("dist") is not None:
                props["distance_map"] = cached["dist"]
            return cached["data"], cached["seg"], props

        # Load from disk
        data, seg, dist_map = self._load_from_disk(identifier)
        properties = self._load_properties(identifier)

        # Lazy caching: store on first access
        if self._cache_mode == "lazy":
            entry_bytes = self._entry_bytes(data, seg, dist_map)
            if not self._would_exceed_limit(entry_bytes):
                self._array_cache[identifier] = {
                    "data": np.array(data),
                    "seg": np.array(seg) if seg is not None else None,
                    "dist": np.array(dist_map) if dist_map is not None else None,
                }
                self._cache_bytes += entry_bytes

        if dist_map is not None:
            properties["distance_map"] = dist_map

        return data, seg, properties

    def cleanup_shared_memory(self):
        """Release POSIX shared memory segments.

        Only takes effect in the process that created the segments (the main
        process).  Safe to call multiple times.
        """
        if os.getpid() != self._creator_pid:
            return
        for shm in self._shm_objects:
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass
        self._shm_objects.clear()
        self._array_cache.clear()
        self._cache_bytes = 0

    # ------------------------------------------------------------------
    # Internal: disk I/O
    # ------------------------------------------------------------------

    def _load_from_disk(
        self, identifier: str
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Load raw arrays from disk (no caching)."""
        npz_path = os.path.join(self.folder, identifier + ".npz")
        npy_data_path = os.path.join(self.folder, identifier + ".npy")
        npy_seg_path = os.path.join(self.folder, identifier + "_seg.npy")
        npy_dist_path = os.path.join(self.folder, identifier + "_dist.npy")

        # Prefer unpacked .npy files (faster) over .npz
        data_from_npy = os.path.isfile(npy_data_path)
        seg_from_npy = os.path.isfile(npy_seg_path)
        dist_from_npy = self.load_distance_maps and os.path.isfile(npy_dist_path)

        data = np.load(npy_data_path, mmap_mode="r") if data_from_npy else None
        seg = np.load(npy_seg_path, mmap_mode="r") if seg_from_npy else None
        dist_map = np.load(npy_dist_path, mmap_mode="r") if dist_from_npy else None

        needs_npz = (
            not data_from_npy
            or not seg_from_npy
            or (self.load_distance_maps and not dist_from_npy)
        )
        if needs_npz and os.path.isfile(npz_path):
            with np.load(npz_path) as npz:
                if data is None:
                    data = npz["data"]
                if not seg_from_npy:
                    seg = npz["seg"] if "seg" in npz else None
                if self.load_distance_maps and dist_map is None and "distance_map" in npz:
                    dist_map = npz["distance_map"]

        if data is None:
            raise ValueError(f"Data not found for case {identifier} in either .npy or .npz format.")
        return data, seg, dist_map

    def _load_properties(self, identifier: str) -> dict:
        if identifier not in self._properties_cache:
            pkl_path = os.path.join(self.folder, identifier + ".pkl")
            with open(pkl_path, "rb") as fh:
                self._properties_cache[identifier] = pickle.load(fh)
        return self._properties_cache[identifier]

    # ------------------------------------------------------------------
    # Internal: caching helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _entry_bytes(data, seg, dist_map) -> int:
        n = data.nbytes
        if seg is not None:
            n += seg.nbytes
        if dist_map is not None:
            n += dist_map.nbytes
        return n

    def _would_exceed_limit(self, entry_bytes: int) -> bool:
        return (
            self._cache_max_bytes > 0
            and (self._cache_bytes + entry_bytes) > self._cache_max_bytes
        )

    def _preload_eager(self):
        """Pre-populate the array cache before workers fork.

        On Linux, forked workers share the data pages via copy-on-write, so
        total physical memory stays close to 1x the cached size.
        """
        logger.info("Eager cache: loading %d cases into RAM...", len(self.identifiers))
        # Pre-load properties so workers inherit them via COW too
        for ident in self.identifiers:
            self._load_properties(ident)
        for ident in self.identifiers:
            if self._cache_max_bytes > 0 and self._cache_bytes >= self._cache_max_bytes:
                break
            data, seg, dist = self._load_from_disk(ident)
            entry_bytes = self._entry_bytes(data, seg, dist)
            if not self._would_exceed_limit(entry_bytes):
                self._array_cache[ident] = {
                    "data": np.array(data),
                    "seg": np.array(seg) if seg is not None else None,
                    "dist": np.array(dist) if dist is not None else None,
                }
                self._cache_bytes += entry_bytes
        logger.info(
            "Cached %d/%d cases (%.1f GB).",
            len(self._array_cache), len(self.identifiers),
            self._cache_bytes / 1024**3,
        )

    def _preload_shared(self):
        """Pre-populate the array cache using POSIX shared memory.

        All forked workers share the exact same physical memory — no COW
        overhead at all for the bulk array data.
        """
        logger.info("Shared memory cache: loading %d cases...", len(self.identifiers))
        # Pre-load properties so workers inherit them via COW
        for ident in self.identifiers:
            self._load_properties(ident)
        for ident in self.identifiers:
            if self._cache_max_bytes > 0 and self._cache_bytes >= self._cache_max_bytes:
                break
            data, seg, dist = self._load_from_disk(ident)
            entry_bytes = self._entry_bytes(data, seg, dist)
            if not self._would_exceed_limit(entry_bytes):
                self._array_cache[ident] = {
                    "data": self._to_shared_array(data),
                    "seg": self._to_shared_array(seg) if seg is not None else None,
                    "dist": self._to_shared_array(dist) if dist is not None else None,
                }
                self._cache_bytes += entry_bytes
        logger.info(
            "Loaded %d/%d cases into shared memory (%.1f GB).",
            len(self._array_cache), len(self.identifiers),
            self._cache_bytes / 1024**3,
        )
        atexit.register(self.cleanup_shared_memory)

    def _to_shared_array(self, arr: np.ndarray) -> np.ndarray:
        """Copy *arr* into a new POSIX shared memory segment."""
        from multiprocessing.shared_memory import SharedMemory

        buf = np.ascontiguousarray(arr)
        shm = SharedMemory(create=True, size=buf.nbytes)
        shared = np.ndarray(buf.shape, dtype=buf.dtype, buffer=shm.buf)
        shared[:] = buf
        shared.flags.writeable = False
        self._shm_objects.append(shm)
        return shared

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.identifiers)

    def __del__(self):
        # Close our local handles to shared memory segments.
        # unlink() is handled by atexit in the creator process only.
        for shm in self._shm_objects:
            try:
                shm.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_identifiers(folder: str) -> List[str]:
        """Discover case identifiers from ``.npz`` files in *folder*."""
        return sorted(f[:-4] for f in os.listdir(folder) if f.endswith(".npz"))

    @staticmethod
    def unpack_dataset(
        folder: str,
        overwrite_existing: bool = False,
        num_processes: int = 4,
        verify: bool = True,
        delete_npz: bool = False,
    ):
        """Unpack ``.npz`` → ``.npy`` for faster loading.

        Creates ``{id}.npy`` (data) and ``{id}_seg.npy`` (seg) next to the
        original ``.npz``. If *delete_npz* is True, the source ``.npz`` is
        removed once its sidecars are written; cases that were already
        unpacked on a previous run also have their redundant ``.npz`` cleaned
        up (provided every key in the archive has its .npy sidecar present).
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed

        npz_files = [f for f in os.listdir(folder) if f.endswith(".npz")]
        tasks = []
        already_unpacked: List[str] = []

        for fname in npz_files:
            identifier = fname[:-4]
            npy_data = os.path.join(folder, identifier + ".npy")

            if not overwrite_existing and os.path.isfile(npy_data):
                already_unpacked.append(identifier)
                continue
            tasks.append((folder, identifier))

        if tasks:
            logger.info("Unpacking %d .npz files to .npy...", len(tasks))
            with ProcessPoolExecutor(max_workers=num_processes) as pool:
                futures = [
                    pool.submit(_unpack_one, f, i, delete_npz) for f, i in tasks
                ]
                for fut in as_completed(futures):
                    fut.result()  # raise on error

        if delete_npz and already_unpacked:
            removed = sum(
                _try_delete_redundant_npz(folder, ident) for ident in already_unpacked
            )
            if removed:
                logger.info(
                    "Removed %d redundant .npz (sidecars already present).", removed,
                )
