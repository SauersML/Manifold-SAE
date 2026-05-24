"""Activation loader for cogito-L40 with DistributedSampler support.

Expected file layout:
    X_L40.npy   — (N, D=7168) float32 memmap
    (optional)
    meta.json   — per-row metadata (color name, template idx, etc.)

For K=1M training we expect N to be at least ~10M rows (≥10× over-completeness
on a 7168-dim residual stream), so memmap is mandatory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


@dataclass
class DataConfig:
    activations_path: str
    batch_size: int = 256
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True
    # Optional fp16 / bf16 cast at load time (saves bandwidth).
    cast_dtype: str = "float32"     # "float32" | "bfloat16" | "float16"


class ActivationDataset(Dataset):
    """Memory-mapped activations. Lazy per-row read; safe across workers."""

    def __init__(self, path: str, cast_dtype: str = "float32"):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Activations file not found: {path}\n"
                f"Expected cogito-L40 X_L40.npy (N, 7168) float32 memmap."
            )
        # mmap_mode='r' so we don't load into RAM.
        self._arr = np.load(path, mmap_mode="r")
        if self._arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {self._arr.shape}")
        self.cast_dtype = cast_dtype

    def __len__(self) -> int:
        return self._arr.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        row = np.asarray(self._arr[idx])               # materialise this slice only
        t = torch.from_numpy(row.copy())               # .copy() severs the mmap view
        if self.cast_dtype == "bfloat16":
            t = t.to(torch.bfloat16)
        elif self.cast_dtype == "float16":
            t = t.to(torch.float16)
        return t


def build_dataloader(
    cfg: DataConfig,
    *,
    rank: int = 0,
    world_size: int = 1,
    shuffle: bool = True,
    seed: int = 0,
) -> tuple[DataLoader, DistributedSampler | None]:
    """Build a distributed dataloader. When world_size==1, sampler is None."""
    ds = ActivationDataset(cfg.activations_path, cfg.cast_dtype)
    if world_size > 1:
        sampler = DistributedSampler(
            ds, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=seed,
            drop_last=cfg.drop_last,
        )
        shuffle = False  # sampler owns shuffling
    else:
        sampler = None
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
    )
    return loader, sampler
