from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numpy import load
import torch
from pathlib import Path


def load_data(train_path: str | Path, valid_path: str | Path, ratio: float = 0.1):
    if valid_path is None:
        train_dataset = load(train_path, mmap_mode="r")
        train_num = int(len(train_dataset) * (1 - ratio))
        return train_dataset[:train_num], train_dataset[train_num:]

    train_dataset = load(train_path, mmap_mode="r")
    valid_dataset = load(valid_path, mmap_mode="r")
    return train_dataset, valid_dataset


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    starting_idxs = torch.randint(len(dataset) - context_length, (batch_size,))
    x = torch.stack([
            torch.from_numpy((dataset[i : i + context_length]).astype(np.int64))
            for i in starting_idxs
    ])  # fmt: skip
    y = torch.stack(
        [
            torch.from_numpy((dataset[i + 1 : i + 1 + context_length]).astype(np.int64))
            for i in starting_idxs
        ]
    )  # fmt: skip
    if "cuda" in device:
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y
