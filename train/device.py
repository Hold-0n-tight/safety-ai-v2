"""Device picker: cuda > mps > cpu."""
from __future__ import annotations

import torch


def pick_device() -> torch.device:
    """Return the best available torch device.

    Priority: CUDA > Apple Silicon MPS > CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")
