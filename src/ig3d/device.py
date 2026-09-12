"""Device selection."""
from __future__ import annotations

import torch


def get_device(prefer: str | None = None) -> torch.device:
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe(dev: torch.device) -> str:
    if dev.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        return "{} ({:.1f} GB free of {:.1f} GB)".format(
            torch.cuda.get_device_name(0), free / 1e9, total / 1e9)
    return dev.type
