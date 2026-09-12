"""Geometric features the ground-truth cost fields are built from."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

from .volumes import window_mask


def clearance(free: np.ndarray) -> np.ndarray:
    """Euclidean distance from each free voxel to the nearest occupied voxel."""
    return distance_transform_edt(free).astype(np.float32)


def openness(edt: np.ndarray, clearance_ref: float) -> np.ndarray:
    """Normalised clearance in [0, 1]: 0 against a surface, 1 fully open."""
    return np.clip(edt / clearance_ref, 0.0, 1.0).astype(np.float32)


def window_fields(vol, influence_r: float, width_ref: float):
    """(is_opening, rel_width), both (Z, Y, X) float32 in [0, 1]."""
    shape = vol.shape
    core = np.zeros(shape, bool)
    width = np.zeros(shape, np.float32)
    for w in vol.windows:
        m = window_mask(w, shape)
        core |= m
        width[m] = w.width
    if not core.any():
        return np.zeros(shape, np.float32), np.ones(shape, np.float32)
    d, (iz, iy, ix) = distance_transform_edt(~core, return_indices=True)
    is_opening = np.clip(1.0 - d / influence_r, 0.0, 1.0).astype(np.float32)
    is_opening[~vol.free] = 0.0
    rel_width = np.clip(width[iz, iy, ix] / width_ref, 0.0, 1.0).astype(np.float32)
    return is_opening, rel_width
