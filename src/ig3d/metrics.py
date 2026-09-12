"""Path metrics, chosen so they cannot be satisfied by accident."""
from __future__ import annotations

import numpy as np

from .volumes import window_mask

PRIMARY = {
    "open": ("narrow_frac", -1),
    "tight": ("narrow_frac", +1),
    "low": ("mean_alt", -1),
    "high": ("mean_alt", +1),
    "shortest": ("length", -1),
    "level": ("vertical", -1),
    "climb_open": ("climb_clear", +1),
    "descend_open": ("descend_clear", +1),
}


def densify(path, step: float = 0.5) -> np.ndarray:
    """Resample a polyline at ~`step` spacing. A no-op in effect for grid paths."""
    path = np.asarray(path, float)
    if len(path) < 2:
        return path
    out = [path[:1]]
    for a, b in zip(path[:-1], path[1:]):
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / step)) + 1)
        t = np.linspace(0.0, 1.0, n)[1:, None]
        out.append(a * (1 - t) + b * t)
    return np.concatenate(out)


def _vox(pts, shape):
    return np.clip(np.rint(pts).astype(int), 0, np.asarray(shape) - 1)


def windows_used(vol, path):
    """[(wall_id, kind)] for every window the path passes through."""
    v = _vox(densify(path), vol.shape)
    used = []
    for w in vol.windows:
        if window_mask(w, vol.shape)[v[:, 0], v[:, 1], v[:, 2]].any():
            used.append((w.wall_id, w.kind))
    return used


def behaviour(vol, path, edt: np.ndarray) -> dict:
    """Every behavioural metric for one path."""
    p = densify(path)
    v = _vox(p, vol.shape)
    d = np.diff(p, axis=0)
    dz = d[:, 0]
    seg_len = np.linalg.norm(d, axis=1)
    mid = _vox(0.5 * (p[:-1] + p[1:]), vol.shape)
    clr_mid = edt[mid[:, 0], mid[:, 1], mid[:, 2]]
    clr = edt[v[:, 0], v[:, 1], v[:, 2]]

    used = windows_used(vol, path)
    n_narrow = sum(1 for _, k in used if k == "narrow")
    up, dn = dz > 0, dz < 0
    rise, fall = float(dz[up].sum()), float(-dz[dn].sum())

    def wmean(mask, w):
        return float((clr_mid[mask] * w[mask]).sum() / w[mask].sum()) if mask.any() else np.nan

    return dict(
        length=float(seg_len.sum()),
        mean_clear=float(clr.mean()),
        min_clear=float(clr.min()),
        mean_alt=float(p[:, 0].mean() / (vol.shape[0] - 1)),
        rise=rise, fall=fall, vertical=rise + fall,
        climb_clear=wmean(up, dz),
        descend_clear=wmean(dn, -dz),
        n_windows=len(used), n_narrow=n_narrow,
        narrow_frac=(n_narrow / len(used)) if used else np.nan,
    )


def regret(cost_under_truth: float, optimal_truth_cost: float) -> float:
    return float(cost_under_truth / optimal_truth_cost - 1.0)
