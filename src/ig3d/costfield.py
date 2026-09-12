"""The anisotropic cost field: representation, edge costs, and ground truth."""
from __future__ import annotations

import itertools

import numpy as np

from . import features as F
from .config import CostConfig

W_OPEN, W_CLEAR = 0.6, 0.4

CHANNEL_NAMES = {1: ("c",), 2: ("c", "a_v"), 3: ("c", "a_up", "a_dn")}

_DIRS = [d for d in itertools.product((-1, 0, 1), repeat=3) if d != (0, 0, 0)]
BETA = (sum(abs(d[0]) * np.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
            for d in _DIRS if d[0] > 0)
        / sum(d[0] ** 2 + d[1] ** 2 + d[2] ** 2 for d in _DIRS))


def combine(m: np.ndarray, dz: np.ndarray, lv: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Vectorised step cost. m: (K, E) mean coefficients; dz, lv, L: (E,)."""
    k = m.shape[0]
    w = m[0] * L
    if k == 2:
        w = w + m[1] * lv
    elif k == 3:
        w = w + np.where(dz > 0, m[1], np.where(dz < 0, m[2], 0.0)) * lv
    return w


def step_cost(field: np.ndarray, u, v) -> float:
    """Cost of one grid step u -> v (scalar reference implementation)."""
    dz, dy, dx = (int(v[i]) - int(u[i]) for i in range(3))
    L = float(np.sqrt(dz * dz + dy * dy + dx * dx))
    m = 0.5 * (field[:, u[0], u[1], u[2]] + field[:, v[0], v[1], v[2]])
    return float(combine(m[:, None], np.array([dz]), np.array([abs(dz)], float),
                         np.array([L]))[0])


def segment_costs(field: np.ndarray, A: np.ndarray, B: np.ndarray, step: float = 1.0):
    """Costs of straight segments A[i] -> B[i] and B[i] -> A[i]: (forward, backward)."""
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    d = B - A
    L = np.linalg.norm(d, axis=1)
    dz = d[:, 0]
    lv = np.abs(dz)
    S = max(2, int(np.ceil(L.max() / step)) + 1) if len(L) else 2
    t = np.linspace(0.0, 1.0, S)
    P = A[:, None, :] + t[None, :, None] * d[:, None, :]
    hi = np.asarray(field.shape[1:]) - 1
    idx = np.clip(np.rint(P).astype(np.int64), 0, hi)
    m = field[:, idx[..., 0], idx[..., 1], idx[..., 2]].mean(axis=2)
    return combine(m, dz, lv, L), combine(m, -dz, lv, L)


def polyline_cost(field: np.ndarray, pts, step: float = 1.0) -> float:
    """Total cost of a polyline (e.g. a PRM path) under a field."""
    pts = np.asarray(pts, float)
    if len(pts) < 2:
        return 0.0
    fwd, _ = segment_costs(field, pts[:-1], pts[1:], step)
    return float(fwd.sum())


def volume_features(vol, cfg: CostConfig) -> dict:
    """Instruction-independent features, computed once per volume."""
    edt = F.clearance(vol.free)
    is_opening, rel_width = F.window_fields(vol, cfg.influence_r, cfg.width_ref)
    return dict(edt=edt, openness=F.openness(edt, cfg.clearance_ref),
                is_opening=is_opening, rel_width=rel_width)


def gt_field_from_features(occ: np.ndarray, feats: dict, cls: str, cfg: CostConfig,
                           k: int = 3) -> np.ndarray:
    """(k, Z, Y, X) float32 ground-truth field for one (volume, instruction class)."""
    Z = occ.shape[0]
    base, span = cfg.base, cfg.span
    io, rw, op = feats["is_opening"], feats["rel_width"], feats["openness"]
    zz = np.broadcast_to((np.arange(Z, dtype=np.float32) / (Z - 1))[:, None, None],
                         occ.shape)
    c = np.full(occ.shape, base, np.float32)
    a_up = np.zeros(occ.shape, np.float32)
    a_dn = np.zeros(occ.shape, np.float32)

    if cls == "open":
        c = base + span * (W_OPEN * io * (1 - rw) + W_CLEAR * (1 - op))
    elif cls == "tight":
        c = base + span * (W_OPEN * io * rw + W_CLEAR * op)
    elif cls == "low":
        c = base + span * zz
    elif cls == "high":
        c = base + span * (1 - zz)
    elif cls == "shortest":
        pass
    elif cls == "level":
        a_up[:] = span
        a_dn[:] = span
    elif cls == "climb_open":
        a_up = span * (1 - op)
    elif cls == "descend_open":
        a_dn = span * (1 - op)
    else:
        raise ValueError(cls)

    field3 = np.stack([np.asarray(c, np.float32), a_up, a_dn]).astype(np.float32)
    field3[0][occ == 1] = cfg.c_max
    field3[1:, occ == 1] = 0.0
    return project_to_k(field3, k)


def gt_field(vol, cls: str, cfg: CostConfig, k: int = 3) -> np.ndarray:
    return gt_field_from_features(vol.occ, volume_features(vol, cfg), cls, cfg, k)


def project_to_k(field3: np.ndarray, k: int) -> np.ndarray:
    """Best per-voxel least-squares k-channel approximation of a 3-channel field."""
    if k == 3:
        return field3
    c, a_up, a_dn = field3
    if k == 2:
        return np.stack([c, 0.5 * (a_up + a_dn)]).astype(np.float32)
    if k == 1:
        return (c + BETA * (a_up + a_dn))[None].astype(np.float32)
    raise ValueError(k)


def channel_ranges(k: int, cfg: CostConfig):
    """(lo, hi) per channel, for normalising network targets into [0, 1]."""
    if k == 1:
        return [(cfg.c_min, cfg.c_max + 2 * BETA * cfg.span)]
    if k == 2:
        return [(cfg.c_min, cfg.c_max), (0.0, cfg.span)]
    return [(cfg.c_min, cfg.c_max), (0.0, cfg.span), (0.0, cfg.span)]


def normalise(field: np.ndarray, cfg: CostConfig) -> np.ndarray:
    out = np.empty_like(field, dtype=np.float32)
    for i, (lo, hi) in enumerate(channel_ranges(field.shape[0], cfg)):
        out[i] = np.clip((field[i] - lo) / (hi - lo), 0.0, 1.0)
    return out


def denormalise(unit: np.ndarray, cfg: CostConfig) -> np.ndarray:
    out = np.empty_like(unit, dtype=np.float32)
    for i, (lo, hi) in enumerate(channel_ranges(unit.shape[0], cfg)):
        out[i] = lo + (hi - lo) * np.clip(unit[i], 0.0, 1.0)
    return out
