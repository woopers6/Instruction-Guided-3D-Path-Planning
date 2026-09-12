"""Synthetic 3D environments."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .config import VolumeConfig

_SIX = ndimage.generate_binary_structure(3, 1)


@dataclass
class Window:
    """One opening in a wall, labelled so evaluation can ask which one was used."""
    wall_id: int
    kind: str
    box: tuple[int, int, int, int, int, int]
    width: int

    @property
    def z_center(self) -> float:
        return 0.5 * (self.box[0] + self.box[1])


@dataclass
class Volume:
    occ: np.ndarray
    start: tuple[int, int, int]
    goal: tuple[int, int, int]
    windows: list[Window] = field(default_factory=list)
    walls: list[tuple[int, int]] = field(default_factory=list)

    @property
    def free(self) -> np.ndarray:
        return self.occ == 0

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.occ.shape


def window_mask(win: Window, shape) -> np.ndarray:
    """Boolean mask of the voxels forming one opening."""
    m = np.zeros(shape, bool)
    z0, z1, y0, y1, x0, x1 = win.box
    m[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] = True
    return m


def connected(free: np.ndarray, start, goal) -> bool:
    """6-connected reachability via connected-component labelling."""
    lab, _ = ndimage.label(free, structure=_SIX)
    return bool(lab[start] != 0 and lab[start] == lab[goal])


def _spaced_positions(rng, lo, hi, n, sep, tries=300):
    """n integers in [lo, hi], pairwise at least `sep` apart, or None."""
    if hi < lo:
        return None
    for _ in range(tries):
        p = np.sort(rng.integers(lo, hi + 1, size=n))
        if n == 1 or np.all(np.diff(p) >= sep):
            return [int(v) for v in p]
    return None


def _try_sample(rng, cfg: VolumeConfig) -> Volume | None:
    Z, Y, X = cfg.size
    b, t = cfg.border, cfg.wall_thickness

    occ = np.zeros((Z, Y, X), np.uint8)
    occ[:b] = occ[-b:] = 1
    occ[:, :b] = occ[:, -b:] = 1
    occ[:, :, :b] = occ[:, :, -b:] = 1

    n = int(rng.integers(cfg.n_walls[0], cfg.n_walls[1] + 1))
    ys = _spaced_positions(rng, b + 8, Y - b - 8 - t, n, cfg.min_wall_separation)
    if ys is None:
        return None
    walls = [(y, y + t - 1) for y in ys]

    windows: list[Window] = []
    halves = [(b + 1, X // 2 - 1), (X // 2 + 1, X - b - 2)]
    for wid, (y0, y1) in enumerate(walls):
        wn = int(rng.integers(cfg.narrow_window[0], cfg.narrow_window[1] + 1))
        ww = int(rng.integers(cfg.wide_window[0], cfg.wide_window[1] + 1))
        order = [("narrow", wn), ("wide", ww)]
        if rng.random() < 0.5:
            order.reverse()
        for (kind, w), (xl, xh) in zip(order, halves):
            x0 = int(rng.integers(xl, xh - w + 2))
            z0 = int(rng.integers(b + 1, Z - b - 1 - w + 1))
            windows.append(Window(wid, kind, (z0, z0 + w - 1, y0, y1, x0, x0 + w - 1), w))

    n_pil = int(rng.integers(cfg.n_pillars[0], cfg.n_pillars[1] + 1))
    keep_out = [(y0 - 2, y1 + 2) for (y0, y1) in walls]
    pillars = []
    for _ in range(200):
        if len(pillars) >= n_pil:
            break
        py = int(rng.integers(b + 5, Y - b - 7))
        px = int(rng.integers(b + 2, X - b - 3))
        if any(not (py + 1 < lo or py > hi) for (lo, hi) in keep_out):
            continue
        pillars.append((py, px))

    for (y0, y1) in walls:
        occ[:, y0:y1 + 1, :] = 1
    for (py, px) in pillars:
        occ[b:Z - b, py:py + 2, px:px + 2] = 1
    for win in windows:
        occ[window_mask(win, occ.shape)] = 0

    zs = int(rng.integers(b + 2, Z - b - 2))
    start = (zs, b + 2, int(rng.integers(b + 2, X - b - 2)))
    goal = (zs, Y - b - 3, int(rng.integers(b + 2, X - b - 2)))
    for (z, y, x) in (start, goal):
        occ[max(b, z - 1):min(Z - b, z + 2),
            max(b, y - 1):min(Y - b, y + 2),
            max(b, x - 1):min(X - b, x + 2)] = 0

    if not connected(occ == 0, start, goal):
        return None
    return Volume(occ=occ, start=start, goal=goal, windows=windows, walls=walls)


def sample_volume(rng: np.random.Generator, cfg: VolumeConfig | None = None,
                  max_tries: int = 100) -> Volume:
    cfg = cfg or VolumeConfig()
    for _ in range(max_tries):
        v = _try_sample(rng, cfg)
        if v is not None:
            return v
    raise RuntimeError("volume rejection sampling failed after %d tries" % max_tries)


def generate(n: int, seed: int, cfg: VolumeConfig | None = None) -> list[Volume]:
    rng = np.random.default_rng(seed)
    return [sample_volume(rng, cfg) for _ in range(n)]
