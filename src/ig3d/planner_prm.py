"""3D PRM with cost-weighted sampling and anisotropic, DIRECTED edge costs."""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from .config import PRMConfig
from .costfield import segment_costs


def sample_nodes(occ, field, start, goal, cfg: PRMConfig, rng):
    """(N, 3) float nodes; index 0 is start, index 1 is goal."""
    free_idx = np.flatnonzero(occ.ravel() == 0)
    w = 1.0 / (field[0].ravel()[free_idx] + cfg.inv_eps)
    w = w / w.sum()
    n = min(cfg.n_nodes, free_idx.size)
    pick = rng.choice(free_idx, size=n, replace=False, p=w)
    pts = np.stack(np.unravel_index(pick, occ.shape), axis=1).astype(float)
    jitter = rng.uniform(-0.45, 0.45, size=pts.shape)
    if cfg.integer_z:
        jitter[:, 0] = 0.0
    pts += jitter
    return np.concatenate([np.asarray([start, goal], float), pts])


def collision_free(A, B, occ, step: float = 0.5, chunk: int = 20000) -> np.ndarray:
    """Boolean per segment: True if no sample along A->B lands in an occupied voxel."""
    out = np.empty(len(A), bool)
    hi = np.asarray(occ.shape) - 1
    for s in range(0, len(A), chunk):
        a, b = A[s:s + chunk], B[s:s + chunk]
        d = b - a
        S = max(2, int(np.ceil(np.linalg.norm(d, axis=1).max() / step)) + 1)
        t = np.linspace(0.0, 1.0, S)
        P = a[:, None, :] + t[None, :, None] * d[:, None, :]
        idx = np.clip(np.rint(P).astype(np.int64), 0, hi)
        out[s:s + chunk] = ~occ[idx[..., 0], idx[..., 1], idx[..., 2]].astype(bool).any(1)
    return out


def build_roadmap(pts, occ, field, cfg: PRMConfig):
    """Directed sparse graph over the nodes."""
    n = len(pts)
    k = min(cfg.k_neighbours + 1, n)
    dist, nb = cKDTree(pts).query(pts, k=k)
    I = np.repeat(np.arange(n), k - 1)
    J = nb[:, 1:].ravel()
    D = dist[:, 1:].ravel()
    keep = np.isfinite(D) & (D <= cfg.max_edge_len)
    a = np.minimum(I[keep], J[keep])
    b = np.maximum(I[keep], J[keep])
    pairs = np.unique(np.stack([a, b], axis=1), axis=0)
    pairs = pairs[collision_free(pts[pairs[:, 0]], pts[pairs[:, 1]], occ,
                                 cfg.collision_step)]
    fwd, bwd = segment_costs(field, pts[pairs[:, 0]], pts[pairs[:, 1]], cfg.cost_step)
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    return csr_matrix((np.concatenate([fwd, bwd]).astype(np.float64), (rows, cols)),
                      shape=(n, n))


def plan(occ, field, start, goal, cfg: PRMConfig | None = None, rng=None):
    """(path (N,3) float, cost) or (None, inf)."""
    cfg = cfg or PRMConfig()
    rng = rng if rng is not None else np.random.default_rng(0)
    pts = sample_nodes(occ, field, start, goal, cfg, rng)
    G = build_roadmap(pts, occ, field, cfg)
    dist, pred = dijkstra(G, directed=True, indices=0, return_predecessors=True)
    if not np.isfinite(dist[1]):
        return None, float("inf")
    seq = [1]
    while seq[-1] != 0:
        seq.append(int(pred[seq[-1]]))
    return pts[seq[::-1]], float(dist[1])


def densify(path, step: float = 0.5) -> np.ndarray:
    """Resample a polyline at ~`step` spacing, for voxel-level metrics."""
    path = np.asarray(path, float)
    out = [path[:1]]
    for a, b in zip(path[:-1], path[1:]):
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / step)) + 1)
        t = np.linspace(0.0, 1.0, n)[1:, None]
        out.append(a * (1 - t) + b * t)
    return np.concatenate(out)
