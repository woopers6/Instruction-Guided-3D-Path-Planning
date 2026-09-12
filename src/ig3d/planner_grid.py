"""Exact 26-connected 3D Dijkstra under an anisotropic field."""
from __future__ import annotations

import itertools

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .costfield import combine

OFFSETS = [d for d in itertools.product((-1, 0, 1), repeat=3) if d != (0, 0, 0)]


class GridGraph:
    """Topology of one volume, reusable across every cost field planned on it."""

    def __init__(self, occ: np.ndarray):
        self.shape = occ.shape
        Z, Y, X = occ.shape
        self.N = Z * Y * X
        free = occ == 0
        pad = np.zeros((Z + 2, Y + 2, X + 2), bool)
        pad[1:-1, 1:-1, 1:-1] = free

        def shifted(d):
            dz, dy, dx = d
            return pad[1 + dz:1 + dz + Z, 1 + dy:1 + dy + Y, 1 + dx:1 + dx + X]

        strides = (Y * X, X, 1)
        U, V, DZ, LV, L = [], [], [], [], []
        for d in OFFSETS:
            ok = free & shifted(d)
            nz = [i for i in range(3) if d[i] != 0]
            for r in range(1, len(nz)):
                for sub in itertools.combinations(nz, r):
                    ok &= shifted(tuple(d[i] if i in sub else 0 for i in range(3)))
            u = np.flatnonzero(ok)
            U.append(u)
            V.append(u + sum(d[i] * strides[i] for i in range(3)))
            DZ.append(np.full(u.size, d[0], np.int8))
            LV.append(np.full(u.size, abs(d[0]), np.float32))
            L.append(np.full(u.size, np.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2),
                             np.float32))
        self.U = np.concatenate(U)
        self.V = np.concatenate(V)
        self.DZ = np.concatenate(DZ)
        self.LV = np.concatenate(LV)
        self.L = np.concatenate(L)

    def weights(self, field: np.ndarray) -> np.ndarray:
        ff = field.reshape(field.shape[0], -1)
        m = 0.5 * (ff[:, self.U] + ff[:, self.V])
        return combine(m, self.DZ, self.LV, self.L)

    def plan(self, field: np.ndarray, start, goal):
        """(path (N,3) int in (z, y, x), total cost) or (None, inf)."""
        w = self.weights(field).astype(np.float64)
        G = csr_matrix((w, (self.U, self.V)), shape=(self.N, self.N))
        s = int(np.ravel_multi_index(start, self.shape))
        g = int(np.ravel_multi_index(goal, self.shape))
        dist, pred = dijkstra(G, directed=True, indices=s, return_predecessors=True)
        if not np.isfinite(dist[g]):
            return None, float("inf")
        seq = [g]
        while seq[-1] != s:
            seq.append(int(pred[seq[-1]]))
        path = np.stack(np.unravel_index(np.asarray(seq[::-1]), self.shape), axis=1)
        return path.astype(int), float(dist[g])


def grid_plan(occ, field, start, goal):
    return GridGraph(occ).plan(field, start, goal)


def path_cost(field: np.ndarray, path) -> float:
    """Total cost of an EXISTING voxel path under a (possibly different) field."""
    p = np.asarray(path, int)
    if len(p) < 2:
        return 0.0
    d = np.diff(p, axis=0)
    L = np.sqrt((d ** 2).sum(1)).astype(float)
    ff = field.reshape(field.shape[0], -1)
    U = np.ravel_multi_index(p[:-1].T, field.shape[1:])
    V = np.ravel_multi_index(p[1:].T, field.shape[1:])
    m = 0.5 * (ff[:, U] + ff[:, V])
    return float(combine(m, d[:, 0], np.abs(d[:, 0]).astype(float), L).sum())


def ascent_descent(path) -> tuple[float, float]:
    """(total rise, total fall) along a path."""
    dz = np.diff(np.asarray(path, float)[:, 0])
    return float(dz[dz > 0].sum()), float(-dz[dz < 0].sum())


def horizontal_length(path) -> float:
    d = np.diff(np.asarray(path, float), axis=0)
    return float(np.hypot(d[:, 1], d[:, 2]).sum())


def length(path) -> float:
    d = np.diff(np.asarray(path, float), axis=0)
    return float(np.sqrt((d ** 2).sum(1)).sum())
