import numpy as np
import pytest

from ig3d.costfield import gt_field_from_features, volume_features
from ig3d.planner_grid import GridGraph, ascent_descent, path_cost


def test_straight_corridor_known_optimum():
    occ = np.ones((5, 5, 12), np.uint8)
    occ[2, 2, 1:11] = 0
    f = np.full((1, 5, 5, 12), 0.5, np.float32)
    path, cost = GridGraph(occ).plan(f, (2, 2, 1), (2, 2, 10))
    assert len(path) == 10
    assert cost == pytest.approx(9 * 0.5)


def test_dijkstra_cost_matches_path_cost(vols, cfg):
    v = vols[0]
    f = gt_field_from_features(v.occ, volume_features(v, cfg.cost), "climb_open", cfg.cost)
    path, cost = GridGraph(v.occ).plan(f, v.start, v.goal)
    assert path_cost(f, path) == pytest.approx(cost, rel=1e-6)
    assert not v.occ[tuple(path.T)].any()


def test_uniform_split_is_degenerate(vols):
    v = vols[0]
    G = GridGraph(v.occ)
    flds = []
    for s in (0.2, 0.8):
        f = np.zeros((3,) + v.shape, np.float32)
        f[0], f[1], f[2] = 0.25, 1.5 * s, 1.5 * (1 - s)
        flds.append(f)
    (p0, _), (_, c1) = G.plan(flds[0], v.start, v.goal), G.plan(flds[1], v.start, v.goal)
    assert path_cost(flds[1], p0) == pytest.approx(c1, rel=1e-6)
    rise, fall = ascent_descent(p0)
    assert rise == pytest.approx(fall)
