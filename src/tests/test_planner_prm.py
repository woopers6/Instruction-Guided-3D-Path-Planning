import numpy as np
import pytest

from ig3d.costfield import gt_field_from_features, polyline_cost, volume_features
from ig3d.metrics import densify
from ig3d.planner_grid import GridGraph
from ig3d.planner_prm import plan


@pytest.mark.parametrize("cls", ["shortest", "level", "climb_open"])
def test_prm_finds_collision_free_near_optimal_paths(vols, cfg, cls):
    v = vols[0]
    f = gt_field_from_features(v.occ, volume_features(v, cfg.cost), cls, cfg.cost)
    _, cg = GridGraph(v.occ).plan(f, v.start, v.goal)
    path, cost = plan(v.occ, f, v.start, v.goal, cfg.prm, np.random.default_rng(0))
    assert path is not None
    vox = np.rint(densify(path, 0.5)).astype(int)
    assert not v.occ[vox[:, 0], vox[:, 1], vox[:, 2]].any()
    assert 0.85 * cg < cost < 2.0 * cg
    assert polyline_cost(f, path) == pytest.approx(cost, rel=0.02)
