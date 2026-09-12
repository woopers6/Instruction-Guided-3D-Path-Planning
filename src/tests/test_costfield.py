import numpy as np
import pytest

from ig3d import costfield as CF
from ig3d.instructions import CLASSES


def field(k, c=0.5, a_up=0.0, a_dn=0.0, shape=(4, 4, 4)):
    f = np.zeros((k,) + shape, np.float32)
    f[0] = c
    if k == 2:
        f[1] = a_up
    if k == 3:
        f[1], f[2] = a_up, a_dn
    return f


def test_step_cost_hand_values():
    f = field(3, c=0.5, a_up=0.3, a_dn=0.1)
    assert CF.step_cost(f, (1, 1, 1), (1, 1, 2)) == pytest.approx(0.5)
    assert CF.step_cost(f, (1, 1, 1), (2, 1, 1)) == pytest.approx(0.8)
    assert CF.step_cost(f, (2, 1, 1), (1, 1, 1)) == pytest.approx(0.6)
    assert CF.step_cost(f, (1, 1, 1), (2, 2, 2)) == pytest.approx(0.5 * 3 ** 0.5 + 0.3)


def test_k1_is_k3_with_zero_vertical_extras():
    f3, f1 = field(3, c=0.7), field(1, c=0.7)
    for u, v in (((1, 1, 1), (2, 2, 2)), ((1, 1, 1), (1, 2, 2)), ((2, 1, 1), (1, 1, 2))):
        assert CF.step_cost(f3, u, v) == pytest.approx(CF.step_cost(f1, u, v))


def test_isotropic_diagonal_is_euclidean():
    assert CF.step_cost(field(3, c=1.0), (1, 1, 1), (2, 2, 1)) == pytest.approx(2 ** 0.5)


def test_segment_costs_are_directed():
    f = field(3, c=0.5, a_up=0.4, a_dn=0.0, shape=(12, 4, 4))
    fwd, bwd = CF.segment_costs(f, np.array([[1.0, 1, 1]]), np.array([[9.0, 1, 1]]))
    assert fwd[0] == pytest.approx(8 * 0.9)
    assert bwd[0] == pytest.approx(8 * 0.5)


def test_gt_fields_obey_bounds(vols, cfg):
    v = vols[0]
    feats = CF.volume_features(v, cfg.cost)
    for cls in CLASSES:
        f = CF.gt_field_from_features(v.occ, feats, cls, cfg.cost, 3)
        free = v.free
        assert f[0][free].min() >= cfg.cost.c_min - 1e-6 > 0, "rule R1: base cost > 0"
        assert f[0][free].max() <= cfg.cost.c_max + 1e-6
        assert f[1:][:, free].min() >= 0
        assert f[1:][:, free].max() <= cfg.cost.span + 1e-6


def test_projection_is_exact_for_isotropic_fields(vols, cfg):
    v = vols[0]
    f3 = CF.gt_field_from_features(v.occ, CF.volume_features(v, cfg.cost), "open", cfg.cost)
    assert np.allclose(CF.project_to_k(f3, 1)[0], f3[0])


def test_k2_cannot_tell_climb_from_descend(vols, cfg):
    v = vols[0]
    feats = CF.volume_features(v, cfg.cost)
    a = CF.gt_field_from_features(v.occ, feats, "climb_open", cfg.cost, 2)
    b = CF.gt_field_from_features(v.occ, feats, "descend_open", cfg.cost, 2)
    assert np.allclose(a, b)


def test_normalise_round_trip(vols, cfg):
    for cls in ("open", "level", "climb_open"):
        f = CF.gt_field(vols[0], cls, cfg.cost, 3)
        assert np.allclose(CF.denormalise(CF.normalise(f, cfg.cost), cfg.cost), f, atol=1e-5)
