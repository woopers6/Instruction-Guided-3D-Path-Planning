import numpy as np

from ig3d import features as F
from ig3d.volumes import window_mask


def test_clearance_is_bounded_by_the_shell(vols):
    for v in vols:
        edt = F.clearance(v.free)
        assert edt[v.occ == 1].max() == 0
        assert edt[v.free].min() >= 1.0
        assert edt.max() < 30, "a huge maximum means the shell is missing"


def test_openness_range(vols, cfg):
    op = F.openness(F.clearance(vols[0].free), cfg.cost.clearance_ref)
    assert op.min() >= 0 and op.max() <= 1


def test_window_fields_separate_narrow_from_wide(vols, cfg):
    for v in vols:
        io, rw = F.window_fields(v, cfg.cost.influence_r, cfg.cost.width_ref)
        for w in v.windows:
            assert np.allclose(io[window_mask(w, v.shape)], 1.0)
        narrow = np.mean([rw[window_mask(w, v.shape)].mean()
                          for w in v.windows if w.kind == "narrow"])
        wide = np.mean([rw[window_mask(w, v.shape)].mean()
                        for w in v.windows if w.kind == "wide"])
        assert narrow < wide
