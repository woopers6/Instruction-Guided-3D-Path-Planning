import numpy as np

from ig3d.volumes import connected, window_mask


def test_shape_and_dtype(vols, cfg):
    for v in vols:
        assert v.occ.shape == cfg.volume.size
        assert v.occ.dtype == np.uint8
        assert set(np.unique(v.occ)) <= {0, 1}


def test_shell_is_closed(vols):
    for v in vols:
        for face in (v.occ[0], v.occ[-1], v.occ[:, 0], v.occ[:, -1],
                     v.occ[:, :, 0], v.occ[:, :, -1]):
            assert face.all(), "shell must be solid on all six faces"


def test_start_goal_free_same_height_connected(vols):
    for v in vols:
        assert v.occ[v.start] == 0 and v.occ[v.goal] == 0
        assert v.start[0] == v.goal[0], "same height keeps rise and fall free to trade"
        assert connected(v.free, v.start, v.goal)


def test_each_wall_has_one_narrow_one_wide_open_window(vols, cfg):
    for v in vols:
        for wid in range(len(v.walls)):
            ws = [w for w in v.windows if w.wall_id == wid]
            assert sorted(w.kind for w in ws) == ["narrow", "wide"]
            for w in ws:
                assert not v.occ[window_mask(w, v.shape)].any(), "window must be open"
                rng = cfg.volume.narrow_window if w.kind == "narrow" else cfg.volume.wide_window
                assert rng[0] <= w.width <= rng[1]


def test_walls_are_otherwise_solid(vols):
    for v in vols:
        for wid, (y0, y1) in enumerate(v.walls):
            band = v.occ[:, y0:y1 + 1, :].copy()
            for w in v.windows:
                if w.wall_id == wid:
                    z0, z1, _, _, x0, x1 = w.box
                    band[z0:z1 + 1, :, x0:x1 + 1] = 1
            assert band.all()
