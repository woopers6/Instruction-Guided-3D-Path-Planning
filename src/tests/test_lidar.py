import numpy as np

from ig3d.lidar import SIZE, block_occupancy, place_endpoints
from ig3d.volumes import connected


def test_block_occupancy_fills_below_surface_and_closes_shell():
    top = np.full((SIZE, SIZE), 100.0, np.float32)
    top[20:28, 10:30] = 140.0
    occ, h = block_occupancy(top, base=96.0, voxel_m=4.0)
    assert occ.shape == (SIZE, SIZE, SIZE)
    assert h[5, 5] == 1 and h[24, 20] == 11
    assert occ[:11, 24, 20].all() and not occ[11:SIZE - 1, 24, 20].any()
    for face in (occ[0], occ[-1], occ[:, 0], occ[:, -1], occ[:, :, 0], occ[:, :, -1]):
        assert face.all()


def test_endpoints_same_height_free_and_connected_over_a_wall():
    top = np.full((SIZE, SIZE), 100.0, np.float32)
    top[20:28, :] = 160.0
    occ, h = block_occupancy(top, base=96.0, voxel_m=4.0)
    ends = place_endpoints(occ, h, np.random.default_rng(0))
    assert ends is not None
    start, goal = ends
    assert start[0] == goal[0]
    assert occ[start] == 0 and occ[goal] == 0
    assert connected(occ == 0, start, goal)


def test_no_endpoints_when_a_column_reaches_the_ceiling_everywhere():
    top = np.full((SIZE, SIZE), 100.0, np.float32)
    top[20:28, :] = 1000.0
    occ, h = block_occupancy(top, base=96.0, voxel_m=4.0)
    assert place_endpoints(occ, h, np.random.default_rng(0)) is None
