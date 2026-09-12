import numpy as np

from ig3d.dataset import FieldDataset, VolumeBank
from ig3d.encoder import OneHotEncoder
from ig3d.instructions import ALL_SENTENCES, split_indices


def test_onehot_table():
    enc = OneHotEncoder()
    assert enc.table.shape == (len(ALL_SENTENCES), 8)
    assert np.allclose(enc.table.sum(1), 1)


def test_dataset_items(cfg):
    bank = VolumeBank.build(2, seed=5, vol_cfg=cfg.volume, cost_cfg=cfg.cost)
    tr, _ = split_indices()
    ds = FieldDataset(bank, tr, OneHotEncoder().table, 3, cfg.cost, samples_per_volume=2)
    occ, emb, tgt, _ = ds[0]
    assert occ.shape == (1,) + cfg.volume.size
    assert emb.shape == (8,)
    assert tgt.shape == (3,) + cfg.volume.size
    assert float(tgt.min()) >= 0 and float(tgt.max()) <= 1
