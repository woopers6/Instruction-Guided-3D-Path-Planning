"""Volumes + features on disk, and a torch Dataset that pairs them with instructions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import CostConfig, VolumeConfig
from .costfield import gt_field_from_features, normalise, volume_features
from .instructions import ALL_LABELS, CLASSES
from .volumes import generate

from .paths import CACHE as CACHE_DIR
FEATURE_KEYS = ("openness", "is_opening", "rel_width")


class VolumeBank:
    """N volumes as stacked arrays: occupancy plus the per-volume features."""

    def __init__(self, occ, feats):
        self.occ = occ
        self.feats = feats

    def __len__(self):
        return len(self.occ)

    def features(self, i) -> dict:
        return {k: self.feats[k][i].astype(np.float32) for k in FEATURE_KEYS}

    @classmethod
    def build(cls, n, seed, vol_cfg: VolumeConfig, cost_cfg: CostConfig):
        vols = generate(n, seed, vol_cfg)
        occ = np.stack([v.occ for v in vols])
        feats = {k: np.empty(occ.shape, np.float16) for k in FEATURE_KEYS}
        for i, v in enumerate(vols):
            f = volume_features(v, cost_cfg)
            for k in FEATURE_KEYS:
                feats[k][i] = f[k]
        return cls(occ, feats)

    @classmethod
    def load_or_build(cls, n, seed, vol_cfg, cost_cfg, tag=""):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        key = "bank_{}_n{}_s{}_{}x{}x{}_r{}.npz".format(
            tag, n, seed, *vol_cfg.size, cost_cfg.influence_r)
        path = CACHE_DIR / key
        if path.exists():
            z = np.load(path)
            return cls(z["occ"], {k: z[k] for k in FEATURE_KEYS})
        bank = cls.build(n, seed, vol_cfg, cost_cfg)
        np.savez(path, occ=bank.occ, **bank.feats)
        return bank


class FieldDataset(Dataset):
    """One item = one volume paired with one instruction drawn from `sent_idx`."""

    def __init__(self, bank: VolumeBank, sent_idx, cond_table: np.ndarray, k: int,
                 cost_cfg: CostConfig, samples_per_volume: int = 4, seed: int = 0):
        self.bank = bank
        self.sent_idx = np.asarray(sent_idx)
        self.cond = cond_table.astype(np.float32)
        self.k = k
        self.cost_cfg = cost_cfg
        self.spv = samples_per_volume
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.bank) * self.spv

    def set_epoch(self, e):
        self.epoch = e

    def __getitem__(self, i):
        vi = i % len(self.bank)
        rng = np.random.default_rng((self.seed, self.epoch, i))
        si = int(rng.choice(self.sent_idx))
        cls = CLASSES[int(ALL_LABELS[si])]
        occ = self.bank.occ[vi]
        field = gt_field_from_features(occ, self.bank.features(vi), cls, self.cost_cfg,
                                       self.k)
        return (torch.from_numpy(occ.astype(np.float32))[None],
                torch.from_numpy(self.cond[si]),
                torch.from_numpy(normalise(field, self.cost_cfg)),
                si)
