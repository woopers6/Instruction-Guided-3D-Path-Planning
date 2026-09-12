import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ig3d.instructions import ALL_SENTENCES, CLASSES, split_indices  # noqa: E402
from ig3d.unet3d import CostNet, UNet3D  # noqa: E402


def test_unet_preserves_spatial_shape():
    net = UNet3D(in_ch=4, out_ch=3, base=8, depth=2)
    x = torch.zeros(1, 4, 16, 16, 16)
    y = net(x)
    assert y.shape == (1, 3, 16, 16, 16)


def test_costnet_output_has_k_channels():
    for k in (1, 2, 3):
        net = CostNet(emb_dim=8, k_channels=k, base=8, depth=2)
        occ = torch.zeros(2, 1, 16, 16, 16)
        emb = torch.zeros(2, 8)
        assert net(occ, emb).shape == (2, k, 16, 16, 16)


def test_conditioning_actually_changes_the_output():
    torch.manual_seed(0)
    net = CostNet(emb_dim=8, k_channels=3, base=8, depth=2).eval()
    occ = torch.rand(1, 1, 16, 16, 16).round()
    a = net(occ, torch.zeros(1, 8))
    b = net(occ, torch.ones(1, 8) * 3.0)
    assert not torch.allclose(a, b, atol=1e-4), "output is independent of the instruction"


def test_costnet_runs_on_the_real_device():
    from ig3d.device import get_device
    dev = get_device()
    net = CostNet(emb_dim=8, k_channels=3, base=8, depth=2).to(dev).eval()
    occ = torch.zeros(1, 1, 16, 16, 16, device=dev)
    emb = torch.zeros(1, 8, device=dev)
    with torch.no_grad():
        out = net(occ, emb)
    assert out.shape == (1, 3, 16, 16, 16) and torch.isfinite(out).all()


def test_instruction_split_is_disjoint_and_stratified():
    tr, held = split_indices(n_heldout_per_class=4, seed=0)
    assert len(set(tr) & set(held)) == 0
    assert len(tr) + len(held) == len(ALL_SENTENCES)
    from ig3d.instructions import ALL_LABELS
    for ci in range(len(CLASSES)):
        assert (ALL_LABELS[held] == ci).sum() == 4


def test_onehot_encoder_shape():
    from ig3d.encoder import OneHotEncoder
    enc = OneHotEncoder(len(CLASSES))
    v = enc(np.array([0, 20, 100]))
    assert v.shape == (3, len(CLASSES))
    assert np.allclose(v.sum(axis=1), 1.0)


@pytest.mark.slow
def test_nli_encoder_separates_the_tier3_antonym_pair():
    from ig3d.config import Config
    from ig3d.encoder import NLIEncoder
    from ig3d.instructions import BY_CLASS
    enc = NLIEncoder(Config().model.nli_model)
    up = enc(BY_CLASS["climb_open"])
    dn = enc(BY_CLASS["descend_open"])
    assert up.shape[0] == len(BY_CLASS["climb_open"])
    sep = np.abs(up.mean(axis=0) - dn.mean(axis=0)).max()
    assert sep > 0.3, "no axis separates the tier-3 pair (max gap {:.3f})".format(sep)
