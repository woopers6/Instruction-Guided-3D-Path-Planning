import torch

from ig3d.unet3d import CostNet


def test_output_shape_and_conditioning_matters():
    torch.manual_seed(0)
    net = CostNet(emb_dim=8, k_channels=3, base=4, depth=2).eval()
    occ = torch.zeros(2, 1, 16, 16, 16)
    e1, e2 = torch.zeros(2, 8), torch.zeros(2, 8)
    e2[:, 3] = 1.0
    with torch.no_grad():
        a, b = net(occ, e1), net(occ, e2)
    assert a.shape == (2, 3, 16, 16, 16)
    assert (a - b).abs().max() > 0


def test_adapter_conditions_and_round_trips():
    torch.manual_seed(0)
    net = CostNet(emb_dim=12, k_channels=3, base=4, depth=2, adapter=16).eval()
    net.set_standardisation(torch.full((12,), -6.0), torch.full((12,), 2.0))
    occ = torch.zeros(2, 1, 16, 16, 16)
    e1 = torch.full((2, 12), -6.0)
    e2 = e1.clone()
    e2[:, 0] = 0.0
    with torch.no_grad():
        a, b = net(occ, e1), net(occ, e2)
    assert a.shape == (2, 3, 16, 16, 16)
    assert (a - b).abs().max() > 0
    fresh = CostNet(emb_dim=12, k_channels=3, base=4, depth=2, adapter=16).eval()
    fresh.load_state_dict(net.state_dict())
    with torch.no_grad():
        assert torch.equal(fresh(occ, e2), b)


def test_default_has_no_adapter_parameters():
    keys = CostNet(emb_dim=8, k_channels=3, base=4, depth=2).state_dict().keys()
    assert not any(k.startswith(("adapter", "cond_")) for k in keys)
