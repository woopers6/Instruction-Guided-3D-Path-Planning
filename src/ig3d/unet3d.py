"""3D U-Net predicting a K-channel cost field."""
from __future__ import annotations

import torch
import torch.nn as nn


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm3d(cout),
        nn.ReLU(inplace=True),
        nn.Conv3d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm3d(cout),
        nn.ReLU(inplace=True),
    )


class UNet3D(nn.Module):
    """Standard 3D U-Net. depth=3 on 48^3 goes 48 -> 24 -> 12 -> 6."""

    def __init__(self, in_ch: int, out_ch: int, base: int = 16, depth: int = 3):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.downs = nn.ModuleList()
        c = in_ch
        for i in range(depth):
            self.downs.append(conv_block(c, chs[i]))
            c = chs[i]
        self.pool = nn.MaxPool3d(2)
        self.bottom = conv_block(c, chs[depth])
        self.ups = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        c = chs[depth]
        for i in reversed(range(depth)):
            self.ups.append(nn.ConvTranspose3d(c, chs[i], 2, stride=2))
            self.up_blocks.append(conv_block(2 * chs[i], chs[i]))
            c = chs[i]
        self.head = nn.Conv3d(c, out_ch, 1)

    def forward(self, x):
        skips = []
        for d in self.downs:
            x = d(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottom(x)
        for up, blk, skip in zip(self.ups, self.up_blocks, reversed(skips)):
            x = blk(torch.cat([skip, up(x)], dim=1))
        return self.head(x)


class CostNet(nn.Module):
    """Occupancy + broadcast instruction vector -> K-channel cost field (logits)."""

    def __init__(self, emb_dim: int, k_channels: int, base: int = 16, depth: int = 3,
                 adapter: int = 0):
        super().__init__()
        self.emb_dim = emb_dim
        self.tiled = ADAPTER_OUT if adapter else emb_dim
        self.net = UNet3D(1 + self.tiled, k_channels, base=base, depth=depth)
        self.adapter = None
        if adapter:
            self.register_buffer("cond_mean", torch.zeros(emb_dim))
            self.register_buffer("cond_std", torch.ones(emb_dim))
            self.adapter = nn.Sequential(
                nn.Linear(emb_dim, adapter), nn.ReLU(inplace=True),
                nn.Linear(adapter, adapter), nn.ReLU(inplace=True),
                nn.Linear(adapter, ADAPTER_OUT))

    def set_standardisation(self, mean, std):
        """Fixed input statistics for the adapter -- from TRAINING sentences only."""
        self.cond_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.cond_std.copy_(torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-3))

    def forward(self, occ, emb):
        if self.adapter is not None:
            emb = self.adapter((emb - self.cond_mean) / self.cond_std)
        e = emb.view(emb.shape[0], self.tiled, 1, 1, 1).expand(-1, -1, *occ.shape[2:])
        return self.net(torch.cat([occ, e], dim=1))


ADAPTER_OUT = 16
