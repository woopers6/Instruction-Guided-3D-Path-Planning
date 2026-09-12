"""Every tunable number in one place."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VolumeConfig:
    """Synthetic environments: the 3D analogue of IG-PRM's 2D maps."""
    size: tuple[int, int, int] = (48, 48, 48)
    border: int = 1
    n_walls: tuple[int, int] = (1, 3)
    wall_thickness: int = 2
    narrow_window: tuple[int, int] = (3, 4)
    wide_window: tuple[int, int] = (8, 11)
    n_pillars: tuple[int, int] = (0, 4)
    min_wall_separation: int = 10


@dataclass
class CostConfig:
    """Ground-truth cost field design."""
    c_max: float = 1.0
    detour_ratio: float = 8.0
    influence_r: float = 12.0
    clearance_ref: float = 6.0
    width_ref: float = 12.0

    @property
    def base(self) -> float:
        """Price of open space: the same for every instruction (rule R2)."""
        return self.c_max / self.detour_ratio

    @property
    def c_min(self) -> float:
        return self.base

    @property
    def span(self) -> float:
        return self.c_max - self.base


@dataclass
class ModelConfig:
    k_channels: int = 3
    base_channels: int = 16
    depth: int = 3
    conditioning: str = "onehot"
    emb_dim: int = 16

    encoder_model: str = "sentence-transformers/all-mpnet-base-v2"
    nli_model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


@dataclass
class TrainConfig:
    train_volumes: int = 600
    val_volumes: int = 60
    epochs: int = 20
    batch_size: int = 4
    lr: float = 1e-3
    seed: int = 0
    samples_per_volume: int = 4


@dataclass
class PRMConfig:
    """PRM settings."""
    n_nodes: int = 5000
    k_neighbours: int = 30
    max_edge_len: float = 12.0
    inv_eps: float = 0.1
    collision_step: float = 0.25
    cost_step: float = 1.0
    integer_z: bool = True


@dataclass
class Config:
    volume: VolumeConfig = field(default_factory=VolumeConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    prm: PRMConfig = field(default_factory=PRMConfig)


DEFAULT = Config()
