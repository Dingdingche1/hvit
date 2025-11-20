import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Sequence, Tuple

from src.model.transformation import SpatialTransformer

ndims = 3


class ResidualConvUnit(nn.Module):
    """Simple residual block with instance normalization for 3D inputs."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
        )
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.residual(x))


class SmoothnessLoss(nn.Module):
    """First-order smoothness regularization for displacement fields."""

    def __init__(self, penalty: str = "l2") -> None:
        super().__init__()
        if penalty not in {"l1", "l2"}:
            raise ValueError("penalty must be either 'l1' or 'l2'")
        self.penalty = penalty

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        dy = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]
        dx = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]
        dz = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]

        if self.penalty == "l2":
            dy = dy.pow(2)
            dx = dx.pow(2)
            dz = dz.pow(2)
        else:
            dy = dy.abs()
            dx = dx.abs()
            dz = dz.abs()

        return (dx.mean() + dy.mean() + dz.mean()) / 3.0


class BendingEnergyLoss(nn.Module):
    """Second-order bending energy regularization for displacement fields."""

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        dxx = flow[:, :, 2:, :, :] - 2 * flow[:, :, 1:-1, :, :] + flow[:, :, :-2, :, :]
        dyy = flow[:, :, :, 2:, :] - 2 * flow[:, :, :, 1:-1, :] + flow[:, :, :, :-2, :]
        dzz = flow[:, :, :, :, 2:] - 2 * flow[:, :, :, :, 1:-1] + flow[:, :, :, :, :-2]

        dxy = flow[:, :, 1:, 1:, :] - flow[:, :, 1:, :-1, :] - flow[:, :, :-1, 1:, :] + flow[:, :, :-1, :-1, :]
        dxz = flow[:, :, 1:, :, 1:] - flow[:, :, 1:, :, :-1] - flow[:, :, :-1, :, 1:] + flow[:, :, :-1, :, :-1]
        dyz = flow[:, :, :, 1:, 1:] - flow[:, :, :, 1:, :-1] - flow[:, :, :, :-1, 1:] + flow[:, :, :, :-1, :-1]

        squared_terms = [
            dxx.pow(2).mean(), dyy.pow(2).mean(), dzz.pow(2).mean(),
            dxy.pow(2).mean(), dxz.pow(2).mean(), dyz.pow(2).mean(),
        ]
        return sum(squared_terms) / len(squared_terms)


class DiffuseMorphField(nn.Module):
    """DiffuseMorph deformation head producing displacement fields from multi-scale features."""

    def __init__(
        self,
        base_channels: int,
        pyramid_levels: Sequence[str],
        target_size: Sequence[int],
        upsample_mode: str = "trilinear",
        use_svf: bool = False,
        integration_steps: int = 5,
        smoothness_weight: float = 0.0,
        bending_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.pyramid_levels: List[str] = sorted(pyramid_levels, key=lambda name: int(name[1:]), reverse=True)
        self.target_size: Sequence[int] = target_size
        self.upsample_mode: str = upsample_mode
        self.use_svf: bool = use_svf
        self.integration_steps: int = integration_steps
        self.smoothness_weight: float = smoothness_weight
        self.bending_weight: float = bending_weight

        fusion_blocks: List[nn.Module] = []

        for _ in range(len(self.pyramid_levels) - 1):
            fusion_blocks.append(ResidualConvUnit(base_channels + base_channels, base_channels))

        self.fusion_blocks = nn.ModuleList(fusion_blocks)
        self.preprocess = ResidualConvUnit(base_channels, base_channels)
        self.flow_head = nn.Conv3d(base_channels, ndims, kernel_size=3, padding=1)

        self.spatial_transformer = SpatialTransformer(target_size)
        self.smooth_reg = SmoothnessLoss("l2") if smoothness_weight > 0 else None
        self.bending_reg = BendingEnergyLoss() if bending_weight > 0 else None

    def forward(self, pyramid_features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        features: List[torch.Tensor] = [pyramid_features[level] for level in self.pyramid_levels if level in pyramid_features]
        if not features:
            raise ValueError("No matching pyramid levels found for DiffuseMorphField input.")

        x = self.preprocess(features[0])
        for idx, skip in enumerate(features[1:]):
            x = self._upsample_to(x, skip.shape[2:])
            x = torch.cat([x, skip], dim=1)
            x = self.fusion_blocks[idx](x)

        x = self._upsample_to(x, self.target_size)
        flow = self.flow_head(x)

        if self.use_svf:
            flow = self._integrate_velocity(flow)

        reg_terms: Dict[str, torch.Tensor] = {}
        if self.smooth_reg is not None:
            reg_terms["smoothness"] = self.smoothness_weight * self.smooth_reg(flow)
        if self.bending_reg is not None:
            reg_terms["bending"] = self.bending_weight * self.bending_reg(flow)

        return flow, reg_terms

    def _upsample_to(self, tensor: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
        align_corners = self.upsample_mode in {"trilinear", "bilinear"}
        return F.interpolate(tensor, size=size, mode=self.upsample_mode, align_corners=align_corners)

    def _integrate_velocity(self, velocity: torch.Tensor) -> torch.Tensor:
        displacement = velocity / (2 ** self.integration_steps)
        for _ in range(self.integration_steps):
            displacement = displacement + self.spatial_transformer(displacement, displacement)
        return displacement
