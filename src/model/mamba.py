import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm import Mamba
from typing import List, Optional, Dict


class MambaBlock(nn.Module):
    """Single Mamba residual block for 3D inputs.

    The block flattens a 3D feature map into a sequence, applies a Mamba state
    space update, and then reshapes it back to volumetric format with a
    residual connection. Dropout is applied after the Mamba operator to keep the
    hierarchy stable when stacking many layers.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, d = x.shape
        seq = rearrange(x, "b c h w d -> b (h w d) c")
        residual = seq
        seq = self.norm(seq)
        seq = self.mamba(seq)
        seq = self.dropout(seq)
        seq = seq + residual
        x = rearrange(seq, "b (h w d) c -> b c h w d", h=h, w=w, d=d)
        return x


class SkipFusion(nn.Module):
    """Skip/fusion block to align encoder features with decoder states.

    The block adapts the encoder features to a target channel dimension and
    optionally fuses them with a decoder feature (which can arrive at a
    different spatial resolution). It keeps the interface compatible with
    volumetric deformation field decoders that expect aligned skip inputs.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.adapter = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.gate = nn.Conv3d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.mix = nn.Conv3d(out_channels * 2, out_channels, kernel_size=3, padding=1)

    def forward(
        self, encoder_feat: torch.Tensor, decoder_feat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        aligned = self.adapter(encoder_feat)
        if decoder_feat is None:
            return aligned

        decoder_resized = F.interpolate(
            decoder_feat,
            size=aligned.shape[2:],
            mode="trilinear",
            align_corners=False,
        )
        gate = torch.sigmoid(self.gate(torch.cat([aligned, decoder_resized], dim=1)))
        fused = torch.cat([aligned, gate * decoder_resized], dim=1)
        return self.mix(fused)


class MambaStage(nn.Module):
    """A stack of Mamba blocks operating at a single resolution."""

    def __init__(
        self,
        dim: int,
        depth: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [MambaBlock(dim, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class HierarchicalMamba(nn.Module):
    """Hierarchical multi-scale Mamba encoder for DDPM features.

    The encoder consumes a DDPM noise prediction or intermediate feature map and
    produces a top-down or bottom-up hierarchy of Mamba-refined features.

    Args:
        in_channels: Number of channels in the DDPM noise/features.
        stage_dims: Channel widths for each hierarchical stage. Provide them in
            [high, mid, low] order when ``direction="top_down"`` or the reverse
            for bottom-up propagation.
        stage_depths: Number of Mamba blocks per stage.
        direction: "top_down" (default) processes high→mid→low with strided
            downsampling. "bottom_up" reverses the pass and upsamples as it
            propagates back toward high-level abstraction.
        fusion_channels: Output channels for the skip/fusion adapters. If not
            provided, the stage dimension is used.
    """

    def __init__(
        self,
        in_channels: int,
        stage_dims: List[int],
        stage_depths: List[int],
        direction: str = "top_down",
        fusion_channels: Optional[List[int]] = None,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert len(stage_dims) == len(stage_depths), "stage_dims and stage_depths must align"
        self.direction = direction
        self.stage_dims = stage_dims
        self.stage_depths = stage_depths
        self.in_projection = nn.Conv3d(in_channels, stage_dims[0], kernel_size=1)

        self.stages = nn.ModuleList()
        self.resamplers = nn.ModuleList()
        self.fusions = nn.ModuleList()

        fusion_channels = fusion_channels or stage_dims

        for idx, (dim, depth) in enumerate(zip(stage_dims, stage_depths)):
            self.stages.append(
                MambaStage(dim, depth, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout)
            )
            self.fusions.append(SkipFusion(dim, fusion_channels[idx]))

            if idx < len(stage_dims) - 1:
                if direction == "top_down":
                    self.resamplers.append(
                        nn.Conv3d(dim, stage_dims[idx + 1], kernel_size=3, stride=2, padding=1)
                    )
                elif direction == "bottom_up":
                    self.resamplers.append(
                        nn.ConvTranspose3d(
                            dim, stage_dims[idx + 1], kernel_size=2, stride=2, output_padding=0
                        )
                    )
                else:
                    raise ValueError("direction must be either 'top_down' or 'bottom_up'")

    def forward(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        x = self.in_projection(x)
        encoder_features: List[torch.Tensor] = []
        fused_skips: List[torch.Tensor] = []

        for idx, stage in enumerate(self.stages):
            x = stage(x)
            encoder_features.append(x)
            fused_skips.append(self.fusions[idx](x))
            if idx < len(self.resamplers):
                x = self.resamplers[idx](x)

        if self.direction == "bottom_up":
            fused_skips = self._propagate_bottom_up(fused_skips)

        self._last_encoder_features = encoder_features
        return {"encoder": encoder_features, "skips": fused_skips}

    def align_with_decoder(self, decoder_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Fuse stored encoder features with decoder states for deformation decoding."""
        assert len(decoder_features) == len(self.fusions), "Decoder features must match encoder stages"
        fused: List[torch.Tensor] = []
        for fusion, enc, dec in zip(self.fusions, self._ordered_encoder_features(), decoder_features):
            fused.append(fusion(enc, dec))
        return fused

    def _propagate_bottom_up(self, skips: List[torch.Tensor]) -> List[torch.Tensor]:
        refined: List[torch.Tensor] = []
        carry: Optional[torch.Tensor] = None
        for fusion, skip in zip(reversed(self.fusions), reversed(skips)):
            carry = fusion(skip, carry) if carry is not None else skip
            refined.insert(0, carry)
        return refined

    def _ordered_encoder_features(self) -> List[torch.Tensor]:
        if not hasattr(self, "_last_encoder_features"):
            raise RuntimeError("Encoder features are not cached; run a forward pass first.")
        return self._last_encoder_features

    def forward_with_cache(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        outputs = self.forward(x)
        self._last_encoder_features = outputs["encoder"]
        return outputs
