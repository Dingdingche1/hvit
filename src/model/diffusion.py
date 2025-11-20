from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal embedding for diffusion timesteps."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_dim // 2
        exponent = -math.log(10000) / (half_dim - 1)
        device = timesteps.device
        emb = torch.exp(torch.arange(half_dim, device=device) * exponent)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=1)


class DiffusionScheduler(nn.Module):
    """Utility holding forward/backward diffusion coefficients."""

    def __init__(self, num_steps: int, beta_start: float, beta_end: float) -> None:
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, num_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.num_steps = num_steps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas", torch.sqrt(alphas))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.num_steps, (batch_size,), device=device)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        return (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t

    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor, noise_pred: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_t)

        betas_t = self.betas[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_alpha_t = self.sqrt_alphas[t].view(-1, 1, 1, 1, 1)
        pred_x0 = (x_t - sqrt_one_minus_alphas_cumprod_t * noise_pred) / sqrt_alpha_t

        alpha_cumprod_prev = torch.where(
            t > 0, self.alphas_cumprod[t - 1], torch.ones_like(t, dtype=x_t.dtype, device=x_t.device)
        )
        coef_x0 = torch.sqrt(alpha_cumprod_prev.clamp(min=1e-6)).view(-1, 1, 1, 1, 1)
        coef_noise = torch.sqrt(1 - alpha_cumprod_prev.clamp(min=1e-6)).view(-1, 1, 1, 1, 1)
        mean = coef_x0 * pred_x0 + coef_noise * noise_pred
        nonzero_mask = (t != 0).float().view(-1, 1, 1, 1, 1)
        return mean + nonzero_mask * torch.sqrt(betas_t) * noise


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels)
        )
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.residual_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        time_emb = self.time_mlp(t).view(t.shape[0], -1, 1, 1, 1)
        h = h + time_emb
        h = self.block2(h)
        return h + self.residual_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_dim)
        self.down = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.res(x, t)
        return self.down(h), h


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.res = ResidualBlock(out_channels * 2, out_channels, time_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.res(x, t)


class DiffusionUNet(nn.Module):
    """Simple 3D UNet-like denoiser following G_theta."""

    def __init__(self, in_channels: int, base_channels: int, time_emb_dim: Optional[int] = None, out_channels: int = 3) -> None:
        super().__init__()
        self.time_emb_dim = time_emb_dim or base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(self.time_emb_dim),
            nn.Linear(self.time_emb_dim, self.time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(self.time_emb_dim * 2, self.time_emb_dim),
        )

        self.init_conv = nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1)
        self.down1 = DownBlock(base_channels, base_channels * 2, self.time_emb_dim)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, self.time_emb_dim)
        self.mid = ResidualBlock(base_channels * 4, base_channels * 4, self.time_emb_dim)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, self.time_emb_dim)
        self.up1 = UpBlock(base_channels * 2, base_channels, self.time_emb_dim)
        self.out_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        if cond is not None:
            x = torch.cat([x, cond], dim=1)

        t_emb = self.time_mlp(t)
        x = self.init_conv(x)
        d1, skip1 = self.down1(x, t_emb)
        d2, skip2 = self.down2(d1, t_emb)
        mid = self.mid(d2, t_emb)
        u2 = self.up2(mid, skip2, t_emb)
        u1 = self.up1(u2, skip1, t_emb)
        return self.out_conv(u1)


class DiffusionRegistration(nn.Module):
    """DDPM wrapper used inside the registration pipeline."""

    def __init__(
        self,
        num_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        cond_channels: int = 0,
        base_channels: int = 64,
        flow_channels: int = 3,
    ) -> None:
        super().__init__()
        self.cond_channels = cond_channels
        self.scheduler = DiffusionScheduler(num_steps, beta_start, beta_end)
        self.denoiser = DiffusionUNet(
            in_channels=flow_channels + cond_channels,
            base_channels=base_channels,
            out_channels=flow_channels,
        )

    def forward(
        self,
        flow: torch.Tensor,
        source: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        feature: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cond = self._prepare_condition(flow, source, target, feature)
        batch_size = flow.shape[0]
        device = flow.device
        t = self.scheduler.sample_timesteps(batch_size, device)
        noise = torch.randn_like(flow)
        noisy_flow = self.scheduler.q_sample(flow, t, noise)

        pred_noise = self.denoiser(noisy_flow, t, cond)
        flow_denoised = self.scheduler.predict_start_from_noise(noisy_flow, t, pred_noise)
        ddpm_loss = F.mse_loss(pred_noise, noise)
        return flow_denoised, ddpm_loss

    def sample(self, shape: Tuple[int, int, int, int, int], cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        flow = torch.randn(shape, device=self.denoiser.out_conv.weight.device)
        for step in reversed(range(self.scheduler.num_steps)):
            t = torch.full((shape[0],), step, device=flow.device, dtype=torch.long)
            noise_pred = self.denoiser(flow, t, cond)
            flow = self.scheduler.p_sample(flow, t, noise_pred)
        return flow

    def _prepare_condition(
        self,
        flow: torch.Tensor,
        source: Optional[torch.Tensor],
        target: Optional[torch.Tensor],
        feature: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if self.cond_channels == 0:
            return None

        cond_tensors = []
        for tensor in (source, target, feature):
            if tensor is None:
                continue
            if tensor.shape[2:] != flow.shape[2:]:
                tensor = F.interpolate(tensor, size=flow.shape[2:], mode="trilinear", align_corners=False)
            cond_tensors.append(tensor)

        if cond_tensors:
            cond = torch.cat(cond_tensors, dim=1)
        else:
            cond = torch.zeros(
                flow.shape[0], self.cond_channels, *flow.shape[2:], device=flow.device, dtype=flow.dtype
            )

        if cond.shape[1] < self.cond_channels:
            pad = torch.zeros(
                cond.shape[0], self.cond_channels - cond.shape[1], *cond.shape[2:], device=cond.device, dtype=cond.dtype
            )
            cond = torch.cat([cond, pad], dim=1)
        elif cond.shape[1] > self.cond_channels:
            cond = cond[:, : self.cond_channels]
        return cond
