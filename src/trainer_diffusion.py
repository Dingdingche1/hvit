import math
from typing import Dict, List

import torch
import torch.nn as nn

import torch.nn.functional as F
from lightning import LightningModule

from src.model.hvit import HViT
from src.model.hvit_light import HViT_Light
from src.loss import CombinedLoss, DiceScore
from src.utils import get_one_hot


dtype_map = {
    'bf16': torch.bfloat16,
    'fp32': torch.float32,
    'fp16': torch.float16
}


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, time_emb_dim: int):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, channels),
            nn.SiLU()
        )
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        time_condition = self.time_mlp(time_emb).view(time_emb.shape[0], -1, 1, 1, 1)
        return F.silu(out + time_condition)


class NoisePredictorUNet(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 32, time_emb_dim: int = 128):
        super().__init__()
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )

        self.encoder1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU()
        )
        self.res1 = ResidualBlock(base_channels, time_emb_dim)

        self.encoder2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU()
        )
        self.res2 = ResidualBlock(base_channels * 2, time_emb_dim)

        self.decoder1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv3d(base_channels, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_embedding(timesteps)
        x1 = self.encoder1(x)
        x1 = self.res1(x1, time_emb)
        x2 = self.encoder2(x1)
        x2 = self.res2(x2, time_emb)
        x3 = self.decoder1(x2)
        x3 = x3 + x1
        return self.out(x3)


class MambaRefiner(nn.Module):
    """Lightweight refinement block inspired by sequence modeling layers."""

    def __init__(self, channels: int = 32):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.depthwise = nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.pointwise = nn.Conv3d(channels, channels, kernel_size=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, d = x.shape
        x_flat = x.permute(0, 2, 3, 4, 1).contiguous().view(-1, c)
        x_norm = self.norm(x_flat).view(b, h, w, d, c).permute(0, 4, 1, 2, 3)
        x_depth = self.depthwise(x_norm)
        x_point = self.pointwise(self.act(x_depth))
        return x_point


class DiffusionRegistrationModule(LightningModule):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        model_cfg = config.get('model', {})
        loss_cfg = config.get('loss', {})
        self.optim_cfg = config.get('optimization', {})
        noise_cfg = config.get('noise', {})

        self.num_labels = model_cfg.get('num_labels', 36)
        self.noise_steps = noise_cfg.get('timesteps', 1000)
        beta_start = noise_cfg.get('beta_start', 1e-4)
        beta_end = noise_cfg.get('beta_end', 0.02)

        betas = torch.linspace(beta_start, beta_end, self.noise_steps)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alpha_cumprod', alpha_cumprod)

        self.ddpm_precision = dtype_map.get(self.optim_cfg.get('ddpm_precision', 'bf16'), torch.float32)
        self.mamba_precision = dtype_map.get(self.optim_cfg.get('mamba_precision', 'bf16'), torch.float32)

        self.ddpm = NoisePredictorUNet(in_channels=model_cfg.get('in_channels', 1),
                                       base_channels=model_cfg.get('ddpm_base_channels', 32),
                                       time_emb_dim=model_cfg.get('ddpm_time_dim', 128))
        self.hvit = HViT_Light(config) if model_cfg.get('hvit_light', True) else HViT(config)
        self.use_mamba = model_cfg.get('use_mamba', False)
        self.mamba_refiner = MambaRefiner(model_cfg.get('mamba_channels', 32)) if self.use_mamba else None

        self.loss_fn = CombinedLoss(reg_loss=loss_cfg.get('registration', 'ncc'),
                                    weights=loss_cfg.get('weights'),
                                    num_classes=self.num_labels)
        self.metrics: List[str] = config.get('metrics', ['dice'])

    def _sample_noisy_inputs(self, x: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = torch.sqrt(self.alpha_cumprod[timesteps]).view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha = torch.sqrt(1 - self.alpha_cumprod[timesteps]).view(-1, 1, 1, 1, 1)
        return sqrt_alpha * x + sqrt_one_minus_alpha * noise

    def _warp_segmentation(self, src_seg: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        src_one_hot = get_one_hot(src_seg, self.num_labels)
        warped = [self.hvit.spatial_trans(src_one_hot[:, i:i+1].float(), flow.float())
                  for i in range(self.num_labels)]
        return torch.cat(warped, dim=1)

    def forward(self, source: torch.Tensor, target: torch.Tensor, src_seg: torch.Tensor, tgt_seg: torch.Tensor):
        noise = torch.randn_like(source)
        t = torch.randint(0, self.noise_steps, (source.shape[0],), device=source.device)
        with torch.autocast(device_type=source.device.type, dtype=self.ddpm_precision):
            noisy_source = self._sample_noisy_inputs(source, noise, t)
            predicted_noise = self.ddpm(noisy_source, t)

        with torch.autocast(device_type=source.device.type, dtype=self.mamba_precision):
            moved, flow = self.hvit(source, target)
            if self.use_mamba:
                flow = flow + self.mamba_refiner(flow)
                moved = self.hvit.spatial_trans(source, flow)
            moved_seg = self._warp_segmentation(src_seg, flow)

        return predicted_noise, noise, moved, moved_seg, flow, t

    def training_step(self, batch, batch_idx):
        source, target, src_seg, tgt_seg = batch
        predicted_noise, noise, moved, moved_seg, flow, _ = self(source, target, src_seg, tgt_seg)

        losses = self.loss_fn(predicted_noise, noise, moved, target, moved_seg, tgt_seg, flow)

        with torch.no_grad():
            if 'dice' in self.metrics:
                dice = DiceScore(moved_seg, tgt_seg.long(), self.num_labels).mean()
                self.log('train_dice', dice, prog_bar=True, on_step=True, on_epoch=True)

        for name, value in losses.items():
            self.log(f'train_{name}', value, prog_bar=name == 'total', on_step=True, on_epoch=True)
        return losses['total']

    def validation_step(self, batch, batch_idx):
        source, target, src_seg, tgt_seg = batch
        predicted_noise, noise, moved, moved_seg, flow, _ = self(source, target, src_seg, tgt_seg)
        losses = self.loss_fn(predicted_noise, noise, moved, target, moved_seg, tgt_seg, flow)

        metrics = {}
        if 'dice' in self.metrics:
            dice = DiceScore(moved_seg, tgt_seg.long(), self.num_labels).mean()
            metrics['dice'] = dice
            self.log('val_dice', dice, prog_bar=True, on_step=False, on_epoch=True)

        for name, value in losses.items():
            self.log(f'val_{name}', value, prog_bar=name == 'total', on_step=False, on_epoch=True)
        return {'loss': losses['total'], **metrics}

    def configure_optimizers(self):
        lr = self.optim_cfg.get('lr', 1e-4)
        weight_decay = self.optim_cfg.get('weight_decay', 1e-4)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        return optimizer
