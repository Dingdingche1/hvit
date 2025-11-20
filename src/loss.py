import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import math
import torch.nn as nn


class NCCLoss(nn.Module):
    """Local normalized cross-correlation loss for 3D volumes."""

    def __init__(self, window_size: int = 9, eps: float = 1e-5):
        super().__init__()
        self.window_size = window_size
        self.eps = eps

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        if y_true.dim() != 5:
            raise ValueError("NCCLoss currently supports 3D volumes with shape (B, C, H, W, D)")

        padding = self.window_size // 2
        window_volume = float(self.window_size ** 3)
        channels = y_true.shape[1]

        kernel = torch.ones((channels, 1, self.window_size, self.window_size, self.window_size),
                            device=y_true.device, dtype=y_true.dtype)

        y_true_mean = F.conv3d(y_true, kernel, padding=padding, groups=channels) / (window_volume + self.eps)
        y_pred_mean = F.conv3d(y_pred, kernel, padding=padding, groups=channels) / (window_volume + self.eps)

        y_true_var = F.conv3d((y_true - y_true_mean) ** 2, kernel, padding=padding, groups=channels)
        y_pred_var = F.conv3d((y_pred - y_pred_mean) ** 2, kernel, padding=padding, groups=channels)

        cross = F.conv3d((y_true - y_true_mean) * (y_pred - y_pred_mean),
                         kernel,
                         padding=padding,
                         groups=channels)

        cc = cross / torch.sqrt((y_true_var + self.eps) * (y_pred_var + self.eps))
        return -torch.mean(cc)


class Grad3D(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super().__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad
    


class DiceLoss(nn.Module):
    """Dice loss"""

    def __init__(self, num_class=36):
        super().__init__()
        self.num_class = num_class

    def forward(self, y_pred, y_true):
        y_true = nn.functional.one_hot(y_true, num_classes=self.num_class)
        y_true = torch.squeeze(y_true, 1)
        y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()
        intersection = y_pred * y_true
        intersection = intersection.sum(dim=[2, 3, 4])
        union = torch.pow(y_pred, 2).sum(dim=[2, 3, 4]) + torch.pow(y_true, 2).sum(dim=[2, 3, 4])
        dsc = (2.*intersection) / (union + 1e-5)
        dsc_loss = (1-torch.mean(dsc))
        return dsc_loss

def DiceScore(y_pred, y_true, num_class):
    y_true = nn.functional.one_hot(y_true, num_classes=num_class)
    y_true = torch.squeeze(y_true, 1)
    y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()
    intersection = y_pred * y_true
    intersection = intersection.sum(dim=[2, 3, 4])
    union = torch.pow(y_pred, 2).sum(dim=[2, 3, 4]) + torch.pow(y_true, 2).sum(dim=[2, 3, 4])
    dsc = (2.*intersection) / (union + 1e-5)
    return dsc


loss_functions = {
    "mse": nn.MSELoss(),
    "dice": DiceLoss(),
    "grad": Grad3D(penalty='l2')
}


class CombinedLoss(nn.Module):
    """Joint loss for diffusion-based registration."""

    def __init__(self, reg_loss: str = "ncc", weights=None, num_classes: int = 36):
        super().__init__()
        weights = weights or {"ddpm": 1.0, "reg": 1.0, "grad": 0.02}
        self.weights = weights
        self.num_classes = num_classes
        self.ddpm_loss = nn.MSELoss()
        self.reg_loss_type = reg_loss
        self.dice_loss = DiceLoss(num_class=num_classes)
        self.ncc_loss = NCCLoss()
        self.grad_loss = Grad3D(penalty='l2')

    def forward(self, predicted_noise: torch.Tensor, target_noise: torch.Tensor,
                moved: torch.Tensor, target: torch.Tensor,
                moved_seg: torch.Tensor, target_seg: torch.Tensor,
                flow: torch.Tensor):
        ddpm = self.ddpm_loss(predicted_noise, target_noise)
        if self.reg_loss_type == "dice":
            reg = self.dice_loss(moved_seg, target_seg.long())
            reg_name = "dice"
        else:
            reg = self.ncc_loss(moved, target)
            reg_name = "ncc"
        smooth = self.grad_loss(flow)

        total = self.weights.get("ddpm", 1.0) * ddpm
        total = total + self.weights.get("reg", 1.0) * reg
        total = total + self.weights.get("grad", 0.02) * smooth

        return {
            "ddpm": ddpm,
            reg_name: reg,
            "grad": smooth,
            "total": total
        }

