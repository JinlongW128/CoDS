"""Reusable expert, sampling, alignment, and routing blocks for S-MoE."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_num_groups(channels: int, max_groups: int = 32) -> int:
    """Return the largest valid GroupNorm group count."""
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class DepthwiseSeparableConv(nn.Module):
    """Depthwise-separable convolution with optional dilation."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, dilation: int = 1,
                 norm: str = 'gn'):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, padding=padding,
            dilation=dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False)

        if norm == 'gn':
            self.norm = nn.GroupNorm(
                _get_num_groups(out_channels), out_channels)
        elif norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.activation(x)


class MultiGranularityBEVSlicer(nn.Module):
    """Sample BEV features at the expert-specific range and resolution."""

    def __init__(self, base_grid_conf: dict, task_grid_conf: dict,
                 mode: str = 'identity'):
        super().__init__()
        if mode not in ('identity', 'downsample', 'crop'):
            raise ValueError(f'Unsupported BEV slicing mode: {mode}')
        self.mode = mode
        if mode == 'identity':
            return

        base_x_range = base_grid_conf['xbound']
        base_y_range = base_grid_conf['ybound']
        task_x_range = task_grid_conf['xbound']
        task_y_range = task_grid_conf['ybound']

        base_resolution = base_x_range[2]
        task_resolution = task_x_range[2]
        if mode == 'downsample':
            self.target_h = int(
                (base_y_range[1] - base_y_range[0]) / task_resolution)
            self.target_w = int(
                (base_x_range[1] - base_x_range[0]) / task_resolution)
        else:
            self.crop_x_start = int(
                (task_x_range[0] - base_x_range[0]) / base_resolution)
            self.crop_x_end = int(
                (task_x_range[1] - base_x_range[0]) / base_resolution)
            self.crop_y_start = int(
                (task_y_range[0] - base_y_range[0]) / base_resolution)
            self.crop_y_end = int(
                (task_y_range[1] - base_y_range[0]) / base_resolution)
            self.target_h = int(
                (task_y_range[1] - task_y_range[0]) / task_resolution)
            self.target_w = int(
                (task_x_range[1] - task_x_range[0]) / task_resolution)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        original_hw = x.shape[-2:]
        if self.mode == 'identity':
            return x, original_hw
        if self.mode == 'crop':
            x = x[:, :, self.crop_y_start:self.crop_y_end,
                  self.crop_x_start:self.crop_x_end]
        x = F.interpolate(
            x, size=(self.target_h, self.target_w),
            mode='bilinear', align_corners=True)
        return x, original_hw


class FeatureAligner(nn.Module):
    """Align an expert output to the shared S-MoE feature resolution."""

    def __init__(self, in_channels: int, out_channels: int,
                 align_type: str = 'upsample'):
        super().__init__()
        self.align_type = align_type
        if align_type == 'upsample':
            self.residual_align = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.GroupNorm(_get_num_groups(out_channels), out_channels),
            )
        elif align_type == 'resample':
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.GroupNorm(_get_num_groups(out_channels), out_channels),
            )
        else:
            self.proj = (
                nn.Conv2d(in_channels, out_channels, 1, bias=False)
                if in_channels != out_channels else nn.Identity())

    def forward(self, x: torch.Tensor, target_hw: Tuple[int, int],
                original_feature: torch.Tensor = None) -> torch.Tensor:
        if self.align_type == 'upsample':
            x = F.interpolate(
                x, size=target_hw, mode='bilinear', align_corners=True)
            if original_feature is not None:
                x = x + self.residual_align(original_feature)
            return x
        if self.align_type == 'resample':
            x = self.proj(x)
            if x.shape[-2:] != target_hw:
                x = F.interpolate(
                    x, size=target_hw, mode='bilinear', align_corners=True)
            return x
        return self.proj(x)


class FGExpert(nn.Module):
    """Foreground expert using local 3x3 and shape-aware 5x5 branches."""

    def __init__(self, in_channels: int, out_channels: int,
                 dropout: float = 0.1):
        super().__init__()
        hidden_channels = max(in_channels // 2, 64)
        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(hidden_channels, affine=True),
            nn.ReLU(inplace=True),
        )
        self.branch_5x5 = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, 5, padding=2,
                groups=in_channels, bias=False),
            nn.InstanceNorm2d(in_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.InstanceNorm2d(hidden_channels, affine=True),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, out_channels, 1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.Dropout2d(dropout),
        )
        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.InstanceNorm2d(out_channels, affine=True),
            )
            if in_channels != out_channels else nn.Identity())
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        local_feature = self.branch_3x3(x)
        shape_feature = self.branch_5x5(x)
        feature = self.fuse(torch.cat([local_feature, shape_feature], dim=1))
        return self.activation(feature + identity)


class BGExpert(nn.Module):
    """Background expert using full-resolution ASPP and global context."""

    def __init__(self, in_channels: int, out_channels: int,
                 dropout: float = 0.1, dilations: tuple = (1, 3, 6)):
        super().__init__()
        num_branches = len(dilations) + 2
        branch_channels = max(out_channels // num_branches, 32)

        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1, bias=False),
            nn.GroupNorm(
                _get_num_groups(branch_channels), branch_channels),
            nn.ReLU(inplace=True),
        )
        self.aspp_branches = nn.ModuleList([
            DepthwiseSeparableConv(
                in_channels, branch_channels, kernel_size=3,
                dilation=dilation, norm='gn')
            for dilation in dilations
        ])
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, branch_channels, 1, bias=False),
            nn.GroupNorm(
                _get_num_groups(branch_channels), branch_channels),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(
                branch_channels * num_branches,
                out_channels, 1, bias=False),
            nn.GroupNorm(_get_num_groups(out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.GroupNorm(_get_num_groups(out_channels), out_channels),
            )
            if in_channels != out_channels else nn.Identity())
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        features = [self.branch_1x1(x)]
        features.extend(branch(x) for branch in self.aspp_branches)
        global_feature = F.interpolate(
            self.global_context(x), size=x.shape[-2:],
            mode='bilinear', align_corners=True)
        features.append(global_feature)
        feature = self.proj(torch.cat(features, dim=1))
        return self.activation(feature + identity)


class SharedExpert(nn.Module):
    """Task-neutral expert using dual-pool channel attention."""

    def __init__(self, in_channels: int, out_channels: int,
                 dropout: float = 0.1, reduction: int = 16):
        super().__init__()
        hidden_channels = max(in_channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.se_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels, bias=False),
            nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, 3, padding=1,
                groups=in_channels, bias=False),
            nn.GroupNorm(1, in_channels),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.Dropout2d(dropout),
        )
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels else nn.Identity())
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        batch_size, channels = x.shape[:2]
        pooled = (
            self.avg_pool(x).view(batch_size, channels)
            + self.max_pool(x).view(batch_size, channels))
        channel_weights = self.se_mlp(pooled).view(
            batch_size, channels, 1, 1)
        feature = self.spatial(x * channel_weights)
        return self.activation(self.proj(feature) + identity)


class SpatialAwareRouter(nn.Module):
    """Blend semantic routing priors with learned local spatial routing."""

    IDX_FG = 0
    IDX_BG = 1
    IDX_SHARED = 2

    def __init__(self, in_channels: int, num_experts: int = 3,
                 task: str = 'det'):
        super().__init__()
        if task not in ('det', 'seg'):
            raise ValueError(f'Unsupported routing task: {task}')
        self.task = task
        self.num_experts = num_experts

        initial_alpha = 0.7 if task == 'det' else 0.45
        self.alpha = nn.Parameter(torch.tensor(initial_alpha))
        self.task_query = nn.Parameter(
            torch.randn(1, in_channels, 1, 1) * 0.01)
        hidden_channels = in_channels // 4
        self.local_attn = nn.Sequential(
            nn.Conv2d(
                in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(
                _get_num_groups(hidden_channels), hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_experts, 1, bias=True),
        )

        bias = self.local_attn[-1].bias.data
        if task == 'det':
            bias[self.IDX_FG] = 1.5
            bias[self.IDX_BG] = -3.0
            bias[self.IDX_SHARED] = 0.0
        else:
            bias[self.IDX_FG] = 0.5
            bias[self.IDX_BG] = 0.5
            bias[self.IDX_SHARED] = 0.0

        self.register_buffer(
            'fixed_bias', torch.zeros(1, num_experts, 1, 1))
        if task == 'det':
            self.fixed_bias[0, self.IDX_BG, 0, 0] = -3.0

    def forward(self, p_fg: torch.Tensor, p_bg: torch.Tensor,
                feature: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = p_fg.shape
        prior = torch.zeros(
            batch_size, self.num_experts, height, width,
            device=p_fg.device, dtype=p_fg.dtype)
        if self.task == 'det':
            prior[:, self.IDX_FG] = p_fg.squeeze(1)
            prior[:, self.IDX_SHARED] = (1.0 - p_fg).squeeze(1)
        else:
            uncertainty = 1.0 - torch.maximum(p_fg, p_bg)
            prior[:, self.IDX_FG] = p_fg.squeeze(1)
            prior[:, self.IDX_BG] = p_bg.squeeze(1)
            prior[:, self.IDX_SHARED] = uncertainty.squeeze(1)
        prior = prior + self.fixed_bias

        modulated_feature = feature * torch.sigmoid(self.task_query)
        learned = self.local_attn(modulated_feature)
        alpha = torch.clamp(self.alpha, 0.0, 1.0)
        gate = F.softmax(
            alpha * prior + (1.0 - alpha) * learned, dim=1)
        return gate.unsqueeze(2)


class TaskAdaptor(nn.Module):
    """Apply a lightweight task-specific residual projection."""

    def __init__(self, channels: int):
        super().__init__()
        self.adapt = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(_get_num_groups(channels), channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.adapt(x)
