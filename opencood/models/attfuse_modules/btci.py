"""BTCI module used by CoDS."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BTCI(nn.Module):
    """Exchange conservative channel hints between detection and segmentation."""

    def __init__(self, channels: int, reduction: int = 8,
                 alpha: float = 0.05, use_spatial_reliability: bool = False):
        super().__init__()
        self.use_spatial_reliability = use_spatial_reliability
        hidden_channels = max(channels // reduction, 16)

        self.det_to_seg = self._make_hint_mlp(channels, hidden_channels)
        self.seg_to_det = self._make_hint_mlp(channels, hidden_channels)
        self.register_buffer('alpha', torch.tensor(alpha))

    @staticmethod
    def _make_hint_mlp(channels: int, hidden_channels: int) -> nn.Sequential:
        module = nn.Sequential(
            nn.Linear(channels, hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels, bias=True),
            nn.Sigmoid(),
        )
        nn.init.zeros_(module[-2].weight)
        nn.init.zeros_(module[-2].bias)
        return module

    def _reliability_gate(self, reliability: torch.Tensor,
                          target_hw: Tuple[int, int]) -> torch.Tensor:
        if self.use_spatial_reliability:
            if reliability.shape[-2:] != target_hw:
                reliability = F.interpolate(
                    reliability, size=target_hw, mode='bilinear',
                    align_corners=True)
            return reliability
        return reliability.mean(dim=(-2, -1), keepdim=True)

    def _apply_hint(self, source: torch.Tensor, target: torch.Tensor,
                    mlp: nn.Module, reliability_gate: torch.Tensor
                    ) -> torch.Tensor:
        batch_size, channels = source.shape[:2]
        weights = mlp(source.mean(dim=(-2, -1)))
        weights = weights.view(batch_size, channels, 1, 1)
        hint = 1.0 + self.alpha * reliability_gate * (weights - 0.5)
        return target * hint

    def forward(self, feature_det: torch.Tensor, feature_seg: torch.Tensor,
                reliability: Optional[torch.Tensor] = None,
                p_fg: Optional[torch.Tensor] = None,
                p_bg: Optional[torch.Tensor] = None):
        del p_fg, p_bg
        batch_size, _, height, width = feature_det.shape
        if reliability is None:
            reliability_gate = torch.full(
                (batch_size, 1, 1, 1), 0.5,
                device=feature_det.device, dtype=feature_det.dtype)
        else:
            reliability_gate = self._reliability_gate(
                reliability, (height, width)).detach()

        # Detaching both source-task features prevents cross-task gradients.
        feature_seg_out = self._apply_hint(
            feature_det.detach(), feature_seg,
            self.det_to_seg, reliability_gate)
        feature_det_out = self._apply_hint(
            feature_seg.detach(), feature_det,
            self.seg_to_det, reliability_gate)
        return feature_det_out, feature_seg_out
