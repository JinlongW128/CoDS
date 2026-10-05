"""S-MoE used by CoDS."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.moe_modules.s_moe_components import (
    BGExpert,
    FGExpert,
    FeatureAligner,
    MultiGranularityBEVSlicer,
    SharedExpert,
    SpatialAwareRouter,
    TaskAdaptor,
    _get_num_groups,
)


class ReliabilityConfidenceHead(nn.Module):
    """Predict independent foreground and background confidence maps.

    The released implementation follows the actual experiments: ``P_fg`` and
    ``P_bg`` are produced by two parallel branches. They are not constrained to
    be complementary, despite the simplified ``P_bg = 1 - P_fg`` expression in
    the paper's Eq. (5).
    """

    def __init__(self, in_channels: int, hidden_channels: int = 32,
                 use_reliability: bool = True):
        super().__init__()
        self.use_reliability = use_reliability
        actual_in = in_channels + 1 if use_reliability else in_channels

        self.fg_net = nn.Sequential(
            nn.Conv2d(actual_in, hidden_channels, 1, bias=False),
            nn.GroupNorm(_get_num_groups(hidden_channels), hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        self.bg_net = nn.Sequential(
            nn.Conv2d(actual_in, hidden_channels, 1, bias=False),
            nn.GroupNorm(_get_num_groups(hidden_channels), hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
        )

        nn.init.constant_(self.fg_net[-1].bias, -2.0)
        nn.init.constant_(self.bg_net[-1].bias, 1.0)
        nn.init.kaiming_normal_(self.fg_net[-1].weight, mode='fan_out')
        nn.init.kaiming_normal_(self.bg_net[-1].weight, mode='fan_out')

    @staticmethod
    def _normalize(logit: torch.Tensor) -> torch.Tensor:
        batch_size = logit.size(0)
        flat = logit.view(batch_size, -1)
        minimum = flat.min(1, keepdim=True)[0].view(batch_size, 1, 1, 1)
        maximum = flat.max(1, keepdim=True)[0].view(batch_size, 1, 1, 1)
        normalized = (logit - minimum) / (maximum - minimum + 1e-6)
        return torch.sigmoid(normalized * 6.0 - 3.0)

    def forward(self, x: torch.Tensor,
                reliability: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_reliability:
            if reliability is None:
                reliability = torch.ones(
                    x.size(0), 1, x.size(2), x.size(3),
                    dtype=x.dtype, device=x.device)
            elif reliability.shape[-2:] != x.shape[-2:]:
                reliability = F.interpolate(
                    reliability, size=x.shape[-2:], mode='bilinear',
                    align_corners=True)
            confidence_input = torch.cat([x, reliability], dim=1)
        else:
            confidence_input = x

        p_fg = self._normalize(self.fg_net(confidence_input))
        p_bg = self._normalize(self.bg_net(confidence_input))
        return p_fg, p_bg


class SMoE(nn.Module):
    """Reliability-guided S-MoE with FG, BG, and shared experts."""

    def __init__(self, args: dict):
        super().__init__()
        in_channels = args.get('in_channels', 256)
        out_channels = args.get('out_channels', 256)
        dropout = args.get('dropout', 0.1)
        use_reliability = args.get('use_reliability', True)

        base_grid = args.get('base_grid_conf', {
            'xbound': [-140.8, 140.8, 0.4],
            'ybound': [-51.2, 51.2, 0.4],
        })
        fg_grid = args.get('fg_grid_conf', {
            'xbound': [-140.8, 140.8, 0.4],
            'ybound': [-51.2, 51.2, 0.4],
        })
        bg_grid = args.get('bg_grid_conf', {
            'xbound': [-30.0, 30.0, 0.15],
            'ybound': [-15.0, 15.0, 0.15],
        })

        self.slicer_fg = MultiGranularityBEVSlicer(
            base_grid, fg_grid, mode='downsample')
        self.slicer_bg = MultiGranularityBEVSlicer(
            base_grid, bg_grid, mode='crop')
        self.aligner_fg = FeatureAligner(
            out_channels, out_channels, align_type='resample')
        self.aligner_bg = FeatureAligner(
            out_channels, out_channels, align_type='resample')

        self.confidence_head = ReliabilityConfidenceHead(
            in_channels,
            hidden_channels=args.get('confidence_channels', 32),
            use_reliability=use_reliability,
        )
        self.expert_fg = FGExpert(in_channels, out_channels, dropout)
        self.expert_bg = BGExpert(in_channels, out_channels, dropout)
        self.expert_shared = SharedExpert(in_channels, out_channels, dropout)

        self.router_det = SpatialAwareRouter(
            out_channels, num_experts=3, task='det')
        self.router_seg = SpatialAwareRouter(
            out_channels, num_experts=3, task='seg')
        self.adapt_det = TaskAdaptor(out_channels)
        self.adapt_seg = TaskAdaptor(out_channels)
        self.proj_original = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor,
                reliability: Optional[torch.Tensor] = None):
        target_hw = x.shape[-2:]
        original_feature = self.proj_original(x)
        p_fg, p_bg = self.confidence_head(x, reliability)

        x_fg, _ = self.slicer_fg(x)
        x_bg, _ = self.slicer_bg(x)
        e_fg = self.expert_fg(x_fg)
        e_bg = self.expert_bg(x_bg)
        e_shared = self.expert_shared(x)

        e_fg = self.aligner_fg(e_fg, target_hw, original_feature)
        e_bg = self.aligner_bg(e_bg, target_hw, None)
        expert_stack = torch.stack([e_fg, e_bg, e_shared], dim=1)

        gate_det = self.router_det(p_fg, p_bg, e_shared)
        gate_seg = self.router_seg(p_fg, p_bg, e_shared)
        feature_det = self.adapt_det((expert_stack * gate_det).sum(dim=1))
        feature_seg = self.adapt_seg((expert_stack * gate_seg).sum(dim=1))

        return feature_det, feature_seg, p_fg, p_bg, gate_det, gate_seg
