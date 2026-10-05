"""BEV segmentation prediction head."""

import torch
import torch.nn as nn


class BevSegHead(nn.Module):
    """Predict static, dynamic, or both BEV segmentation maps."""

    def __init__(self, target: str, input_dim: int,
                 dynamic_output_class: int = 2,
                 static_output_class: int = 3):
        super().__init__()
        self.target = target
        if target in ('dynamic', 'both'):
            self.dynamic_head = nn.Conv2d(
                input_dim, dynamic_output_class, kernel_size=3, padding=1)
        if target in ('static', 'dynamic', 'both'):
            self.static_head = nn.Conv2d(
                input_dim, static_output_class, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        if self.target == 'dynamic':
            dynamic_map = self.dynamic_head(x)
            static_map = torch.zeros_like(dynamic_map)
        elif self.target == 'static':
            static_map = self.static_head(x)
            dynamic_map = torch.zeros_like(static_map)
        else:
            dynamic_map = self.dynamic_head(x)
            static_map = self.static_head(x)
        return static_map, dynamic_map
