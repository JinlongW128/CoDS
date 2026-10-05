"""Naive BEV decoder used by the CoDS segmentation head."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class NaiveDecoder(nn.Module):
    """Decode the cropped BEV feature with convolutional upsampling blocks."""

    def __init__(self, params: dict):
        super().__init__()
        self.num_ch_dec = params['num_ch_dec']
        self.num_layer = params['num_layer']
        self.input_dim = params['input_dim']
        assert len(self.num_ch_dec) == self.num_layer

        self.convs = OrderedDict()
        for index in range(self.num_layer - 1, -1, -1):
            in_channels = (
                self.input_dim
                if index == self.num_layer - 1
                else self.num_ch_dec[index + 1])
            out_channels = self.num_ch_dec[index]
            self.convs[('upconv', index, 0)] = nn.Conv2d(
                in_channels, out_channels, 3, 1, 1)
            self.convs[('norm', index, 0)] = nn.BatchNorm2d(out_channels)
            self.convs[('relu', index, 0)] = nn.ReLU(True)
            self.convs[('upconv', index, 1)] = nn.Conv2d(
                out_channels, out_channels, 3, 1, 1)
            self.convs[('norm', index, 1)] = nn.BatchNorm2d(out_channels)
            self.convs[('relu', index, 1)] = nn.ReLU(True)
        self.decoder = nn.ModuleList(list(self.convs.values()))

    @staticmethod
    def upsample(x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2, mode='nearest')

    def forward(self, x: torch.Tensor,
                use_upsample: bool = True) -> torch.Tensor:
        for index in range(self.num_layer - 1, -1, -1):
            x = self.convs[('upconv', index, 0)](x)
            x = self.convs[('norm', index, 0)](x)
            x = self.convs[('relu', index, 0)](x)

            # This branch matches the stride used by the shrink header.
            if use_upsample and index == 0:
                x = self.upsample(x)

            x = self.convs[('upconv', index, 1)](x)
            x = self.convs[('norm', index, 1)](x)
            x = self.convs[('relu', index, 1)](x)
        return x
