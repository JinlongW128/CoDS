"""Collaborative Reliability Map (CoRM) backbone used by CoDS."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.coalign_modules.fusion_in_one import AttFusion
from opencood.tools.comm_utils import CommRecorder


def compute_corm(x: torch.Tensor, record_len: torch.Tensor) -> torch.Tensor:
    """Compute per-pixel reliability from cross-agent feature variance.

    Official CoDS configurations require ``proj_first: true``. Consequently,
    every agent feature entering this function is already represented in the
    ego coordinate system; the following attention fusion still uses the
    normalized pairwise matrices exactly as in the original implementation.
    """
    cumulative = torch.cumsum(record_len, dim=0)
    starts = torch.cat([
        torch.zeros(1, dtype=torch.long, device=x.device), cumulative[:-1]
    ])
    reliability_maps = []
    _, _, height, width = x.shape

    for batch_index in range(len(record_len)):
        start = starts[batch_index].item()
        agent_count = record_len[batch_index].item()
        scene_features = x[start:start + agent_count]
        if agent_count == 1:
            reliability = torch.ones(
                1, 1, height, width, device=x.device, dtype=x.dtype)
        else:
            variance = scene_features.var(dim=0, unbiased=False)
            dispersion = variance.norm(dim=0, keepdim=True)
            reliability = (1.0 / (1.0 + dispersion)).unsqueeze(0)
        reliability_maps.append(reliability)

    return torch.cat(reliability_maps, dim=0)


class _ImprovedCompressor(nn.Module):
    """Optional spatial-channel compression retained from the experiment code."""

    def __init__(self, input_dim: int, compression_dim: int, stride: int = 4):
        super().__init__()
        compressed_dim = input_dim // compression_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, compressed_dim, 3, stride=stride, padding=1),
            nn.BatchNorm2d(compressed_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                compressed_dim, input_dim, 3, stride=stride, padding=1,
                output_padding=stride - 1),
            nn.BatchNorm2d(input_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(input_dim, input_dim, 3, padding=1),
            nn.BatchNorm2d(input_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, use_fp16: bool = False) -> torch.Tensor:
        x = self.encoder(x)
        if use_fp16:
            x = x.to(torch.float16).to(torch.float32)
        return self.decoder(x)


class AttBEVBackboneCoRM(nn.Module):
    """Attention-fusion BEV backbone augmented with the final-scale CoRM."""

    def __init__(self, model_cfg: dict, input_channels: int):
        super().__init__()
        self.model_cfg = model_cfg
        self.discrete_ratio = model_cfg['voxel_size'][0]
        self.downsample_rate = 1

        self.compress = False
        self.compression_ratio = 1
        self.compress_layers = nn.ModuleList()
        if model_cfg.get('compression_dim', 0) > 0:
            compression_stride = model_cfg.get('compression_stride', 1)
            self.compression_ratio = (
                model_cfg['compression_dim'] * compression_stride ** 2)
            for channels in model_cfg['num_filters']:
                self.compress_layers.append(_ImprovedCompressor(
                    channels, model_cfg['compression_dim'],
                    compression_stride))
            self.compress = True

        layer_nums = model_cfg.get('layer_nums', [])
        layer_strides = model_cfg.get('layer_strides', [])
        num_filters = model_cfg.get('num_filters', [])
        assert len(layer_nums) == len(layer_strides) == len(num_filters)

        upsample_strides = model_cfg.get('upsample_strides', [])
        upsample_filters = model_cfg.get('num_upsample_filter', [])
        assert len(upsample_strides) == len(upsample_filters)

        self.blocks = nn.ModuleList()
        self.fuse_modules = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        channel_inputs = [input_channels, *num_filters[:-1]]

        for index in range(len(layer_nums)):
            layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    channel_inputs[index], num_filters[index], 3,
                    stride=layer_strides[index], padding=0, bias=False),
                nn.BatchNorm2d(
                    num_filters[index], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ]
            self.fuse_modules.append(AttFusion(num_filters[index]))
            for _ in range(layer_nums[index]):
                layers.extend([
                    nn.Conv2d(
                        num_filters[index], num_filters[index], 3,
                        padding=1, bias=False),
                    nn.BatchNorm2d(
                        num_filters[index], eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                ])
            self.blocks.append(nn.Sequential(*layers))

            if upsample_strides:
                stride = upsample_strides[index]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[index], upsample_filters[index],
                            stride, stride=stride, bias=False),
                        nn.BatchNorm2d(
                            upsample_filters[index], eps=1e-3, momentum=0.01),
                        nn.ReLU(),
                    ))
                else:
                    stride = int(np.round(1 / stride))
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[index], upsample_filters[index],
                            stride, stride=stride, bias=False),
                        nn.BatchNorm2d(
                            upsample_filters[index], eps=1e-3, momentum=0.01),
                        nn.ReLU(),
                    ))

        output_channels = sum(upsample_filters)
        if len(upsample_strides) > len(layer_nums):
            stride = upsample_strides[-1]
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(
                    output_channels, output_channels, stride,
                    stride=stride, bias=False),
                nn.BatchNorm2d(output_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))
        self.num_bev_features = output_channels

    def forward(self, data_dict: dict, comm_record: bool = True) -> dict:
        spatial_features = data_dict['spatial_features']
        record_len = data_dict['record_len']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        x = spatial_features
        height, width = x.shape[2:]

        pairwise_t_matrix = pairwise_t_matrix[
            :, :, :, [0, 1], :][:, :, :, :, [0, 1, 3]]
        pairwise_t_matrix[..., 0, 1] *= height / width
        pairwise_t_matrix[..., 1, 0] *= width / height
        pairwise_t_matrix[..., 0, 2] /= (
            self.downsample_rate * self.discrete_ratio * width / 2)
        pairwise_t_matrix[..., 1, 2] /= (
            self.downsample_rate * self.discrete_ratio * height / 2)

        upsampled = []
        last_reliability = None
        num_levels = len(self.blocks)
        for index in range(num_levels):
            x = self.blocks[index](x)
            if self.compress:
                x = self.compress_layers[index](
                    x, use_fp16=not self.training)

            if comm_record:
                _, channels, feature_height, feature_width = x.shape
                recorder = CommRecorder()
                recorder.add_feature_map(
                    channels, feature_height, feature_width,
                    nums=len(record_len), ratio=1.0 / self.compression_ratio,
                    bytes_per_element=2 if self.compress else 4)
                recorder.add_pose_bytes(nums=len(record_len))

            if index == num_levels - 1:
                last_reliability = compute_corm(x, record_len)
            fused = self.fuse_modules[index](
                x, record_len, pairwise_t_matrix)
            upsampled.append(
                self.deblocks[index](fused) if self.deblocks else fused)

        if comm_record:
            CommRecorder().increase_frame_counter(num=len(record_len))

        x = torch.cat(upsampled, dim=1) if len(upsampled) > 1 else upsampled[0]
        if len(self.deblocks) > num_levels:
            x = self.deblocks[-1](x)
        data_dict['spatial_features_2d'] = x

        output_hw = x.shape[-2:]
        if last_reliability is None:
            last_reliability = torch.ones(
                len(record_len), 1, *output_hw,
                dtype=x.dtype, device=x.device)
        elif last_reliability.shape[-2:] != output_hw:
            last_reliability = F.interpolate(
                last_reliability, size=output_hw,
                mode='bilinear', align_corners=True)
        data_dict['reliability_map'] = last_reliability
        return data_dict
