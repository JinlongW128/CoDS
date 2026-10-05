"""PointPillar implementation of CoDS for joint detection and segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.attfuse_modules.btci import BTCI
from opencood.models.attfuse_modules.corm import AttBEVBackboneCoRM
from opencood.models.common_modules.downsample_conv import DownsampleConv
from opencood.models.common_modules.pillar_vfe import PillarVFE
from opencood.models.common_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.moe_modules.s_moe import SMoE
from opencood.models.seg_modules.bev_seg_head import BevSegHead
from opencood.models.seg_modules.naive_decoder import NaiveDecoder


class PointPillarCoDS(nn.Module):
    """CoDS with CoRM, S-MoE, BTCI, and complete det/seg heads."""

    def __init__(self, args: dict):
        super().__init__()
        self.pillar_vfe = PillarVFE(
            args['pillar_vfe'],
            num_point_features=4,
            voxel_size=args['voxel_size'],
            point_cloud_range=args['lidar_range'],
        )
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = AttBEVBackboneCoRM(
            args['base_bev_backbone'], input_channels=64)

        self.out_channel = 384
        self.shrink_flag = 'shrink_header' in args
        if self.shrink_flag:
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]

        self.s_moe = SMoE(args.get('s_moe_args', {}))
        self.use_btci = args.get('use_btci', True)
        if self.use_btci:
            self.btci = BTCI(
                channels=self.out_channel,
                reduction=args.get('btci_reduction', 8),
                alpha=args.get('btci_alpha', 0.05),
                use_spatial_reliability=args.get(
                    'btci_use_spatial_reliability', False),
            )

        self.cls_head = nn.Conv2d(
            self.out_channel, args['anchor_number'], kernel_size=1)
        self.reg_head = nn.Conv2d(
            self.out_channel, 7 * args['anchor_number'], kernel_size=1)
        self.use_dir = 'dir_args' in args
        if self.use_dir:
            self.dir_head = nn.Conv2d(
                self.out_channel,
                args['dir_args']['num_bins'] * args['anchor_number'],
                kernel_size=1,
            )

        self.decoder = NaiveDecoder(args['decoder'])
        self.target = args['target']
        if self.target == 'both':
            self.seg_head = BevSegHead(
                self.target, args['seg_head_dim'],
                args['dynamic_output_class'], args['static_output_class'])
        elif self.target == 'dynamic':
            self.seg_head = BevSegHead(
                self.target, args['seg_head_dim'],
                args['dynamic_output_class'])
        elif self.target == 'static':
            self.seg_head = BevSegHead(
                self.target, args['seg_head_dim'],
                static_output_class=args['static_output_class'])
        else:
            raise ValueError(
                "Segmentation target must be 'dynamic', 'static', or 'both'.")

    @staticmethod
    def crop_for_det(feature: torch.Tensor) -> torch.Tensor:
        target_height = 96
        crop_y = (feature.shape[-2] - target_height) // 2
        return feature[:, :, crop_y:crop_y + target_height, :]

    @staticmethod
    def crop_for_seg(feature: torch.Tensor) -> torch.Tensor:
        target_height, target_width = 128, 128
        crop_y = (feature.shape[-2] - target_height) // 2
        crop_x = (feature.shape[-1] - target_width) // 2
        return feature[
            :, :, crop_y:crop_y + target_height,
            crop_x:crop_x + target_width]

    def forward(self, data_dict: dict) -> dict:
        processed_lidar = data_dict['processed_lidar']
        batch_dict = {
            'voxel_features': processed_lidar['voxel_features'],
            'voxel_coords': processed_lidar['voxel_coords'],
            'voxel_num_points': processed_lidar['voxel_num_points'],
            'record_len': data_dict['record_len'],
            'pairwise_t_matrix': data_dict['pairwise_t_matrix'],
        }
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        feature = batch_dict['spatial_features_2d']
        reliability = batch_dict['reliability_map']
        if self.shrink_flag:
            feature = self.shrink_conv(feature)
            if reliability.shape[-2:] != feature.shape[-2:]:
                reliability = F.interpolate(
                    reliability, size=feature.shape[-2:],
                    mode='bilinear', align_corners=True)

        (feature_det, feature_seg, p_fg, p_bg,
         gate_det, gate_seg) = self.s_moe(
            feature, reliability=reliability)
        if self.use_btci:
            feature_det, feature_seg = self.btci(
                feature_det, feature_seg, reliability, p_fg, p_bg)

        det_feature = self.crop_for_det(feature_det)
        output_dict = {
            'psm': self.cls_head(det_feature),
            'rm': self.reg_head(det_feature),
        }
        if self.use_dir:
            output_dict['dir_preds'] = self.dir_head(det_feature)

        seg_feature = self.decoder(self.crop_for_seg(feature_seg))
        static_map, dynamic_map = self.seg_head(seg_feature)
        output_dict.update({
            'static_seg': static_map,
            'dynamic_seg': dynamic_map,
            'reliability_map': reliability,
            'P_fg': p_fg,
            'P_bg': p_bg,
            'gate_det': gate_det,
            'gate_seg': gate_seg,
        })
        return output_dict
