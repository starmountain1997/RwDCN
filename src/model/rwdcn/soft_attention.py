import torch
from mmcv.cnn import ConvModule
from torch import nn


class SpatialAttention(nn.Module):
    def __init__(self, n_feats=64, act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super(SpatialAttention, self).__init__()
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.spatial_attn1 = ConvModule(n_feats * 2, n_feats, 1, act_cfg=act_cfg)
        self.spatial_attn2 = ConvModule(
            n_feats * 2, n_feats, 1, act_cfg=act_cfg)
        self.spatial_attn3 = ConvModule(
            n_feats, n_feats, 3, padding=1, act_cfg=act_cfg)
        self.spatial_attn4 = ConvModule(
            n_feats, n_feats, 1, act_cfg=act_cfg)
        self.spatial_attn5 = nn.Conv2d(
            n_feats, n_feats, 3, padding=1)

        self.spatial_attn_l1 = ConvModule(
            n_feats, n_feats, 1, act_cfg=act_cfg)
        self.spatial_attn_l2 = ConvModule(
            n_feats * 2, n_feats, 3, padding=1, act_cfg=act_cfg)
        self.spatial_attn_l3 = ConvModule(
            n_feats, n_feats, 3, padding=1, act_cfg=act_cfg)
        self.spatial_attn_add1 = ConvModule(
            n_feats, n_feats, 1, act_cfg=act_cfg)
        self.spatial_attn_add2 = nn.Conv2d(n_feats, n_feats, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)  # 1, 128, 192, 192
        attn = self.spatial_attn1(x)  # 1, 64, 192, 192
        attn_max = self.max_pool(attn)  # 1, 64, 96, 96
        attn_avg = self.avg_pool(attn)  # 1, 64, 48, 48
        attn = self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1))
        # pyramid levels
        attn_level = self.spatial_attn_l1(attn)
        attn_max = self.max_pool(attn_level)
        attn_avg = self.avg_pool(attn_level)
        attn_level = self.spatial_attn_l2(
            torch.cat([attn_max, attn_avg], dim=1))
        attn_level = self.spatial_attn_l3(attn_level)
        attn_level = self.upsample(attn_level)

        attn = self.spatial_attn3(attn) + attn_level
        attn = self.spatial_attn4(attn)
        attn = self.upsample(attn)
        attn = self.spatial_attn5(attn)
        attn_add = self.spatial_attn_add2(self.spatial_attn_add1(attn))
        attn = torch.sigmoid(attn)

        return attn, attn_add  # 1, 64, 192, 192
