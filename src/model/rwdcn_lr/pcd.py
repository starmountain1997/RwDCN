import torch
from mmcv.cnn import ConvModule
from torch import nn

from mmedit.models.backbones.sr_backbones.edvr_net import ModulatedDCNPack


class PCD(nn.Module):
    def __init__(self, deform_groups=8, act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super(PCD, self).__init__()
        # Pyramid has three levels:
        # L3: level 3, 1/4 spatial size 256 chs
        # L2: level 2, 1/2 spatial size 128 chs
        # L1: level 1, original spatial size 64 chs
        self.offset_conv1 = nn.ModuleDict({
            'l3': ConvModule(512, 256, 3, 1, 1, act_cfg=act_cfg),
            'l2': ConvModule(256, 128, 3, 1, 1, act_cfg=act_cfg),
            'l1': ConvModule(128, 64, 3, 1, 1, act_cfg=act_cfg),
        })
        self.offset_conv2 = nn.ModuleDict({
            'l3': ConvModule(256, 256, 3, 1, 1, act_cfg=act_cfg),
            'l2': ConvModule(256, 128, 3, 1, 1, act_cfg=act_cfg),
            'l1': ConvModule(128, 64, 3, 1, 1, act_cfg=act_cfg),
        })
        self.offset_conv3 = nn.ModuleDict({
            'l2': ConvModule(128, 128, 3, 1, 1, act_cfg=act_cfg),
            'l1': ConvModule(64, 64, 3, 1, 1, act_cfg=act_cfg),
        })
        self.dcn_pack = nn.ModuleDict({
            'l3': ModulatedDCNPack(256, 256, 3, padding=1, deform_groups=deform_groups),
            'l2': ModulatedDCNPack(128, 128, 3, padding=1, deform_groups=deform_groups),
            'l1': ModulatedDCNPack(64, 64, 3, padding=1, deform_groups=deform_groups),
        })
        self.feat_conv = nn.ModuleDict({
            'l2': ConvModule(256, 128, 3, 1, 1, act_cfg=act_cfg),
            'l1': ConvModule(128, 64, 3, 1, 1, act_cfg=act_cfg),
        })
        self.bottleneck = nn.ModuleDict({
            'l3': ConvModule(256, 128, 3, 1, 1, act_cfg=act_cfg),
            'l2': ConvModule(128, 64, 3, 1, 1, act_cfg=act_cfg),
        })

        # self.upsample = nn.PixelShuffle(2)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, refsr, lrsr, ref):
        aligned_feat = []
        upsampled_offset, upsampled_feat = None, None
        for i in range(3, 0, -1):
            level = f'l{i}'
            offset = torch.cat([refsr[i - 1], lrsr[i - 1]], dim=1)
            offset = self.offset_conv1[level](offset)
            if i == 3:
                offset = self.offset_conv2[level](offset)
            else:
                offset = self.offset_conv2[level](torch.cat([offset, upsampled_offset], dim=1))
                offset = self.offset_conv3[level](offset)
            feat = self.dcn_pack[level](ref[i - 1], offset)
            if i == 3:
                feat = self.lrelu(feat)
                aligned_feat.append(feat)
            else:
                feat = self.feat_conv[level](torch.cat([feat, upsampled_feat], dim=1))
                aligned_feat.append(feat)
            if i > 1:
                upsampled_offset = self.upsample(offset) * 2
                upsampled_offset = self.bottleneck[level](upsampled_offset)
                upsampled_feat = self.upsample(feat)
                upsampled_feat = self.bottleneck[level](upsampled_feat)
        return aligned_feat
