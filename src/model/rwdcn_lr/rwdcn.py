import torch.nn.functional as F
from torch import nn

from src.model.rwdcn_lr.lte import LTE
from src.model.rwdcn_lr.pcd import PCD
from src.model.rwdcn_lr.texture_transfer import TextureTransfer


class RwDCN(nn.Module):
    def __init__(self, scale_factor=4):
        super(RwDCN, self).__init__()
        self.conv_first = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.lte = LTE()
        self.scale_factor = scale_factor
        self.pcd = PCD()
        self.tt = TextureTransfer()

    def forward(self, lr, ref):
        # 1, 64, 512, 512
        # 1, 128, 256, 256
        # 1, 256, 128, 128
        lrsr = F.interpolate(lr, scale_factor=self.scale_factor)
        refsr = F.interpolate(ref, scale_factor=1. / self.scale_factor)
        refsr = F.interpolate(refsr, scale_factor=self.scale_factor)
        lrsr = self.lte(lrsr)
        refsr = self.lte(refsr)
        ref = self.lte(ref)
        lr = self.conv_first(lr)
        # 1, 256, 128, 128
        # 1, 128, 256, 256
        # 1, 64, 512, 512
        aligned_feat = self.pcd(refsr, lrsr, ref)
        sr = self.tt(lr, aligned_feat)
        return sr
