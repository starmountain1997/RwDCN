import torch.nn.functional as F
from PIL import Image
from mmcv.cnn.resnet import conv3x3
from mmedit.models import make_layer, ResidualBlockNoBN
from torch import nn
from torchvision import transforms

from src.model.rwdcn.pcd import PCD
from src.model.rwdcn.soft_attention import SpatialAttention
from src.model.rwdcn.texture_transfer import TextureTransferAttention
from src.utils.vgg import VGGExtractor


class SFE(nn.Module):
    def __init__(self, n_feats=256, n_res_blocks=16):
        super(SFE, self).__init__()
        self.conv_head = conv3x3(3, n_feats)
        self.conv_body = make_layer(ResidualBlockNoBN, n_res_blocks, mid_channels=n_feats)
        self.conv_tail = conv3x3(n_feats, n_feats)

    def forward(self, x):
        x = F.relu(self.conv_head(x))
        x1 = x
        x = self.conv_body(x)
        x = self.conv_tail(x)
        x = x + x1
        return x


class RwDCNAttention(nn.Module):
    def __init__(self, scale_factor=4):
        super(RwDCNAttention, self).__init__()
        self.lte = VGGExtractor(require_grad=True, return_type='List').cuda()
        self.scale_factor = scale_factor
        self.sa1 = SpatialAttention(n_feats=64).cuda()
        self.sa2 = SpatialAttention(n_feats=128).cuda()
        self.sa3 = SpatialAttention(n_feats=256).cuda()
        self.pcd = PCD().cuda()
        self.tt = TextureTransferAttention().cuda()
        self.sfe = SFE().cuda()

    def forward(self, lr, ref):
        lrsr = F.interpolate(lr, scale_factor=self.scale_factor)
        refsr = F.interpolate(ref, scale_factor=1. / self.scale_factor)
        refsr = F.interpolate(refsr, scale_factor=self.scale_factor)
        lrsr = self.lte(lrsr)
        refsr = self.lte(refsr)
        ref = self.lte(ref)
        a1, aa1 = self.sa1(lrsr[0], refsr[0])  # 1, 64, 320, 320
        a2, aa2 = self.sa2(lrsr[1], refsr[1])  # 1, 128, 180, 180
        a3, aa3 = self.sa3(lrsr[2], refsr[2])  # 1, 256, 90, 90
        lr = self.sfe(lr)  # 1, 64, 48, 48
        aligned_feat = self.pcd(refsr, lrsr, ref)  # 1, 256, 48, 48
        # sr = self.tt(lr, aligned_feat)
        sr = self.tt(lr, aligned_feat, [a1, a2, a3], [aa1, aa2, aa3])
        return sr


if __name__ == '__main__':
    lr = Image.open('/home/usrs/gzr1997/DS/REDS/DATA/train/sharp_bicubic/003/00000000.png')
    ref = Image.open('/home/usrs/gzr1997/DS/REDS/DATA/train/sharp/003/00000000.png')

    lr = transforms.ToTensor()(lr)
    lr = transforms.CenterCrop((48, 48))(lr)
    ref = transforms.ToTensor()(ref)
    ref = transforms.CenterCrop((192, 192))(ref)
    lr = lr.unsqueeze(0).cuda()
    ref = ref.unsqueeze(0).cuda()
    rwdcn = RwDCNAttention()
    res = rwdcn(lr, ref)
