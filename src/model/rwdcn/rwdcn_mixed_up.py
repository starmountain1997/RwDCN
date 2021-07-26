import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms

from src.model.rwdcn.pcd import PCD
from src.model.srntt.srntt import ContentExtractor, TextureTransfer
from src.utils.vgg import VGGExtractor


class RwDCNMixedUp(nn.Module):
    def __init__(self, ngf=64, n_blocks=16, scale_factor=4):
        super(RwDCNMixedUp, self).__init__()
        self.content_extractor = ContentExtractor(ngf, n_blocks).cuda()
        self.texture_transfer = TextureTransfer(ngf, n_blocks, False).cuda()
        self.scale_factor = scale_factor
        self.lte = VGGExtractor(require_grad=True, return_type='List').cuda()
        self.pcd = PCD().cuda()

    # @torchsnooper.snoop()
    def forward(self, lr, ref):
        lrsr = F.interpolate(lr, scale_factor=self.scale_factor, mode='bicubic')
        refsr = F.interpolate(ref, scale_factor=1. / self.scale_factor)
        refsr = F.interpolate(refsr, scale_factor=self.scale_factor)

        lrsr = self.lte(lrsr)
        refsr = self.lte(refsr)
        ref = self.lte(ref)

        aligned_feat = self.pcd(refsr, lrsr, ref)

        base = F.interpolate(lr, scale_factor=self.scale_factor, mode='bicubic')
        upscale_plain, content_feat = self.content_extractor(lr)
        upscale_srntt = self.texture_transfer(content_feat, aligned_feat)
        return upscale_plain + base


if __name__ == '__main__':
    lr = Image.open('/home/usrs/gzr1997/DS/REDS/DATA/train/sharp_bicubic/003/00000000.png')
    ref = Image.open('/home/usrs/gzr1997/DS/REDS/DATA/train/sharp/003/00000000.png')

    lr = transforms.ToTensor()(lr)
    lr = transforms.CenterCrop((48, 48))(lr)
    ref = transforms.ToTensor()(ref)
    ref = transforms.CenterCrop((192, 192))(ref)
    lr = lr.unsqueeze(0).cuda()
    ref = ref.unsqueeze(0).cuda()
    rwdcn = RwDCNMixedUp()
    res = rwdcn(lr, ref)
