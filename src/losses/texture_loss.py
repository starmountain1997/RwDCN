import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms

from src.utils.vgg import VGGExtractor

TARGET_LAYER = {1: 'relu3_1', 6: 'relu2_1', 11: 'relu1_1'}


def gram_matrix(features):
    n, c, h, w = features.size()
    feat_reshaped = features.view(n, c, -1)
    gram = torch.bmm(feat_reshaped, feat_reshaped.transpose(1, 2))
    return gram


class TextureLoss(nn.Module):
    def __init__(self, use_weights=False):
        super(TextureLoss, self).__init__()
        self.use_weights = use_weights

        self.model = VGGExtractor(return_type='Dict')
        self.register_buffer('a', torch.tensor(-20., requires_grad=True))
        self.register_buffer('b', torch.tensor(.65, requires_grad=True))

    def forward(self, sr, gt, weights=None):
        input_size = sr.shape[-1]
        sr_feat = self.model(sr)
        gt_feat = self.model(gt)
        if self.use_weights:
            weights = F.pad(weights, (1, 1, 1, 1), mode='replicate')
            for idx, l in enumerate(['relu3_1', 'relu2_1', 'relu1_1']):
                weights_scaled = F.interpolate(
                    weights, None, 2 ** idx, 'bicubic', True
                )
                coeff = weights_scaled * self.a.detach() + self.b.detach()
                coeff = torch.sigmoid(coeff)
                gt_feat[l] = gt_feat[l] * coeff
                sr_feat[l] = sr_feat[l] * coeff
        loss_relu1_1 = torch.norm(gram_matrix(sr_feat['relu1_1']) - gram_matrix(gt_feat['relu1_1']), ) / 4. / (
                (input_size * input_size * 1024) ** 2)
        loss_relu2_1 = torch.norm(gram_matrix(sr_feat['relu2_1']) - gram_matrix(gt_feat['relu2_1']), ) / 4. / (
                (input_size * input_size * 512) ** 2)
        loss_relu3_1 = torch.norm(gram_matrix(sr_feat['relu3_1']) - gram_matrix(gt_feat['relu3_1'])) / 4. / (
                (input_size * input_size * 256) ** 2)
        loss = (loss_relu1_1 + loss_relu2_1 + loss_relu3_1) / 3.
        return loss


if __name__ == '__main__':
    sr = Image.open('/home/usrs/gzr1997/DS/REDS/DATA/train/sharp/003/00000004.png')
    ref = Image.open('/home/usrs/gzr1997/DS/REDS/DATA/train/sharp/003/00000000.png')

    sr = transforms.ToTensor()(sr)
    sr = transforms.CenterCrop((192, 192))(sr)
    ref = transforms.ToTensor()(ref)
    ref = transforms.CenterCrop((192, 192))(ref)
    sr = sr.unsqueeze(0).cuda()
    ref = ref.unsqueeze(0).cuda()
    texture_loss = TextureLoss().cuda()
    loss = texture_loss(sr, ref)
