import torch
from torch import nn

from torchvision.models.resnet import conv3x3


class ResBlock(nn.Module):
    def __init__(self, in_chs, out_chs, stride=1, downsample=None, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(in_chs, out_chs, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_chs, out_chs)

    def forward(self, x):
        x1 = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale + x1
        return out


class TextureTransfer(nn.Module):
    def __init__(self, n_feats=64, n_resblocks=16):
        super(TextureTransfer, self).__init__()
        self.head_small = nn.Sequential(
            nn.Conv2d(n_feats + 256, n_feats, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True)
        )

        self.body_small = nn.Sequential(
            *[ResBlock(n_feats, n_feats) for _ in range(n_resblocks)],
        )
        self.tail_small = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, True)
        )
        self.head_medium = nn.Sequential(
            nn.Conv2d(n_feats + 128, n_feats, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True)
        )
        self.body_medium = nn.Sequential(
            *[ResBlock(n_feats, n_feats) for _ in range(n_resblocks)],
        )
        self.tail_medium = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, True)
        )
        self.head_large = nn.Sequential(
            nn.Conv2d(n_feats + 64, n_feats, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True)
        )
        self.body_large = nn.Sequential(
            *[ResBlock(n_feats, n_feats) for _ in range(n_resblocks)],
        )
        self.tail_large = nn.Sequential(
            nn.Conv2d(n_feats, n_feats // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats // 2, 3, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, lr, aligned_feats):
        # 1, 256, 128, 128
        # 1, 128, 256, 256
        # 1, 64, 512, 512
        h = torch.cat([lr, aligned_feats[0]], 1)  # 1, 512, 128, 128
        h = self.head_small(h)
        h = self.body_small(h) + lr
        lr = self.tail_small(h)  # 1, 64, 256, 256

        h = torch.cat([lr, aligned_feats[1]], 1)  # 1, 192, 256, 256
        h = self.head_medium(h)  # 1, 64, 256, 256
        h = self.body_medium(h) + lr
        lr = self.tail_medium(h)

        h = torch.cat([lr, aligned_feats[2]], 1)
        h = self.head_large(h)
        h = self.body_large(lr) + lr
        lr = self.tail_large(h)

        return lr
