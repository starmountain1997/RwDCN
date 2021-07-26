import torch
from mmedit.models import make_layer, ResidualBlockNoBN
from torch import nn


class TextureTransfer(nn.Module):
    def __init__(self, n_feats=64, n_resblocks=16):
        super(TextureTransfer, self).__init__()
        self.head_small = nn.Sequential(
            nn.Conv2d(n_feats + 256, n_feats, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True)
        )
        self.body_small = make_layer(ResidualBlockNoBN, n_resblocks, mid_channels=n_feats)
        self.tail_small = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, True)
        )

        self.head_medium = nn.Sequential(
            nn.Conv2d(n_feats + 128, n_feats, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True)
        )
        self.body_medium = make_layer(ResidualBlockNoBN, n_resblocks, mid_channels=n_feats)
        self.tail_medium = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, True)
        )

        self.head_large = nn.Sequential(
            nn.Conv2d(n_feats + 64, n_feats, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True)
        )
        self.body_large = make_layer(ResidualBlockNoBN, n_resblocks, mid_channels=n_feats)
        self.tail_large = nn.Sequential(
            nn.Conv2d(n_feats, n_feats // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats // 2, 3, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, lr, aligned_feats):
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
        h = self.body_large(h) + lr
        lr = self.tail_large(h)

        return lr


class TextureTransferAttention(nn.Module):
    def __init__(self, n_feats=256, n_resblocks=16):
        super(TextureTransferAttention, self).__init__()
        self.head_small = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True)
        )
        self.body_small = make_layer(ResidualBlockNoBN, n_resblocks, mid_channels=256)
        self.tail_small = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, True)
        )

        self.head_medium = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True)
        )
        self.body_medium = make_layer(ResidualBlockNoBN, n_resblocks, mid_channels=128)
        self.tail_medium = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, True)
        )
        self.head_large = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True)
        )
        self.body_large = make_layer(ResidualBlockNoBN, n_resblocks, mid_channels=64)
        self.tail_large = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, lr, aligned_feats, a, aa):
        # 1, 64, 192, 192
        # 1, 128, 96, 96
        # 1, 256, 48, 48
        h = torch.cat([lr, aligned_feats[0]], 1)  # 1, 512, 48, 48
        h = self.head_small(h)  # 1, 256, 48, 48
        h = self.body_small(h) * a[2] * 2 + aa[2]  # 1, 256, 48, 48
        lr = self.tail_small(h)  # 1, 64, 96, 96

        h = torch.cat([lr, aligned_feats[1]], 1)  # 1, 64+128, 96,
        h = self.head_medium(h)  # 1, 128, 96, 96
        h = self.body_medium(h) * a[1] * 2 + aa[1]  # 1, 128, 96, 96
        lr = self.tail_medium(h)

        h = torch.cat([lr, aligned_feats[2]], 1)
        h = self.head_large(h)
        h = self.body_large(h) * a[0] * 2 + aa[0]
        lr = self.tail_large(h)

        return lr
