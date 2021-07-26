from torch import nn

from src.model.rcan.rcab import default_conv
from src.model.rcan.rg import ResidualGroup
from src.model.rcan.utils import MeanShift, Upsampler


class RCAN(nn.Module):
    def __init__(self,
                 rgb_std=None,
                 rgb_mean=None,
                 n_resgroups=10,
                 n_resblocks=20,
                 n_feats=64,
                 reduction=16,
                 scale_factor=4,
                 rgb_range=1.,
                 ):
        super(RCAN, self).__init__()
        if rgb_mean is None:
            rgb_mean = [0.485, 0.456, 0.406]
        if rgb_std is None:
            rgb_std = [0.229, 0.224, 0.225]
        self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)
        modules_head = [default_conv(3, n_feats, 3)]
        kernel_size = 3
        modules_body = [
            ResidualGroup(n_feats=n_feats, kernel_size=kernel_size, reduction=reduction, n_resblocks=n_resblocks)
            for _ in range(n_resgroups)
        ]
        modules_body.append(default_conv(n_feats, n_feats, kernel_size))
        modules_tail = [
            Upsampler(default_conv, scale_factor, n_feats, act=False),
            default_conv(n_feats, 3, 3)
        ]
        self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        x = self.add_mean(x)
        return x
