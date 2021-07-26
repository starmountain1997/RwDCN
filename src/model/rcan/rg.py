from torch import nn

from src.model.rcan.rcab import RCAB
from .rcab import default_conv


class ResidualGroup(nn.Module):
    def __init__(self, n_feats, kernel_size, reduction, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(n_feats=n_feats, kernel_size=kernel_size, reduction=reduction)
            for _ in range(n_resblocks)
        ]
        modules_body.append(default_conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res
