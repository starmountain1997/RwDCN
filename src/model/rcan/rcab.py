from torch import nn

from src.model.rcan.ca import CALayer


def default_conv(in_chs, out_chs, kernel_size, bias=True):
    return nn.Conv2d(in_chs, out_chs, kernel_size, padding=(kernel_size // 2), bias=bias)


class RCAB(nn.Module):
    def __init__(
            self, n_feats, kernel_size, reduction, bias=True, bn=False, res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(default_conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                modules_body.append(nn.ReLU(True))
        modules_body.append(CALayer(n_feats, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res
