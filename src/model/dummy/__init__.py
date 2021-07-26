import torch.nn.functional as F

from src.pl.module import SRModule


class DummyModule(SRModule):
    def __init__(self, **kwargs):
        super(DummyModule, self).__init__(**kwargs)

    def forward(self, batch):
        lr = batch['lr']
        sr = F.interpolate(lr, scale_factor=4, mode='bicubic')
        return sr
