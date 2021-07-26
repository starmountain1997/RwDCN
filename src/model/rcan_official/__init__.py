import torch

from src.model.rcan.rcan import RCAN
from src.pl.module import SRModule


class RCANModule(SRModule):
    def __init__(self, **kwargs):
        super(RCANModule, self).__init__(**kwargs)
        state_dict = torch.load('/home/usrs/gzr1997/CODE/RwDCN/pretrained/RCAN_BIX4.pt')
        self.rcan = RCAN()
        self.rcan.load_state_dict(state_dict, strict=False)
        self.configure_loss()

    def forward(self, batch):
        lr = batch['lr']
        sr = self.rcan(lr)
        return sr
