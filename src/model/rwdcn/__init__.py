import torch

from src.model.rwdcn.rwdcn import RwDCNAttention
from src.model.rwdcn.rwdcn_mixed_up import RwDCNMixedUp
from src.pl.module import SRModule


class RwDCNAttentionModule(SRModule):
    def __init__(self, ):
        super(RwDCNAttentionModule, self).__init__()
        self.rwdcn_attention = RwDCNAttention()
        self.configure_loss()

    def forward(self, batch):
        sr = self.rwdcn_attention(batch['lr'], batch['ref'])
        return sr


class RwDCNMixedUpModule(SRModule):
    def __init__(self):
        super(RwDCNMixedUpModule, self).__init__()
        self.rwdcn_mixed_up = RwDCNMixedUp()
        state_dict = torch.load('/home/usrs/gzr1997/CODE/RwDCN/pretrained/netG_100.pth')
        self.rwdcn_mixed_up.load_state_dict(state_dict, strict=False)
        self.configure_loss()

    def forward(self, batch):
        sr = self.rwdcn_mixed_up(batch['lr'], batch['ref'])
        return sr
