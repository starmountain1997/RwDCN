from src.model.rwdcn_lr.rwdcn import RwDCN
from src.pl.module import SRModule


class RwDCNLRModule(SRModule):
    def __init__(self):
        super(RwDCNLRModule, self).__init__()
        self.rwdcn = RwDCN()

    def forward(self, batch):
        sr = self.rwdcn(batch['lr'], batch['ref'])
        return sr
