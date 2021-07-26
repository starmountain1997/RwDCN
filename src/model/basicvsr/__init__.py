from mmedit.models import BasicVSRNet

from src.pl.module import SRModule


class BasicVSRModule(SRModule):
    def __init__(self, ):
        super(BasicVSRModule, self).__init__()
        self.basic_vsr = BasicVSRNet()
        self.configure_loss()

    def forward(self, batch):
        sr = self.basic_vsr(batch['lrs'])
        return sr
