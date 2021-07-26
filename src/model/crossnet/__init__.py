from src.model.crossnet.crossnet import CrossNet
from src.pl.module import SRModule


class CrossNetModule(SRModule):
    def __init__(self, ):
        super(CrossNetModule, self).__init__()
        self.crossnet = CrossNet()
        self.configure_loss()

    def forward(self, batch):
        sr = self.crossnet(batch['lr'], batch['ref'])
        return sr
