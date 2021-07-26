from src.model.rcan.rcan import RCAN
from src.pl.module import SRModule


class RCANModule(SRModule):
    def __init__(self, img_save_dir=None, weight_decay=None, milestones=None, gamma=None, rgb_std=None,
                 rgb_mean=None,
                 **kwargs):
        super(RCANModule, self).__init__(**kwargs)
        self.rcan = RCAN()

    def forward(self, batch):
        lr = batch['lr']
        sr = self.rcan(lr)
        return sr
