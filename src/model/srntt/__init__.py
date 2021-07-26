import torch
import torch.nn.functional as F

from src.model.srntt.srntt import SRNTT
from src.model.srntt.swapper import Swapper
from src.model.srntt.vgg import VGG
from src.pl.module import SRModule

TARGET_LAYERS = ['relu3_1', 'relu2_1', 'relu1_1']


class SRNTTModule(SRModule):
    def __init__(self, **kwargs):
        super(SRNTTModule, self).__init__(**kwargs)
        self.srntt = SRNTT()
        self.swapper = Swapper()
        self.vgg = VGG(model_type='vgg19')
        self.scale_factor = 4

    def forward(self, batch):
        lr = batch['lr']
        ref = batch['ref']
        lrsr = F.interpolate(lr, scale_factor=self.scale_factor)
        refsr = F.interpolate(ref, scale_factor=1. / self.scale_factor)
        refsr = F.interpolate(refsr, scale_factor=self.scale_factor)
        lrsr = self.vgg(lrsr, TARGET_LAYERS)
        refsr = self.vgg(refsr, TARGET_LAYERS)
        ref = self.vgg(ref, TARGET_LAYERS)
        maps, weights, correspondences = self.swapper(lrsr, ref, refsr)
        maps = {k: torch.tensor(v).unsqueeze(0).cuda() for k, v in maps.items()}
        _, sr = self.srntt(lr, maps)
        return sr
