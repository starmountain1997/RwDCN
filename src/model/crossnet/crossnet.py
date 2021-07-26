import torch.nn.functional as F

from src.model.crossnet.backward_warp_layer import Backward_warp
from src.model.crossnet.flow_net import FlowNet
from src.model.crossnet.utils import *


class CrossNet(nn.Module):

    def __init__(self):
        super(CrossNet, self).__init__()
        self.FlowNet = FlowNet(6)
        self.Backward_warp = Backward_warp()
        self.Encoder = Encoder(3)
        self.UNet_decoder_2 = UNet_decoder_2()

    def forward(self, lr, ref):
        lrsr = F.interpolate(lr, scale_factor=4)
        flow = self.FlowNet(ref, lrsr)
        flow_12_1 = flow['flow_12_1']
        flow_12_2 = flow['flow_12_2']
        flow_12_3 = flow['flow_12_3']
        flow_12_4 = flow['flow_12_4']

        SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.Encoder(lrsr)
        HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.Encoder(ref)

        warp_21_conv1 = self.Backward_warp(HR2_conv1, flow_12_1)
        warp_21_conv2 = self.Backward_warp(HR2_conv2, flow_12_2)
        warp_21_conv3 = self.Backward_warp(HR2_conv3, flow_12_3)
        warp_21_conv4 = self.Backward_warp(HR2_conv4, flow_12_4)

        sythsis_output = self.UNet_decoder_2(SR_conv1, SR_conv2, SR_conv3, SR_conv4, warp_21_conv1, warp_21_conv2,
                                             warp_21_conv3, warp_21_conv4)
        return sythsis_output
