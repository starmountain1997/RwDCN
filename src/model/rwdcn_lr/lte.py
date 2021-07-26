import torch
from torchvision.models import vgg19

TARGET_LAYER = {1: 'relu3_1', 6: 'relu2_1', 11: 'relu1_1'}


class LTE(torch.nn.Module):
    def __init__(self, require_grad=True):
        super(LTE, self).__init__()
        features = list(vgg19(pretrained=True).features)[:12]
        self.features = torch.nn.ModuleList(features)
        if not require_grad:
            self.features.eval()

    def forward(self, x):
        results = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in TARGET_LAYER.keys():
                results.append(x)
        return results
