import torch
from torch import nn
from torchvision.models import vgg19

TARGET_LAYER = {1: 'relu3_1', 6: 'relu2_1', 11: 'relu1_1'}


class VGGExtractor(nn.Module):
    def __init__(self, return_type='List', require_grad=False):
        super(VGGExtractor, self).__init__()
        features = list(vgg19(pretrained=True).features)[:12]
        self.features = torch.nn.ModuleList(features)
        if not require_grad:
            self.features.eval()
        self.return_type = return_type

    def forward(self, x):
        if self.return_type == 'List':
            results = []
        elif self.return_type == 'Dict':
            results = {}
        else:
            raise ValueError(f'invalid return_type: {self.return_type}')
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in TARGET_LAYER.keys():
                if self.return_type == 'List':
                    results.append(x)
                elif self.return_type == 'Dict':
                    results[TARGET_LAYER[idx]] = x
        return results
