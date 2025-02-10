import torch
import torch.nn as nn
from models.core.depthwise_conv import DepthwiseSeparableConv

class DLFModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            DepthwiseSeparableConv(3, 32, stride=2),
            nn.ReLU(),
            DepthwiseSeparableConv(32, 64, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
