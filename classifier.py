import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from timm.models.layers import SelectAdaptivePool2d
from timm.models.layers.se import EffectiveSEModule, SEModule
from timm.models.layers.separable_conv import SeparableConv2d


class ClassifierBlock(nn.Module):
    def __init__(self, num_channels):
        super(ClassifierBlock, self).__init__()

        self.conv0 = SeparableConv2d(num_channels, (num_channels // 2))
        self.bn0 = nn.BatchNorm2d((num_channels // 2))
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.dropout = nn.Dropout2d(p=0.3)

        self.conv1 = SeparableConv2d((num_channels // 2), 1792)
        self.bn1 = nn.BatchNorm2d(1792)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1)

        self.se0 = SEModule(1792)
        self.global_pool = SelectAdaptivePool2d(pool_type="max")

        self.relu = nn.ReLU(inplace=True)
        self.linear = nn.Linear(1792, 1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu(x)
        x = self.bn0(x)
        x = self.pool0(x)
        x = self.dropout(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.dropout(x)

        x = self.se0(x)
        x = self.dropout(x)
        x = self.global_pool(x)

        x = x.flatten(1)
        x = self.linear(x)

        return x
