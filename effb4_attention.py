import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from timm.models.layers.se import EffectiveSEModule
from timm.models.layers.separable_conv import SeparableConv2d


class Efficient_Attention(nn.Module):
    def __init__(self):
        super(Efficient_Attention, self).__init__()

        self.base_model = timm.create_model(
            "tf_efficientnet_b4_ns", pretrained=True, num_classes=1
        )
        self.drop_rate = self.base_model.drop_rate

        self.head = nn.Sequential(*list(self.base_model.children())[:3])
        self.block_head = nn.Sequential(*list(self.base_model.blocks.children())[:4])

        self.eca_layer = EffectiveSEModule(112)
        self.att_conv = SeparableConv2d(112, 1)

        self.block_end = nn.Sequential(*list(self.base_model.blocks.children())[4:])
        self.trailer = nn.Sequential(*list(self.base_model.children())[-5:-1])
        self.classifier = self.base_model.classifier

        del self.base_model

    def forward(self, x):
        x = self.head(x)
        x = self.block_head(x)

        mask = self.eca_layer(x)
        mask = self.att_conv(mask)
        mask = torch.sigmoid(mask)
        x = x * mask

        x = self.block_end(x)
        x = self.trailer(x)
        x = x.flatten(1)
        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate)
        x = self.classifier(x)

        return x, mask
