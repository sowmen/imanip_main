import sys
sys.path.append('../image_manipulation/')

import torch
from torch import nn
import timm
from segmentation.srm_kernel import setup_srm_layer
from segmentation.timm_efficientnet import EfficientNet
import gc

class SRM_Classifer(nn.Module):
    def __init__(self, in_channels=3):
        super(SRM_Classifer, self).__init__()
        
        self.srm_conv = setup_srm_layer(3)
        
        self.bayer_conv = nn.Conv2d(in_channels, out_channels=3, kernel_size=5, padding=2, bias=False)
        nn.init.xavier_uniform_(self.bayer_conv.weight)
        
        self.rgb_conv = nn.Conv2d(in_channels, out_channels=16, kernel_size=5, padding=2, bias=False)
        nn.init.xavier_uniform_(self.rgb_conv.weight)
        
        self.ela_net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        nn.init.xavier_uniform_(self.ela_net[0].weight)
        nn.init.xavier_uniform_(self.ela_net[3].weight)

        base_model = EfficientNet(in_channels=54)
        self.encoder = base_model.encoder
        self.classifier = base_model.classifier

        del base_model
        gc.collect()
        
    def forward(self, im, ela):
        x1 = self.srm_conv(im)
        x2 = self.bayer_conv(im)
        x3 = self.rgb_conv(im)
        x_ela = self.ela_net(ela)
        x = torch.cat([x1, x2, x3, x_ela], dim=1)

        x, _, _ = self.encoder(x)
        x = self.classifier(x)

        return x