import sys
sys.path.append('../image_manipulation/')

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import gc
from segmentation.timm_efficientnet import EfficientNet
from segmentation.srm_kernel import setup_srm_layer

class SMP_SRM(nn.Module):
    
    def __init__(self, in_channels=3):
        super(SMP_SRM, self).__init__()
        
        self.in_channels = in_channels
        
        self.srm_conv = setup_srm_layer(self.in_channels)
        
        self.bayer_conv = nn.Conv2d(self.in_channels, out_channels=3, kernel_size=5, padding=2, bias=False)
        nn.init.xavier_uniform_(self.bayer_conv.weight)
        
        self.rgb_conv = nn.Conv2d(self.in_channels, out_channels=16, kernel_size=5, padding=2, bias=False)
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
        
        
        self.base_model = smp.DeepLabV3('timm-efficientnet-b4', in_channels=54, classes=1, encoder_weights='noisy-student') 
        
        
        
    def forward(self, im, ela):
        x1 = self.srm_conv(im)
        x2 = self.bayer_conv(im)
        x3 = self.rgb_conv(im)
        x_ela = self.ela_net(ela)
        _merged_input = torch.cat([x1, x2, x3, x_ela], dim=1)
        
        out = self.base_model(_merged_input)
        
        return out