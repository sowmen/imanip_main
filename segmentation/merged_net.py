import sys
sys.path.append('../image_manipulation/')

import torch
from torch import nn
import timm
from segmentation.srm_kernel import setup_srm_layer

class SRM_Classifer(nn.Module):
    def __init__(self, in_channels):
        super(SRM_Classifer, self).__init__()
        
        self.srm_conv = setup_srm_layer(3)
        
        self.bayer_conv = nn.Conv2d(in_channels, out_channels=3, kernel_size=5, padding=2, bias=False)
        nn.init.xavier_uniform_(self.bayer_conv.weight)
        
        self.rgb_conv = nn.Conv2d(in_channels, out_channels=10, kernel_size=5, padding=2, bias=False)
        nn.init.xavier_uniform_(self.rgb_conv.weight)
        
        self.base_model = timm.create_model('tf_efficientnet_b4_ns', pretrained=True, in_chans=16, num_classes=1)
        
    def forward(self, input):
        x1 = self.srm_conv(input)
        x2 = self.bayer_conv(input)
        x3 = self.rgb_conv(input)
        x = torch.cat([x1, x2, x3], dim=1)
        
        x = self.base_model(x)
        return x