import sys
sys.path.append('../image_manipulation/')

import torch
from torch import nn
import timm
from segmentation.srm_kernel import setup_srm_layer
from segmentation.timm_efficientnet import EfficientNet
import gc
import copy
from collections import OrderedDict

class SRM_Classifer(nn.Module):
    def __init__(self, in_channels=3, encoder_checkpoint="", freeze_encoder=False):
        super(SRM_Classifer, self).__init__()
        
        self.in_channels = in_channels
        
        self.srm_conv = setup_srm_layer(self.in_channels)
        
        self.bayer_conv = nn.Conv2d(self.in_channels, out_channels=3, kernel_size=5, padding=2, bias=False)
        nn.init.xavier_uniform_(self.bayer_conv.weight)
        
        self.rgb_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, out_channels=16, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels=16, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        nn.init.xavier_uniform_(self.rgb_conv[0].weight)
        nn.init.xavier_uniform_(self.rgb_conv[1].weight)
        
        self.ela_net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(16),
            # nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        nn.init.xavier_uniform_(self.ela_net[0].weight)
        nn.init.xavier_uniform_(self.ela_net[1].weight)

        self.dft_net = nn.Sequential(
            nn.Conv2d(18, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        nn.init.xavier_uniform_(self.dft_net[0].weight)
        nn.init.xavier_uniform_(self.dft_net[3].weight)


        base_model = EfficientNet(in_channels=86)
        self.encoder = base_model.encoder
        self.classifier = base_model.classifier

        del base_model
        gc.collect()

        if freeze_encoder:
            self.freeze()
        if encoder_checkpoint:
            self.load_weights(encoder_checkpoint)
        
    def forward(self, im, ela, dft_dwt):
        x1 = self.srm_conv(im)
        x2 = self.bayer_conv(im)
        x3 = self.rgb_conv(im)
        x_ela = self.ela_net(ela)
        x_dft = self.dft_net(dft_dwt)
        _merged_input = torch.cat([x1, x2, x3, x_ela, x_dft], dim=1)
        
        enc_out, (start, end), _ = self.encoder(_merged_input)
        x = self.classifier(enc_out)
        
        return x, (_merged_input, enc_out, start, end)
    
    def freeze(self):
        for param in super().parameters():
            param.requires_grad = False
        print('--------- SRM Frozen -----------')
            
    def unfreeze(self):
        for param in super().parameters():
            param.requires_grad = True
        print('--------- SRM Opened -----------')
    
    def load_weights(self, checkpoint=""):
        print(f'--------- Loaded Checkpoint: {checkpoint} ----------')
        checkpoint = torch.load(checkpoint)
        encoder_dict = OrderedDict()
        for item in checkpoint.items():
            key = item[0].split('.',1)[-1]
            if('base_model' in key):
                s = key.replace('base_model.','')
                encoder_dict[s] = item[1]
            else:
                encoder_dict[key] = item[1]
        print(super().load_state_dict(encoder_dict))
