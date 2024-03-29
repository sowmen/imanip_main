import sys
sys.path.append('../image_manipulation/')

import torch
from torch import nn
from timm.models.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
from segmentation.srm_kernel import setup_srm_layer
from segmentation.timm_efficientnet import EfficientNet
import gc
import re
from collections import OrderedDict

class SRM_Classifer(nn.Module):
    def __init__(self, in_channels=3, encoder_checkpoint="", freeze_encoder=False, num_classes=1):
        super(SRM_Classifer, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        self.srm_conv = setup_srm_layer(self.in_channels)
        
        self.bayer_conv = nn.Conv2d(self.in_channels, out_channels=3, kernel_size=5, padding=2, bias=False)
        nn.init.xavier_uniform_(self.bayer_conv.weight)
        
        self.rgb_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, out_channels=16, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(16, out_channels=16, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        nn.init.xavier_uniform_(self.rgb_conv[0].weight)
        nn.init.xavier_uniform_(self.rgb_conv[1].weight)

        
        self.ela_net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        nn.init.xavier_uniform_(self.ela_net[0].weight)
        nn.init.xavier_uniform_(self.ela_net[1].weight)


        base_model = EfficientNet(in_channels=54)
        self.encoder = base_model.encoder


        self.reducer = nn.Sequential(
            SelectAdaptivePool2d(pool_type="avg", flatten=True),
            nn.Dropout(0.3),
            nn.Linear(1792, 448),
            nn.ReLU(inplace=True),
            nn.Linear(448, 256),
            nn.ReLU(inplace=True),
        )
        nn.init.xavier_uniform_(self.reducer[2].weight)
        nn.init.xavier_uniform_(self.reducer[4].weight)
        
        self.classifier = nn.Linear(256, self.num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)

        del base_model
        gc.collect()

        if freeze_encoder:
            self.freeze()
        if encoder_checkpoint:
            self.load_weights(encoder_checkpoint)
    
    # @torch.cuda.amp.autocast()
    def forward(self, im, ela):

        x1 = self.srm_conv(im)
        x2 = self.bayer_conv(im)
        x3 = self.rgb_conv(im)
        x_ela = self.ela_net(ela)


        _merged_input = torch.cat([x1, x2, x3, x_ela], dim=1)
        
        enc_out, (start, end), _ = self.encoder(_merged_input)
        reduced_feat = self.reducer(enc_out)

        x = self.classifier(reduced_feat)
        
        return x, (reduced_feat, _merged_input, enc_out, start, end)
    
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        print('--------- SRM Frozen -----------')
            
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
        print('--------- SRM Opened -----------')
    
    def load_weights(self, checkpoint=""):
        print(f'--------- Loaded Checkpoint: {checkpoint} ----------')
        pretrained_dict = torch.load(checkpoint)
        del pretrained_dict['module.classifier.weight']
        del pretrained_dict['module.classifier.bias']
        
        encoder_dict = {re.sub("^module.", "", k): v for k, v in pretrained_dict.items()}
        
        print(self.load_state_dict(encoder_dict, strict=False))
        
        
