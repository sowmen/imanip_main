import sys
sys.path.append('../image_manipulation/')

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import gc
from segmentation.srm_kernel import setup_srm_layer

class SMP_SRM_UPP(nn.Module):
    
    def __init__(self, in_channels=3, classifier_only=False):
        super(SMP_SRM_UPP, self).__init__()
        
        self.in_channels = in_channels
        self.classifier_only = classifier_only
        

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
        
        self.aux_params = dict(
            pooling='avg',             # one of 'avg', 'max'
            dropout=0.2,               # dropout ratio, default is None
            activation=None,      # activation function, default is None
            classes=1,                 # define number of output labels
        )
        base_model = smp.UnetPlusPlus(
            in_channels=54,
            classes=1,
            encoder_name='timm-efficientnet-b4',
            encoder_weights=None,
            decoder_attention_type='scse',
            aux_params=self.aux_params
        )
        self.encoder = base_model.encoder

        if classifier_only == False:
            self.decoder = base_model.decoder
            self.segmentation_head = base_model.segmentation_head

        self.classification_head = base_model.classification_head

        del(base_model)
        gc.collect()

        
    def forward(self, im, ela):
        x1 = self.srm_conv(im)
        x2 = self.bayer_conv(im)
        x3 = self.rgb_conv(im)
        x_ela = self.ela_net(ela)
        _merged_input = torch.cat([x1, x2, x3, x_ela], dim=1)
        
        features = self.encoder(_merged_input)

        if self.classifier_only == False:
            decoder_output = self.decoder(*features)
            masks = self.segmentation_head(decoder_output)

        labels = self.classification_head(features[-1])
        
        if self.classifier_only == True:
            return labels
            
        return masks, labels
