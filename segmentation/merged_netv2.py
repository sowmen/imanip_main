import sys
sys.path.append('../image_manipulation/')

import torch
from torch import nn
from segmentation.srm_kernel import setup_srm_layer
from segmentation.timm_efficientnet_encoder import TimmEfficientNetBaseEncoder, timm_efficientnet_encoder_params
from segmentation import layers

import re
from tqdm import tqdm


class Mani_FeatX(nn.Module):
    def __init__(self,  in_channels=3, num_classes=1, 
                        model_name='tf_efficientnet_b4_ns', freeze=False, checkpoint=None):
        super(Mani_FeatX, self).__init__()
        

        self.in_channels = in_channels
        
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


        self.encoder = TimmEfficientNetBaseEncoder(in_channels=54, model_name=model_name, pretrained=True)
        self.encoder_params = timm_efficientnet_encoder_params[model_name.rsplit('_', 1)[0]]['params']

        self.conv_reducer = nn.Sequential(
            nn.Conv2d(self.encoder_params['out_channels'][-1], 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.SiLU(inplace=True)
        )
        
        self.classifier = layers.ClassificationHead(in_channels=256, classes=num_classes, dropout=self.encoder_params['drop_rate'])

        if freeze:
            self.freeze()
        if checkpoint:
            self.load_weights(checkpoint)
    

    def forward(self, im, ela):

        x1 = self.srm_conv(im)
        x2 = self.bayer_conv(im)
        x3 = self.rgb_conv(im)
        x_ela = self.ela_net(ela)


        _merged_input = torch.cat([x1, x2, x3, x_ela], dim=1)
        
        stage_outputs = self.encoder(_merged_input)
        reduced_feat = self.conv_reducer(stage_outputs[-1])
        class_tensor = self.classifier(reduced_feat)
        
        return class_tensor, (reduced_feat, _merged_input, stage_outputs)
    
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        print('--------- SRM Frozen -----------')
            
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
        print('--------- SRM Opened -----------')
    
    def load_weights(self, checkpoint):
        pretrained_dict = torch.load(checkpoint)
        
        filtered_dict = {re.sub("^module.", "", k): v for k, v in tqdm(pretrained_dict.items(), desc=f"Filtering Weights[{checkpoint}]")\
             if re.sub("^module.", "", k) in self.state_dict().keys()}
        
        print(self.load_state_dict(filtered_dict, strict=False))