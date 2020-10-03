import sys
sys.path.append('../image_manipulation/')

import segmentation_models_pytorch as smp
import torch.nn as nn
import gc
from segmentation.timm_efficientnet import EfficientNet

class SMP_DIY(nn.Module):
    
    def __init__(self, num_classes, encoder_checkpoint="", freeze_encoder=False):
        super(SMP_DIY, self).__init__()
        
        self.encoder = EfficientNet(encoder_checkpoint=encoder_checkpoint, freeze_encoder=freeze_encoder).get_encoder()
        self.model = smp.Unet('timm-efficientnet-b4', classes=num_classes, encoder_weights='noisy-student') 
        self.decoder = self.model.decoder 
        self.segmentation_head = self.model.segmentation_head
        
        del self.model 
        gc.collect() 
        
        
        
    def forward(self, x):
        _, _, features = self.encoder(x)
        x = self.decoder(*features)
        x = self.segmentation_head(x)
        
        return x