import timm
from timm.models.layers.classifier import create_classifier
from timm.models.layers.separable_conv import SeparableConvBnAct
from timm.models.layers.activations import Swish
import torch
import torch.nn as nn
import gc

from collections import OrderedDict

class EfficientNet(nn.Module):
    def __init__(
        self, model_name='tf_efficientnet_b4_ns', in_channels=3, num_classes=1, encoder_checkpoint="", freeze_encoder=False
    ):
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        base_model_sequential = timm.create_model(
            model_name=self.model_name,
            pretrained=True,
            in_chans=self.in_channels,
            num_classes=self.num_classes,
        ).as_sequential()

        self.encoder = self.Encoder(self.model_name, base_model_sequential[:13])
        if encoder_checkpoint:
            self.encoder.load_weights(encoder_checkpoint)
        if freeze_encoder:
            self.encoder.freeze()
            
        self.classifier = base_model_sequential[13:]

        del base_model_sequential
        gc.collect()

        # self.reduce_channels = nn.Sequential(
        #     # SeparableConvBnAct(1792, 1792//4, act_layer=Swish)
        #     nn.Conv2d(448, 448//4, kernel_size=1, stride=1, bias=False),
        #     nn.BatchNorm2d(448//4),
        #     Swish(),
        # )
        
        # self.global_pool, self.classifier = create_classifier(
        #     448//4, self.num_classes, pool_type='avg')
        

    def forward(self, x):
        x, _, _ = self.encoder(x)
        # x = self.reduce_channels(smp[-1])
        # x = self.global_pool(x)
        x = self.classifier(x)

        return x

    class Encoder(nn.Module):
        def __init__(self, model_name, layers):
            super().__init__()

            self.name = model_name
            self.stem = layers[:3]
            self.blocks = layers[3:10]
            self.head = layers[10:13]

        def forward(self, x):
            

            start_outputs = []
            end_outputs = []
            smp_outputs = []
            
            smp_outputs.append(x) # input
            
            x = self.stem(x)
            smp_outputs.append(x) # conv_stem
            
            idx = 0
            for (i, block) in enumerate(self.blocks):
                for inner_block in block:
                    x = inner_block(x)

                    if idx in [0, 2, 6, 10, 22]:
                        start_outputs.append(x)
                    if idx in [1, 5, 9, 21, 31]:
                        end_outputs.append(x)
                    if idx in [5, 9, 21, 31]:
                        smp_outputs.append(x)
                    idx += 1

            x = self.head(x)

            return x, (start_outputs, end_outputs), smp_outputs 
        
        def load_weights(self, checkpoint=""):
            # print(f'--------- Loaded Checkpoint: {checkpoint} ----------')
            checkpoint = torch.load(checkpoint)
            encoder_dict = OrderedDict()
            for item in checkpoint.items():
                if('encoder' in item[0]):
                    s = item[0]
                    encoder_dict[s[s.find('encoder')+len('encoder')+1:]] = item[1]
            super().load_state_dict(encoder_dict) 
            
        
        def freeze(self):
            for param in super().parameters():
                param.requires_grad = False
            print('--------- Encoder Frozen -----------')
                
        def unfreeze(self):
            for param in super().parameters():
                param.requires_grad = True
            print('--------- Encoder Opened -----------')

    def get_encoder(self):
        return self.encoder
