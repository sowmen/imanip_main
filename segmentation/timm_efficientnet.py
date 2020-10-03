import timm
import torch
import torch.nn as nn
import gc

from collections import OrderedDict

class EfficientNet(nn.Module):
    def __init__(
        self, model_name='tf_efficientnet_b4_ns', num_classes=1, encoder_checkpoint="", freeze_encoder=False
    ):
        super().__init__()
        base_model_sequential = timm.create_model(
            model_name=model_name,
            pretrained=True,
            num_classes=num_classes,
        ).as_sequential()

        self.encoder = self.Encoder(model_name, base_model_sequential[:13])
        self.classifier = base_model_sequential[13:]

        del base_model_sequential
        gc.collect()

        if encoder_checkpoint:
            self.encoder.load_weights(encoder_checkpoint)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        x, _ = self.encoder(x)
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
            

            start_outputs = OrderedDict()
            end_outputs = OrderedDict()
            smp_outputs = []
            
            smp_outputs.append(x) # input
            
            x = self.stem(x)
            smp_outputs.append(x) # conv_stem
            
            idx = 0
            for (i, block) in enumerate(self.blocks):
                for inner_block in block:
                    x = inner_block(x)

                    if idx in [0, 2, 6, 10]:
                        start_outputs[f"block_{i}_layer_{idx}"] = x
                    if idx in [1, 5, 9, 21]:
                        end_outputs[f"block_{i}_layer_{idx}"] = x
                    if idx in [5, 9, 21, 31]:
                        smp_outputs.append(x)
                    idx += 1

            x = self.head(x)

            return x, (start_outputs, end_outputs), smp_outputs 
        
        def load_weights(self, checkpoint=""):
            checkpoint = torch.load(checkpoint)
            encoder_dict = OrderedDict()
            for item in checkpoint.items():
                if('encoder' in item[0]):
                    s = item[0]
                    encoder_dict[s[s.find('encoder')+len('encoder')+1:]] = item[1]
            super().load_state_dict(encoder_dict) 

    def get_encoder(self):
        return self.encoder
