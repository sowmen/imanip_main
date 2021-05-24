import timm
import torch.nn as nn
import gc
from segmentation_models_pytorch.base import modules as md

class TimmEfficientNetBaseEncoder(nn.Module):
        def __init__(self, model_name="tf_efficientnet_b4_ns", in_channels=3, pretrained=True, attention=None):
            super().__init__()

            self.model_name = model_name
            self.in_channels = in_channels
            self.attention = attention
            self.params = timm_efficientnet_encoder_params[model_name.rsplit('_', 1)[0]]['params']


            base_model_sequential = timm.create_model(
                model_name=model_name,
                pretrained=pretrained,
                in_chans=in_channels,
            ).as_sequential()             

            self.stem = base_model_sequential[:3]
            self.blocks = base_model_sequential[3:10]

            del base_model_sequential
            gc.collect()

            
            self.attention_modules = nn.ModuleList()
            for num_channel in self.params['out_channels']:
                if self.attention is not None:
                    self.attention_modules.append(md.Attention(attention, in_channels=num_channel))
                else:
                    self.attention_modules.append(nn.Identity())


        def forward(self, x):
            """
            Output shapes for EfficientNet-B4
            stage_outputs = [  48  x 128 x 128,     -> output after stem
                               24  x 128 x 128,     -> Block 1
                               32  x 64  x 64 ,     -> Block 5
                               56  x 32  x 32 ,     -> Block 9
                               160 x 16  x 16 ,     -> Block 21
                               448 x 8   x 8        -> Block 31, before conv_head
                            ]
            """
            stage_outputs = []
            
            x = self.stem(x)
            x = self.attention_modules[0](x)

            stage_outputs.append(x)
            
            attn_idx, idx = 1, 0
            for block in self.blocks:
                for inner_block in block:
                    x = inner_block(x)

                    if idx in self.params['stage_idxs']:
                        x = self.attention_modules[attn_idx](x)
                        stage_outputs.append(x)
                        attn_idx += 1
                    idx += 1

            return stage_outputs



timm_efficientnet_encoder_params = {

    "tf_efficientnet_b4": {
        "params": {
            "out_channels": (48, 24, 32, 56, 160, 448),
            "stage_idxs": (1, 5, 9, 21, 31),
            "drop_rate": 0.4,
        },
    },

    "tf_efficientnet_b5": {
        "params": {
            "out_channels": (48, 24, 40, 64, 176, 512),
            "stage_idxs": (2, 7, 12, 26, 38),
            "drop_rate": 0.4,
        },
    },

    "tf_efficientnet_b6": {
        "params": {
            "out_channels": (56, 32, 40, 72, 200, 576),
            "stage_idxs": (2, 8, 14, 30, 44),
            "drop_rate": 0.5,
        },
    },

    "tf_efficientnet_b7": {
        "params": {
            "out_channels": (64, 32, 48, 80, 224, 640),
            "stage_idxs": (3, 10, 17, 37, 54),
            "drop_rate": 0.5,
        },
    },
}