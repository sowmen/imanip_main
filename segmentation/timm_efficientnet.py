import timm
import torch
import torch.nn as nn

from collections import OrderedDict

class EfficientNet(nn.Module):
    def __init__(
        self, model_name, num_classes=1, encoder_checkpoint="", freeze_encoder=False
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

        if encoder_checkpoint:
            checkpoint = torch.load(encoder_checkpoint)
            self.encoder.load_state_dict(checkpoint)

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
            x = self.stem(x)

            start_outputs = OrderedDict()
            end_outputs = OrderedDict()

            idx = 0
            for (i, block) in enumerate(self.blocks):
                for inner_block in block:
                    x = inner_block(x)

                    if idx in [0, 2, 6, 10]:
                        start_outputs[f"block_{i}_layer_{idx}"] = x
                    elif idx in [1, 5, 9, 21]:
                        end_outputs[f"block_{i}_layer_{idx}"] = x

                    idx += 1

            x = self.head(x)

            return x, (start_outputs, end_outputs)

    def get_encoder(self):
        return self.encoder
