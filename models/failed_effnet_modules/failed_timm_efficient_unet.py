import sys
sys.path.append('../image_manipulation/')

from models.timm_efficientnet import EfficientNet
from models.layers import *
from collections import OrderedDict

__all__ = [
    "EfficientUnet",
    "get_efficientunet",
]


class EfficientUnet(nn.Module):
    def __init__(self, encoder, out_channels=2, concat_input=True):
        super().__init__()

        self.encoder = encoder
        self.concat_input = concat_input

        self.up_conv1 = up_conv(self.n_channels, 512)
        self.double_conv1 = double_conv(self.size[0], 512)
        self.up_conv2 = up_conv(512, 256)
        self.double_conv2 = double_conv(self.size[1], 256)
        self.up_conv3 = up_conv(256, 128)
        self.double_conv3 = double_conv(self.size[2], 128)
        self.up_conv4 = up_conv(128, 64)
        self.double_conv4 = double_conv(self.size[3], 64)

        if self.concat_input:
            self.up_conv_input = up_conv(64, 32)
            self.double_conv_input = double_conv(self.size[4], 32)

        self.final_conv = nn.Conv2d(self.size[5], out_channels, kernel_size=1)

    @property
    def n_channels(self):
        n_channels_dict = {
            "tf_efficientnet_b0_ns": 1280,
            "tf_efficientnet_b1_ns": 1280,
            "tf_efficientnet_b2_ns": 1408,
            "tf_efficientnet_b3_ns": 1536,
            "tf_efficientnet_b4_ns": 1792,
            "tf_efficientnet_b5_ns": 2048,
            "tf_efficientnet_b6_ns": 2304,
            "tf_efficientnet_b7_ns": 2560,
        }
        return n_channels_dict[self.encoder.name]

    @property
    def size(self):
        size_dict = {
            "tf_efficientnet_b0_ns": [592, 296, 152, 80, 35, 32],
            "tf_efficientnet_b1_ns": [592, 296, 152, 80, 35, 32],
            "tf_efficientnet_b2_ns": [600, 304, 152, 80, 35, 32],
            "tf_efficientnet_b3_ns": [608, 304, 160, 88, 35, 32],
            "tf_efficientnet_b4_ns": [624, 312, 160, 88, 35, 32],
            "tf_efficientnet_b5_ns": [640, 320, 168, 88, 35, 32],
            "tf_efficientnet_b6_ns": [656, 328, 168, 96, 35, 32],
            "tf_efficientnet_b7_ns": [672, 336, 176, 96, 35, 32],
        }
        return size_dict[self.encoder.name]

    def forward(self, x):
        input_ = x

        x, (start, end), _ = self.encoder(x)

        # print(x.size())
        x = self.up_conv1(x)
        # print(x.size())
        # print("---", list(start)[-1], " -", start[list(start)[-1]].size())
        x = torch.cat([x, start.popitem()[1]], dim=1)
        # print(x.size())
        x = self.double_conv1(x)
        # print(x.size())

        x = self.up_conv2(x)
        # print(x.size())
        # print("---", list(start)[-1], " -", start[list(start)[-1]].size())
        x = torch.cat([x, start.popitem()[1]], dim=1)
        # print(x.size())
        x = self.double_conv2(x)
        # print(x.size())

        x = self.up_conv3(x)
        # print(x.size())
        # print("---", list(start)[-1], " -", start[list(start)[-1]].size())
        x = torch.cat([x, start.popitem()[1]], dim=1)
        # print(x.size())
        x = self.double_conv3(x)
        # print(x.size())

        x = self.up_conv4(x)
        # print(x.size())
        # print("---", list(start)[-1], " -", start[list(start)[-1]].size())
        x = torch.cat([x, start.popitem()[1]], dim=1)
        # print(x.size())
        x = self.double_conv4(x)
        # print(x.size())

        if self.concat_input:
            x = self.up_conv_input(x)
            # print(x.size())
            # print("---", input_.size())
            x = torch.cat([x, input_], dim=1)
            # print(x.size())
            x = self.double_conv_input(x)
            # print(x.size())

        x = self.final_conv(x)
        # print(x.size())

        return x


def get_efficientunet(
    model_name, out_channels=1, encoder_checkpoint='', freeze_encoder=False, concat_input=True
):
    encoder = EfficientNet(
        model_name=model_name,
        encoder_checkpoint=encoder_checkpoint,
        freeze_encoder=freeze_encoder,
    ).get_encoder()
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model
