import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    def __init__(self, name=None):
        super().__init__()
        self.name = name

    def forward(self, x):
        return x * torch.sigmoid(x)


class Conv2dSamePadding(nn.Conv2d):
    """2D Convolutions with same padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True, name=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation, groups=groups,
                         bias=bias)
        self.stride = self.stride if len(self.stride) == 2 else [
            self.stride[0]] * 2
        self.name = name

    def forward(self, x):
        input_h, input_w = x.size()[2:]
        kernel_h, kernel_w = self.weight.size()[2:]
        stride_h, stride_w = self.stride
        output_h, output_w = math.ceil(
            input_h / stride_h), math.ceil(input_w / stride_w)
        pad_h = max(
            (output_h - 1) * self.stride[0] + (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max(
            (output_w - 1) * self.stride[1] + (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w //
                          2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)



def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels,
                  kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels,
                  kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def up_conv(in_channels, out_channels):
    return nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size=2, stride=2
    )


def upsize2(x, sampling='nearest'):
    x = F.interpolate(x, scale_factor=2, mode=sampling)
    return x

    
    
class Decode(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Decode, self).__init__()

        self.top = nn.Sequential(
            nn.Conv2d(in_channel, out_channel//2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d( out_channel//2),
            Swish(), #nn.ReLU(inplace=True),
            #nn.Dropout(0.1),

            nn.Conv2d(out_channel//2, out_channel//2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel//2),
            Swish(), #nn.ReLU(inplace=True),
            #nn.Dropout(0.1),

            nn.Conv2d(out_channel//2, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channel),
            Swish(), #nn.ReLU(inplace=True),
        )
        nn.init.xavier_uniform_(self.top[0].weight)
        nn.init.xavier_uniform_(self.top[3].weight)
        nn.init.xavier_uniform_(self.top[6].weight)


    def forward(self, x):
        x = self.top(torch.cat(x, 1))
        return x



class BnInception(nn.Module):
    def __init__(self,  in_channels, out_channels, kernel_size=[1,3,5], padding =[0,1,2]):
        super(BnInception, self).__init__()
        
        self.conv_c0 = nn.Conv2d(in_channels, out_channels, kernel_size[0], padding=padding[0])
        nn.init.xavier_uniform_(self.conv_c0.weight)
        self.conv_c1 = nn.Conv2d(in_channels, out_channels, kernel_size[1], padding=padding[1])
        nn.init.xavier_uniform_(self.conv_c1.weight)
        self.conv_c2 = nn.Conv2d(in_channels, out_channels, kernel_size[2], padding=padding[2])
        nn.init.xavier_uniform_(self.conv_c2.weight)
        self.batchNorm1 = nn.BatchNorm2d(int(out_channels*3))
        self.relu = nn.ReLU(inplace=True)

        self.final_conv = nn.Conv2d(3*out_channels, out_channels, kernel_size=1)
        nn.init.xavier_uniform_(self.final_conv.weight)
        self.batchNorm2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = torch.cat(x, dim=1)
        x0 = self.conv_c0(x)
        x1 = self.conv_c1(x)
        x2 = self.conv_c2(x)
        x = torch.cat([x0,x1,x2], dim=1)
        x = self.batchNorm1(x)
        x = self.relu(x)
        
        x = self.final_conv(x)
        x = self.batchNorm2(x)
        x = self.relu(x)
        # print(x.size())
        return x



from segmentation_models_pytorch.base import modules as md

class AttentionDecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        self.attention1 = md.Attention('scse', in_channels=in_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        self.attention2 = md.Attention('scse', in_channels=out_channels)

    def forward(self, x):
        x = torch.cat(x, 1)
        
        x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class ClassificationHead(nn.Sequential):

    def __init__(self, in_channels, classes, pooling="avg", dropout=0.2, activation=None):
        if pooling not in ("max", "avg"):
            raise ValueError("Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
        pool = nn.AdaptiveAvgPool2d(1) if pooling == 'avg' else nn.AdaptiveMaxPool2d(1)
        flatten = md.Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, classes, bias=True)
        activation = md.Activation(activation)
        super().__init__(pool, flatten, dropout, linear, activation)
