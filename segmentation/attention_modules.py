import torch
import torch.nn as nn

from segmentation.gc_block import GlobalContext
from timm.models.layers.norm import LayerNorm2d

class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class AttentionGate(nn.Module):
    """
    Attention Block from "Attention U-Net: Learning Where to Look for the Pancreas"
    """

    def __init__(self, F_l, F_g, F_int):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_l = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, l, g): 
        # l -> current layer all skip connections
        # g -> output of below layer after upsize. Acts as gate
        g1 = self.W_g(g)
        l1 = self.W_l(l)
        psi = self.relu(g1 + l1)
        psi = self.psi(psi)
        out = l * psi
        return out



class GatedContextAttention(nn.Module):
    """
    Attention Block from "Attention U-Net: Learning Where to Look for the Pancreas"
    """

    def __init__(self, layer_channels, gate_channels, inter_channels):
        super().__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(inter_channels)
        )

        self.layer_gcb_context = GlobalContext(layer_channels)
        self.W_layer = nn.Sequential(
            nn.Conv2d(layer_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(inter_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, l, g): 
        # l -> current layer all skip connections
        # g -> output of below layer after upsize. Acts as gate

        l1 = self.layer_gcb_context(l)
        l1 = self.W_layer(l1)

        g1 = self.W_gate(g)
        
        psi = self.relu(g1 + l1)
        psi = self.psi(psi)
        
        out = l * psi
        return out


class Attention(nn.Module):

    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == 'scse':
            self.attention = SCSEModule(**params)
        elif name == 'gcb':
            self.attention = GlobalContext(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)



##############---------------- Recursive Residual Attention ----------##############
class Recurrent_block(nn.Module):
    """
    Recurrent Block for R2Unet_CNN
    """
    def __init__(self, out_ch, t=2):
        super(Recurrent_block, self).__init__()

        self.t = t
        self.out_ch = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x = self.conv(x)
            out = self.conv(x + x)
        return out


class RRCNN_block(nn.Module):
    """
    Recurrent Residual Convolutional Neural Network Block
    """
    def __init__(self, in_ch, out_ch, t=2):
        super(RRCNN_block, self).__init__()

        self.RCNN = nn.Sequential(
            Recurrent_block(out_ch, t=t),
            Recurrent_block(out_ch, t=t)
        )
        self.Conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv(x)
        x2 = self.RCNN(x1)
        out = x1 + x2
        return out

############################################################################################