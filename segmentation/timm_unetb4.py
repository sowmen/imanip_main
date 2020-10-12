import sys
sys.path.append('../image_manipulation/')

import torch
import torch.nn as nn
from segmentation.layers import Decode, upsize2, BnInception
import copy

class UnetB4(nn.Module):
    def __init__(self, encoder, in_channels=3, num_classes=1, layer='end', sampling='nearest'):
        super(UnetB4, self).__init__()
        
        self.encoder = encoder
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.layer = layer
        self.sampling = sampling
        
        # BOTTOM UP -> LOWEST DECODER IS 0
        if self.layer == 'start':
            self.size = [272,112,56,32,24]
        elif self.layer == 'end':
            self.size = [448,160,56,32,24]
        
        self.decode0 = Decode(self.size[0] + self.size[1], 512)
        self.decode1 = Decode(512 + self.size[2], 256)
        self.decode2 = Decode(256 + self.size[3], 128)
        self.decode3 = Decode(128 + self.size[4], 64)
        self.decode_input = Decode(64 + self.in_channels, 32)
        
        self.final_conv = nn.Conv2d(32, self.num_classes, kernel_size=1)
        # self.final_conv = nn.Conv2d(32, self.num_classes, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.final_conv.weight)

    def forward(self, inp, ela):
        # _input = copy.deepcopy(x)
        
        x, (_merged_input, feat, start, end) = self.encoder(inp, ela)
        
        if self.layer == 'start':
            layer = start
        else:
            layer = end
            
        x = self.decode0([upsize2(layer[-1], self.sampling), layer[-2]]) # 1792x8x8 -> 1792x16x16 + 112x16x16 => 512x16x16
        x = self.decode1([upsize2(x, self.sampling), layer[-3]])   # 512x16x16 -> 512x32x32 + 56x32x32 => 256x32x32
        x = self.decode2([upsize2(x, self.sampling), layer[-4]])   # 256x32x32 -> 256x64x64 + 32x64x64 => 128x64x64
        x = self.decode3([upsize2(x, self.sampling), layer[-5]])   # 128x64x64 -> 128x128x128 + 24x128x128 => 64x128x128
        x = self.decode_input([upsize2(x, self.sampling), _merged_input]) # 64x128x128 -> 64x256x256 + 54x256x256 => 32x256x256
        
        x = self.final_conv(x) # 32x256x256 => 1x256x256
        
        return x
    
class UnetB4_Inception(nn.Module):
    def __init__(self, encoder, in_channels=3, num_classes=1, layer='end', sampling='nearest'):
        super(UnetB4_Inception, self).__init__()
        
        self.encoder = encoder
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.layer = layer
        self.sampling = sampling
        
        # BOTTOM UP -> LOWEST DECODER IS 0
        if self.layer == 'start':
            self.size = [272,112,56,32,24]
        elif self.layer == 'end':
            self.size = [448,160,56,32,24]
        
        self.decode0 = BnInception(self.size[0] + self.size[1], 256)
        self.decode1 = BnInception(256 + self.size[2], 128)
        self.decode2 = BnInception(128 + self.size[3], 64)
        self.decode3 = BnInception(64 + self.size[4], 32)
        self.decode_input = BnInception(32 + self.in_channels, 16)
        
        self.final_conv = nn.Conv2d(16, self.num_classes, kernel_size=1)
        # self.final_conv = nn.Conv2d(32, self.num_classes, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.final_conv.weight)
        
    def forward(self, inp, ela):
        # _input = copy.deepcopy(x)
        
        x, (_merged_input, feat, start, end) = self.encoder(inp, ela)
        
        if self.layer == 'start':
            layer = start
        else:
            layer = end
            
        x = self.decode0([upsize2(layer[-1], self.sampling), layer[-2]]) # 1792x8x8 -> 1792x16x16 + 112x16x16 => 512x16x16
        x = self.decode1([upsize2(x, self.sampling), layer[-3]])   # 512x16x16 -> 512x32x32 + 56x32x32 => 256x32x32
        x = self.decode2([upsize2(x, self.sampling), layer[-4]])   # 256x32x32 -> 256x64x64 + 32x64x64 => 128x64x64
        x = self.decode3([upsize2(x, self.sampling), layer[-5]])   # 128x64x64 -> 128x128x128 + 24x128x128 => 64x128x128
        x = self.decode_input([upsize2(x, self.sampling), _merged_input]) # 64x128x128 -> 64x256x256 + 54x256x256 => 32x256x256
        
        x = self.final_conv(x) # 32x256x256 => 1x256x256
        
        return x
        