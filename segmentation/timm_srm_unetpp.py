import sys
sys.path.append('../image_manipulation/')

import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation.layers import upsize2, Decode
# from segmentation.layers import BnInception as Decode
import copy

class UnetPP(nn.Module):
    def __init__(self, encoder, num_classes=1, layer='end', sampling='nearest'):
        super(UnetPP, self).__init__()
        
        self.encoder = encoder
        self.num_classes = num_classes
        self.layer = layer
        self.sampling = sampling
        
        # TOP DOWN -> TOP DECODER IS 0
        if self.layer == 'start':
            self.size = [24,32,56,112,272]
        elif self.layer == 'end':
            self.size = [24,32,56,160,448]
        
        self.mix = nn.Parameter(torch.FloatTensor(5))
        self.mix.data.fill_(1)
        
        self.decode0_1 = Decode(self.size[0] + self.size[1], 32)
        self.decode1_1 = Decode(self.size[1] + self.size[2], 64)
        self.decode2_1 = Decode(self.size[2] + self.size[3], 128)
        self.decode3_1 = Decode(self.size[3] + self.size[4], 512)
        
        self.decode0_2 = Decode(self.size[0] +32+64, 64)
        self.decode1_2 = Decode(self.size[1] +64+128, 128)
        self.decode2_2 = Decode(self.size[2] +128+512, 512)
        
        self.decode0_3 = Decode(self.size[0] +32+64+128, 128)
        self.decode1_3 = Decode(self.size[1] +64+128+512, 512)
        
        self.decode0_4 = Decode(self.size[0] +32+64+128+512, 512)
        
        # self.logit0 = nn.Conv2d(self.size[0], self.num_classes, kernel_size=1)
        self.logit1 = nn.Conv2d(32, self.num_classes, kernel_size=1)
        nn.init.xavier_uniform_(self.logit1.weight)
        self.logit2 = nn.Conv2d(64, self.num_classes, kernel_size=1)
        nn.init.xavier_uniform_(self.logit2.weight)
        self.logit3 = nn.Conv2d(128, self.num_classes, kernel_size=1)
        nn.init.xavier_uniform_(self.logit3.weight)
        self.logit4 = nn.Conv2d(512, self.num_classes, kernel_size=1)
        nn.init.xavier_uniform_(self.logit4.weight)
        
        
    def forward(self, inp, ela):
        batch_size, C, H, W = inp.shape
        
        x, (_, _merged_input, enc_out, start, end) = self.encoder(inp, ela)
        if self.layer == 'start':
            layer = start
        else:
            layer = end
            
        x0_0 = layer[0]
        x1_0 = layer[1]
        x2_0 = layer[2]
        x3_0 = layer[3]
        x4_0 = layer[4]

        x0_1 = self.decode0_1([x0_0, upsize2(x1_0, self.sampling)])
        x1_1 = self.decode1_1([x1_0, upsize2(x2_0, self.sampling)])
        x2_1 = self.decode2_1([x2_0, upsize2(x3_0, self.sampling)])
        x3_1 = self.decode3_1([x3_0, upsize2(x4_0, self.sampling)])
        
        x0_2 = self.decode0_2([x0_0, x0_1, upsize2(x1_1, self.sampling)])
        x1_2 = self.decode1_2([x1_0, x1_1, upsize2(x2_1, self.sampling)])
        x2_2 = self.decode2_2([x2_0, x2_1, upsize2(x3_1, self.sampling)])
        
        x0_3 = self.decode0_3([x0_0, x0_1, x0_2, upsize2(x1_2, self.sampling)])
        x1_3 = self.decode1_3([x1_0, x1_1, x1_2, upsize2(x2_2, self.sampling)])
        
        x0_4 = self.decode0_4([x0_0, x0_1, x0_2, x0_3, upsize2(x1_3, self.sampling)])
        
        # deep supervision
        # logit0 = self.logit0(x0_0)
        logit1 = self.logit1(x0_1)
        logit2 = self.logit2(x0_2)
        logit3 = self.logit3(x0_3)
        logit4 = self.logit4(x0_4)
        
        logit = self.mix[1]*logit1 + self.mix[2]*logit2 + self.mix[3]*logit3 + self.mix[4]*logit4
        logit = F.interpolate(logit, size=(H,W), mode='bilinear', align_corners=False)
        
        return logit
        