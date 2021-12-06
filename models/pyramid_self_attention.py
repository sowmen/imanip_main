import torch
import torch.nn as nn
from timm.models.layers import Conv2dSame

"""
PixelAttention implementation from SPAN
"""
class PixelAttention(nn.Module):
    def __init__(self, in_channels, shift=1, useBN=False, useRes=False):
        super(PixelAttention, self).__init__()

        self.in_channels = in_channels
        self.kernel_range = [3,3]
        self.shift = shift
        self.ff_kernel = [3,3]
        self.useBN = useBN
        self.useRes = useRes
        
        n_p = self.kernel_range[0] * self.kernel_range[1]
        
        self.K_P = Conv2dSame(self.in_channels, self.in_channels*n_p, kernel_size=1, stride=1, bias=False)
        nn.init.xavier_uniform_(self.K_P.weight)
        
        self.V_P = Conv2dSame(self.in_channels, self.in_channels*n_p, kernel_size=1, stride=1, bias=False)
        nn.init.xavier_uniform_(self.V_P.weight)
        
        self.Q_P = Conv2dSame(self.in_channels, self.in_channels, kernel_size=1, stride=1, bias=False)
        nn.init.xavier_uniform_(self.Q_P.weight)
        
        self.ff1_kernel = Conv2dSame(self.in_channels, self.in_channels, kernel_size=3, stride=1, bias=True)
        nn.init.xavier_uniform_(self.ff1_kernel.weight)
        
        self.ff2_kernel = Conv2dSame(self.in_channels, 2*self.in_channels, kernel_size=3, stride=1, bias=True)
        nn.init.xavier_uniform_(self.ff2_kernel.weight)
        
        self.ff3_kernel = Conv2dSame(2*self.in_channels, self.in_channels, kernel_size=3, stride=1, bias=True)
        nn.init.xavier_uniform_(self.ff3_kernel.weight)
        
        
    def forward(self, x):
        h_half = self.kernel_range[0]//2
        w_half = self.kernel_range[1]//2
        B, C, H, W = x.shape
        
        x_k = self.K_P(x)
        x_v = self.V_P(x)
        x_q = self.Q_P(x)
        
        # pad B,C,H,W in order W,H,C,B
        paddings = (w_half*self.shift,w_half*self.shift, h_half*self.shift,h_half*self.shift, 0,0, 0,0)
        x_k = nn.functional.pad(x_k, paddings, mode="constant")
        x_v = nn.functional.pad(x_v, paddings, mode="constant")
        mask_x = torch.ones(B,1,H,W)
        mask_pad = nn.functional.pad(mask_x, paddings, mode="constant")

        k_ls = list()
        v_ls = list()
        masks = list()
        
        c_x, c_y = h_half*self.shift, w_half*self.shift
        layer=0
        for i in range(-h_half, h_half+1):
            for j in range(-w_half, w_half+1):
                k_t = x_k[:, layer*C:(layer+1)*C, c_x+i*self.shift:c_x+i*self.shift+H, c_y+j*self.shift:c_y+j*self.shift+W]
                k_ls.append(k_t)
                
                v_t = x_v[:, layer*C:(layer+1)*C, c_x+i*self.shift:c_x+i*self.shift+H, c_y+j*self.shift:c_y+j*self.shift+W]
                v_ls.append(v_t)
                
                _m = mask_pad[:, :, c_x+i*self.shift:c_x+i*self.shift+H, c_y+j*self.shift:c_y+j*self.shift+W]
                masks.append(_m)
                
                layer+=1
  
        m_stack = torch.hstack(masks)
        m_vec = m_stack.view(B*H*W, self.kernel_range[0]*self.kernel_range[1], 1)
        
        k_stack = torch.hstack(k_ls)
        v_stack = torch.hstack(v_ls)
        
        k = k_stack.view(B*H*W, self.kernel_range[0]*self.kernel_range[1], C)
        v = v_stack.view(B*H*W, self.kernel_range[0]*self.kernel_range[1], C)
        q = x_q.view(B*H*W, 1, C)
        
        alpha = torch.softmax(torch.matmul(k, q.transpose(1,2))*m_vec/8, axis=1) #s[0]*s[1]*s[2]*9
        __res = torch.matmul(alpha.transpose(1,2), v)
        _res = __res.view(B, C, H, W)
        
        if self.useRes:
            t = x +_res
        else:
            t = _res
        if self.useBN:
            t = nn.BatchNorm2d(t.shape(-1))(t)

        _t = t
        t = self.ff1_kernel(t)
        t = self.ff2_kernel(t)
        t = self.ff3_kernel(t)
        
        if self.useRes:
            t = _t + t
        if self.useBN:
            res = nn.BatchNorm2d(t.shape(-1))(t)
        else:
            res = t
        
        return res
    
p = PixelAttention(256)
p(torch.randn(1,256,256,256))