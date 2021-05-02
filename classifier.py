import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from timm.models import layers
from timm.models.layers import SelectAdaptivePool2d
from timm.models.layers.se import EffectiveSEModule, SEModule
from timm.models.layers.separable_conv import SeparableConvBnAct
from timm.models.layers import activations
from torch.nn.functional import pad


class ClassifierConv(nn.Module):
    def __init__(self, in_channels):
        super(ClassifierConv, self).__init__()

        self.sconv0 = SeparableConvBnAct(in_channels, in_channels//4, act_layer=nn.ReLU)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.dropout = nn.Dropout2d(p=0.4)

        self.sconv1 = SeparableConvBnAct(in_channels//4, in_channels//6, act_layer=nn.ReLU)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1)

        self.se0 = SEModule((in_channels // 6))
        
        self.sconv2 = SeparableConvBnAct(in_channels//6, in_channels//8, act_layer=nn.ReLU)
        # self.pool2 = nn.MaxPool2d(kernel_size=3, stride=1)
        
        self.sconv3 = SeparableConvBnAct(in_channels//8, in_channels//8, act_layer=nn.ReLU)
        self.sconv4 = SeparableConvBnAct(in_channels//8, in_channels//16, act_layer=nn.ReLU)
        self.se1 = SEModule((in_channels // 16))
        self.sconv5 = SeparableConvBnAct(in_channels//16, in_channels//32, act_layer=nn.ReLU)

        self.global_pool = SelectAdaptivePool2d(pool_type="avg")
        
        self.relu = nn.ReLU(inplace=True)
        self.linear = nn.Linear((in_channels // 32), 1)

    def forward(self, x):
        x = self.sconv0(x)
        x = self.pool0(x)
        x = self.dropout(x)

        x = self.sconv1(x)
        x = self.pool1(x)
        x = self.dropout(x)

        x = self.se0(x)
        
        x = self.sconv2(x)
        # x = self.pool2(x)
        x = self.dropout(x)
        
        x = self.sconv3(x)
        x = self.dropout(x)
        
        x = self.sconv4(x)
        x = self.se1(x)
        x = self.sconv5(x)

        x = self.global_pool(x)

        x = x.flatten(1)
        x = self.linear(x)

        return x


class Classifier3(nn.Module):
    def __init__(self, in_channels):
        super(Classifier3, self).__init__()

        self.conv0 = nn.Conv2d(in_channels, in_channels//6, kernel_size=3, stride=2)
        self.bn0 = nn.BatchNorm2d(in_channels//6)
        # self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv1 = nn.Conv2d(in_channels//6, in_channels//6, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels//6)

        self.conv2 = nn.Conv2d(in_channels//6, in_channels//8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels//8)

        # self.conv3 = nn.Conv2d(in_channels//4, in_channels//2, kernel_size=1)
        # self.bn3 = nn.BatchNorm2d(in_channels//2)
        
        self.global_pool = SelectAdaptivePool2d(pool_type="avg")
        
        self.relu = nn.ReLU(inplace=True)
        self.linear0 = nn.Linear(in_channels//8, 1792)
        # self.linear1 = nn.Linear(1792, 786)
        self.linear_final = nn.Linear(1792, 1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)
        # x = self.pool1(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.relu(x)

        x = self.global_pool(x)

        x = x.flatten(1)
        x = self.linear0(x)
        x = self.relu(x)
        # x = self.linear1(x)
        # x = self.relu(x)
        x = self.linear_final(x)

        return x
    
class Classifier_GAP(nn.Module):
    def __init__(self, in_channels):
        super(Classifier_GAP, self).__init__()

        self.pool = nn.AdaptiveAvgPool1d(1)     
        ### Fully Connected Multi-Layer Perceptron (FC-MLP)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 16, in_channels, bias=True),
            nn.ReLU(inplace=True)
        )
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.xavier_uniform_(self.mlp[2].weight)

        self.fc = nn.Linear(in_channels, 1)
        nn.init.xavier_uniform_(self.fc.weight)
        
    def forward(self, x): # x -> (B,21,1792)
        # y = torch.mean(x,dim=1) # y -> (B,1792)
        y = torch.sigmoid(x)
        y = self.pool(x.permute(0,2,1)).squeeze() # y -> (B,1792)
        # print(y.shape)
        y = self.mlp(y) # y -> (B,1792)
        # print(y.shape)
        x =  x * y.unsqueeze(1) # x -> (B,21,1792)
        
        x = F.dropout(x, p=0.3)
        # x = torch.mean(x,dim=1) # x -> (B,1,1792)
        x = self.pool(x.permute(0,2,1)).squeeze()
        
        x = self.fc(x)
        
        return x


class Classifier_Linear(nn.Module):
    def __init__(self):
        super(Classifier_Linear, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(5376, 2688, bias=True),
            nn.BatchNorm1d(2688),
            activations.Swish(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(2688, 1792, bias=True),
            nn.BatchNorm1d(1792),
            activations.Swish(inplace=True),
            nn.Linear(1792, 1024, bias=True),
            nn.BatchNorm1d(1024),
            activations.Swish(inplace=True),
        )
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.xavier_uniform_(self.mlp[4].weight)
        nn.init.xavier_uniform_(self.mlp[7].weight)

        self.fc = nn.Linear(1024, 1)
        nn.init.xavier_uniform_(self.fc.weight)
        
    def forward(self, x):
        x = self.mlp(x)        
        x = self.fc(x)
        
        return x
