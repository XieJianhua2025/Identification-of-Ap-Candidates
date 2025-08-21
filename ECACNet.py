# encoding:utf8

# Author: Jianhua Xie
# Date: 2025-08-21

import torch
import torch.nn as nn
import torch.nn.functional as F

# Efficient Channel Attention (ECA) Module
class ECA_Module(nn.Module):
    def __init__(self, channels, k_size=5):
        super(ECA_Module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)  
        y = self.conv(y.squeeze(-1).transpose(-1, -2))  
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  
        return x * y.expand_as(x)

# Basic CNN Block
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, use_eca=False):
        super(CNNBlock, self).__init__()
        stride = (1, 2) if downsample else (1, 1)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), stride=stride, padding=(0, 2))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 5), padding=(0, 2))
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

        self.eca = ECA_Module(out_channels) if use_eca else nn.Identity()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.eca(x)
        return self.relu2(x)

# CNN Model with ECA in the 5th layer
class ECACNet(nn.Module):
    def __init__(self, num_classes=7):
        super(ECACNet, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 5), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))  
        )

        self.layer1 = CNNBlock(16, 32, downsample=True)
        self.layer2 = CNNBlock(32, 64, downsample=True)
        self.layer3 = CNNBlock(64, 128, downsample=True)
        self.layer4 = CNNBlock(128, 256, downsample=True)
        self.layer5 = CNNBlock(256, 256, downsample=True, use_eca=True)  # Add ECA here

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.initial(x)   
        x = self.layer1(x)    
        x = self.layer2(x)    
        x = self.layer3(x)    
        x = self.layer4(x)    
        x = self.layer5(x)    
        x = self.global_pool(x)  
        x = self.fc(x)        
        return x
