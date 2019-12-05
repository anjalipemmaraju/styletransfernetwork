import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import copy
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
from generator import ResidualBlock
from generator import ConvLayer

class SuperResolution(nn.Module):
    def __init__(self):
        super(SuperResolution, self).__init__()
        self.refpad4 = nn.ReflectionPad2d(4)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1)
        self.b1 = nn.BatchNorm2d(64, affine=True)
        
        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        self.res3 = ResidualBlock(64)
        self.res4 = ResidualBlock(64)
        self.conv2 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(64, affine=True)
        self.conv3 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(64, affine=True)
        self.conv4 = nn.Conv2d(64, 3, kernel_size=9, stride=1)
    
    def forward(self, x):
        x = self.refpad4(x)
        x = self.conv1(x)
        #print(x.shape)
        x = self.b1(x)

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)

        #print(x.shape)

        x = self.conv2(x)
        x = self.bn2(x)
        #print(x.shape)

        x = self.conv3(x)
        x = self.bn3(x)
        #print(x.shape)
        x = self.refpad4(x)
        x = self.conv4(x)
        #print(x.shape)
        return x
