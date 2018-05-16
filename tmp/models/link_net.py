import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from math import sqrt
from torch.utils import model_zoo
from torchvision import models
import math

class DecoderBlock(nn.Module):
    def __init__(self, m, n):
        self.conv1 = nn.Conv2d(m, m/4, 1)
        self.upconv2 = nn.ConvTranspose2d(m/4, m/4, 3, 2)
        self.conv2 = nn.Conv2d(m/4, n, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.upconv2(x)
        x = self.conv2(x)
        return x
##TODO: implementation
class LinkNet(nn.Module):
    def __init__(self, num_classes):
        resnet = models.resnet50(pretrained=True)
        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu, resnet.maxpool)
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder1 = DecoderBlock(64, 64)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder4 = DecoderBlock(512, 256)

        self.full_conv1 = nn.ConvTranspose2d(64, 32, 3, 2)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.full_conv2 = nn.ConvTranspose2d(32, num_classes, 2, 2)

    def forward(self, x):
        pass