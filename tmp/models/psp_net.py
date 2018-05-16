import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from math import sqrt
from torch.utils import model_zoo
from torchvision import models
import math

class PyramidPoolingModule(nn.Module):
    def __init__(self, in_size, in_dim, reduction_dim, setting):
        super(PyramidPoolingModule, self).__init__()
        self.features = []
        for s in setting:
            pool_size = (int(math.ceil(float(in_size[0]) / s)), int(math.ceil(float(in_size[1]) / s)))
            self.features.append(nn.Sequential(
                nn.AvgPool2d(kernel_size=pool_size, stride=pool_size, ceil_mode=True),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim, momentum=.95),
                nn.ReLU(),
                nn.Upsample(size=in_size, mode='bilinear')
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        out = [x]
        for f in self.features:
            out.append(f(x))
        out = torch.cat(out, 1)
        return out


class PSP_net(nn.Module):
    def __init__(self, num_classes, input_size, use_aux=False):
        super(PSP_net, self).__init__()
        self.use_aux = use_aux
        self.input_size = input_size
        self.num_classes = num_classes
        resnet = models.resnet101(pretrained=True)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation = (2, 2)
                m.padding = (2, 2)
                m.stride = (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation = (4, 4)
                m.padding = (4, 4)
                m.stride = (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        self.ppm = PyramidPoolingModule((int(math.ceil(input_size[0] / 8.0)), int(math.ceil(input_size[1] / 8.0))),
                                        2048, 512, (1, 2, 3, 6))
        self.final = nn.Sequential(
            nn.Conv2d(4096, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=.95),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )
        if use_aux:
            self.aux_logits = nn.Sequential(
                PyramidPoolingModule((int(math.ceil(input_size[0] / 8.0)), int(math.ceil(input_size[1] / 8.0))),
                                     1024, 256, (1, 2, 3, 6)),
                nn.Conv2d(2048, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256, momentum=.95),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Conv2d(256, num_classes, kernel_size=1)
            )
    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.training and self.use_aux:
            aux = self.aux_logits(x)
        x = self.layer4(x)
        x = self.ppm(x)
        x = self.final(x)
        if self.training and self.use_aux:
            return F.upsample(x, self.input_size, mode='bilinear'), F.upsample(aux, self.input_size, mode='bilinear')
        return F.upsample(x, self.input_size, mode='bilinear')

