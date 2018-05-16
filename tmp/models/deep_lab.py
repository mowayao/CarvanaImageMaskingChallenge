import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from math import sqrt
from torch.utils import model_zoo
from torchvision import models
import math
import numpy as np


class ASPP(nn.Module):
    def __init__(self):
        pass
    def forward(self, x):
        pass

class DeepLab3(nn.Module):
    def __init__(self):
        super(DeepLab3, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.conv_pool = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.block1 = resnet.layer1
        self.block2 = resnet.layer2
        self.block3 = resnet.layer3
        self.block4 = resnet.layer4


    def forward(self, x):
        x = self.conv_pool(x)
        print x.size()
        x = self.block1(x)
        print x.size()
        x = self.block2(x)
        print x.size()
        x = self.block3(x)
        print x.size()
        x = self.block4(x)
        print x.size()
        return x

model = DeepLab3()
print model
test_input = torch.FloatTensor(np.ones(shape=(15, 3, 256, 256),dtype=float))
model.cuda()
from torch.autograd import Variable
test_input = Variable(test_input).cuda()
model(test_input)