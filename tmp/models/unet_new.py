from blocks import *

import torch
import torch.nn as nn
import torch.nn.functional as F


BN_EPS = 1e-4  #1e-4  #1e-5


# 1024x1024
class UNet1024 (nn.Module):
    def __init__(self):
        super(UNet1024, self).__init__()
        C = 3

        #1024
        
        self.down1 = StackEncoder(  C,   24, kernel_size=3)   #512
        self.down2 = StackEncoder( 24,   64, kernel_size=3)   #256
        self.down3 = StackEncoder( 64,  128, kernel_size=3)   #128
        self.down4 = StackEncoder(128,  256, kernel_size=3)   # 64
        self.down5 = StackEncoder(256,  512, kernel_size=3)   # 32
        self.down6 = StackEncoder(512,  768, kernel_size=3)   # 16
        
        self.center = nn.Sequential(
            ConvBnRelu2d(768, 768, kernel_size=3, padding=1, stride=1 )
        )
        # 8
        # x_big_channels, x_channels, y_channels
           
        self.up6 = StackDecoder(768,  768, 512, kernel_size=3)  # 16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3)  # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3)  # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3)  #128
        self.up2 = StackDecoder(  64,  64,  24, kernel_size=3)  #256
        self.up1 = StackDecoder(  24,  24,  24, kernel_size=3)  #512
       
      
        self.classify = nn.Conv2d(24, 1, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, x):

        out = x                      
                                      
        down1,out = self.down1(out) 
        down2,out = self.down2(out)  
        down3,out = self.down3(out)  
        down4,out = self.down4(out) 
        down5,out = self.down5(out) 
        down6,out = self.down6(out)
                  
        out = self.center(out)
        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)
        out = self.up1(down1, out)
        #1024

        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        return out
