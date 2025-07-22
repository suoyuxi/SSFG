import warnings

import e2cnn.nn as enn
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm
import torch

from mmdet.models.builder import MODELS
from ..utils import (build_enn_divide_feature, build_enn_norm_layer,
                     build_enn_trivial_feature, ennAvgPool, ennConv,
                     ennMaxPool, ennReLU, ennTrivialConv)

ROTATED_PYRAMIDS = MODELS

class CatConv(nn.Module):
    '''
    1*1.conv+norm+relu
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cat_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.cat_conv(x)

class DoubleConv(nn.Module):
    '''
    (conv+norm+relu)*2
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2,x1], dim=1)
        return self.conv(x)

@ROTATED_PYRAMIDS.register_module()
class PyramidNet(nn.Module):
    """
    cat the feature maps from the resnet backbone and the pyramid backbone
    """
    def __init__(self, in_channels=1, out_channels=256):
        super(PyramidNet, self).__init__()
        
        self.prec0 = DoubleConv(1, 16)
        self.down0 = Down(16, 64)

        self.prec1 = DoubleConv(1, 16)
        self.down1 = Down(80, 64)

        self.prec2 = DoubleConv(1, 16)
        self.down2 = Down(80, 64)

        self.down3 = Down(64, 64)
        self.down4 = Down(64, 64)
        
        self.up3 = Up(128, 64)
        self.up2 = Up(128, 64)
        self.up1 = Up(128, 64)
        self.up0 = Up(128, 64)

        self.cat_conv0 = CatConv(320, 256)
        self.cat_conv1 = CatConv(320, 256)
        self.cat_conv2 = CatConv(320, 256)
        self.cat_conv3 = CatConv(320, 256)
        self.cat_conv4 = CatConv(320, 256)

    def forward(self, 
                x,
                subaperture2x,
                subaperture4x,
                subaperture8x):
        print(subaperture2x[0,:,:].shape)
        print(subaperture4x.shape)
        print(subaperture8x.shape)
        import cv2
        import numpy as np
        s2x = subaperture2x.cpu().detach().numpy()
        s2x = (s2x[0,:,:] + 1.0) * 255.0
        cv2.imwrite('/workspace/syx/models/TEST/s2x.png', s2x.astype(np.uint8))
        s4x = subaperture4x.cpu().detach().numpy()
        s4x = (s4x[0,:,:] + 1.0) * 255.0
        cv2.imwrite('/workspace/syx/models/TEST/s4x.png', s4x.astype(np.uint8))
        s8x = subaperture8x.cpu().detach().numpy()
        s8x = (s8x[0,:,:] + 1.0) * 255.0
        cv2.imwrite('/workspace/syx/models/TEST/s8x.png', s8x.astype(np.uint8))

        p0_1 = self.prec0(subaperture2x.unsqueeze(dim=1)) # b 16 512*512
        p0_2 = self.down0(p0_1) # b 64 256*256

        p1_1 = self.prec1(subaperture4x.unsqueeze(dim=1)) # b 16 256*256
        p1_2 = self.down1( torch.cat([p0_2, p1_1], dim=1) ) # b 64 128*128

        p2_1 = self.prec2(subaperture8x.unsqueeze(dim=1)) # b 16 128*128
        p2_2 = self.down2( torch.cat([p1_2, p2_1], dim=1) ) # b 64 64*64

        p3_2 = self.down3( p2_2 ) # b 64 32*32
        p4_3 = self.down4( p3_2 ) # b 64 16*16

        p3_3 = self.up3( p4_3,p3_2 ) # 64 32*32
        p2_3 = self.up2( p3_3,p2_2 ) # 64 64*64
        p1_3 = self.up1( p2_3,p1_2  ) # 64 128*128
        p0_3 = self.up0( p1_3,p0_2 ) # 64 256*256

        outs = []
        outs.append( self.cat_conv0( torch.cat([p0_3, x[0]], dim=1 ) ) )
        outs.append( self.cat_conv1( torch.cat([p1_3, x[1]], dim=1 ) ) )
        outs.append( self.cat_conv2( torch.cat([p2_3, x[2]], dim=1 ) ) )
        outs.append( self.cat_conv3( torch.cat([p3_3, x[3]], dim=1 ) ) )
        outs.append( self.cat_conv4( torch.cat([p4_3, x[4]], dim=1 ) ) )

        return tuple(outs)
