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

ROTATED_ANALYSES = MODELS

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

@ROTATED_ANALYSES.register_module()
class STFTNet(nn.Module):
    """
    cat the feature maps from the resnet backbone and the pyramid backbone
    """
    def __init__(self, in_channels=5, out_channels=256):
        super(STFTNet, self).__init__()
        
        self.prec = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.down0 = Down(64, 64)
        self.down1 = Down(64, 64)
        self.down2 = Down(64, 64)
        self.down3 = Down(64, 64)
        self.down4 = Down(64, 64)
        
        self.up3 = Up(128, 64)
        self.up2 = Up(128, 64)
        self.up1 = Up(128, 64)
        self.up0 = Up(128, 64)

        self.cat_conv0 = CatConv(320, out_channels)
        self.cat_conv1 = CatConv(320, out_channels)
        self.cat_conv2 = CatConv(320, out_channels)
        self.cat_conv3 = CatConv(320, out_channels)
        self.cat_conv4 = CatConv(320, out_channels)

    def forward(self, features, img_stft):

        stft_map = self.prec(img_stft) # b 16(x)64 512*512

        map0 = self.down0(stft_map) # b 64 256*256
        map1 = self.down1( map0 ) # b 64 128*128
        map2 = self.down2( map1 ) # b 64 64*64
        map3 = self.down3( map2 ) # b 64 32*32
        map4 = self.down4( map3 ) # b 64 16*16

        map3 = self.up3( map4,map3 ) # 64 32*32
        map2 = self.up2( map3,map2 ) # 64 64*64
        map1 = self.up1( map2,map1 ) # 64 128*128
        map0 = self.up0( map1,map0 ) # 64 256*256
        print('map0',map0.shape)
        # s0=map0.shape
        # s1=map1.shape
        # s2=map2.shape
        # s3=map3.shape
        # shape_str = f"shape of tensors: {s0},{s1},{s2},{s3}"
        # with open('tensor_shape_res.txt','a') as file:
        #     file.write(shape_str)
        # exit()

        outs = []
        outs.append( self.cat_conv0( torch.cat([map0, features[0]], dim=1 ) ) )
        outs.append( self.cat_conv1( torch.cat([map1, features[1]], dim=1 ) ) )
        outs.append( self.cat_conv2( torch.cat([map2, features[2]], dim=1 ) ) )
        outs.append( self.cat_conv3( torch.cat([map3, features[3]], dim=1 ) ) )
        outs.append( self.cat_conv4( torch.cat([map4, features[4]], dim=1 ) ) )

        return tuple(outs)
