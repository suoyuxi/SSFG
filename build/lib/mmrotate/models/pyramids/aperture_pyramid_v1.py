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

class PyramidBackbone(nn.Module):
    """
    backbone of the aperture pyramid
    """

    def __init__(self,
                 in_channels=1,
                 out_channels=256):
        super(PyramidBackbone, self).__init__()
        
        self.Pre = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,1)
        )
        self.DoubleConv1 = DoubleConv(out_channels//4, out_channels//2)
        self.DoubleConv2 = DoubleConv(out_channels//2, out_channels)

    def forward(self, x):

        x = self.Pre(x)
        x = self.DoubleConv1(x)
        x = self.DoubleConv2(x)

        return x

@ROTATED_PYRAMIDS.register_module()
class PyramidNet(nn.Module):
    """
    cat the feature maps from the resnet backbone and the pyramid backbone
    """
    def __init__(self, in_channels=1, out_channels=256):
        super(PyramidNet, self).__init__()
        
        self.pyramid_conv1 = PyramidBackbone(in_channels, out_channels)
        self.cat_conv1 = CatConv(out_channels*2, out_channels)

        self.pyramid_conv2 = PyramidBackbone(in_channels, out_channels)
        self.cat_conv2 = CatConv(out_channels*2, out_channels)

        self.pyramid_conv3 = PyramidBackbone(in_channels, out_channels)
        self.cat_conv3 = CatConv(out_channels*2, out_channels)

    def forward(self, 
                x,
                subaperture2x,
                subaperture4x,
                subaperture8x):

        outs = []
        outs.append( x[0] )

        p1 = self.pyramid_conv1(subaperture2x.unsqueeze(dim=1))
        feature1 = torch.cat([p1, x[1]], dim=1)
        outs.append( self.cat_conv1(feature1) )

        p2 = self.pyramid_conv2(subaperture4x.unsqueeze(dim=1))
        feature2 = torch.cat([p2, x[2]], dim=1)
        outs.append( self.cat_conv2(feature2) )

        p3 = self.pyramid_conv3(subaperture8x.unsqueeze(dim=1))
        feature3 = torch.cat([p3, x[3]], dim=1)
        outs.append( self.cat_conv3(feature3) )

        outs.append( x[4] )

        return tuple(outs)
