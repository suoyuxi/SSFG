# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from cfg import backbone_cfg, neck_cfg
from mmdet.models.builder import MODELS

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels , in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.up(x)

        return torch.sigmoid(x)

class merge(nn.Module):

    def __init__(self):
        super(merge, self).__init__()

        self.up2 = Up(2048, 1024)
        self.up1 = Up(1024, 512)
        self.up0 = Up(512, 256)

        self.out_conv = OutConv(256, 1)

    def forward(self, x):

        out2 = self.up2(x[3], x[2])
        out1 = self.up1(out2, x[1])
        out0 = self.up0(out1, x[0]) # b*256*256*256

        out = self.out_conv(out0)

        return out

class MSELoss(nn.Module):

    def __init__(self):
        super(MSELoss, self).__init__()
        self.author = 'Yuxi_Suo'

    def forward(self, x, gt):
        return torch.mean( (x-gt)**2 * (0.01+gt**0.5) )

class pretrain_model(nn.Module):

    def __init__(self, backbone_cfg):
        super(pretrain_model, self).__init__()
        self.backbone = MODELS.build(backbone_cfg)
        self.merge = merge()
        
        self.criterion = MSELoss()

    def forward_train(self, x, gt):
        x = self.backbone(x)
        x = self.merge(x)

        return self.criterion(x, gt)

    def forward(self, x):
        x = self.backbone(x)
        x = self.merge(x)

        return x

if __name__ == '__main__':

    pm = pretrain_model(backbone_cfg)

    x = torch.randn([2,3,1024,1024])
    x = pm(x)
    print( x.size() )
    
