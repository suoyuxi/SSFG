import os
import cv2
import sys
import copy
import logging
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F

from datetime import datetime
from resnet import pretrain_model
from dataset import predst
from cfg import backbone_cfg, neck_cfg

def vis(img, x):
    '''
    '''
    import cv2
    import numpy as np
    from datetime import datetime

    files = os.listdir("./map/map/")
    cnt = len(files)

    img = img.squeeze().detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    
    feature_map = F.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=False)
    feature_map = feature_map.cpu().squeeze(0).detach().numpy()
    feature_map = np.sum(feature_map, axis=0)
    feature_map = ((feature_map - feature_map.min()) / (feature_map.max() - feature_map.min()) * 255).astype(np.uint8)
        
    # 将特征图转换为彩色图
    feature_map_colored = cv2.applyColorMap(feature_map, cv2.COLORMAP_JET)
    # 叠加特征图，使用 cv2.addWeighted 控制透明度
    overlay = cv2.addWeighted(img, 0.5, feature_map_colored, 0.8, 0)
    # 获取当前时间
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    # 保存图像
    cv2.imwrite(f"./map/map/{str(cnt)}.png", overlay)

def inference(net, device):

    #set up testset
    testset = predst(sl_dir='/media/syx/新加卷/MyResearch/VOC-SL/test/SLCMats/',fsi_dir='/media/syx/新加卷/MyResearch/VOC-FSI/test/SLCMats/')
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    
    for sample, gt, img in test_loader:

        # load data to GPU or CPU
        sample = sample.to(device=device, dtype=torch.float32)
        # forward inference
        gau_map = net(sample)

        vis(img, gau_map)

if __name__ == '__main__':
    
    # # make save dir
    # os.makedirs('/workspace/mmrotate/pretrain/checkpoints/20241210/results/', exist_ok=True)
    # lock GPU or CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #load the net
    net = pretrain_model(backbone_cfg)
    net.load_state_dict( torch.load('/media/syx/新加卷/MyResearch/mmrotate/pretrain/checkpoints/MCAStdAveQuan/model/model.pth', map_location=torch.device('cpu')) )
    net.to(device=device)

    # inference
    inference(net, device)
