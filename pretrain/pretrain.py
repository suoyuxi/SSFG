import os
import sys
import logging
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from resnet import pretrain_model
from dataset import predst
from cfg import backbone_cfg, neck_cfg

def train_net(net, device, args):

    #set up trainset
    trainset = predst()
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    # optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9,0.99)) # Adam 
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9) # SGD 
    scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1) 
    
    # logging
    logger.info('Starting training:\
        Start Epoch:     {}\
        Epochs:          {}\
        Batch size:      {}\
        Learning rate:   {}\
        Device:          {}'.format(args.epoch_start,args.epochs,args.batch_size,args.lr,device.type))

    for epoch in range(args.epoch_start, args.epoch_start+args.epochs):
        # trianing each epoch
        epoch_loss = 0
        with tqdm(total=len(trainset), desc='Epoch {}/{}'.format(epoch+1,args.epochs+args.epoch_start)) as pbar:
            
            for sample, gt in train_loader:
                # load data to GPU or CPU
                sample = sample.to(device=device, dtype=torch.float32)
                gt = gt.to(device=device, dtype=torch.float32)
                # forward inference
                loss = net.forward_train(sample, gt)
                epoch_loss += loss.item()
                # show the loss and process
                pbar.set_postfix({'real-time loss': loss.item()})
                # bp
                optimizer.zero_grad()
                loss.backward()
                
                optimizer.step()
                # update par
                pbar.update(args.batch_size)

                # break
                
        scheduler.step()
        logger.info('epoch : {} learning rate : {} total loss : {}'.format(epoch + 1, scheduler.get_lr()[0], epoch_loss))
        torch.save(net.backbone.state_dict(), os.path.join(args.checkpoints, 'backbone','resnet_{}.pth'.format(epoch + 1)))
        torch.save(net.state_dict(), os.path.join(args.checkpoints, 'model','model_{}.pth'.format(epoch + 1)))

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--checkpoints', dest='checkpoints', type=str, default='./checkpoints/20250105', help='save model in a file folder')
    parser.add_argument('-s', '--epoch_start', dest='epoch_start', type=int, default=0, help='start of epoch')
    parser.add_argument('-e', '--epochs', dest='epochs', type=int, default=12, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('-l', '--learning_rate', dest='lr', type=float, default=1.0, help='Learning rate')

    return parser.parse_args()


if __name__ == '__main__':
    # parse the args
    args = get_args()
    # make checkpoint dir
    os.makedirs(os.path.join(args.checkpoints, 'backbone'), exist_ok=True)
    os.makedirs(os.path.join(args.checkpoints, 'model'), exist_ok=True)
    
    # set up log
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger()
    # log out
    txt_handler = logging.FileHandler(os.path.join(args.checkpoints, 'log.txt'), mode='a')
    txt_handler.setLevel(logging.INFO)
    logger.addHandler(txt_handler)
    # lock GPU or CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Using device {}'.format(device))

    #load the net
    net = pretrain_model(backbone_cfg)
    logger.info('Initialize ResNet-50')
    net.to(device=device)

    # train net
    train_net(net, device, args)
