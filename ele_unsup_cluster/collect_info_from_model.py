#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# import sim_model.loader
# import sim_model.model_stage1
from simsiam.sim_model.dataloader_stage1 import dataloader_ele_my
from simsiam.sim_model.dataloader_stage1_vit import dataloader_ele_DINO
from simsiam.sim_model.model_stage1 import SimSiam
from simsiam.sim_model.model_stage_1_vit import vit_simsiam

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('--DIR', metavar='DIR', default=r'D:\1111\target_stage1', type=str,
parser.add_argument('--DIR', metavar='DIR', default=r'/root/autodl-tmp/data/target_stage1_small', type=str,
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
###############################################################################
# 源：32个
parser.add_argument('--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
###############################################################################
# 源：512
parser.add_argument('-b', '--batch_size', default=240, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.95, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=0.001, type=float,
# parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=5, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
# parser.add_argument('--resume', default='/root/autodl-tmp/dino_ele/simsiam/checkpoint_res50_240_0094.pth.tar', type=str, metavar='PATH',
# parser.add_argument('--resume', default='/root/autodl-tmp/dino_ele/simsiam/checkpoint_vit_240_0040.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--gpu', default=True, type=bool, help='GPU id to use.')


# simsiam specific configs:
parser.add_argument('--out_dim', default=20, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--in_dim', default=2, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--pred_dim', default=15, type=int,
                    help='hidden dimension of the predictor (default: 512)')
parser.add_argument('--fix-pred-lr', action='store_true',
                    help='Fix learning rate for the predictor')


# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def main():
    args = parser.parse_args()

    # print model name
    # print("=> creating model '{}'".format(args.arch))
    model = SimSiam(args.in_dim, args.out_dim, args.pred_dim)
    # model = vit_simsiam()

    init_lr = args.lr

    print('cuda devices count is:{}'.format(torch.cuda.device_count()))
    print('args gpu setting is :{}'.format(args.gpu))


    if args.fix_pred_lr:
        optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
                        {'params': model.module.predictor.parameters(), 'fix_lr': True}]
    else:
        optim_params = model.parameters()


    optimizer = torch.optim.SGD(optim_params, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)


    if torch.cuda.is_available() and args.gpu:
        if torch.cuda.device_count() == 1:
            DEVICE = torch.device("cuda:" + str(torch.cuda.current_device()))
            print('single GPU processing:{}'.format(DEVICE))
            model = model.to(DEVICE)
        elif torch.cuda.device_count() > 1:
            model.cuda()
            device_ids = range(torch.cuda.device_count())
            print('multiple GPU processing:{}'.format(device_ids))
            # model = torch.nn.DataParallel(model)  # 前提是model已经.cuda()
            model = torch.nn.DataParallel(model)

            # if args.resume:
            if os.path.isfile(args.resume):
                # print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                args.start_epoch = checkpoint['epoch']
                print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

                # model.predictor = nn.Sequential(nn.Linear(2048, 2048, bias=False),
                #                                nn.BatchNorm1d(2048),
                #                                nn.ReLU(inplace=True),  # hidden layer
                #                                nn.Linear(2048, 2048))  # output layer
                #
                # model = torch.nn.DataParallel(model)
    else:
        DEVICE = torch.device("cpu")
        print('CPU processing:{}'.format(DEVICE))
        model = model.to(DEVICE)
        args.batch_size = 2
        args.workers = 1

    print('init_learing_rate:{}, epohs:{}, batch_size:{}, start_epoch:{}, workers:{}'.format(init_lr, args.epochs, args.batch_size, args.start_epoch, args.workers))

    ###########################################################################################################
    # print(model) # print model after SyncBatchNorm
    criterion = nn.CosineSimilarity(dim=1)


    # # optionally resume from a checkpoint
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(args.resume))
    #         if args.gpu and torch.cuda.is_available():
    #             checkpoint = torch.load(args.resume)
    #         else:
    #             # Map model to be loaded to specified single gpu.
    #             loc = 'cuda:{}'.format(args.gpu)
    #             checkpoint = torch.load(args.resume, map_location=loc)
    #         args.start_epoch = checkpoint['epoch']
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         print("=> loaded checkpoint '{}' (epoch {})"
    #               .format(args.resume, checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))
    # else:
    #     print('start train a new model............')



    # torch.backends.cudnn.benchmark=True 将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，
    # 进而实现网络的加速。适用场景是网络结构固定（不是动态变化的），网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的，其实也就是一般情况下都比较适用。
    cudnn.benchmark = True

    # dist.init_process_grop('nccl', init_method='file:///myfile', work_size=1, rank=0)
    # train_dataset_2 = dataloader_ele_simsiam(path=args.DIR)

    # torch.cuda.set_device(0)
    # dist.barrier()

    train_dataset_2 = dataloader_ele_my(path=args.DIR)
    # train_dataset_2 = dataloader_ele_DINO(path=args.DIR)
    print('train data num is :{}'.format(train_dataset_2.__len__()))
    # sampler = torch.utils.data.DistributedSampler(train_dataset_2, shuffle=True)
    train_loader_2 = torch.utils.data.DataLoader(
        train_dataset_2, batch_size=args.batch_size,
        # sampler=sampler,
        shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)


    model.eval()
    feature_all = []

    for batch, (imgL, imgR, labels) in enumerate(train_loader_2):

        # print(imgL.shape)
        imgL = torch.reshape(imgL, (args.batch_size, 2, 224, 224)).float()
        imgR = torch.reshape(imgR, (args.batch_size, 2, 224, 224)).float()

        if torch.cuda.is_available() and args.gpu:
            imgL = imgL.cuda()
            imgR = imgR.cuda()

        # compute output and loss
        data_dict = model(x1=imgL, x2=imgR)
        z2 = data_dict['z2']

        z2.cpu().detach().numpy()

        if len(feature_all)==0:
            feature_all = z2.cpu().detach().numpy()
        else:
            feature_all = np.append(feature_all, z2, axis=1)



if __name__ == '__main__':
    # 对模型进行无监督训练
    main()
