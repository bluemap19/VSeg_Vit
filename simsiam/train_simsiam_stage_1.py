#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import math
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from simsiam.sim_model.dataloader_stage1 import dataloader_ele_my
from simsiam.sim_model.model_stage1 import SimSiam
from simsiam.sim_model.model_stage_1_vit import vit_simsiam

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('--DIR', metavar='DIR', default=r'D:\1111\target_stage1', type=str,
parser.add_argument('--DIR', metavar='DIR', default=r'/root/autodl-tmp/data/target_stage1_small_big_mix', type=str,
                    help='path to dataset')
# parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
parser.add_argument('-a', '--arch', metavar='ARCH', default='vit',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')

###############################################################################
# 源：32个
parser.add_argument('--workers', default=10, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
###############################################################################

# 源：512
parser.add_argument('-b', '--batch_size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.95, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=0.0001, type=float,
# parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,`
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
parser.add_argument('--in_dim', default=2, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--out_dim', default=512, type=int,
# parser.add_argument('--out_dim', default=20, type=int,
                    help='feature dimension (default: 2048)')
# small：20
parser.add_argument('--pred_dim', default=384, type=int,
# parser.add_argument('--pred_dim', default=15, type=int,
                    help='hidden dimension of the predictor (default: 512)')
# small：15
parser.add_argument('--fix-pred-lr', action='store_true',
                    help='Fix learning rate for the predictor')


# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main():
    args = parser.parse_args()

    # print model name
    # print("=> creating model '{}'".format(args.arch))
    if args.arch.__contains__('resnet50'):
        model = SimSiam(args.in_dim, args.out_dim, args.pred_dim)
    else:
        # model = vit_simsiam(in_chans=2, patch_size=16, out_dim=args.out_dim, pred_dim=args.pred_dim)
        model = vit_simsiam(in_chans=2, patch_size=8)

    # infer learning rate before changing batch size
    # init_lr = args.lr * args.batch_size / 160
    init_lr = args.lr

    print('cuda devices count is:{}'.format(torch.cuda.device_count()))
    print('args gpu setting is :{}'.format(args.gpu))
    ####################################################################################
    # args.gpu_id = "2,7";  # 指定gpu id
    # args.cuda = torch.cuda.is_available()  # 作为是否使用cpu的判定
    # # 配置环境  也可以在运行时临时指定 CUDA_VISIBLE_DEVICES='2,7' Python train.py
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id  # 这里的赋值必须是字符串，list会报错
    # device_ids = range(torch.cuda.device_count())   # torch.cuda.device_count()=2
    # print('current devices id is :{}'.format(device_ids))
    # # device_ids = [0,1]                                # 这里的0 就是上述指定 2，是主gpu,  1就是7,模型和数据由主gpu分发
    # if args.cuda:
    #     # 单GPU方式训练
    #     model = model.cuda()  # 这里将模型复制到gpu ,默认是cuda('0')，即转到第一个GPU 2
    #     DEVICE = torch.device("cuda:" + str(torch.cuda.current_device()))
    #     print('now using devices is :{}'.format(DEVICE))
    #     model = model.to(DEVICE)
    # if len(device_ids) > 1:
    #     # 以多GPU并行方式训练
    #     print('now using devices is :{}'.format(device_ids))
    #     model = torch.nn.DaraParallel(model);  # 前提是model已经.cuda()
    # else:
    #     # 以CPU方式训练
    #     DEVICE = torch.device("cpu")
    #     print('now using device is :{}'.format(DEVICE))
    #     model = model.to(DEVICE)
    ###########################################################################################################
    # 直接使用CUDA的多GPU
    # model.cuda()
    # # DistributedDataParallel will divide and allocate batch_size to all
    # # available GPUs if device_ids are not set
    # model = model = torch.nn.DataParallel(model)
    ###########################################################################################################

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
            # torch.distributed.init_process_group(backend='nccl', init_method='env://')
            # local_rank = torch.distributed.get_rank()
            # print("local rank: ", local_rank)
            # torch.cuda.set_device(local_rank)
            # device = torch.device("cuda", local_rank)
            # print("device:", device)


            # model.cuda()
            device_ids = range(torch.cuda.device_count())
            print('multiple GPU processing:{}'.format(device_ids))
            # model = torch.nn.DataParallel(model)  # 前提是model已经.cuda()
            # model.cuda()
            # model = torch.nn.DataParallel(model)

            model = model.cuda()
            device_ids = [0, 1]  # id为0和1的两块显卡
            model = torch.nn.DataParallel(model, device_ids=device_ids)

            # optimizer = nn.DataParallel(optimizer, device_ids=device_ids)

            # # if args.resume:
            # if os.path.isfile(args.resume):
            #     # print("=> loading checkpoint '{}'".format(args.resume))
            #     checkpoint = torch.load(args.resume)
            #     model.load_state_dict(checkpoint['state_dict'])
            #     optimizer.load_state_dict(checkpoint['optimizer'])
            #     args.start_epoch = checkpoint['epoch']
            #     print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            #
            #     # model.predictor = nn.Sequential(nn.Linear(2048, 2048, bias=False),
            #     #                                nn.BatchNorm1d(2048),
            #     #                                nn.ReLU(inplace=True),  # hidden layer
            #     #                                nn.Linear(2048, 2048))  # output layer
            #     #
            #     # model = torch.nn.DataParallel(model)
    else:
        DEVICE = torch.device("cpu")
        print('CPU processing:{}'.format(DEVICE))
        model = model.to(DEVICE)
        args.batch_size = 2
        args.workers = 1

    print('arch:{}, init_learing_rate:{}, epohs:{}, batch_size:{}, start_epoch:{}, workers:{}, outdim:{}'.format(
        args.arch, init_lr, args.epochs, args.batch_size, args.start_epoch, args.workers, args.out_dim))


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

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        train(train_loader_2, model, optimizer, epoch, args)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'model_dict': model.module.state_dict(),
        }, is_best=False, filename='checkpoint_{}_batch{}_dim{}_epoch{:04d}.pth.tar'.format(args.arch, args.batch_size, args.out_dim, epoch))


def train(train_loader, model, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    learning_rate_s = AverageMeter('Learning_Rate', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, learning_rate_s],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for batch, (imgL, imgR, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # print(imgL.shape)
        imgL = torch.reshape(imgL, (args.batch_size, 2, 224, 224)).float()
        imgR = torch.reshape(imgR, (args.batch_size, 2, 224, 224)).float()

        # print(imgR.shape)

        # print(type(imgL))
        # imgL = torch.from_numpy(imgL).float()
        # imgR = torch.from_numpy(imgR).float()
        # print(imgL[0][0][0].dtype)

        # if args.gpu is not None:
        #     # images[0] = images[0].cuda(args.gpu, non_blocking=True)
        #     imgL = imgL.cuda(args.gpu, non_blocking=True)
        #     # images[1] = images[1].cuda(args.gpu, non_blocking=True)
        #     imgR = imgR.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available() and args.gpu:
            imgL = imgL.cuda()
            imgR = imgR.cuda()

        # compute output and loss
        data_dict = model(x1=imgL, x2=imgR)
        loss = data_dict['loss'].mean()

        losses.update(loss.item(), imgL.size(0))
        learning_rate_s.update(optimizer.state_dict()['param_groups'][0]['lr'], imgL.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch % args.print_freq == 0:
            progress.display(batch)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    # args.epochs
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * (epoch%15) / 15))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


if __name__ == '__main__':
    # 对模型进行无监督训练
    main()
