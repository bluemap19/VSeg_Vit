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

from ele_unsup_cluster.get_pca_model_ele import get_pca_model_ele_info
from simsiam.sim_model.dataloader_stage3 import DatasetFolder
from simsiam.sim_model.model_stage1 import SimSiam
from simsiam.sim_model.model_stage3 import model_simSiam_stage3
from simsiam.sim_model.model_stage_1_vit import vit_simsiam

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--DIR_in', metavar='DIR', default=r'/root/autodl-tmp/data/target_stage3_small_p', type=str,
                    help='path to dataset')
########################################################################################
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
###############################################################################
# 源：32个
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
###############################################################################
# 源：512
parser.add_argument('-b', '--batch_size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning_rate', default=0.002, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.95, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight_decay', default=0.0001, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=5, type=int,
                    metavar='N', help='print frequency (default: 10)')
# /root/autodl-nas
# parser.add_argument('--resume', default=r'/root/autodl-tmp/dino_ele/simsiam/checkpoint_res50_240_0100.pth.tar', type=str, metavar='PATH',
# parser.add_argument('--resume', default=r'/root/autodl-tmp/dino_ele/simsiam/checkpoint_vit_240_0051.pth.tar', type=str, metavar='PATH',
# parser.add_argument('--resume', default=r'/root/autodl-nas/checkpoint_VIT_8patch_56_0014.pth.tar', type=str, metavar='PATH',
parser.add_argument('--resume', default=r'/root/autodl-nas/checkpoint_res50_batch240_dim384_epoch0039.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# parser.add_argument('--resume_predictor', default=r'checkpoint_0030.pth.tar', type=str, metavar='PATH',
# parser.add_argument('--resume_predictor', default=r'checkpoint_0030.pth.tar', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')
parser.add_argument('--PCA_PATH', default=r'/root/autodl-nas/stage1_all_feature_512_res50.txt', type=str, metavar='PCA_PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--gpu', default=True, type=bool, help='GPU id to use.')

# simsiam specific configs:
parser.add_argument('--out_dim', default=512, type=int,
# parser.add_argument('--out_dim', default=20, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--in_dim', default=2, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--pred_dim', default=384, type=int,
# parser.add_argument('--pred_dim', default=15, type=int,
                    help='hidden dimension of the predictor (default: 512)')
parser.add_argument('--fix_pred_lr', action='store_true',
                    help='Fix learning rate for the predictor')


def main():

    # 这个程序加上预测头，简单预测下，模型的分类效果
    # 保存结果是Endocer+Predictor整体的模型

    args = parser.parse_args()

    if torch.cuda.is_available():
        pass
    else:
        args.resume = r'D:\Data\target_answer\250X250\checkpoint_res50_batch240_dim384_epoch0039.pth.tar'
        # args.resume = r'D:\Data\target_answer\250X250\checkpoint_res50_batch240_dim15_epoch0016.pth.tar'
        # args.resume = r'D:/Data/target_answer/256-dyna-stat/checkpoint_VIT_8patch_56_0014.pth.tar'
        args.batch_size = 32
        args.workers = 1
        args.DIR_in = r'D:\Data\target_stage3_small_p'
        args.epochs = 2
        args.PCA_PATH = r'D:\GitHubProj\dino\ele_unsup_cluster\stage1_all_feature_512_res50.txt'
        # args.PCA_PATH = r'D:\GitHubProj\dino\ele_unsup_cluster\stage1_all_feature_20_res50.txt'
        args.out_dim = 512
        args.pred_dim = 384
        # args.out_dim = 20
        # args.pred_dim = 15



    ########### ResNet50 主干模型载入
    model_pre = SimSiam(args.in_dim, args.out_dim, args.pred_dim)
    # prev_dim = model_pre.encoder.fc[6].out_features
    ###### print(prev_dim)

    # ##### VIT 主干模型载入
    # model_pre = vit_simsiam()
    # prev_dim = model_pre.arch_model.head[6].out_features
    ########## print(prev_dim)

    # # 预测头模型载入
    # # model = model_simSiam_stage3(class_num=4, in_dim=16)
    # model = model_simSiam_stage3(class_num=1, in_dim=16)

    # 先把主干模型进行冻结
    for name, param in model_pre.named_parameters():
        param.requires_grad = False

    # PCA模型载入
    PCA = get_pca_model_ele_info(path=args.PCA_PATH, dim=6)

    # 如果没有阶段一的模型文件的话，直接退出，没办法进行阶段三
    if args.resume:
        # 判断模型文件是否存在
        if os.path.isfile(args.resume):
            # 使用GPU运行
            if torch.cuda.is_available() and args.gpu:
                if torch.cuda.device_count() >= 1:
                    DEVICE = torch.device("cuda:" + str(torch.cuda.current_device()))
                    print('putting model on single GPU:{}'.format(DEVICE))
                    # 单GPU该怎么载入模型
                    model_pre.cuda()
                    # model.cuda()

                    print('load encoder from:{}'.format(args.resume))
                    # model_pre = model_pre.to(DEVICE)
                    checkpoint = torch.load(args.resume)
                    model_pre.load_state_dict(checkpoint['model_dict'])

                    # model = model.to(DEVICE)

                    # if args.resume_predictor:
                    #     print('load predictor from:{}'.format(args.resume_predictor))
                    #     checkpoint = torch.load(args.resume_predictor)
                    #     model.load_state_dict(checkpoint['state_dict'])

                # elif torch.cuda.device_count() > 1:
                #     # 多GPU该怎么载入模型
                #     model_pre.cuda()
                #     model.cuda()
                #     device_ids = range(torch.cuda.device_count())
                #     print('multiple GPU processing:{}'.format(device_ids))
                #     model_pre = torch.nn.DataParallel(model_pre);  # 前提是model已经.cuda()
                #     checkpoint = torch.load(args.resume)
                #     model_pre.load_state_dict(checkpoint['state_dict'])
            else:
                DEVICE = torch.device("cpu")
                print('CPU processing:{}'.format(DEVICE))
                checkpoint = torch.load(args.resume, map_location=DEVICE)
                model_pre.to(DEVICE)
                model_pre.load_state_dict(checkpoint['model_dict'])    # model_dict

                # args.start_epoch = checkpoint['epoch']
                # optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        print('NO RESUME FILE')
        exit(0)

    # model_pre = torch.nn.Module(model_pre)
    model_pre = model_pre.encoder
    # model_pre = torch.nn.DataParallel(model_pre)
    # model_pre = model_pre.arch_model

    # # # torch.backends.cudnn.benchmark=True 将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，
    # # # 进而实现网络的加速。适用场景是网络结构固定（不是动态变化的），网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的，其实也就是一般情况下都比较适用。
    cudnn.benchmark = True

    # print('load imgs from path:{}'.format(args.DIR_in))
    train_dataset_2 = DatasetFolder(path=args.DIR_in+'/train')
    print('train data num is :{}'.format(train_dataset_2.__len__()))
    train_loader_2 = torch.utils.data.DataLoader(
        train_dataset_2, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    feature_all = []
    for epoch in range(args.start_epoch, args.epochs):
        f_t = model_train_data(train_loader_2, model_pre, PCA)
        if len(feature_all)==0:
            feature_all = f_t
        else:
            feature_all = np.append(feature_all, f_t, axis=0)
        print('current feature shape :{}'.format(feature_all.shape))

    np.savetxt('feature_all_{}_{}_{}.txt'.format(args.epochs, args.out_dim, 'res50'), feature_all, fmt='%.4f', delimiter='\t', comments='')




def model_train_data(train_loader, model_pre, PCA):
    # switch to train mode
    # model.eval()
    model_pre.eval()

    img_feature = []
    for batch, (imgL, labels, color_feature) in enumerate(train_loader):
        if torch.cuda.is_available():
            imgL = imgL.cuda()
            color_feature = color_feature.cuda()

        x1 = model_pre(imgL)

        x1 = x1.cpu().detach().numpy()

        x1 = PCA.transform(x1)
        x1 = torch.from_numpy(x1).float()

        if torch.cuda.is_available():
            x1 = x1.cuda()

        x1 = torch.cat((x1, color_feature), 1)

        feature_in = torch.cat((x1, labels), 1).numpy()
        print(x1.shape, labels.shape, feature_in.shape)
        if len(img_feature)==0:
            img_feature = feature_in
        else:
            img_feature = np.append(img_feature, feature_in, axis=0)

    return np.array(img_feature)



if __name__ == '__main__':
    # 对模型进行无监督训练
    main()

