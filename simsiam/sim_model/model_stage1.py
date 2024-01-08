# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
import argparse
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize
        z = F.normalize(z, dim=1) # l2-normalize
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, in_dim=2, out_dim=512, pred_dim=384):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = models.__dict__['resnet50'](num_classes=out_dim, zero_init_residual=True)
        # self.encoder = base_encoder(num_classes=out_dim, zero_init_residual=True)

        self.encoder.conv1 = nn.Conv2d(in_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False)
        # (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        # fc层 是 resnet最后一层，原始设置是 (fc): Linear(in_features=2048, out_features=out_dim, bias=True)
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True),      # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True),      # second layer
                                        self.encoder.fc,            # 等于 (6): Linear(in_features=2048, out_features=2048, bias=True)
                                        nn.BatchNorm1d(out_dim, affine=False)) # output layer
        # print(self.encoder.fc[6])
        self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        # self.predictor = nn.Sequential(nn.Linear(out_dim, pred_dim, bias=False),
        #                                 nn.BatchNorm1d(pred_dim),
        #                                 nn.ReLU(inplace=True), # hidden layer
        #                                 nn.Linear(pred_dim, out_dim)) # output layer
        self.predictor = nn.Sequential(nn.Linear(out_dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, out_dim)) # output layer

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        L = D(p1, z2) / 2 + D(p2, z1) / 2

        return {'loss': L, 'z1': z1, 'z2': z2}
# ssh -p 46320 root@region-41.seetacloud.com
# import torchvision.models as models
# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))
#
# parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('--DIR', metavar='DIR', default='imgfile', type=str,
#                     help='path to dataset')
# parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
#                     choices=model_names,
#                     help='model architecture: ' +
#                         ' | '.join(model_names) +
#                         ' (default: resnet50)')
# parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
#                     help='number of data loading workers (default: 32)')
# parser.add_argument('--epochs', default=100, type=int, metavar='N',
#                     help='number of total epochs to run')
# parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
#                     help='manual epoch number (useful on restarts)')
# parser.add_argument('-b', '--batch-size', default=512, type=int,
#                     metavar='N',
#                     help='mini-batch size (default: 512), this is the total '
#                          'batch size of all GPUs on the current node when '
#                          'using Data Parallel or Distributed Data Parallel')
# parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
#                     metavar='LR', help='initial (base) learning rate', dest='lr')
# parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                     help='momentum of SGD solver')
# parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
#                     metavar='W', help='weight decay (default: 1e-4)',
#                     dest='weight_decay')
# parser.add_argument('-p', '--print-freq', default=10, type=int,
#                     metavar='N', help='print frequency (default: 10)')
# parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')
# parser.add_argument('--world-size', default=-1, type=int,
#                     help='number of nodes for distributed training')
# parser.add_argument('--rank', default=-1, type=int,
#                     help='node rank for distributed training')
# parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
#                     help='url used to set up distributed training')
# parser.add_argument('--dist-backend', default='nccl', type=str,
#                     help='distributed backend')
# parser.add_argument('--seed', default=None, type=int,
#                     help='seed for initializing training. ')
# parser.add_argument('--gpu', default=None, type=int,
#                     help='GPU id to use.')
# parser.add_argument('--multiprocessing-distributed', action='store_true',
#                     help='Use multi-processing distributed training to launch '
#                          'N processes per node, which has N GPUs. This is the '
#                          'fastest way to use PyTorch for either single node or '
#                          'multi node data parallel training')
#
# # simsiam specific configs:
# parser.add_argument('--dim', default=2048, type=int,
#                     help='feature dimension (default: 2048)')
# parser.add_argument('--pred-dim', default=512, type=int,
#                     help='hidden dimension of the predictor (default: 512)')
# parser.add_argument('--fix-pred-lr', action='store_true',
#                     help='Fix learning rate for the predictor')
#
# args = parser.parse_args()
#
#
# import torchvision.models as models
# model = SimSiam()
# print(model)

# # # print(args)
# # print(model.encoder)
# # # print(model.predictor)