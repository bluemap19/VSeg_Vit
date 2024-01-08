# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
import argparse

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from simsiam.sim_model.model_stage1 import SimSiam


class model_simSiam_stage3(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, in_dim=384, class_num=3):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(model_simSiam_stage3, self).__init__()

        a = in_dim//2
        # b = a//2

        self.predictor = nn.Sequential(nn.Linear(in_dim, in_dim, bias=False),
                                nn.BatchNorm1d(in_dim),
                                nn.ReLU(inplace=True), # first layer
                                nn.Linear(in_dim, a, bias=False),
                                nn.BatchNorm1d(a),
                                nn.ReLU(inplace=True), # second layer
                                # # nn.Linear(1024, class_num, bias=False)) # output layer
                                # nn.Linear(128, 64, bias=False),
                                # nn.BatchNorm1d(64),
                                # nn.ReLU(inplace=True), # second layer
                                nn.Linear(a, class_num, bias=False)) # output layer

    def forward(self, x1):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        # z1 = self.charter_encoder(x1) # NxC
        # z2 = self.charter_encoder(x2) # NxC

        p = self.predictor(x1) # NxC

        return p


class Model_simple_linear(nn.Module):

    def __init__(self, in_dim=10, out_dim=1):
        super(Model_simple_linear, self).__init__()
        # 是因为bias一般为False的时候，nn.Conv2d()后面通常接nn.BatchNorm2d(output)。
        a = in_dim // 2
        self.predictor = nn.Sequential(nn.Linear(in_dim, in_dim, bias=True),
                                       # nn.BatchNorm1d(in_dim),
                                       nn.ReLU(inplace=True),  # first layer
                                       nn.Linear(in_dim, a, bias=True),
                                       # nn.BatchNorm1d(a),
                                       nn.ReLU(inplace=True),  # second layer
                                       nn.Linear(a, a, bias=True),
                                       # nn.BatchNorm1d(a),
                                       nn.ReLU(inplace=True),  # third layer
                                       nn.Linear(a, out_dim, bias=False))  # output layer

    def forward(self, x):
        y_pred = self.predictor(x)

        return y_pred

# a = model_simSiam_stage3(class_num=3)
# print(a)
# b = models.__dict__['resnet50'](num_classes=66666, zero_init_residual=True)
# print(b)


