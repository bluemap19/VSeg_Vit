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
from simsiam.sim_model.dataloader_stage3 import DatasetFolder
from simsiam.sim_model.model_stage1 import SimSiam
from simsiam.sim_model.model_stage3 import model_simSiam_stage3
from simsiam.sim_model.model_stage_1_vit import vit_simsiam

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--DIR_in', metavar='DIR', default=r'/root/autodl-tmp/data/tar_s3', type=str,
                    help='path to dataset')
########################################################################################
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
###############################################################################
# 源：32个
parser.add_argument('--workers', default=24, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
###############################################################################
# 源：512
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning_rate', default=0.006, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.8, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight_decay', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=5, type=int,
                    metavar='N', help='print frequency (default: 10)')
# /root/autodl-nas
parser.add_argument('--resume', default=r'/root/autodl-nas/models/checkpoint_res50_256_0029.pth.tar', type=str, metavar='PATH',
# parser.add_argument('--resume', default=r'/root/autodl-nas/checkpoint_0108_dyst_0087_dyna_128.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# parser.add_argument('--resume_predictor', default=r'checkpoint_0030.pth.tar', type=str, metavar='PATH',
# parser.add_argument('--resume_predictor', default=r'checkpoint_0030.pth.tar', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')

parser.add_argument('--gpu', default=True, type=bool, help='GPU id to use.')

# simsiam specific configs:
parser.add_argument('--out_dim', default=512, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--in_dim', default=2, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--pred_dim', default=384, type=int,
                    help='hidden dimension of the predictor (default: 512)')
parser.add_argument('--fix_pred_lr', action='store_true',
                    help='Fix learning rate for the predictor')


class CombinedModel(nn.Module):
    def __init__(self, submodel1, submodel2):
        super(CombinedModel, self).__init__()
        self.submodel1 = submodel1
        self.submodel2 = submodel2

    def forward(self, x):
        x = self.submodel1(x)
        x = self.submodel2(x)
        return x

def main():
    args = parser.parse_args()

    # print model name
    print("=> creating model '{}'".format(args.arch))
    # model = model_simSiam_stage3(class_num=1)
    model_pre = SimSiam(args.in_dim, args.out_dim, args.pred_dim)
    prev_dim = model_pre.encoder.fc[6].out_features
    # print(prev_dim)

    # model_pre = vit_simsiam()
    # prev_dim = model_pre.arch_model.head[6].out_features
    # # print(prev_dim)

    model = model_simSiam_stage3(class_num=4, in_dim=prev_dim)

    print('args gpu setting is :{}'.format(args.gpu))
    print('cuda devices count is:{}'.format(torch.cuda.device_count()))

    # # criterion = nn.CosineSimilarity(dim=1)
    # # criterion = nn.Softmax()
    # # criterion = nn.CrossEntropyLoss()
    # # criterion = nn.BCELoss(size_average=True)
    # # criterion = nn.LogSoftmax()
    # # criterion = nn.NLLLoss()
    criterion = nn.MSELoss()
    # # criterion = nn.BCELoss()
    # # criterion = nn.BCEWithLogitsLoss()


    if torch.cuda.is_available():
        pass
    else:
        args.resume = r'D:\Data\target_answer\models\checkpoint_res50_256_0028.pth.tar'
        args.batch_size = 16
        args.workers = 1
        args.DIR_in = r'D:\Data\tar_s3'
        args.epochs = 20
        pass


    # # 如果没有阶段一的模型文件的话，直接退出，没办法进行阶段三
    # # 判断模型文件是否存在
    # print('load encoder from:{}'.format(args.resume))
    # if os.path.isfile(args.resume):
    #     # 使用GPU运行
    #     if torch.cuda.device_count() >= 1:
    #         DEVICE = torch.device("cuda:" + str(torch.cuda.current_device()))
    #         print('putting model on single GPU:{}'.format(DEVICE))
    #         # 单GPU该怎么载入模型
    #         model_pre.cuda()
    #         # model_pre = model_pre.to(DEVICE)
    #         checkpoint = torch.load(args.resume)
    #         model_pre.load_state_dict(checkpoint['model_dict'])
    #
    #     else:
    #         # # CPu 如何载入模型
    #         DEVICE = torch.device("cpu")
    #         print('CPU processing:{}'.format(DEVICE))
    #         checkpoint = torch.load(args.resume, map_location=DEVICE)
    #         model_pre.to(DEVICE)
    #         model_pre.load_state_dict(checkpoint['model_dict'])    # model_dict
    #         print("=> loaded checkpoint '{}' (epoch {})"
    #               .format(args.resume, checkpoint['epoch']))
    # else:
    #     print("=> no checkpoint found at '{}'".format(args.resume))
    #     exit(0)
    #
    #
    #
    # path_pred = r'checkpoint_predictor_0014.pth.tar'
    # if os.path.isfile(path_pred):
    #     model.cuda()
    #     checkpoint = torch.load(path_pred)
    #     model.load_state_dict(checkpoint['model_dict'])
    # else:
    #     print("=> no checkpoint found at '{}'".format(args.resume))
    #     exit(0)


    model_pre = model_pre.encoder
    model_final = CombinedModel(model_pre, model)
    model_final.cuda()
    model_final = torch.nn.DataParallel(model_final)
    for name, param in model_final.named_parameters():
        # print(name)
        param.requires_grad = True

    init_lr = 0.01
    args.weight_decay = 0.0001
    args.momentum = 0.9
    optim_params = model_final.parameters()
    optimizer = torch.optim.SGD(optim_params, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # # # torch.backends.cudnn.benchmark=True 将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，
    # # # 进而实现网络的加速。适用场景是网络结构固定（不是动态变化的），网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的，其实也就是一般情况下都比较适用。
    cudnn.benchmark = True

    # print('load imgs from path:{}'.format(args.DIR_in))
    train_dataset_2 = DatasetFolder(path=args.DIR_in+'/train')
    print('train data num is :{}'.format(train_dataset_2.__len__()))
    train_loader_2 = torch.utils.data.DataLoader(
        train_dataset_2, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    val_dataset_2 = DatasetFolder(path=args.DIR_in+'/train')
    print('val data num is :{}'.format(val_dataset_2.__len__()))
    val_loader_2 = torch.utils.data.DataLoader(
        val_dataset_2, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)



    print('init_learing_rate:{}, epohs:{}, batch_size:{}, start_epoch:{}, workers:{}'.format(
        init_lr, args.epochs, args.batch_size, args.start_epoch, args.workers))

    # 解冻前一部分，整体进行训练
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        train_final(train_loader_2, model_final, criterion, optimizer, epoch, args)

        if epoch%args.print_freq == 0:
            # validate(val_loader_2, model, criterion, args)
            validate_final(train_loader_2, model_final, args)


        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model_final.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'model_dict': model_final.module.state_dict(),
        }, is_best=False, filename='checkpoint_final_{:04d}.pth.tar'.format(epoch))


def train_final(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time_final', ':6.3f')
    data_time = AverageMeter('Data_final', ':6.3f')
    losses = AverageMeter('Loss_final', ':.6f')
    learning_rate_s = AverageMeter('Learning_Rate_final', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, learning_rate_s],
        prefix="Epoch_final: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for batch, (imgL, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        labels = torch.reshape(labels, (args.batch_size, -1)).float()

        if torch.cuda.is_available() and args.gpu:
            imgL = imgL.cuda()
            labels = labels.cuda()

        y_pred = model(imgL)

        loss = criterion(y_pred.squeeze(dim=1), labels.squeeze(dim=1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), imgL.size(0))
        learning_rate_s.update(optimizer.state_dict()['param_groups'][0]['lr'], imgL.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if batch % args.print_freq == 0:
            progress.display(batch)


def validate_final(val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    count_charter_all = np.array([0, 0, 0], dtype=np.int32)
    count_charter_corr_all = np.array([0, 0, 0], dtype=np.int32)

    count_labels_all = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
    count_labels_corr_all = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for batch, (imgL, labels) in enumerate(val_loader):
            labels = torch.reshape(labels, (args.batch_size, -1)).float()

            if torch.cuda.is_available() and args.gpu:
                imgL = imgL.cuda()
                labels = labels.cuda()

            y_pred = model(imgL)

            if labels.shape[1] > 1:
                y_pred = y_pred.long().cpu().detach().numpy()
                labels = labels.long().cpu().detach().numpy()

                count_charter_corr, count_charter, count_labels_corr, count_labels = get_multi_charter_acc(y_pred, labels)
                count_charter_all += count_charter
                count_charter_corr_all += count_charter_corr
                count_labels_all += count_labels
                count_labels_corr_all += count_labels_corr

            batch_time.update(time.time() - end)
            end = time.time()

    print('charter acc:{}, class acc:{}'.format(count_charter_corr_all/(count_charter_all+1), count_labels_corr_all/(count_labels_all+1)))
    print('charter num:{}, {}, class num:{}, {}'.format(count_charter_corr_all, count_charter_all, count_labels_corr_all, count_labels_all))



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
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr



def accuracy(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        batch_size = target.size(0)

        # print(type(output), type(target))
        # print(output.view(1, -1))
        # print(target.view(1, -1))
        correct = torch.eq(output.view(1, -1).long(), target.view(1, -1).long())
        # print(correct.reshape(-1).float().sum(0, keepdim=True).mul_(100.0 / batch_size))

        return correct.reshape(-1).float().sum(0, keepdim=True).mul_(100.0 / batch_size)
        # _, pred = output.topk(maxk, 1, True, True)
        # pred = pred.t()
        # correct = pred.eq(target.view(1, -1).expand_as(pred))
        #
        # res = []
        # for k in topk:
        #     correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        #     res.append(correct_k.mul_(100.0 / batch_size))
        # return res

def accuracy_np(y_pred, labels):
    y_pred = y_pred.ravel()
    labels = labels.ravel()

    if y_pred.shape[0] != labels.shape[0]:
        print('error............:{}-{}'.format(y_pred.shape[0], labels.shape[0]))
        exit(0)

    acc = 0
    for i in range(y_pred.shape[0]):
        if y_pred[i] == labels[i]:
            acc += 1

    acc = acc/y_pred.shape[0]*100
    return acc


def get_multi_charter_acc(y_pred, labels):
    count_charter = np.array([0, 0, 0], dtype=np.int32)
    count_charter_corr = np.array([0, 0, 0], dtype=np.int32)

    count_labels = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
    count_labels_corr = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)

    for i in range(labels.shape[0]):
        # 计算判断类别的正确个数
        label_class = labels[i, -1]
        count_labels[label_class] += 1
        if (y_pred[i][-1]==labels[i,-1]):
            count_labels_corr[label_class] += 1

        # 计算 特征提取能力的 整体个数个数
        count_charter = count_charter + labels[i, :3].ravel()
        # 计算 特征提取能力 正确个数
        for j in range(3):
            if (int(labels[i][j]) != 0) and (int(labels[i][j]) == int(y_pred[i][j])):
                count_charter_corr[j] += 1

    count_charter = np.array(count_charter)
    count_charter_corr = np.array(count_charter_corr)
    count_labels = np.array(count_labels)
    count_labels_corr = np.array(count_labels_corr)

    return count_charter_corr, count_charter, count_labels_corr, count_labels

def get_single_charter_acc(y_pred, labels):
    count_labels = [0, 0, 0, 0, 0, 0]
    count_labels_corr = [0, 0, 0, 0, 0, 0]
    for i in range(labels.shape[0]):
        class_index = labels[i][0]
        count_labels[class_index] += 1

        if labels[i][0] == y_pred[i][0]:
            count_labels_corr[class_index] += 1

    count_labels = np.array(count_labels)
    count_labels_corr = np.array(count_labels_corr)

    return count_labels_corr / (count_labels + 1)


if __name__ == '__main__':
    # 对模型进行无监督训练
    main()
