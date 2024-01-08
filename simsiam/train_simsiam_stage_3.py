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
        args.epochs = 20
        args.PCA_PATH = r'D:\GitHubProj\dino\ele_unsup_cluster\stage1_all_feature_512_res50.txt'
        # args.PCA_PATH = r'D:\GitHubProj\dino\ele_unsup_cluster\stage1_all_feature_20_res50.txt'
        args.out_dim = 512
        args.pred_dim = 384
        # args.out_dim = 20
        # args.pred_dim = 15
        pass

    # arc 模型载入，有两种办法，一个是Res50， 另一个是VIT模型

    ########### ResNet50 主干模型载入
    model_pre = SimSiam(args.in_dim, args.out_dim, args.pred_dim)
    prev_dim = model_pre.encoder.fc[6].out_features
    ###### print(prev_dim)

    # ##### VIT 主干模型载入
    # model_pre = vit_simsiam()
    # prev_dim = model_pre.arch_model.head[6].out_features
    ########## print(prev_dim)

    # 预测头模型载入
    # model = model_simSiam_stage3(class_num=4, in_dim=16)
    model = model_simSiam_stage3(class_num=1, in_dim=16)

    # 先把主干模型进行冻结
    for name, param in model_pre.named_parameters():
        # print(name)
        param.requires_grad = False

    # print(model[-1])

    # infer learning rate before changing batch size
    init_lr = args.lr

    print('args gpu setting is :{}'.format(args.gpu))
    print('cuda devices count is:{}'.format(torch.cuda.device_count()))

    # # criterion = nn.CosineSimilarity(dim=1)
    # criterion = nn.Softmax()
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss(size_average=True)
    # criterion = nn.LogSoftmax()
    # criterion = nn.NLLLoss()
    # criterion = nn.MSELoss()
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()

    if args.fix_pred_lr:
        optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
                        {'params': model.module.predictor.parameters(), 'fix_lr': True}]
    else:
        optim_params = model.parameters()

    optimizer = torch.optim.SGD(optim_params, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

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
                    model.cuda()

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
                # # CPu 如何载入模型
                # DEVICE = torch.device("cpu")
                # model = model.to(DEVICE)
                # model_pre = model_pre.to(DEVICE)
                # print("=> loading checkpoint '{}'".format(args.resume))
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

    print('init_learing_rate:{}, epohs:{}, batch_size:{}, start_epoch:{}, workers:{}'.format(
        init_lr, args.epochs, args.batch_size, args.start_epoch, args.workers))

    # # # torch.backends.cudnn.benchmark=True 将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，
    # # # 进而实现网络的加速。适用场景是网络结构固定（不是动态变化的），网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的，其实也就是一般情况下都比较适用。
    cudnn.benchmark = True

    # print('load imgs from path:{}'.format(args.DIR_in))
    train_dataset_2 = DatasetFolder(path=args.DIR_in+'/train')
    print('train data num is :{}'.format(train_dataset_2.__len__()))
    train_loader_2 = torch.utils.data.DataLoader(
        train_dataset_2, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    val_dataset_2 = DatasetFolder(path=args.DIR_in+'/val')
    print('val data num is :{}'.format(val_dataset_2.__len__()))
    val_loader_2 = torch.utils.data.DataLoader(
        val_dataset_2, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)



    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        model_pre = torch.nn.DataParallel(model_pre)
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        train(train_loader_2, model, criterion, optimizer, epoch, args, model_pre, PCA)

        if (epoch+1)%args.print_freq == 0:
            acc1 = validate(train_loader_2, model, criterion, args, model_pre, PCA)
            # acc2 = validate(val_loader_2, model, criterion, args, model_pre, PCA)

        save_checkpoint({
            'epoch': epoch + 1,
            'model_dict': model.state_dict()
        }, is_best=False, filename='checkpoint_predictor_{:04d}.pth.tar'.format(epoch))



def train(train_loader, model, criterion, optimizer, epoch, args, model_pre, PCA):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.6f')
    learning_rate_s = AverageMeter('Learning_Rate', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, learning_rate_s],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    model_pre.eval()

    end = time.time()
    for batch, (imgL, labels, color_feature) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # imgL = torch.reshape(imgL, (args.batch_size, 1, 196, 196))
        # imgR = torch.reshape(imgR, (args.batch_size, 1, 196, 196))
        # print(labels.shape)
        # print(imgL.shape)

        labels = labels[:, 1]
        # print(labels.shape)

        labels = torch.reshape(labels, (args.batch_size, -1)).float()
        # print(labels.shape)
        # print(labels)

        # print(imgR.shape)

        # print(type(imgL))
        # imgL = torch.from_numpy(imgL).float()
        # imgR = torch.from_numpy(imgR).float()
        # print(imgL[0][0][0].dtype)

        if torch.cuda.is_available() and args.gpu:
            imgL = imgL.cuda()
            labels = labels.cuda()
            color_feature = color_feature.cuda()

        x1 = model_pre(imgL)

        x1 = x1.cpu().detach().numpy()
        # print(x1.shape)
        # print(PCA)
        x1 = PCA.transform(x1)
        # print(x1.shape)
        x1 = torch.from_numpy(x1).float()

        if torch.cuda.is_available() and args.gpu:
            x1 = x1.cuda()

        # print(x1.shape, color_feature.shape)
        x1 = torch.cat((x1, color_feature), 1)
        # print(x1.shape)

        y_pred = model(x1)

        print('labels:{}'.format(labels.long().view(-1)))
        print('pred  :{}'.format(y_pred.long().view(-1)))
        # # compute output and loss
        # data_dict = model(x1=imgL, x2=imgR)
        # loss = data_dict['loss'].mean()
        # print(a.shape, labels.shape)
        loss = criterion(y_pred.squeeze(dim=1), labels.squeeze(dim=1))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), imgL.size(0))
        learning_rate_s.update(optimizer.state_dict()['param_groups'][0]['lr'], imgL.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        if batch % args.print_freq == 0:
            progress.display(batch)



def validate(val_loader, model, criterion, args, model_pre, PCA):

    count_charter_all = np.array([0, 0, 0], dtype=np.int32)
    count_charter_corr_all = np.array([0, 0, 0], dtype=np.int32)

    count_labels_all = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
    count_labels_corr_all = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)

    # switch to evaluate mode
    model.eval()
    model_pre.eval()

    pred_count = 0
    correct_count = 0

    with torch.no_grad():
        for batch, (imgL, labels, color_feature) in enumerate(val_loader):
            # imgL = torch.reshape(imgL, (args.batch_size, 1, 196, 196))
            # imgR = torch.reshape(imgR, (args.batch_size, 1, 196, 196))
            labels = labels[:, 1]

            labels = labels.float()
            # print(imgL.shape, type(imgL))
            # labels = torch.from_numpy(labels).float()
            # print(labels.shape)
            # imgL = torch.from_numpy(imgL).reshape(-1, 2, 224, 224)
            # print(imgL.shape, type(imgL))

            if torch.cuda.is_available() and args.gpu:
                imgL = imgL.cuda()
                # imgR = imgR.cuda()
                labels = labels.cuda()
                color_feature = color_feature.cuda()

            # x1, x2 = model_pre(imgL, imgR)['z1'], model_pre(imgL, imgR)['z2']
            x1 = model_pre(imgL)

            x1 = x1.cpu().detach().numpy()
            x1 = PCA.transform(x1)
            x1 = torch.from_numpy(x1).float()
            if torch.cuda.is_available() and args.gpu:
                x1 = x1.cuda()

            x1 = torch.cat((x1, color_feature), 1)

            y_pred = model(x1)

            # if args.gpu is not None:
            #     images = images.cuda(args.gpu, non_blocking=True)
            # target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            # output = model(images)
            # loss = criterion(y_pred, labels)

            # measure accuracy and record loss
            # 大于1个维度的数据输出
            if (y_pred.shape[0] > 32):
                # print('multiple classify evalute')
                y_pred = y_pred.long().cpu().detach().numpy()
                labels = labels.long().cpu().detach().numpy()

                count_charter_corr, count_charter, count_labels_corr, count_labels = get_multi_charter_acc(y_pred, labels)
                count_charter_corr_all += count_charter_corr
                count_charter_all += count_charter
                count_labels_corr_all += count_labels_corr
                count_labels_all += count_labels
            # 维度为1的数预测据的准确率计算
            else:
                y_p_t = []
                y_pred = y_pred.cpu().detach().numpy()
                for i in range(32):
                    if y_pred[i] <= 0.5:
                        y_p_t.append(0)
                    elif y_pred[i] >= 0.5:
                        y_p_t.append(1)
                    else:
                        print('wrong answer:{}'.format(y_pred[i]))
                        exit(0)
                y_p_t = np.array(y_p_t)
                correct_count += torch.sum(torch.eq(torch.from_numpy(y_p_t).view(-1).long(), labels.view(-1).long()))
                pred_count += labels.shape[0]

                # accracy = np.mean((torch.argmax(y_pred,1)==torch.argmax(labels,1)).numpy())
                pass

    if correct_count > 0:
        print('acc:{}'.format((correct_count+1)/pred_count))
        pass
    else:
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
    """ Decay the learning rate based on schedule """
    # cur_lr = init_lr - args.weight_decay
    # cur_lr = (init_lr * args.momentum) + (1-args.momentum)*cur_lr
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs / 2))
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
