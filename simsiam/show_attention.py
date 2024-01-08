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
import vision_transformer as vits


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--DIR_in', metavar='DIR', default=r'/root/autodl-tmp/data/target_stage3', type=str,
                    help='path to dataset')
########################################################################################
parser.add_argument('-a', '--arch', metavar='ARCH', default='vit_small',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
###############################################################################
# 源：32个
parser.add_argument('--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=40, type=int, metavar='N',
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
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.2, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight_decay', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
# /root/autodl-nas
parser.add_argument('--resume', default=r'/root/autodl-nas/checkpoint_0148_dyst_0137_dyna_256.pth.tar', type=str, metavar='PATH',
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
parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')


def main():
    args = parser.parse_args()

    if torch.cuda.is_available():
        pass
    else:
        args.resume = r'D:\Data\target_answer\models\checkpoint_vit_256_0014.pth.tar'
        args.batch_size = 16
        args.workers = 1
        args.DIR_in = r'D:\Data\tar_s3'
        args.epochs = 20
        pass
    # print model name
    # print("=> creating model '{}'".format(args.arch))
    # model = model_simSiam_stage3(class_num=1)
    # model_pre = SimSiam(args.in_dim, args.out_dim, args.pred_dim)
    # prev_dim = model_pre.encoder.fc[6].out_features
    # print(prev_dim)

    model_pre = vit_simsiam()
    # prev_dim = model_pre.arch_model.head[6].out_features
    # # print(prev_dim)

    model = vits.__dict__[args.arch](patch_size=args.patch_size, in_chans=2, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    # model = model_simSiam_stage3(class_num=4, in_dim=prev_dim)
    #
    # for name, param in model_pre.named_parameters():
    #     # print(name)
    #     param.requires_grad = False

    # print(model[-1])

    # # infer learning rate before changing batch size
    # init_lr = args.lr * args.batch_size / 256
    #
    # print('args gpu setting is :{}'.format(args.gpu))
    # print('cuda devices count is:{}'.format(torch.cuda.device_count()))
    #
    # # # criterion = nn.CosineSimilarity(dim=1)
    # # # criterion = nn.Softmax()
    # # # criterion = nn.CrossEntropyLoss()
    # # # criterion = nn.BCELoss(size_average=True)
    # # # criterion = nn.LogSoftmax()
    # # # criterion = nn.NLLLoss()
    # criterion = nn.MSELoss()
    # # # criterion = nn.BCELoss()
    # # # criterion = nn.BCEWithLogitsLoss()
    # #
    # if args.fix_pred_lr:
    #     optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
    #                     {'params': model.module.predictor.parameters(), 'fix_lr': True}]
    # else:
    #     optim_params = model.parameters()
    #
    # optimizer = torch.optim.SGD(optim_params, init_lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    #
    # if torch.cuda.is_available():
    #     pass
    # else:
    #     args.resume = r'D:\Data\target_answer\models\checkpoint_res50_256_0028.pth.tar'
    #     args.batch_size = 16
    #     args.workers = 1
    #     args.DIR_in = r'D:\Data\tar_s3'
    #     args.epochs = 20
    #     pass
    #
    #
    # 如果没有阶段一的模型文件的话，直接退出，没办法进行阶段三
    if args.resume:
        # 判断模型文件是否存在
        if os.path.isfile(args.resume):
            # 使用GPU运行
            if torch.cuda.is_available() and args.gpu:
                if torch.cuda.device_count() == 1:
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

                elif torch.cuda.device_count() > 1:
                    # 多GPU该怎么载入模型
                    model_pre.cuda()
                    device_ids = range(torch.cuda.device_count())
                    print('multiple GPU processing:{}'.format(device_ids))
                    model_pre = torch.nn.DataParallel(model_pre);  # 前提是model已经.cuda()
                    checkpoint = torch.load(args.resume)
                    model_pre.load_state_dict(checkpoint['state_dict'])
            else:
                # # CPu 如何载入模型
                # DEVICE = torch.device("cpu")
                # print('CPU processing:{}'.format(DEVICE))
                # model = model.to(DEVICE)
                # model_pre = model_pre.to(DEVICE)
                # print("=> loading checkpoint '{}'".format(args.resume))
                DEVICE = torch.device("cpu")
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


    model_pre = model_pre.arch_model
    print()

    # pretrained_dict = {key: value for key, value in model_pre.named_parameters() if (key in model)}
    # print(pretrained_dict)


    pretrained_dict = {}
    for name, param in model.named_parameters():
        for name_vice, param_vice in model_pre.named_parameters():
            if name_vice.__contains__(name):
                pretrained_dict.update({name:param_vice})
    model.load_state_dict(pretrained_dict)
    # print(model.cls_token)


    # # # torch.backends.cudnn.benchmark=True 将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，
    # # # 进而实现网络的加速。适用场景是网络结构固定（不是动态变化的），网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的，其实也就是一般情况下都比较适用。
    cudnn.benchmark = True

    # print('load imgs from path:{}'.format(args.DIR_in))
    train_dataset_2 = DatasetFolder(path=args.DIR_in+'/train')
    print('train data num is :{}'.format(train_dataset_2.__len__()))
    train_loader_2 = torch.utils.data.DataLoader(
        train_dataset_2, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    pic = train_dataset_2[0]
    print(pic[0].shape)
    print(pic[1])
    pic = torch.from_numpy(pic[0]).reshape((1, 2, 224, 224))

    attentions = model.get_last_selfattention(pic)
    nh = attentions.shape[1]  # number of head
    w_featmap = pic.shape[-2] // args.patch_size
    h_featmap = pic.shape[-1] // args.patch_size

    print(type(attentions), attentions.shape)

    attentions = attentions.reshape(nh, w_featmap, h_featmap)

    # val_dataset_2 = DatasetFolder(path=args.DIR_in+'/train')
    # print('val data num is :{}'.format(val_dataset_2.__len__()))
    # val_loader_2 = torch.utils.data.DataLoader(
    #     val_dataset_2, batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.workers, pin_memory=True, drop_last=True)
    #
    # for epoch in range(args.start_epoch, args.epochs):
    #     adjust_learning_rate(optimizer, init_lr, epoch, args)
    #
    #     # train for one epoch
    #     train(train_loader_2, model, criterion, optimizer, epoch, args, model_pre)
    #
    #     if epoch%args.print_freq == 0:
    #         acc1 = validate(val_loader_2, model, criterion, args, model_pre)
    #         # print('acc1:{}'.format(acc1))
    #         # acc2 = validate(train_loader_2, model, criterion, args, model_pre)
    #         # print('acc2:{}'.format(acc2))
    #
    # #     # acc1 = validate(val_loader_2, model, criterion, args, model_pre)
    # #     # acc2 = validate(train_loader_2, model, criterion, args, model_pre)
    # #     # print('svaing model.....')
    #     save_checkpoint({
    #         'epoch': epoch + 1,
    #         'state_dict': model.state_dict()
    #     }, is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch))


def train(train_loader, model, criterion, optimizer, epoch, args, model_pre):
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

    end = time.time()
    for batch, (imgL, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # imgL = torch.reshape(imgL, (args.batch_size, 1, 196, 196))
        # imgR = torch.reshape(imgR, (args.batch_size, 1, 196, 196))
        # print(labels.shape)
        labels = torch.reshape(labels, (args.batch_size, -1)).float()
        # print(labels.shape)
        # print(labels)

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
            # imgR = imgR.cuda()
            labels = labels.cuda()


        # 这个地方可能出问题
        x1 = model_pre(imgL)

        y_pred = model(x1)

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

        # if batch%30 == 0:
        #     print(torch.squeeze(y_pred)[:10].cpu().detach().numpy())
        #     print(torch.squeeze(labels)[:10].cpu().detach().numpy())

        if batch % args.print_freq == 0:
            progress.display(batch)
            # print(y_pred)
            # print(labels)

def validate(val_loader, model, criterion, args, model_pre):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    # # top_list = [AverageMeter('Acc@1', ':6.2f'), AverageMeter('Acc@2', ':6.2f'), AverageMeter('Acc@3', ':6.2f'),
    # #             AverageMeter('Acc@4', ':6.2f'), AverageMeter('Acc@5', ':6.2f'), AverageMeter('Acc@6', ':6.2f')]
    # class_0 = AverageMeter('class0acc', ':.4f')
    # class_1 = AverageMeter('class1acc', ':.4f')
    # class_2 = AverageMeter('class2acc', ':.4f')
    # class_3 = AverageMeter('class3acc', ':.4f')
    # class_4 = AverageMeter('class4acc', ':.4f')
    # class_5 = AverageMeter('class5acc', ':.4f')
    #
    # # acc_temp = [AverageMeter('acc_temp', ':6.2f'), AverageMeter('acc_temp', ':6.2f'), AverageMeter('acc_temp', ':6.2f')]
    # acc_0 = AverageMeter('charter0acc', ':.4f')
    # acc_1 = AverageMeter('charter1acc', ':.4f')
    # acc_2 = AverageMeter('charter2acc', ':.4f')

    count_charter_all = np.array([0, 0, 0], dtype=np.int32)
    count_charter_corr_all = np.array([0, 0, 0], dtype=np.int32)

    count_labels_all = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
    count_labels_corr_all = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)

    acc_my = AverageMeter('ACC', ':6.2f')
    # progress = ProgressMeter(
    #     len(val_loader),
    #     [batch_time, losses, acc_my, class_0, class_1, class_2, class_3, class_4, class_5, acc_0, acc_1, acc_2],
    #     prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    model_pre.eval()

    with torch.no_grad():
        end = time.time()
        for batch, (imgL, labels) in enumerate(val_loader):
            # imgL = torch.reshape(imgL, (args.batch_size, 1, 196, 196))
            # imgR = torch.reshape(imgR, (args.batch_size, 1, 196, 196))
            labels = torch.reshape(labels, (args.batch_size, -1)).float()

            if torch.cuda.is_available() and args.gpu:
                imgL = imgL.cuda()
                imgR = imgR.cuda()
                labels = labels.cuda()

            # x1, x2 = model_pre(imgL, imgR)['z1'], model_pre(imgL, imgR)['z2']
            x1 = model_pre(imgL)
            # x2 = model_pre(imgR)
            # print(x1.shape, x2.shape)
            y_pred = model(x1)

            # if args.gpu is not None:
            #     images = images.cuda(args.gpu, non_blocking=True)
            # target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            # output = model(images)
            # loss = criterion(y_pred, labels)

            # measure accuracy and record loss
            if labels.shape[1] > 1:
                # print('multiple classify evalute')
                y_pred = y_pred.long().cpu().detach().numpy()
                labels = labels.long().cpu().detach().numpy()
                # for i in range(y_pred.shape[1]):
                #     acc = accuracy_np(y_pred[:, i], labels[:, i])
                #     top_list[i].update(acc, imgR.size[0])

                count_charter_corr, count_charter, count_labels_corr, count_labels = get_multi_charter_acc(y_pred, labels)
                count_charter_all += count_charter
                count_charter_corr_all += count_charter_corr
                count_labels_all += count_labels
                count_labels_corr_all += count_labels_corr
                # print('1111111111:{}'.format(acc_s1))
                # print('2222222222:{}'.format(acc_s2))
                # print(acc_s1)
                # acc_0.update(float(acc_s1[0]), imgR.size[0])
                # acc_1.update(acc_s1[1], imgR.size[0])
                # acc_2.update(acc_s1[2], imgR.size[0])
                #
                # class_0.update(acc_s2[0], imgR.size[0])
                # class_1.update(acc_s2[1], imgR.size[0])
                # class_2.update(acc_s2[2], imgR.size[0])
                # class_3.update(acc_s2[3], imgR.size[0])
                # class_4.update(acc_s2[4], imgR.size[0])
                # class_5.update(acc_s2[5], imgR.size[0])
                # for i in range(6):
                #     top_list[i].update(acc_s2[i], imgR.size[0])
                #
                # for j in range(3):
                #     acc_temp[j].update(acc_s1[j], imgR.size[0])
                # pass
            else:
                acc = accuracy(y_pred, labels, topk=(1, 5))
                acc_my.update(acc.item(), imgR.size[0])
            # losses.update(loss.item(), imgL.size(0))
            # top1.update(acc1[0], imgL.size(0))
            # top5.update(acc5[0], imgL.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if batch % args.print_freq == 0:
            #     progress.display(batch)

        # # TODO: this should also be done with the ProgressMeter
        # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        #       .format(top1=top1, top5=top5))
    print('charter acc:{}, class acc:{}'.format(count_charter_corr_all/(count_charter_all+1), count_labels_corr_all/(count_labels_all+1)))
    print('charter num:{}, {}, class num:{}, {}'.format(count_charter_corr_all, count_charter_all, count_labels_corr_all, count_labels_all))
    return acc_my.avg


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

    # print(y_pred[:, -1])
    # print(labels[:, -1])

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

    # for i in range(labels.shape[0]):
    #     if (labels[i, :].ravel()==np.array([0, 0, 0])).all():
    #         count_labels[0] += 1
    #         if (y_pred[i, :].ravel()==labels[i, :].ravel()).all():
    #             count_labels_corr[0] += 1
    #         # count_charter = count_charter + labels[i, :].ravel()
    #         # count_charter_corr = count_charter_corr + (y_pred[i, :].ravel()==labels[i, :].ravel()) - (labels[i, :].ravel()==np.array([0, 0, 0], dtype=np.int64))
    #
    #     elif (labels[i, :].ravel()==np.array([1, 0, 0])).all():
    #         count_labels[1] += 1
    #         if (y_pred[i, :].ravel()==labels[i, :].ravel()).all():
    #             count_labels_corr[1] += 1
    #         # count_charter = count_charter + labels[i, :].ravel()
    #         # count_charter_corr = count_charter_corr + (y_pred[i, :].ravel()==labels[i, :].ravel()) - (labels[i, :].ravel()==np.array([0, 0, 0], dtype=np.int64))
    #
    #     elif (labels[i, :].ravel()==np.array([0, 1, 0])).all():
    #         count_labels[2] += 1
    #         if (y_pred[i, :].ravel()==labels[i, :].ravel()).all():
    #             count_labels_corr[2] += 1
    #         # count_charter = count_charter + labels[i, :].ravel()
    #         # count_charter_corr = count_charter_corr + (y_pred[i, :].ravel()==labels[i, :].ravel()) - (labels[i, :].ravel()==np.array([0, 0, 0], dtype=np.int64))
    #
    #     elif (labels[i, :].ravel()==np.array([1, 1, 0])).all():
    #         count_labels[3] += 1
    #         if (y_pred[i, :].ravel()==labels[i, :].ravel()).all():
    #             count_labels_corr[3] += 1
    #         # count_charter = count_charter + labels[i, :].ravel()
    #         # count_charter_corr = count_charter_corr + (y_pred[i, :].ravel()==labels[i, :].ravel()) - (labels[i, :].ravel()==np.array([0, 0, 0], dtype=np.int64))
    #
    #     elif (labels[i, :].ravel()==np.array([0, 0, 1])).all():
    #         count_labels[4] += 1
    #         if (y_pred[i, :].ravel()==labels[i, :].ravel()).all():
    #             count_labels_corr[4] += 1
    #         # count_charter = count_charter + labels[i, :].ravel()
    #         # count_charter_corr = count_charter_corr + (y_pred[i, :].ravel()==labels[i, :].ravel()) - (labels[i, :].ravel()==np.array([0, 0, 0], dtype=np.int64))
    #
    #     elif (labels[i, :].ravel()==np.array([0, 1, 1])).all():
    #         count_labels[5] += 1
    #         if (y_pred[i, :].ravel()==labels[i, :].ravel()).all():
    #             count_labels_corr[5] += 1
    #         # count_charter = count_charter + labels[i, :].ravel()
    #         # count_charter_corr = count_charter_corr + (y_pred[i, :].ravel()==labels[i, :].ravel()) - (labels[i, :].ravel()==np.array([0, 0, 0], dtype=np.int64))
    #
    # for i in range(3):
    #     count_charter[i] = np.sum(labels[:, i].ravel())
    #     for j in range(labels.shape[0]):
    #         if (y_pred[j][i] == labels[j][i]) and (labels[j][i]==1):
    #             count_charter_corr[i] += 1
    #         pass
    #     # count_charter_corr[i] = np.sum(y_pred[:, i].ravel()==labels[:, i].ravel()) - np.sum(labels[:, i].ravel()==np.zeros((y_pred.shape[0]), dtype=np.int64))

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
