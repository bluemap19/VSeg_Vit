import math
import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from simsiam.ele_seg_2.dataloader_ele_seg import dataloader_bottle, dataloader_up1, dataloader_up2
from simsiam.ele_seg_2.model_fracture_seg import model_S, model_V
from src_ele.pic_opeeration import show_Pic


def adjust_learning_rate(optimizer, init_lr, epoch, all_epoch):
    """Decay the learning rate based on schedule"""
    # args.epochs
    cur_lr = init_lr * (1 + math.cos(math.pi/2 * epoch/all_epoch)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr


def CE_Loss(inputs, target):
    # print('inputs shape:{}, target shape:{}'.format(inputs.shape, target.shape))
    # (batchsize, 1, 224, 224), (batchsize, 224, 224)
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    # temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, 1)
    temp_inputs = inputs.contiguous().view(-1)
    # temp_inputs = inputs.contiguous().view(-1)
    temp_target = target.contiguous().view(-1)
    # print(temp_inputs.shape, temp_target.shape)

    # # print(target.size, target.shape, torch.sum(target))
    # a, b, c = target.shape
    # weight_fracture = (a*b*c)/torch.sum(target).item()
    # # print('weight_fracture:{}, {}, {}'.format(a*b*c, torch.sum(target).item(), weight_fracture))
    # weights = [1.0, weight_fracture]
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # class_weights = torch.FloatTensor(weights).to(device)
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    # # CrossEntropyLoss 函数只能使用 long() 格式的
    # loss_f = nn.CrossEntropyLoss(weight=class_weights)

    loss_f = nn.CrossEntropyLoss()
    loss_t = loss_f(temp_inputs, temp_target)
    return loss_t



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

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def train_up2_by_para(model=model_V(384, 1), batch_size=3, Init_lr=0.002, windows_length=22, base_str='ele_seg_WChipModel_windows', data_path =r'D:\Data\pic_seg_choices\data_train', windows_step=2):
    # ------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=1e-4
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减，可防止过拟合
    #                   adam会导致weight_decay错误，使用adam时建议设置为0。
    # ------------------------------------------------------------------#
    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 0

    # # 设置训练 dataloader 的 batchsize
    # if torch.cuda.is_available():
    #     batch_size = 10
    # else:
    #     batch_size = 2

    # ------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值，默认每个世代都保存
    # ------------------------------------------------------------------#
    save_period = 10

    # Init_lr = 0.002
    # Min_lr = Init_lr * 0.1
    # # 遍历的 窗长大小
    # windows_length = 28
    # # 遍历的 步长大小
    # windows_step = 6
    # # data_path = r'D:\Data\img_seg_data_in\train\1'
    # data_path = r'D:\Data\target_stage3_small_p\pic_seg\1_o_fracture'
    Start_Epoch, Final_Epoch = 0, 20
    print('current model para:\nInit_lr:{}\twindows_length:{}\twindows_step:{}\tepoch:{}-{}\t'.format(
        Init_lr, windows_length, windows_step, Start_Epoch, Final_Epoch))

    # ------------------------------------------------------------------#
    #   num_workers     用于设置是否使用多线程读取数据，1代表关闭多线程
    #                   开启后会加快数据读取速度，但是会占用更多内存
    #                   keras里开启多线程有些时候速度反而慢了许多
    #                   在IO为瓶颈的时候再开启多线程，即GPU运算速度远大于读取图片的速度。
    # ------------------------------------------------------------------#
    num_workers = 0

    # model = WV_model(num_in_dim=384, num_out_dim=1)

    # model = model_V(38, 1)
    # model_str = 'modeel_V_up1'

    # model_path = r'ele_seg_model_batch5_epoch0020.pth.tar'
    # if model_path != '':
    #     print('Load weights {}.'.format(model_path))
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     checkpoint = torch.load(model_path)
    #     model.load_state_dict(checkpoint['state_dict'])
    #     model_dict = model.state_dict()

    model_train = model.train()
    if torch.cuda.is_available():
        # model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()
    else:
        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        device = torch.device("cpu")
        model_train = model_train.to(device)

    if True:
        # ---------------------------------------#
        #   根据optimizer_type选择优化器
        # ---------------------------------------#
        optimizer = {
            'adam': optim.Adam(model.parameters(), Init_lr, betas=(momentum, 0.99), weight_decay=weight_decay),
            # 使用的是这个
            'sgd': optim.SGD(model.parameters(), Init_lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)
        }[optimizer_type]

        train_dataset = dataloader_up2(path=data_path, windows_length=windows_length, windows_step=windows_step)
        print('current train dataset length is:{}'.format(train_dataset.length))
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True)

        # ----开始模型训练-----------------------------------#
        for epoch in range(Start_Epoch, Final_Epoch):
            adjust_learning_rate(optimizer, Init_lr, epoch, Final_Epoch)

            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses = AverageMeter('Loss', ':.6f')
            learning_rate_s = AverageMeter('Learning_Rate', ':.4f')
            progress = ProgressMeter(
                train_dataset.length,
                [batch_time, data_time, losses, learning_rate_s],
                prefix="Epoch: [{}]".format(epoch))

            end = time.time()

            for batch, (pic_all_New, feature_in, feature_in_split, mask_org_split) in enumerate(gen):
                # print('epoch:{},current batch:{}'.format(epoch, batch))
                # print('pic_all_New:{}, feature_in:{}, feature_in_split:{}, mask_org_split:{}'.format(
                #     pic_all_New.shape, feature_in.shape, feature_in_split.shape, mask_org_split.shape))
                # pic_all_new_scaled:[2, 3, 56, 56], feature_in:[2, 1, 8, 56, 56], feature_in_split:[2, 25, 8, 28, 28], mask_org_split:[2, 25, 28, 28]
                data_time.update(time.time() - end)

                num, num_per_pic, dim, w, h = feature_in_split.shape  # b_s, 7*7, 8, win_l, win_s
                x = feature_in_split.reshape(-1, dim, w, h)
                num, num_per_pic, w, h = mask_org_split.shape  # b_s, 7*7, win_l, win_l
                y = mask_org_split.reshape(-1, w, h)

                print(x.shape, y.shape)

                x = x.float()
                y = y.float()
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()

                # print(x.shape, y.shape)
                optimizer.zero_grad()

                outputs = model_train(x)
                loss = CE_Loss(outputs, y)

                loss.backward()
                optimizer.step()

                losses.update(loss.item(), pic_all_New.size(0))
                learning_rate_s.update(optimizer.state_dict()['param_groups'][0]['lr'],
                                       pic_all_New.size(0))  # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                progress.display(batch)

            if (epoch + 1) % save_period == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'windows_length': windows_length,
                    'windows_step': windows_step,
                    'Init_lr': Init_lr,
                    'batch_size': batch_size,
                }, filename='{}_windows_length{}_lr{}_epoch{:04d}.pth.tar'.format(base_str, windows_length, Init_lr,
                                                                                  epoch + 1))


    pass
if __name__ == "__main__":
    if torch.cuda.is_available():
        batch_size = 10
    else:
        batch_size = 3

    train_data_depth = r'D:\Data\pic_seg_choices\data_train'

    # model = model_S(num_in_dim=384, num_out_dim=1)
    # model = model_VL(384, 1)
    model = model_V(8, 1)

    win_len_list = [10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110]
    # win_len_list = [12, 20, 28]


    for i in range(len(win_len_list)):
        train_up2_by_para(model=model, batch_size=2, Init_lr=0.002,
                  windows_length=win_len_list[i], base_str='model_V_up2', data_path=train_data_depth, windows_step=8)
# train_up1_by_para(model=model_V(384, 1), batch_size=3, Init_lr=0.002, windows_length=22, base_str='ele_seg_WChipModel_windows', data_path =r'D:\Data\pic_seg_choices\data_train', windows_step=2)