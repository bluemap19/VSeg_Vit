import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from simsiam.ele_seg.dataloader_ele_seg import dataloader_ele_seg
from simsiam.ele_seg.unet_ele_model import Unet_ele_seg


def f_score(inputs, target, beta=1, smooth=1e-5, threhold=0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    # --------------------------------------------#
    #   计算dice系数
    # --------------------------------------------#
    temp_inputs = torch.gt(temp_inputs, threhold).float()
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score


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


    # CrossEntropyLoss 函数只能使用 long() 格式的
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

if __name__ == "__main__":

    # data_path = r'D:\Data\img_seg_data_in\train\1'
    data_path = r'D:\Data\target_stage3_small_p\train\1_o_fracture'
    Start_Epoch, Final_Epoch = 0, 200

    # ------------------------------------------------------------------#
    #   其它训练参数：学习率、优化器、学习率下降有关
    # ------------------------------------------------------------------#
    # ------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #                   当使用Adam优化器时建议设置  Init_lr=1e-4
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    # ------------------------------------------------------------------#
    Init_lr = 1e-2
    Min_lr = Init_lr * 0.1

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



    # 设置训练 dataloader 的 batchsize
    if torch.cuda.is_available():
        batch_size = 3
    else:
        batch_size = 5


    # ------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有'step'、'cos'
    # ------------------------------------------------------------------#
    lr_decay_type = 'cos'

    # ------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值，默认每个世代都保存
    # ------------------------------------------------------------------#
    save_period = 1

    # ------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    # ------------------------------------------------------------------#
    save_dir = 'logs'


    # ------------------------------------------------------------------#
    #   num_workers     用于设置是否使用多线程读取数据，1代表关闭多线程
    #                   开启后会加快数据读取速度，但是会占用更多内存
    #                   keras里开启多线程有些时候速度反而慢了许多
    #                   在IO为瓶颈的时候再开启多线程，即GPU运算速度远大于读取图片的速度。
    # ------------------------------------------------------------------#
    num_workers = 1

    model = Unet_ele_seg()


    # if not pretrained:
    #     weights_init(model)
    # if model_path != '':
    #     # ------------------------------------------------------#
    #     #   权值文件请看README，百度网盘下载
    #     # ------------------------------------------------------#
    #     print('Load weights {}.'.format(model_path))
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     model_dict = model.state_dict()
    #     pretrained_dict = torch.load(model_path, map_location=device)
    #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    #     model_dict.update(pretrained_dict)
    #     model.load_state_dict(model_dict)


    model_train = model.train()
    if torch.cuda.is_available():
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()
    else:
        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        device = torch.device("cpu")
        model_train = model.to(device)


    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Interval_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#
    if True:
        # ---------------------------------------#
        #   根据optimizer_type选择优化器
        # ---------------------------------------#
        optimizer = {
            'adam': optim.Adam(model.parameters(), Init_lr, betas=(momentum, 0.999), weight_decay=weight_decay),              # 使用的是这个
            'sgd': optim.SGD(model.parameters(), Init_lr, momentum=momentum, nesterov=True,weight_decay=weight_decay)
        }[optimizer_type]


        train_dataset = dataloader_ele_seg(path=data_path)
        print('current train dataset length is:{}'.format(train_dataset.length))
        # val_dataset = dataloader_ele_seg(path=data_path)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True)
        # gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
        #                      drop_last=True, collate_fn=unet_dataset_collate)


        # ---------------------------------------#
        #   开始模型训练
        # ---------------------------------------#
        for epoch in range(Start_Epoch, Final_Epoch):

            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses = AverageMeter('Loss', ':.6f')
            learning_rate_s = AverageMeter('Learning_Rate', ':.4f')
            progress = ProgressMeter(
                train_dataset.length,
                [batch_time, data_time, losses, learning_rate_s],
                prefix="Epoch: [{}]".format(epoch))

            end = time.time()

            for batch, (pic_all_New, attn, in_p_1, in_p_2, in_p_4, in_p_8) in enumerate(gen):
                print('epoch:{},current batch:{},data shape is:{}'.format(epoch, batch, pic_all_New.shape))
                print('pic_all_New:{}, attn:{}, in_p_1:{}, in_p_2:{}, in_p_4:{}'.format(pic_all_New.shape, attn.shape, in_p_1.shape, in_p_2.shape, in_p_4.shape))
                data_time.update(time.time() - end)

                # pic_all_New = torch.from_numpy(pic_all_New).type(torch.FloatTensor)
                # attn = torch.from_numpy(attn).type(torch.FloatTensor)
                # in_p_1 = torch.from_numpy(in_p_1).type(torch.FloatTensor)
                # in_p_2 = torch.from_numpy(in_p_2).type(torch.FloatTensor)
                # in_p_4 = torch.from_numpy(in_p_4).type(torch.FloatTensor)
                # in_p_8 = torch.from_numpy(in_p_8).type(torch.FloatTensor)

                # pic_all_New = (pic_all_New).type(torch.FloatTensor)
                # attn = (attn).type(torch.FloatTensor)
                # in_p_1 = (in_p_1).type(torch.FloatTensor)
                # in_p_2 = (in_p_2).type(torch.FloatTensor)
                # in_p_4 = (in_p_4).type(torch.FloatTensor)
                # in_p_8 = (in_p_8).type(torch.FloatTensor)

                pic_all_New = pic_all_New.float()
                attn = attn.float()
                in_p_1 = in_p_1.float()
                in_p_2 = in_p_2.float()
                in_p_4 = in_p_4.float()
                in_p_8 = in_p_8.float()


                if torch.cuda.is_available():
                    pic_all_New = pic_all_New.cuda()
                    attn = attn.cuda()
                    in_p_1 = in_p_1.cuda()
                    in_p_2 = in_p_2.cuda()
                    in_p_4 = in_p_4.cuda()
                    in_p_8 = in_p_8.cuda()

                optimizer.zero_grad()

                outputs = model_train(attn, in_p_4, in_p_2, in_p_1)
                loss = CE_Loss(outputs, pic_all_New[:, -1, :, :])

                # with torch.no_grad():
                #     # -------------------------------#
                #     #   计算f_score
                #     # -------------------------------#
                #     _f_score = f_score(outputs, labels)

                loss.backward()
                optimizer.step()

                losses.update(loss.item(), pic_all_New.size(0))
                learning_rate_s.update(optimizer.state_dict()['param_groups'][0]['lr'], pic_all_New.size(0))# measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # total_f_score += _f_score.item()

                progress.display(batch)

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename='ele_seg_model_batch{}_epoch{:04d}.pth.tar'.format(batch_size, epoch))

