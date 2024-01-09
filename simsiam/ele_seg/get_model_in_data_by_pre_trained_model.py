import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import nn
import torch.nn.functional as F
from simsiam.ele_seg.unet_ele_model import Unet_ele_seg
from simsiam.get_model_in_data import get_picsegmentation_modelindata_from_dyna_stat_pic
from src_ele.dir_operation import traverseFolder
from src_ele.file_operation import get_ele_data_from_path
from src_ele.pic_opeeration import show_Pic


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
    # data_path = r'D:\Data\pic_seg_choices\img_org'
    data_path = r'D:\Data\pic_seg_choices\DATA_NEW_ADD\img'
    Start_Epoch, Final_Epoch = 0, 1


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

    # 载入预训练好的 分割头模型
    model_path = r'ele_seg_model_batch5_epoch0099.pth.tar'
    print('Load weights {}.'.format(model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrained_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(pretrained_dict['state_dict'])


    model_eval = model.eval()
    if torch.cuda.is_available():
        # model_eval = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_eval = model_eval.cuda()
    else:
        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        device = torch.device("cpu")
        model_eval = model.to(device)

    path_temp = r'D:\Data\pic_seg_choices\img_org'
    path_out_t = r'D:\Data\pic_seg_choices\mask_1'
    file_list = traverseFolder(path_temp)
    # print(file_list)
    for i in range(len(file_list)//2):
        # path_dyna_t = file_list[i*2]
        # path_stat_t = path_dyna_t.replace('dyna', 'stat')

        path_temp = file_list[i*2]
        path_temp_stat = ''

        if path_temp.__contains__('dyna'):
            path_temp_stat = path_temp.replace('dyna', 'stat')
        elif path_temp.__contains__('stat'):
            path_temp_stat = path_temp
            path_temp = path_temp_stat.replace('stat', 'dyna')

        print(path_temp, path_temp_stat)
        pic_dyna, depth = get_ele_data_from_path(path_temp)
        pic_stat, depth = get_ele_data_from_path(path_temp_stat)
        shape_org = (pic_dyna.shape[0], pic_stat.shape[1])

        pic_all_New, attn, in_p_1, in_p_2, in_p_4, in_p_8 = get_picsegmentation_modelindata_from_dyna_stat_pic(pic_dyna, pic_stat, depth, pre_enhance=False)

        pic_all_New = torch.from_numpy(pic_all_New).float().reshape(1, 2, 224, 224)
        attn = torch.from_numpy(attn).float().reshape(1, 6, 28, 28)
        in_p_1 = torch.from_numpy(in_p_1).float().reshape(1, 2, 224, 224)
        in_p_2 = torch.from_numpy(in_p_2).float().reshape(1, 2, 112, 112)
        in_p_4 = torch.from_numpy(in_p_4).float().reshape(1, 2, 56, 56)
        in_p_8 = torch.from_numpy(in_p_8).float().reshape(1, 2, 28, 28)
        outputs = model_eval(attn, in_p_4, in_p_2, in_p_1)
        outputs = outputs.cpu().detach().numpy().reshape((224, 224))

        for i in range(outputs.shape[0]):
            for j in range(outputs.shape[1]):
                if outputs[i][j] < 0.5:
                    outputs[i][j] = 0
                else:
                    outputs[i][j] = 1

        # print(outputs.shape)
        outputs = cv2.resize(outputs, (pic_dyna.shape[1], pic_dyna.shape[0]))
        show_Pic([pic_dyna[:, :], pic_stat[:, :], outputs*256], pic_order='13')
        # exit(0)
        cv2.imwrite(path_out_t+'/'+path_temp_stat.split('/')[-1].replace('stat', 'mask'), outputs.astype(int)*256)
