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
from simsiam.pred_answer_validate import multi_pic_seg_acc_cal
from src_ele.pic_opeeration import show_Pic


if __name__ == "__main__":

    # data_path = r'D:\Data\img_seg_data_in\train\1'
    # data_path = r'D:\Data\target_stage3_small_p\train\1_o_fracture'
    # data_path = r'D:\Data\img_seg_data_in\train\1'
    data_path = r'D:\Data\pic_seg_choices\DATA_NEW_ADD\img'
    Start_Epoch, Final_Epoch = 0, 1


    # ------------------------------------------------------------------#
    #   num_workers     用于设置是否使用多线程读取数据，1代表关闭多线程
    #                   开启后会加快数据读取速度，但是会占用更多内存
    #                   keras里开启多线程有些时候速度反而慢了许多
    #                   在IO为瓶颈的时候再开启多线程，即GPU运算速度远大于读取图片的速度。
    # ------------------------------------------------------------------#
    num_workers = 1

    model = Unet_ele_seg()

    # 载入预训练好的 分割头模型
    # model_path = r'ele_seg_model_batch5_epoch0099.pth.tar'
    model_path = r'ele_seg_model_batch5_epoch0038.pth.tar'
    print('Load weights {}.'.format(model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrained_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(pretrained_dict['state_dict'])


    model_train = model.eval()

    # path_temp = r'D:\Data\target_stage3_small_p\train\1'
    # file_list =
    if True:
        train_dataset = dataloader_ele_seg(path=data_path)
        print('current train dataset length is:{}'.format(train_dataset.length))
        # val_dataset = dataloader_ele_seg(path=data_path)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=2, num_workers=num_workers, pin_memory=True,
                         drop_last=True)

        acc_list_all = []
        miou_list_all = []
        F1_list_all = []
        # ---------------------------------------#
        #   开始查看模型训练效果
        # ---------------------------------------#
        for epoch in range(Start_Epoch, Final_Epoch):

            for batch, (pic_all_New, attn, in_p_1, in_p_2, in_p_4, in_p_8) in enumerate(gen):
                print('epoch:{},current batch:{},data shape is:{}'.format(epoch, batch, pic_all_New.shape))

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

                outputs = model_train(attn, in_p_4, in_p_2, in_p_1)
                pic_all_New = pic_all_New.cpu().detach().numpy()
                pic_mask = pic_all_New[:, -1, :, :].reshape((2, 224, 224))

                outputs = outputs.cpu().detach().numpy().reshape((2, 224, 224))
                # print(outputs.shape)

                acc_list, miou_list, F1_list = multi_pic_seg_acc_cal(outputs, pic_mask)
                print('pic seg miu and acc is:{}, {}, {}'.format(miou_list, acc_list, F1_list))

                acc_list_all.append(acc_list)
                miou_list_all.append(miou_list)
                F1_list_all.append(F1_list)

                # for i in range(outputs.shape[0]):
                #     for j in range(outputs.shape[1]):
                #         for k in range(outputs.shape[2]):
                #             if outputs[i][j][k] < 0.5:
                #                 outputs[i][j][k] = 0
                #             else:
                #                 outputs[i][j][k] = 1
                # show_Pic([pic_all_New[0, 0, :, :]*256, pic_all_New[0, 1, :, :]*256, pic_all_New[0, 2, :, :]*256, outputs[0, :, :]*256], pic_order='22')
                # exit(0)

        acc_list_all = np.array(acc_list_all).ravel()
        miou_list_all = np.array(miou_list_all).ravel()
        F1_list_all = np.array(F1_list_all).ravel()
        print('acc list:{}'.format(miou_list_all))
        print('miou list:{}'.format(miou_list_all))
        print('F1 list:{}'.format(F1_list_all))
        print('final acc mean:{}'.format(np.mean(acc_list_all)))
        print('final miou mean:{}'.format(np.mean(miou_list_all)))
        print('final F1 mean:{}'.format(np.mean(F1_list_all)))